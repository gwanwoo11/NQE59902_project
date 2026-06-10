"""Reconstruct the IAEA-2D loading map from the reference flux fields.

Group-2 diffusion has NO fission source (chi=[1,0]), so per cell:

    Sigma_a2 = ( Sigma_s12 * phi1 + D2 * laplacian(phi2) ) / phi2

with D2, Sigma_s12 known and identical for all fuel (D2=0.4, Ss12=0.02).
We recover Sigma_a2 per assembly from the interior fine cells, classify into
Fuel1 (0.080) / Fuel2 (0.085) / Fuel2+Rod (0.130), and read reflector/outside
from the flux level. The map is then symmetrised (true IAEA-2D is diagonal-sym).

ANALYSIS ONLY -- prints a proposed map; does not edit iaea2d_reference.py.
"""
from __future__ import annotations

import os
import numpy as np

from iaea2d_reference import MATERIAL_MAP_COARSE as M, MATERIAL_NAMES, SIGMA_A

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "2d_IAEA")

NA, NF = 9, 19              # assemblies, fine cells per assembly
H = 20.0 / NF              # cm, forced by 9*19=171 over 9*20cm
MARGIN = 5                 # exclude this many fine cells at each assembly edge

# known, fuel-independent group-2 constants
D2_FUEL, SS12_FUEL = 0.4, 0.02
# canonical Sigma_a2 -> repo material id
TARGETS = [(0.080, 3, "Fuel1"), (0.085, 1, "Fuel2"), (0.130, 2, "Fuel2+Rod")]


def laplacian(a: np.ndarray) -> np.ndarray:
    lap = np.full_like(a, np.nan)
    lap[1:-1, 1:-1] = (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
                       - 4.0 * a[1:-1, 1:-1])
    return lap / (H * H)


def assembly_view(a: np.ndarray) -> np.ndarray:
    """(171,171) -> (9,9,19,19)."""
    return a.reshape(NA, NF, NA, NF).transpose(0, 2, 1, 3)


def recover_sigma_a2(ph1: np.ndarray, ph2: np.ndarray) -> np.ndarray:
    """Per-assembly median Sigma_a2 from interior cells (NaN where no signal)."""
    lap2 = laplacian(ph2)
    sa2_cell = (SS12_FUEL * ph1 + D2_FUEL * lap2) / ph2
    blk = assembly_view(sa2_cell)            # (9,9,19,19)
    inner = blk[:, :, MARGIN:NF - MARGIN, MARGIN:NF - MARGIN]
    return np.nanmedian(inner.reshape(NA, NA, -1), axis=2)


def main() -> None:
    np.set_printoptions(linewidth=130, precision=4, suppress=True)
    ph1 = np.load(os.path.join(REF, "phione.npy"))
    ph2 = np.load(os.path.join(REF, "phitwo.npy"))
    pw = np.load(os.path.join(REF, "power.npy"))

    # per-assembly mean power & flux (state 0)
    pw_a = assembly_view(pw[0]).mean(axis=(2, 3))
    fl_a = assembly_view(ph1[0]).mean(axis=(2, 3))
    fuel = pw_a > 1e-6

    print("per-assembly mean POWER (state0):"); print(pw_a)
    print("\nper-assembly mean PHI1 (state0)  [reflector vs outside]:"); print(fl_a)

    # Sigma_a2 recovery, all 5 states, then averaged & symmetrised
    sa2_states = np.stack([recover_sigma_a2(ph1[s], ph2[s]) for s in range(0, 10, 2)])
    sa2 = np.nanmean(sa2_states, axis=0)
    sa2_sym = np.where(fuel, 0.5 * (sa2 + sa2.T), np.nan)

    print("\nrecovered Sigma_a2 per FUEL assembly (avg over 5 states, symmetrised):")
    print(np.where(fuel, sa2_sym, np.nan))
    print("\nper-state spread (max abs deviation from mean) over fuel cells:",
          float(np.nanmax(np.abs(sa2_states - sa2), axis=0)[fuel].max()))

    vals = sa2_sym[fuel]
    print(f"\nfuel Sigma_a2 stats: min={np.nanmin(vals):.4f} med={np.nanmedian(vals):.4f} "
          f"max={np.nanmax(vals):.4f}")
    print("histogram (edges 0.07..0.14):")
    h, e = np.histogram(vals, bins=np.arange(0.07, 0.141, 0.005))
    for c, lo in zip(h, e):
        print(f"   {lo:.3f}-{lo+0.005:.3f}: {'#'*int(c)} ({c})")

    # classify
    def classify(x):
        return min(TARGETS, key=lambda t: abs(x - t[0]))[1]

    rec = np.zeros((NA, NA), dtype=np.int64)
    refl_thr = 0.3  # flux level separating reflector (~2.5) from outside (~0.04)
    for i in range(NA):
        for j in range(NA):
            if fuel[i, j]:
                rec[i, j] = classify(sa2_sym[i, j])
            elif fl_a[i, j] > refl_thr:
                rec[i, j] = 4   # reflector
            else:
                rec[i, j] = 0   # outside
    # enforce exact diagonal symmetry on the final ids too (resolve any stragglers)
    rec_sym = rec.copy()
    for i in range(NA):
        for j in range(i + 1, NA):
            if rec[i, j] != rec[j, i]:
                # prefer the fuel/reflector classification with stronger evidence:
                # keep the lower-triangle value (arbitrary but consistent) unless one is outside
                a, b = rec[i, j], rec[j, i]
                pick = max(a, b) if 0 in (a, b) else a
                rec_sym[i, j] = rec_sym[j, i] = pick

    print("\n--- RECONSTRUCTED map (0=Out 1=Fuel1 2=Rod 3=Fuel2 4=Refl) ---"); print(rec_sym)
    print("\n--- CURRENT repo map ---"); print(M)
    print("\ndiff (cells changed):")
    diff = np.argwhere(rec_sym != M)
    for r, c in diff:
        print(f"   [{r},{c}] {int(M[r,c])}({MATERIAL_NAMES[int(M[r,c])]}) -> "
              f"{int(rec_sym[r,c])}({MATERIAL_NAMES[int(rec_sym[r,c])]})")
    print(f"\n#changed = {len(diff)}/81   reconstructed symmetric? "
          f"{bool((rec_sym==rec_sym.T).all())}")
    print(f"fuel footprint == reference? "
          f"{bool((np.isin(rec_sym,[1,2,3])==fuel).all())}")


if __name__ == "__main__":
    main()
