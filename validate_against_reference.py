"""External validation of diffusion_solver.py against an IAEA-2D reference.

This is a READ-ONLY diagnostic. It does NOT modify iaea2d_reference.py.

It cross-checks the in-repo "starter" IAEA representation against two external
references:

  1) The canonical IAEA-2D PWR benchmark spec from the literature
     (two-group XS table + reference eigenvalue).
       - XS table: arXiv:2208.13483 Table III (and matching arXiv:2407.10988).
       - Reference k_eff ~= 1.0296  (FreeFem++ refs report 1.02977 / 1.02959;
         classic ANL Benchmark Problem Book value 1.02959).
       - Geometry: 20 cm assembly pitch, 20 cm water reflector,
         quarter core with diagonal (x<->y) symmetry.

  2) The fine-mesh reference SOLUTION fields the user placed in 2d_IAEA/:
       phione.npy / phitwo.npy : group-1 / group-2 flux,   shape (10,171,171)
       power.npy               : assembly power,            shape (10,171,171)
       obs.npy                 : noisy observations         shape (10,171,171)
     171 = 9 assemblies x 19 fine cells.  The 10 slices are 5 distinct states
     (each stored twice); only `obs` differs within a pair (two noisy
     realizations of the same physical state).

Three checks are reported:
  [A] XS cross-check         repo XS  vs  canonical IAEA-2D XS
  [B] geometry / symmetry    repo map vs  reference footprint & diagonal symmetry
  [C] solver behaviour       k_eff mesh-convergence -> canonical k_eff, and
                             flux/power field agreement with the reference.
"""

from __future__ import annotations

import os

import numpy as np
from scipy.sparse.linalg import splu

from diffusion_solver import SolverConfig, solve_two_group_diffusion, _assemble_group_operator
from iaea2d_reference import (
    MATERIAL_MAP_COARSE,
    MATERIAL_NAMES,
    build_xs_fields,
    DIFFUSION_COEFF,
    SIGMA_A,
    NU_SIGMA_F,
    SIGMA_S_21,
    CHI,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(HERE, "2d_IAEA")
OUT_DIR = os.path.join(HERE, "results")

# Canonical IAEA-2D benchmark, indexed by region Omega_1..Omega_4.
# [D1, D2, Sa1, Sa2, nuSf1, nuSf2, Ss_1->2]
CANON = {
    "Omega1 (Fuel1)":      [1.5, 0.4, 0.01, 0.080, 0.0, 0.135, 0.02],
    "Omega2 (Fuel2)":      [1.5, 0.4, 0.01, 0.085, 0.0, 0.135, 0.02],
    "Omega3 (Fuel2+Rod)":  [1.5, 0.4, 0.01, 0.130, 0.0, 0.135, 0.02],
    "Omega4 (Reflector)":  [2.0, 0.3, 0.00, 0.010, 0.0, 0.000, 0.04],
}
# Which repo material id implements each canonical region (matched by Sa2).
CANON_TO_REPO_ID = {
    "Omega1 (Fuel1)": 3,       # repo FUEL2   (Sa2=0.080)
    "Omega2 (Fuel2)": 1,       # repo FUEL1   (Sa2=0.085)
    "Omega3 (Fuel2+Rod)": 2,   # repo FUEL1_ROD (Sa2=0.130)
    "Omega4 (Reflector)": 4,   # repo REFLECTOR
}
CANON_KEFF = 1.0296

N_ASSEMBLY = 9
N_FINE = 19            # fine cells per assembly in the reference (9*19 = 171)
PITCH_CM = 20.0


def hr(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


# ---------------------------------------------------------------------------
# Fast power-iteration eigensolver.
# Uses diffusion_solver's EXACT operator assembly, but factorises L1/L2 once
# (they are constant across power iterations) so we can afford the 171x171
# mesh. Self-checked against solve_two_group_diffusion at the coarse mesh.
# ---------------------------------------------------------------------------
def fast_eigensolve(material_map, d, sigma_a, nu_sigma_f, sigma_s21, chi, cfg):
    nx, ny = material_map.shape
    n = nx * ny
    L1 = _assemble_group_operator(d[0], sigma_a[0] + sigma_s21, material_map, cfg).tocsc()
    L2 = _assemble_group_operator(d[1], sigma_a[1], material_map, cfg).tocsc()
    lu1, lu2 = splu(L1), splu(L2)

    active = material_map.reshape(-1) != 0
    phi1 = np.where(active, 1.0, 0.0)
    phi2 = np.where(active, 1.0, 0.0)
    nf0, nf1 = nu_sigma_f[0].reshape(-1), nu_sigma_f[1].reshape(-1)
    s21 = sigma_s21.reshape(-1)

    k = 1.0
    f_prev = nf0 * phi1 + nf1 * phi2
    f_prev_sum = f_prev[active].sum()
    it = 0
    for it in range(1, cfg.max_iters + 1):
        rhs1 = (chi[0] / k) * f_prev
        rhs2 = (chi[1] / k) * f_prev + s21 * phi1
        rhs1[~active] = 0.0
        rhs2[~active] = 0.0
        p1, p2 = lu1.solve(rhs1), lu2.solve(rhs2)
        p1[~active] = 0.0
        p2[~active] = 0.0

        f_new = nf0 * p1 + nf1 * p2
        f_new_sum = f_new[active].sum()
        k_new = k * (f_new_sum / f_prev_sum)

        norm = f_new_sum if f_new_sum > 0 else 1.0
        p1 /= norm
        p2 /= norm
        f_new /= norm
        f_new_sum /= norm

        fd = max(
            np.linalg.norm(p1 - phi1) / (np.linalg.norm(p1) + 1e-20),
            np.linalg.norm(p2 - phi2) / (np.linalg.norm(p2) + 1e-20),
        )
        kd = abs(k_new - k) / max(abs(k_new), 1e-20)
        phi1, phi2, f_prev, f_prev_sum, k = p1, p2, f_new, f_new_sum, k_new
        if kd < cfg.tol_keff and fd < cfg.tol_flux:
            break
    return float(k), phi1.reshape(nx, ny), phi2.reshape(nx, ny), it


def refine_map(material_map: np.ndarray, factor: int) -> np.ndarray:
    """Upsample the coarse assembly map to `factor` x `factor` fine cells each."""
    return np.kron(material_map, np.ones((factor, factor), dtype=material_map.dtype))


def solve_refined(material_map_coarse: np.ndarray, factor: int):
    fine = refine_map(material_map_coarse, factor)
    xs = build_xs_fields(fine)
    cfg = SolverConfig(h_cm=PITCH_CM / factor)
    k, p1, p2, iters = fast_eigensolve(
        fine, xs["D"], xs["Sigma_a"], xs["nuSigma_f"], xs["Sigma_s21"], xs["chi"], cfg
    )
    return k, p1, p2, iters, fine


def norm_mask(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Normalise to unit mean over `mask` (handles eigenvector scale freedom)."""
    m = field[mask].mean()
    return field / m if m != 0 else field


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    np.set_printoptions(linewidth=120, precision=3, suppress=True)

    obs = np.load(os.path.join(REF_DIR, "obs.npy"))
    ph1 = np.load(os.path.join(REF_DIR, "phione.npy"))
    ph2 = np.load(os.path.join(REF_DIR, "phitwo.npy"))
    pw = np.load(os.path.join(REF_DIR, "power.npy"))
    M = MATERIAL_MAP_COARSE

    # ---------------------------------------------------------------- [A] XS
    hr("[A] Cross-section cross-check  (repo  vs  canonical IAEA-2D)")
    repo_tab = {
        1: ("FUEL1", DIFFUSION_COEFF[1], SIGMA_A[1], NU_SIGMA_F[1], SIGMA_S_21[1]),
        2: ("FUEL1_ROD", DIFFUSION_COEFF[2], SIGMA_A[2], NU_SIGMA_F[2], SIGMA_S_21[2]),
        3: ("FUEL2", DIFFUSION_COEFF[3], SIGMA_A[3], NU_SIGMA_F[3], SIGMA_S_21[3]),
        4: ("REFLECTOR", DIFFUSION_COEFF[4], SIGMA_A[4], NU_SIGMA_F[4], SIGMA_S_21[4]),
    }
    all_match = True
    for region, vals in CANON.items():
        rid = CANON_TO_REPO_ID[region]
        name, D, Sa, nSf, Ss = repo_tab[rid]
        repo_vec = [D[0], D[1], Sa[0], Sa[1], nSf[0], nSf[1], Ss]
        ok = np.allclose(repo_vec, vals, atol=1e-9)
        all_match &= ok
        print(f"  {region:20s} <-> repo id{rid} {name:10s} : {'MATCH' if ok else 'DIFFER'}")
        if not ok:
            print(f"      canonical {vals}")
            print(f"      repo      {repo_vec}")
    print(f"\n  chi: repo {CHI.tolist()} vs canonical [1.0, 0.0] : "
          f"{'MATCH' if np.allclose(CHI,[1,0]) else 'DIFFER'}")
    print(f"  ==> XS VALUES {'all match the canonical IAEA-2D set' if all_match else 'DIFFER'} "
          "(repo Fuel1/Fuel2 naming is swapped vs canonical Omega1/Omega2; values identical).")

    # -------------------------------------------------------- [B] geometry
    hr("[B] Geometry / symmetry  (repo map  vs  reference solution)")
    # reference fuel footprint from power (power>0 <=> fissile cell)
    fuel_ref = (pw[0].reshape(N_ASSEMBLY, N_FINE, N_ASSEMBLY, N_FINE).mean(axis=(1, 3)) > 1e-9)
    fuel_repo = np.isin(M, [1, 2, 3])
    mism = np.argwhere(fuel_ref != fuel_repo)
    print(f"  fuel footprint agreement      : {int((fuel_ref == fuel_repo).sum())}/81 cells")
    for r, c in mism:
        print(f"     mismatch [{r},{c}] repo id={int(M[r,c])} ({MATERIAL_NAMES[int(M[r,c])]}), "
              f"reference power={'>0 (fuel)' if fuel_ref[r,c] else '0 (non-fuel)'}")

    asym = [(r, c) for r, c in np.argwhere(M != M.T) if r < c]
    print(f"\n  repo map diagonal symmetry    : {len(asym)} asymmetric mirror-pairs "
          f"(a true IAEA-2D quarter core is diagonally symmetric)")
    for r, c in asym:
        print(f"     [{r},{c}]={int(M[r,c])}({MATERIAL_NAMES[int(M[r,c])]})  !=  "
              f"[{c},{r}]={int(M[c,r])}({MATERIAL_NAMES[int(M[c,r])]})")
    sym_err = float(np.abs(ph1[0] - ph1[0].T).max())
    print(f"  reference flux symmetry       : max|phi1 - phi1^T| = {sym_err:.4g} "
          "(~0 => reference is diagonally symmetric)")

    # domain check: does the reference carry flux where repo marks id==0 (outside)?
    ph1_assembly = ph1[0].reshape(N_ASSEMBLY, N_FINE, N_ASSEMBLY, N_FINE).mean(axis=(1, 3))
    refl_cells = (M == 4)
    out_cells = (M == 0)
    refl_level = ph1_assembly[refl_cells].mean()
    print(f"\n  reference mean phi1 over repo REFLECTOR cells : {refl_level:8.4f}")
    print(f"  reference mean phi1 over repo OUTSIDE  cells : {ph1_assembly[out_cells].mean():8.4f}"
          f"   ({int(out_cells.sum())} cells)")
    print("  ==> reference flux is non-zero in the repo 'outside' triangle, i.e. the reference")
    print("      solves the FULL square (those cells are reflector), whereas the repo solver")
    print("      excludes id==0 cells (phi=0). Domain shapes differ in the far corner.")

    # -------------------------------------------------------- [C] solver
    hr("[C] Solver behaviour  (k_eff convergence & field agreement)")
    # self-check: fast solver == repo solver at coarse mesh
    xs0 = build_xs_fields(M)
    k_repo, _, _ = solve_two_group_diffusion(
        M, xs0["D"], xs0["Sigma_a"], xs0["nuSigma_f"], xs0["Sigma_s21"], xs0["chi"]
    )
    k_fast, _, _, _ = fast_eigensolve(
        M, xs0["D"], xs0["Sigma_a"], xs0["nuSigma_f"], xs0["Sigma_s21"], xs0["chi"], SolverConfig()
    )
    print(f"  self-check: repo solve_two_group_diffusion k_eff = {k_repo:.8f}")
    print(f"              fast_eigensolve (same operator)  k_eff = {k_fast:.8f}  "
          f"(|diff|={abs(k_repo-k_fast):.2e})")

    print(f"\n  k_eff mesh-convergence (repo loading map+XS) toward canonical {CANON_KEFF}:")
    print(f"     {'cells/assy':>10} {'mesh h(cm)':>10} {'grid':>9} {'k_eff':>12} "
          f"{'k-1 (pcm)':>10} {'iters':>6}")
    fine_sol = None
    for factor in (1, 2, 4, 8, N_FINE):
        k, p1, p2, iters, fine = solve_refined(M, factor)
        g = fine.shape[0]
        print(f"     {factor:>10} {PITCH_CM/factor:>10.3f} {g:>4}x{g:<4} "
              f"{k:>12.6f} {(k-1)*1e5:>10.0f} {iters:>6}")
        if factor == N_FINE:
            fine_sol = (k, p1, p2, fine)

    # field comparison at the matched 171x171 mesh
    k_fine, p1_fine, p2_fine, fine_map = fine_sol
    pw_solver = NU_SIGMA_F[0][0] * 0 + (build_xs_fields(fine_map)["nuSigma_f"][1] * p2_fine
                                        + build_xs_fields(fine_map)["nuSigma_f"][0] * p1_fine)
    active_fine = fine_map != 0
    fuel_fine = build_xs_fields(fine_map)["nuSigma_f"][1] > 0  # fissile cells

    # compare normalised flux to each of the 5 distinct reference states
    print("\n  171x171 field agreement vs each reference state (relative L2 over active cells):")
    print(f"     {'state':>6} {'phi1 relL2':>12} {'phi2 relL2':>12} {'power relL2':>12}")
    best = None
    for s in range(0, 10, 2):
        a1 = norm_mask(ph1[s], active_fine)
        a2 = norm_mask(ph2[s], active_fine)
        ap = norm_mask(pw[s], fuel_fine)
        b1 = norm_mask(p1_fine, active_fine)
        b2 = norm_mask(p2_fine, active_fine)
        bp = norm_mask(pw_solver, fuel_fine)
        e1 = np.linalg.norm((a1 - b1)[active_fine]) / np.linalg.norm(a1[active_fine])
        e2 = np.linalg.norm((a2 - b2)[active_fine]) / np.linalg.norm(a2[active_fine])
        ep = np.linalg.norm((ap - bp)[fuel_fine]) / np.linalg.norm(ap[fuel_fine])
        print(f"     {s:>6} {e1:>12.4f} {e2:>12.4f} {ep:>12.4f}")
        if best is None or ep < best[1]:
            best = (s, ep, e1, e2)
    print(f"  ==> closest reference state = #{best[0]} (power relL2={best[1]:.4f}, "
          f"phi1={best[2]:.4f}, phi2={best[3]:.4f})")

    # ----------------------------------------------------------- plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        s = best[0]
        ref_p = norm_mask(pw[s], fuel_fine)
        sol_p = norm_mask(pw_solver, fuel_fine)
        ref_p = np.where(fuel_fine, ref_p, np.nan)
        sol_p = np.where(fuel_fine, sol_p, np.nan)
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
        im0 = ax[0].imshow(ref_p, cmap="inferno"); ax[0].set_title(f"reference power (state #{s})")
        im1 = ax[1].imshow(sol_p, cmap="inferno"); ax[1].set_title(f"repo solver power (171x171, k={k_fine:.5f})")
        im2 = ax[2].imshow(sol_p - ref_p, cmap="coolwarm"); ax[2].set_title("solver - reference")
        for a, im in zip(ax, (im0, im1, im2)):
            a.set_xticks([]); a.set_yticks([]); fig.colorbar(im, ax=a, fraction=0.046)
        fig.tight_layout()
        out = os.path.join(OUT_DIR, "reference_validation.png")
        fig.savefig(out, dpi=110)
        print(f"\n  wrote {out}")
    except Exception as exc:  # plotting is optional
        print(f"\n  (plot skipped: {exc})")

    # ----------------------------------------------------------- verdict
    hr("SUMMARY")
    print(f"  [A] XS values        : {'MATCH canonical IAEA-2D' if all_match else 'DIFFER'}")
    print(f"  [B] fuel footprint   : {int((fuel_ref==fuel_repo).sum())}/81  | "
          f"repo map diagonal asymmetries: {len(asym)} pair(s)  | "
          f"reference symmetric (err {sym_err:.3g})")
    print(f"  [C] k_eff(coarse 9x9)= {k_repo:.5f}   k_eff(fine 171)= {k_fine:.5f}   "
          f"canonical ~ {CANON_KEFF}")
    print(f"      best field match : state #{best[0]}  power relL2 = {best[1]:.4f}")


if __name__ == "__main__":
    main()
