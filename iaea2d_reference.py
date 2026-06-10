"""IAEA 2D PWR quarter-core reference (numpy-only, simple arrays).

This file stores:
1) A coarse material loading map for the IAEA-2D quarter core.
2) Two-group cross sections (canonical IAEA-2D values; see validation note).

Note:
- The two-group XS below match the canonical IAEA-2D benchmark exactly
  (arXiv:2208.13483 Table III); reference k_eff ~= 1.0296.
- `MATERIAL_MAP_COARSE` was reconstructed (2026-06) from the external IAEA-2D
  reference flux fields in `2d_IAEA/` by inverting the source-free group-2
  diffusion balance, Sigma_a2 = (Sigma_s12*phi1 + D2*lap(phi2)) / phi2, for the
  per-assembly Sigma_a2, then enforcing diagonal (quarter-core) symmetry. Versus
  the earlier hand-drawn starter map this fixed three diagonal-symmetry
  violations (pairs [2,6]/[6,2], [3,7]/[7,3], [4,8]/[8,4]). When the recovery is
  aligned to the correct half-width symmetry-boundary mesh (see below), the four
  control rods at [0,0], [0,4], [4,0], [4,4] all recover cleanly at Sigma_a2 ~=
  0.130. See `reconstruct_material_map.py` and `validate_against_reference.py`.

GEOMETRY NOTE (open issue): the IAEA-2D quarter core has HALF-width assemblies
  on the two reflective (symmetry) edges and a QUARTER assembly at the [0,0]
  corner, because the full core is odd-by-odd and the centre row/column are
  shared. `diffusion_solver.py` currently uses a UNIFORM mesh (every map cell =
  full 20 cm), which over-sizes the central region and flattens the radial power
  (~28% L2 vs the reference). Modelling the half/quarter boundary cells
  collapses that error to ~1.6%. Fixing it requires a non-uniform mesh in the
  solver (and regenerating the training data).
"""

from __future__ import annotations

import numpy as np

# -----------------------------------------------------------------------------
# Material IDs
# -----------------------------------------------------------------------------
# 0 is reserved for "outside / unused" cells in the coarse map.
OUTSIDE = 0
FUEL1 = 1
FUEL1_ROD = 2
FUEL2 = 3
REFLECTOR = 4

MATERIAL_NAMES = {
    OUTSIDE: "Outside",
    FUEL1: "Fuel1",
    FUEL1_ROD: "Fuel1+Rod",
    FUEL2: "Fuel2",
    REFLECTOR: "Reflector",
}


# -----------------------------------------------------------------------------
# Coarse loading pattern (top row -> bottom row), reconstructed from the
# 2d_IAEA/ reference and symmetrised. [0,0] is the core centre (reflective/
# reflective corner); the far bottom-right is the vacuum corner.
# -----------------------------------------------------------------------------
# Legend (id -> repo name -> canonical IAEA-2D region, Sigma_a2):
#   0: Outside   (excluded from the solve, phi=0)
#   1: Fuel1     -> canonical Omega2,      Sigma_a2 = 0.085
#   2: Fuel1+Rod -> canonical Omega3 (rod), Sigma_a2 = 0.130
#   3: Fuel2     -> canonical Omega1,      Sigma_a2 = 0.080
#   4: Reflector -> canonical Omega4,      Sigma_a2 = 0.010
# (repo Fuel1/Fuel2 naming is swapped vs the canonical Omega1/Omega2 labels;
#  the XS *values* per id are correct -- see SIGMA_A below.)
MATERIAL_MAP_COARSE = np.array(
    [
        [2, 1, 1, 1, 2, 1, 1, 3, 4],
        [1, 1, 1, 1, 1, 1, 1, 3, 4],
        [1, 1, 1, 1, 1, 1, 3, 3, 4],
        [1, 1, 1, 1, 1, 1, 3, 4, 4],
        [2, 1, 1, 1, 2, 3, 3, 4, 4],
        [1, 1, 1, 1, 3, 3, 4, 4, 0],
        [1, 1, 3, 3, 3, 4, 4, 0, 0],
        [3, 3, 3, 4, 4, 4, 0, 0, 0],
        [4, 4, 4, 4, 4, 0, 0, 0, 0],
    ],
    dtype=np.int64,
)


# -----------------------------------------------------------------------------
# Two-group constants (from the table in the screenshot)
# -----------------------------------------------------------------------------
# Index order uses material IDs above (0..4). Row 0 (OUTSIDE) is zeros.
#
# Per-group array order:
#   [group1, group2]
#
# Scattering is defined as down-scattering Sigma_s,2<-1 only.
DIFFUSION_COEFF = np.array(
    [
        [0.0, 0.0],   # OUTSIDE
        [1.5, 0.4],   # FUEL1
        [1.5, 0.4],   # FUEL1_ROD
        [1.5, 0.4],   # FUEL2
        [2.0, 0.3],   # REFLECTOR
    ],
    dtype=np.float64,
)

SIGMA_A = np.array(
    [
        [0.0, 0.0],    # OUTSIDE
        [0.010, 0.085],  # FUEL1
        [0.010, 0.130],  # FUEL1_ROD
        [0.010, 0.080],  # FUEL2
        [0.000, 0.010],  # REFLECTOR
    ],
    dtype=np.float64,
)

NU_SIGMA_F = np.array(
    [
        [0.0, 0.0],    # OUTSIDE
        [0.000, 0.135],  # FUEL1
        [0.000, 0.135],  # FUEL1_ROD
        [0.000, 0.135],  # FUEL2
        [0.000, 0.000],  # REFLECTOR
    ],
    dtype=np.float64,
)

SIGMA_S_21 = np.array(
    [
        0.0,   # OUTSIDE
        0.020,  # FUEL1
        0.020,  # FUEL1_ROD
        0.020,  # FUEL2
        0.040,  # REFLECTOR
    ],
    dtype=np.float64,
)

# Common two-group assumption in this benchmark style.
CHI = np.array([1.0, 0.0], dtype=np.float64)


def get_base_xs_tables() -> dict[str, np.ndarray]:
    """Return immutable copies of reference XS tables."""
    return {
        "D_table": DIFFUSION_COEFF.copy(),
        "Sigma_a_table": SIGMA_A.copy(),
        "nuSigma_f_table": NU_SIGMA_F.copy(),
        "Sigma_s21_table": SIGMA_S_21.copy(),
        "chi": CHI.copy(),
    }


def build_xs_fields_from_tables(
    material_map: np.ndarray,
    d_table: np.ndarray,
    sigma_a_table: np.ndarray,
    nu_sigma_f_table: np.ndarray,
    sigma_s21_table: np.ndarray,
    chi: np.ndarray,
) -> dict[str, np.ndarray]:
    """Convert a map + XS tables into cell-wise 2-group fields."""
    m = material_map.astype(np.int64)
    d = np.stack([d_table[m, 0], d_table[m, 1]], axis=0)
    sa = np.stack([sigma_a_table[m, 0], sigma_a_table[m, 1]], axis=0)
    nsf = np.stack([nu_sigma_f_table[m, 0], nu_sigma_f_table[m, 1]], axis=0)
    s21 = sigma_s21_table[m]
    return {
        "D": d,
        "Sigma_a": sa,
        "nuSigma_f": nsf,
        "Sigma_s21": s21,
        "chi": chi.copy(),
    }


def build_xs_fields(material_map: np.ndarray) -> dict[str, np.ndarray]:
    """Convert an integer material map into cell-wise XS fields.

    Returns a dictionary with:
      - D: shape (2, H, W)
      - Sigma_a: shape (2, H, W)
      - nuSigma_f: shape (2, H, W)
      - Sigma_s21: shape (H, W)
      - chi: shape (2,)
    """
    return build_xs_fields_from_tables(
        material_map=material_map,
        d_table=DIFFUSION_COEFF,
        sigma_a_table=SIGMA_A,
        nu_sigma_f_table=NU_SIGMA_F,
        sigma_s21_table=SIGMA_S_21,
        chi=CHI,
    )


if __name__ == "__main__":
    xs = build_xs_fields(MATERIAL_MAP_COARSE)
    print("material_map shape:", MATERIAL_MAP_COARSE.shape)
    print("D shape:", xs["D"].shape)
    print("Sigma_a shape:", xs["Sigma_a"].shape)
    print("nuSigma_f shape:", xs["nuSigma_f"].shape)
    print("Sigma_s21 shape:", xs["Sigma_s21"].shape)
