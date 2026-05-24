"""2-group 2D diffusion solver (FDM 5-point + power iteration)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

BoundaryType = Literal["vacuum", "reflective"]


@dataclass(frozen=True)
class SolverConfig:
    h_cm: float = 20.0
    max_iters: int = 300
    tol_keff: float = 1.0e-10
    tol_flux: float = 1.0e-8
    # User-confirmed quarter-core boundary setting.
    bc_left: BoundaryType = "reflective"
    bc_top: BoundaryType = "reflective"
    bc_right: BoundaryType = "vacuum"
    bc_bottom: BoundaryType = "vacuum"


def _idx(i: int, j: int, ny: int) -> int:
    return i * ny + j


def _harmonic_mean(a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return 2.0 * a * b / (a + b)


def _assemble_group_operator(
    d_group: np.ndarray,
    sigma_remove: np.ndarray,
    material_map: np.ndarray,
    cfg: SolverConfig,
) -> csr_matrix:
    nx, ny = d_group.shape
    n = nx * ny
    h2 = cfg.h_cm * cfg.h_cm
    a = lil_matrix((n, n), dtype=np.float64)

    for i in range(nx):
        for j in range(ny):
            p = _idx(i, j, ny)
            if material_map[i, j] == 0:
                a[p, p] = 1.0
                continue

            diag = sigma_remove[i, j]
            d_c = d_group[i, j]

            if j > 0:
                if material_map[i, j - 1] != 0:
                    c = _harmonic_mean(d_c, d_group[i, j - 1]) / h2
                    diag += c
                    a[p, _idx(i, j - 1, ny)] = -c
                else:
                    diag += d_c / h2
            elif cfg.bc_left == "vacuum":
                diag += d_c / h2

            if j < ny - 1:
                if material_map[i, j + 1] != 0:
                    c = _harmonic_mean(d_c, d_group[i, j + 1]) / h2
                    diag += c
                    a[p, _idx(i, j + 1, ny)] = -c
                else:
                    diag += d_c / h2
            elif cfg.bc_right == "vacuum":
                diag += d_c / h2

            if i > 0:
                if material_map[i - 1, j] != 0:
                    c = _harmonic_mean(d_c, d_group[i - 1, j]) / h2
                    diag += c
                    a[p, _idx(i - 1, j, ny)] = -c
                else:
                    diag += d_c / h2
            elif cfg.bc_top == "vacuum":
                diag += d_c / h2

            if i < nx - 1:
                if material_map[i + 1, j] != 0:
                    c = _harmonic_mean(d_c, d_group[i + 1, j]) / h2
                    diag += c
                    a[p, _idx(i + 1, j, ny)] = -c
                else:
                    diag += d_c / h2
            elif cfg.bc_bottom == "vacuum":
                diag += d_c / h2

            a[p, p] = diag

    return a.tocsr()


def solve_two_group_diffusion(
    material_map: np.ndarray,
    d: np.ndarray,
    sigma_a: np.ndarray,
    nu_sigma_f: np.ndarray,
    sigma_s21: np.ndarray,
    chi: np.ndarray,
    cfg: SolverConfig | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    cfg = cfg or SolverConfig()
    nx, ny = material_map.shape
    n = nx * ny

    sigma_r1 = sigma_a[0] + sigma_s21
    sigma_r2 = sigma_a[1]
    l1 = _assemble_group_operator(d[0], sigma_r1, material_map, cfg)
    l2 = _assemble_group_operator(d[1], sigma_r2, material_map, cfg)

    active = material_map.reshape(-1) != 0
    phi1 = np.ones(n, dtype=np.float64)
    phi2 = np.ones(n, dtype=np.float64)
    phi1[~active] = 0.0
    phi2[~active] = 0.0

    k = 1.0
    f_prev = nu_sigma_f[0].reshape(-1) * phi1 + nu_sigma_f[1].reshape(-1) * phi2
    f_prev_sum = np.sum(f_prev[active])

    for _ in range(cfg.max_iters):
        rhs1 = (chi[0] / k) * f_prev
        rhs2 = (chi[1] / k) * f_prev + sigma_s21.reshape(-1) * phi1
        rhs1[~active] = 0.0
        rhs2[~active] = 0.0

        phi1_new = spsolve(l1, rhs1)
        phi2_new = spsolve(l2, rhs2)
        phi1_new[~active] = 0.0
        phi2_new[~active] = 0.0

        f_new = nu_sigma_f[0].reshape(-1) * phi1_new + nu_sigma_f[1].reshape(-1) * phi2_new
        f_new_sum = np.sum(f_new[active])
        k_new = k * (f_new_sum / f_prev_sum)

        norm = f_new_sum if f_new_sum > 0.0 else 1.0
        phi1_new /= norm
        phi2_new /= norm
        f_new /= norm
        f_new_sum /= norm

        flux_delta = max(
            np.linalg.norm(phi1_new - phi1) / (np.linalg.norm(phi1_new) + 1.0e-20),
            np.linalg.norm(phi2_new - phi2) / (np.linalg.norm(phi2_new) + 1.0e-20),
        )
        k_delta = abs(k_new - k) / max(abs(k_new), 1.0e-20)

        phi1, phi2 = phi1_new, phi2_new
        f_prev = f_new
        f_prev_sum = f_new_sum
        k = k_new

        if k_delta < cfg.tol_keff and flux_delta < cfg.tol_flux:
            break

    return float(k), phi1.reshape(nx, ny), phi2.reshape(nx, ny)
