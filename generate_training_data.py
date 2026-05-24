"""Generate random diffusion training data from reference XS."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from diffusion_solver import SolverConfig, solve_two_group_diffusion
from iaea2d_reference import (
    MATERIAL_MAP_COARSE,
    build_xs_fields,
    build_xs_fields_from_tables,
    get_base_xs_tables,
)


def _perturb_positive(values: np.ndarray, rel: float, rng: np.random.Generator) -> np.ndarray:
    scale = 1.0 + rng.uniform(-rel, rel, size=values.shape)
    out = values * scale
    out = np.where(values > 0.0, np.maximum(out, 1.0e-12), 0.0)
    return out


def _build_random_material_map(rng: np.random.Generator) -> np.ndarray:
    m = MATERIAL_MAP_COARSE.copy()
    active = m != 0
    # Randomly assign one of 4 materials for active cells.
    m[active] = rng.integers(1, 5, size=np.count_nonzero(active))
    return m


def _build_random_xs(material_map: np.ndarray, rel_perturb: float, rng: np.random.Generator) -> dict[str, np.ndarray]:
    base = get_base_xs_tables()
    d_table = _perturb_positive(base["D_table"], rel_perturb, rng)
    sa_table = _perturb_positive(base["Sigma_a_table"], rel_perturb, rng)
    nsf_table = _perturb_positive(base["nuSigma_f_table"], rel_perturb, rng)
    s21_table = _perturb_positive(base["Sigma_s21_table"], rel_perturb, rng)

    # Keep OUTSIDE row/entry exactly zero.
    d_table[0, :] = 0.0
    sa_table[0, :] = 0.0
    nsf_table[0, :] = 0.0
    s21_table[0] = 0.0

    return build_xs_fields_from_tables(
        material_map=material_map,
        d_table=d_table,
        sigma_a_table=sa_table,
        nu_sigma_f_table=nsf_table,
        sigma_s21_table=s21_table,
        chi=base["chi"],
    )


def make_dataset(
    num_samples: int = 1000,
    rel_perturb: float = 0.10,
    seed: int = 42,
    out_dir: Path | None = None,
) -> Path:
    out_dir = out_dir or (Path(__file__).resolve().parent / "train_data")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cfg = SolverConfig()

    h, w = MATERIAL_MAP_COARSE.shape
    x = np.zeros((num_samples, 7, h, w), dtype=np.float32)
    y_flux = np.zeros((num_samples, 2, h, w), dtype=np.float32)
    y_keff = np.zeros((num_samples,), dtype=np.float32)
    maps = np.zeros((num_samples, h, w), dtype=np.int64)

    for n in range(num_samples):
        m = _build_random_material_map(rng)
        xs = _build_random_xs(m, rel_perturb=rel_perturb, rng=rng)
        keff, phi1, phi2 = solve_two_group_diffusion(
            material_map=m,
            d=xs["D"],
            sigma_a=xs["Sigma_a"],
            nu_sigma_f=xs["nuSigma_f"],
            sigma_s21=xs["Sigma_s21"],
            chi=xs["chi"],
            cfg=cfg,
        )

        x[n, 0] = xs["D"][0]
        x[n, 1] = xs["D"][1]
        x[n, 2] = xs["Sigma_a"][0]
        x[n, 3] = xs["Sigma_a"][1]
        x[n, 4] = xs["nuSigma_f"][0]
        x[n, 5] = xs["nuSigma_f"][1]
        x[n, 6] = xs["Sigma_s21"]

        y_flux[n, 0] = phi1
        y_flux[n, 1] = phi2
        y_keff[n] = keff
        maps[n] = m

        if (n + 1) % 50 == 0 or (n + 1) == num_samples:
            print(f"generated {n + 1}/{num_samples}")

    # Save main training set.
    np.save(out_dir / "inputs.npy", x)
    np.save(out_dir / "targets_flux.npy", y_flux)
    np.save(out_dir / "targets_keff.npy", y_keff)
    np.save(out_dir / "material_maps.npy", maps)

    # Also save unperturbed reference sample for final comparison.
    ref_xs = build_xs_fields(MATERIAL_MAP_COARSE)
    ref_keff, ref_phi1, ref_phi2 = solve_two_group_diffusion(
        material_map=MATERIAL_MAP_COARSE,
        d=ref_xs["D"],
        sigma_a=ref_xs["Sigma_a"],
        nu_sigma_f=ref_xs["nuSigma_f"],
        sigma_s21=ref_xs["Sigma_s21"],
        chi=ref_xs["chi"],
        cfg=cfg,
    )
    ref_input = np.stack(
        [
            ref_xs["D"][0],
            ref_xs["D"][1],
            ref_xs["Sigma_a"][0],
            ref_xs["Sigma_a"][1],
            ref_xs["nuSigma_f"][0],
            ref_xs["nuSigma_f"][1],
            ref_xs["Sigma_s21"],
        ],
        axis=0,
    ).astype(np.float32)
    np.save(out_dir / "reference_input.npy", ref_input)
    np.save(out_dir / "reference_flux.npy", np.stack([ref_phi1, ref_phi2], axis=0).astype(np.float32))
    np.save(out_dir / "reference_keff.npy", np.array(ref_keff, dtype=np.float32))

    print(f"saved dataset to: {out_dir}")
    print(f"reference keff: {ref_keff:.6f}")
    return out_dir


if __name__ == "__main__":
    make_dataset(num_samples=1000, rel_perturb=0.10, seed=42)
