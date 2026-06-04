"""Generate diffusion training data with fixed structure + permuted fuel inventory.

Design choices:
- Outside (id=0) and reflector (id=4) positions are STRUCTURAL and never randomized.
  Their positions stay exactly as in MATERIAL_MAP_COARSE.
- Only fuel positions (cells where reference is in {Fuel1, Fuel1+Rod, Fuel2}) get
  reassigned, and the assignment is a **permutation** of a fixed inventory whose
  counts equal those of the reference map. So each generated case has identical
  totals of Fuel1, Fuel1+Rod, Fuel2 — much more physical and a much smaller
  search space than per-cell independent sampling.
- Each sample is one *unique structural pattern* plus a fresh +-rel_perturb XS
  perturbation on the base cross-section tables.
- Held-out structural split: samples are saved in `inputs.npy` in generation order,
  and `split_{train,val,test}.npy` carry contiguous index ranges, so train / val /
  test never share a structural pattern. The unperturbed reference case is also
  saved separately as a final OOD probe.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from diffusion_solver import SolverConfig, solve_two_group_diffusion
from iaea2d_reference import (
    FUEL1,
    FUEL1_ROD,
    FUEL2,
    MATERIAL_MAP_COARSE,
    OUTSIDE,
    REFLECTOR,
    build_xs_fields,
    build_xs_fields_from_tables,
    get_base_xs_tables,
)

FUEL_IDS: tuple[int, ...] = (FUEL1, FUEL1_ROD, FUEL2)


def _perturb_positive(values: np.ndarray, rel: float, rng: np.random.Generator) -> np.ndarray:
    scale = 1.0 + rng.uniform(-rel, rel, size=values.shape)
    out = values * scale
    out = np.where(values > 0.0, np.maximum(out, 1.0e-12), 0.0)
    return out


def _get_fuel_positions() -> tuple[np.ndarray, np.ndarray]:
    fuel_mask = np.isin(MATERIAL_MAP_COARSE, FUEL_IDS)
    return np.where(fuel_mask)


def _get_reference_inventory() -> dict[int, int]:
    return {int(fid): int((MATERIAL_MAP_COARSE == fid).sum()) for fid in FUEL_IDS}


def _make_inventory_labels(inventory: dict[int, int]) -> np.ndarray:
    labels: list[int] = []
    for fid, count in inventory.items():
        labels.extend([fid] * count)
    return np.asarray(labels, dtype=MATERIAL_MAP_COARSE.dtype)


def _permuted_material_map(
    rng: np.random.Generator,
    fuel_positions: tuple[np.ndarray, np.ndarray],
    inventory_labels: np.ndarray,
) -> np.ndarray:
    m = MATERIAL_MAP_COARSE.copy()
    labels = inventory_labels.copy()
    rng.shuffle(labels)
    rows, cols = fuel_positions
    m[rows, cols] = labels
    return m


def _build_random_xs(material_map: np.ndarray, rel_perturb: float, rng: np.random.Generator) -> dict[str, np.ndarray]:
    base = get_base_xs_tables()
    d_table = _perturb_positive(base["D_table"], rel_perturb, rng)
    sa_table = _perturb_positive(base["Sigma_a_table"], rel_perturb, rng)
    nsf_table = _perturb_positive(base["nuSigma_f_table"], rel_perturb, rng)
    s21_table = _perturb_positive(base["Sigma_s21_table"], rel_perturb, rng)

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
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    out_dir: Path | None = None,
) -> Path:
    out_dir = out_dir or (Path(__file__).resolve().parent / "train_data")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cfg = SolverConfig()

    h, w = MATERIAL_MAP_COARSE.shape
    fuel_positions = _get_fuel_positions()
    inventory = _get_reference_inventory()
    inventory_labels = _make_inventory_labels(inventory)

    print("Fixed-structure permutation generator")
    print(f"  outside cells   = {int((MATERIAL_MAP_COARSE == OUTSIDE).sum())}")
    print(f"  reflector cells = {int((MATERIAL_MAP_COARSE == REFLECTOR).sum())}")
    print(f"  fuel cells      = {inventory_labels.size}")
    print(f"  inventory       = {inventory}  (preserved across all samples)")

    x = np.zeros((num_samples, 7, h, w), dtype=np.float32)
    y_flux = np.zeros((num_samples, 2, h, w), dtype=np.float32)
    y_keff = np.zeros((num_samples,), dtype=np.float32)
    maps = np.zeros((num_samples, h, w), dtype=np.int64)

    for n in range(num_samples):
        m = _permuted_material_map(rng, fuel_positions, inventory_labels)
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

    np.save(out_dir / "inputs.npy", x)
    np.save(out_dir / "targets_flux.npy", y_flux)
    np.save(out_dir / "targets_keff.npy", y_keff)
    np.save(out_dir / "material_maps.npy", maps)

    # Held-out structural split: contiguous index ranges. Since every sample is a
    # unique permutation, no structure leaks across splits.
    n_test = int(num_samples * test_ratio)
    n_val = int(num_samples * val_ratio)
    n_train = num_samples - n_val - n_test
    idx = np.arange(num_samples)
    np.save(out_dir / "split_train.npy", idx[:n_train])
    np.save(out_dir / "split_val.npy", idx[n_train : n_train + n_val])
    np.save(out_dir / "split_test.npy", idx[n_train + n_val :])
    print(f"structural split: train={n_train}  val={n_val}  test={n_test}")
    n_unique = len({tuple(m.flatten()) for m in maps})
    print(f"unique material patterns generated: {n_unique}/{num_samples}")

    # Unperturbed reference case (true OOD probe — its structural pattern is the
    # canonical IAEA-style loading, not in the train/val/test set by construction).
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
