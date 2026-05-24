# 2D Neutron Diffusion U-Net Surrogate Project Plan

## Goal
- Build a U-Net CNN surrogate for a 2D two-group neutron diffusion problem.
- Use a diffusion power-iteration solver (NumPy/SciPy) to generate labels.
- Evaluate flux and `k_eff` deviation on the real reference problem.

## Project Gate (Must Pass First)
**Critical first step: acquire and validate the reference problem definition.**

Without a reliable reference dataset/specification, all downstream steps (randomized data generation, solver labels, surrogate evaluation) are uncertain.  
This project starts only after the items below are confirmed.

### Required reference inputs
- Exact geometry and mesh definition of the IAEA 2D LWR two-group case
  - domain size
  - grid dimensions (`Nx`, `Ny`)
  - region/material map
- Boundary conditions (vacuum/reflective/etc. on each side)
- Two-group cross sections per material:
  - `D1`, `D2`
  - `Sigma_a1`, `Sigma_a2`
  - scattering terms (`Sigma_s12`, `Sigma_s21` if used)
  - fission terms (`nuSigma_f1`, `nuSigma_f2`)
  - fission spectrum (`chi1`, `chi2`)
- Reference benchmark outputs (at least one):
  - benchmark `k_eff`
  - optional flux map(s) for sanity checks

## Development Plan

## Phase 0 - Reference Acquisition and Validation (Priority 1)
- Collect official/public source for IAEA 2D LWR 2-group benchmark inputs.
- Convert source data into a machine-readable format (`json` or `yaml`).
- Build a loader:
  - `load_reference_case(path) -> geometry, xs, bc`
- Add validation checks:
  - positivity of required XS
  - physically reasonable ranges
  - shape consistency (`material_map`, `Nx`, `Ny`)
- Produce a short "reference readiness checklist" and mark pass/fail.

**Deliverable:** `reference_case.json` (or `yaml`) + loader + validation report.

## Phase 1 - Synthetic Input Generator
- Create baseline material set from reference XS.
- Create perturbed materials:
  - random perturbation around reference (default `+-10%`, configurable)
  - enforce lower/upper clipping for physical values
- Create random material map generator on reference grid.
- Export one training sample input as tensor-ready channels.

Example input channels:
- group constants fields (`D1`, `D2`, `Sigma_a1`, ...)
- optional one-hot material channels

**Deliverable:** `generate_random_case(...)` producing reproducible samples by seed.

## Phase 2 - Deterministic Label Solver (NumPy/SciPy)
- Implement 2-group diffusion discretization on 2D mesh.
- Implement power iteration for eigenvalue problem:
  - solve fixed-source diffusion per iteration (SciPy sparse linear solver)
  - update fission source and `k_eff`
  - convergence criteria for flux and `k_eff`
- Add numerical safety:
  - normalization
  - max iterations
  - residual logging
- Verify solver against reference benchmark before generating training labels.

**Deliverable:** `solve_diffusion_power_iteration(case) -> phi_g1, phi_g2, k_eff`.

## Phase 3 - Dataset Build
- Generate `N_train`, `N_val`, `N_test` random cases.
- For each case, run solver to create labels:
  - target flux maps (`phi1`, `phi2`)
  - optional scalar target `k_eff`
- Store dataset efficiently (`npz`/`hdf5`) with metadata and seed traceability.

**Deliverable:** reproducible dataset package + manifest.

## Phase 4 - U-Net Surrogate Training
- Build baseline 2D U-Net model.
- Input: per-cell XS/material channels.
- Output options:
  - Option A: flux maps only (`phi1`, `phi2`)
  - Option B: flux maps + `k_eff` head (multi-task)
- Train with normalization and weighted losses.
- Track validation metrics:
  - relative L2 flux error
  - `k_eff` absolute/pcm error

**Deliverable:** trained model checkpoint + training curves.

## Phase 5 - Final Evaluation on Real Reference Problem
- Run trained model on unperturbed real reference case.
- Compare against deterministic solver:
  - flux map error (global and regional)
  - `k_eff` deviation (abs and pcm)
- Generate final plots/report tables.

**Deliverable:** final evaluation report for assignment.

## Risks and Mitigation
- **Risk:** reference spec mismatch  
  **Mitigation:** freeze one canonical reference file and checksum it.
- **Risk:** physically invalid perturbed XS  
  **Mitigation:** bounded perturbation + validation filters.
- **Risk:** solver instability/convergence issues  
  **Mitigation:** robust convergence checks and fallback linear solver settings.
- **Risk:** surrogate overfitting to synthetic distribution  
  **Mitigation:** diverse randomization and strict validation split.

## Immediate Next Actions
1. Acquire official IAEA 2D LWR two-group benchmark inputs (geometry, BC, XS, reference `k_eff`).
2. Freeze them into `reference_case.json` and validate schema.
3. Only then begin solver implementation.

---

If you want, next step I can create:
- `reference_case.schema.json` (strict schema), and
- a `reference_case.template.json` you can fill as soon as source data is found.
