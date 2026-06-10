# IAEA-2D mesh geometry & resolution — finding (DEFERRED)

**Status (2026-06-10): identified and quantified, NOT fixed.** By project decision
the solver and training pipeline are left unchanged. This note records the issue
so it can be picked up later. The diagnostic scripts referenced below are in the
repo and reproduce every number here.

---

## TL;DR

1. **Cross sections are correct.** `iaea2d_reference.py` XS == canonical IAEA-2D
   exactly (arXiv:2208.13483 Table III); reference k_eff ≈ 1.0296.
2. **The loading map was corrected** (already applied): 3 diagonal-symmetry
   violations fixed — `[2,6]` Fuel1→Fuel2, `[3,7]` Fuel2→Reflector,
   `[8,4]` Outside→Reflector. The four control rods (`[0,0],[0,4],[4,0],[4,4]`)
   all recover at Σa2≈0.130; the centre `[0,0]` rod is real and kept.
3. **The solver mesh geometry is wrong (the real fidelity gap).** The IAEA-2D
   quarter core has **HALF-width** assemblies on the two reflective (symmetry)
   edges and a **QUARTER** assembly at `[0,0]` — the full core is 17×17 (odd×odd)
   so the centre row/column are shared and halved. `diffusion_solver.py` uses a
   **uniform full-20cm mesh** for every map cell, over-sizing the central region
   (and the central rods) and flattening the radial power.
4. Fixing the geometry collapses the solver-vs-reference field error from
   **~28% → ~1.6%** (pointwise), correlation **0.93 → 0.9998**.
5. Independently, 9×9 (1 node/assembly) is too coarse; real accuracy needs mesh
   refinement (truncation error ~O(h²)).
6. **Fix deferred** — it requires a solver change *and* regenerating the training
   data *and* retraining, and only matters for fidelity to the *true* IAEA-2D
   (not for the surrogate-imitates-its-own-solver metric).

---

## 1. The geometry issue in detail

A quarter core is bounded by two symmetry lines that pass through the centre of
the full core. Assemblies straddling those lines are shared between quarters:

- `[0,0]` (the reflective/reflective corner = core centre): **quarter** assembly (~10×10 cm region → ~½×½ cells)
- row 0 and column 0 (the two symmetry edges): **half** assemblies (10 cm)
- all interior cells `[1:,1:]`: **full** 20 cm assemblies

The reference grid is `171 = 9×19` only under the *full*-assembly reading; the
correct half-edge reading is `171 ≈ 10 (half) + 8×20 (full)` at h≈1 cm, which is
why the cited papers quote **h = 1 cm**. The repo solver instead treats all 9×9
cells as full 20 cm assemblies (180 cm core), so the central assemblies — and the
strong central control rods — are physically doubled, depressing/flattening the
centre.

**Detection (from the reference φ₂):** the `[0,0]` control-rod flux dip ends at
~10 fine cells (= quarter assembly), while an interior rod (`[4,4]`) dip spans
~20 cells (= full assembly). Confirms half/quarter boundary cells.

---

## 2. Evidence (171×171, solver vs reference state 0)

| solver geometry | power relL2 | φ₁ correlation | radial center/edge |
|---|---|---|---|
| uniform full-20cm (current repo) | 0.281 | 0.932 | 0.96 (too flat) |
| **half/quarter boundary cells** | **0.016** | **0.9998** | 1.83 |
| reference (target) | — | — | 2.08 |

Per-assembly normalized power at the core-centre cell `[0,0]`: reference **0.72**,
current solver **0.34** (corner rod over-sized), geometry-corrected **0.74**.

---

## 3. Why finer mesh = more accurate

The 5-point FDM approximates the leakage term ∇²φ by
`(φ_L+φ_R+φ_U+φ_D−4φ_C)/h²`, whose truncation error is ~O(h²). The flux curves
on cm–10cm scales (centre peak, rod dips, fuel/reflector bends); at h=20 cm
(1 node/assembly) that curvature is invisible, so leakage — and hence the flux
shape and k_eff — are crude. Holding the geometry correct and refining h:

| h (cm) | grid | k_eff | per-assembly shape relL2 vs ref |
|---|---|---|---|
| 10 | 17×17 | 1.0341 | 0.091 |
| 5 | 34×34 | 1.0337 | 0.056 |
| 2 | 85×85 | 1.0339 | 0.017 |
| 1 | 171×171 | 1.0340 | 0.010 |

Shape converges toward 0 as h→0; k_eff (an integral) converges much faster
(already ~converged by h=10). 171 is not special — it is simply fine enough to
be ~converged, hence usable as "truth". (k≈1.034 here matches the reference
*state*; the textbook nominal IAEA-2D is 1.0296 — the downloaded reference is a
perturbed 5-state family, not the bare nominal.)

**Note:** 17×17 fixes the corner and the tilt *direction* (halves max-cell error
0.38→0.17) but over-peaks (center/edge 1.73 vs 1.46 target) — still ~9%. Genuine
accuracy (~1–2%) needs ~85×85+.

---

## 4. Resolution / accuracy / cost (if revisited)

Training needs many samples (the reference has only 5 states, so it is
validation-only). Labels are produced by the solver, so label resolution = solve
resolution, and fine solves are expensive (≈30k samples × per-sample solve):

| resolution | accuracy (relL2) | ~cost @30k samples |
|---|---|---|
| 17×17 | ~9% (over-peaks) | minutes |
| 34×34 | ~5.6% | 1–4 h |
| 85×85 | **~1.7%** | 8–25 h (heavy) |
| 171×171 | ~1% | impractical |

Note the **input** (homogenized per-assembly XS) is intrinsically 9×9 — a finer
*input* is redundant; only the *output* (intra-assembly flux) gains from
refinement.

### Options when revisiting
- **A. Solve fine + coarsen labels to 9×9** (keep the existing 9×9 U-Net), with a
  reduced sample count (e.g. 3–5k) — ~1.7% fidelity, U-Net unchanged. Edit only
  `generate_training_data.py` (half-edge upsample → solve → average to 9×9).
- **B. 9×9 input → 171 output super-resolution U-Net** — captures intra-assembly
  flux, most faithful to the reference; needs fine labels + a decoder change.
- **C. Non-uniform 9×9 mesh in the solver** (half/quarter boundary cells) — keeps
  data-gen cheap but stays coarse (~9%); the quick prototype had a bug
  (fission sum must be volume-weighted) and was discarded.

---

## 5. Implications

This is exactly the "self-referential benchmark" limitation flagged in
`CLAUDE.md` / `PLAN.md` Phase-0: the surrogate is trained to imitate the in-repo
solver, whose labels are ~28% off the true IAEA-2D pointwise (radial tilt;
~10% per-assembly). Consequences:

- The **surrogate-vs-its-own-solver** metric (`results/reference_comparison.png`)
  is *unaffected* by this issue — it is a learning problem of the same difficulty
  regardless of solver geometry.
- The **fidelity to the actual IAEA-2D benchmark** is what is degraded. Fixing it
  matters only if the project's benchmark claim is meant to be physically faithful.

---

## What was changed vs not

- **Changed:** `iaea2d_reference.py` `MATERIAL_MAP_COARSE` (3 cells) + provenance
  docstrings. Added diagnostic scripts `validate_against_reference.py`,
  `reconstruct_material_map.py`.
- **NOT changed:** `diffusion_solver.py`, `generate_training_data.py`, `main.py`,
  the training data, and the trained model. The mesh-geometry fix above is **not**
  applied.

## References
- `2d_IAEA/*.npy` — external reference: 5 states ×2 (obs = 2 noisy realizations),
  φ₁/φ₂/power, 171×171, diagonally symmetric.
- arXiv:2208.13483 (Table III XS), arXiv:2407.10988 (k_eff 1.02977); classic ANL
  value 1.02959.
- Reproduce: `python validate_against_reference.py`, `python reconstruct_material_map.py`.
