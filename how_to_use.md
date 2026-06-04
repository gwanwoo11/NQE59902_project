# How to Use

## 1) Generate synthetic training data
Run this first:

```bash
python generate_training_data.py
```

This creates synthetic dataset files in `train_data/`.

## 2) Train and test the CNN
After data generation is done, run:

```bash
python main.py
```

This trains the model and saves outputs in `results/`:
- `best_model.pt`
- `training_history.png`
- `reference_comparison.png`

## 3) (Optional) Compare against an MLP baseline
After `main.py` has produced `results/best_model.pt`, run:

```bash
python mlp_baseline.py
```

This trains a small MLP on the same data, reloads the existing U-Net checkpoint, and saves a side-by-side comparison in `results/`:
- `mlp_best_model.pt`
- `mlp_training_history.png`
- `comparison_unet_vs_mlp.png`
- `comparison_summary.txt`

If no U-Net checkpoint exists yet, this script will train one from scratch first.

## Notes
- Always run `generate_training_data.py` before `main.py` if you want a fresh dataset.
- Training/validation/test splits are made from synthetic data in `train_data/`.
