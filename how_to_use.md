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

## Notes
- Always run `generate_training_data.py` before `main.py` if you want a fresh dataset.
- Training/validation/test splits are made from synthetic data in `train_data/`.
