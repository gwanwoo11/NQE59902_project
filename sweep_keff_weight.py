"""Sweep keff_weight (lambda) under the unit-norm flux loss.

For each lambda in [10, 3, 1], train U-Net from scratch and record:
- test flux rel-L2 (mean / median / max)
- test k_eff pcm  (mean / median / max)

User criterion: pick the lambda with the lowest flux rel-L2 mean whose test pcm mean
stays inside the acceptable window (~< 1000 pcm).
"""

from __future__ import annotations

import json
from pathlib import Path

from main import (
    TrainConfig,
    build_dataloaders,
    evaluate_physical,
    train_model,
)


def run_one(lam: float) -> dict:
    cfg = TrainConfig(keff_weight=lam)
    print(f"\n{'=' * 12} keff_weight = {lam} {'=' * 12}")
    model, norm, _history = train_model(cfg)
    train_loader, val_loader, test_loader, _ = build_dataloaders(cfg)
    device = next(model.parameters()).device
    splits = {}
    for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        splits[name] = evaluate_physical(model, loader, device, norm)
    return {"lambda": lam, "splits": splits}


def main() -> None:
    lambdas = [10.0, 3.0, 1.0]
    results = [run_one(lam) for lam in lambdas]

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lambda_sweep.json").write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 90)
    print("Lambda sweep summary (unit-norm flux loss)")
    print("=" * 90)
    header = (
        f"{'lambda':>7}  "
        f"{'test rel-L2 mean':>16} {'med':>8} {'max':>8}  "
        f"{'test pcm mean':>14} {'med':>10} {'max':>10}"
    )
    print(header)
    for r in results:
        t = r["splits"]["test"]
        print(
            f"{r['lambda']:>7.2f}  "
            f"{t['flux_rel_l2_mean']:>16.4f} {t['flux_rel_l2_median']:>8.4f} {t['flux_rel_l2_max']:>8.4f}  "
            f"{t['keff_pcm_mean']:>14.1f} {t['keff_pcm_median']:>10.1f} {t['keff_pcm_max']:>10.1f}"
        )


if __name__ == "__main__":
    main()
