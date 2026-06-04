"""Compare TinyUNet against a simple MLP baseline on the same diffusion dataset.

Reuses build_dataloaders / evaluate / infer_reference_case / train_model from main.py,
so normalization, splits, loss weighting, and reference inference are identical
across the two models.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f

from main import (
    TinyUNet,
    TrainConfig,
    build_dataloaders,
    evaluate,
    infer_reference_case,
    report_split_metrics,
    train_model,
)


class TinyMLP(nn.Module):
    """Flattened MLP baseline. Same I/O contract as TinyUNet: returns (flux, keff)."""

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 2,
        height: int = 9,
        width: int = 9,
        hidden: int = 512,
        dropout_p: float = 0.30,
    ) -> None:
        super().__init__()
        self.in_dim = in_channels * height * width
        self.out_channels = out_channels
        self.height = height
        self.width = width
        self.backbone = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
        )
        self.flux_head = nn.Linear(hidden, out_channels * height * width)
        self.keff_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = x.size(0)
        h = self.backbone(x.reshape(b, -1))
        flux = self.flux_head(h).reshape(b, self.out_channels, self.height, self.width)
        keff = self.keff_head(h).squeeze(1)
        return flux, keff


def train_mlp(cfg: TrainConfig) -> tuple[TinyMLP, dict[str, np.ndarray], dict[str, list[float]]]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, test_loader, norm = build_dataloaders(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyMLP(dropout_p=cfg.dropout_p).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    keff_weight = cfg.keff_weight

    history: dict[str, list[float]] = {"train": [], "val": []}
    best_val = float("inf")
    best_path = cfg.out_dir / "mlp_best_model.pt"

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for x_b, y_flux_b, y_keff_b in train_loader:
            x_b, y_flux_b, y_keff_b = x_b.to(device), y_flux_b.to(device), y_keff_b.to(device)
            pred_flux, pred_keff = model(x_b)
            loss_flux = f.mse_loss(pred_flux, y_flux_b)
            loss_keff = f.mse_loss(pred_keff, y_keff_b)
            loss = loss_flux + keff_weight * loss_keff
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= n_batches

        val_loss, _, _ = evaluate(model, val_loader, device=device, keff_weight=keff_weight)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
        if epoch % 20 == 0 or epoch == 1 or epoch == cfg.epochs:
            print(f"[MLP] epoch {epoch:03d} | train {train_loss:.5e} | val {val_loss:.5e}")

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_flux, test_keff = evaluate(model, test_loader, device=device, keff_weight=keff_weight)
    print(f"[MLP] test total={test_loss:.5e} flux={test_flux:.5e} keff={test_keff:.5e}")
    return model, norm, history


def load_or_train_unet(cfg: TrainConfig) -> tuple[TinyUNet, dict[str, np.ndarray]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = cfg.out_dir / "best_model.pt"
    _, _, _, norm = build_dataloaders(cfg)
    if ckpt.exists():
        print(f"[UNet] loading existing checkpoint: {ckpt}")
        model = TinyUNet(
            dropout_p=cfg.dropout_p,
            ch1=cfg.ch1, ch2=cfg.ch2, ch3=cfg.ch3,
            bottleneck_ch=cfg.bottleneck_ch,
        ).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        return model, norm
    print("[UNet] no checkpoint found - training from scratch")
    model, norm, _ = train_model(cfg)
    return model, norm


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def save_comparison(
    cfg: TrainConfig,
    ref_true: np.ndarray,
    unet_pred: np.ndarray,
    mlp_pred: np.ndarray,
    k_ref: float,
    k_unet: float,
    k_mlp: float,
    unet_params: int,
    mlp_params: int,
    mlp_history: dict[str, list[float]],
    unet_split_metrics: dict[str, dict[str, float]],
    mlp_split_metrics: dict[str, dict[str, float]],
) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    panels = [
        (0, 0, ref_true[0], "True phi1", "viridis"),
        (0, 1, unet_pred[0], "U-Net phi1", "viridis"),
        (0, 2, mlp_pred[0], "MLP phi1", "viridis"),
        (1, 0, ref_true[1], "True phi2", "plasma"),
        (1, 1, unet_pred[1], "U-Net phi2", "plasma"),
        (1, 2, mlp_pred[1], "MLP phi2", "plasma"),
    ]
    for r, c, data, title, cmap in panels:
        im = axes[r, c].imshow(data, cmap=cmap, origin="upper")
        axes[r, c].set_title(title)
        fig.colorbar(im, ax=axes[r, c], fraction=0.046, pad=0.04)

    rel_unet = float(np.linalg.norm(unet_pred - ref_true) / (np.linalg.norm(ref_true) + 1.0e-20))
    rel_mlp = float(np.linalg.norm(mlp_pred - ref_true) / (np.linalg.norm(ref_true) + 1.0e-20))
    fig.suptitle(
        f"Reference comparison | k_ref={k_ref:.6f}\n"
        f"U-Net: k={k_unet:.6f} (|dk|={abs(k_unet - k_ref):.3e}, rel-L2 phi={rel_unet:.3e}, params={unet_params:,})\n"
        f"MLP  : k={k_mlp:.6f} (|dk|={abs(k_mlp - k_ref):.3e}, rel-L2 phi={rel_mlp:.3e}, params={mlp_params:,})",
        fontsize=10,
    )
    fig.savefig(cfg.out_dir / "comparison_unet_vs_mlp.png", dpi=180)
    plt.close(fig)

    fig2 = plt.figure(figsize=(6, 4))
    plt.plot(mlp_history["train"], label="train")
    plt.plot(mlp_history["val"], label="val")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("MLP training history")
    plt.legend()
    plt.tight_layout()
    fig2.savefig(cfg.out_dir / "mlp_training_history.png", dpi=180)
    plt.close(fig2)

    def _fmt_split(model_metrics: dict[str, dict[str, float]], model_name: str) -> str:
        lines = [f"  [{model_name}] in-distribution physical-unit metrics"]
        header = f"    {'split':6s} {'n':>4s}  {'flux rel-L2 mean':>16s} {'med':>8s} {'max':>8s}  {'keff pcm mean':>14s} {'med':>10s} {'max':>10s}"
        lines.append(header)
        for split in ("train", "val", "test"):
            m = model_metrics[split]
            lines.append(
                f"    {split:6s} {int(m['n_samples']):>4d}  "
                f"{m['flux_rel_l2_mean']:>16.4f} {m['flux_rel_l2_median']:>8.4f} {m['flux_rel_l2_max']:>8.4f}  "
                f"{m['keff_pcm_mean']:>14.1f} {m['keff_pcm_median']:>10.1f} {m['keff_pcm_max']:>10.1f}"
            )
        return "\n".join(lines)

    summary = (
        "Reference (out-of-distribution) comparison summary\n"
        f"  k_ref               = {k_ref:.6f}\n"
        f"  U-Net k             = {k_unet:.6f}  |dk| = {abs(k_unet - k_ref):.6e}  pcm = {abs(k_unet - k_ref) * 1e5:.1f}\n"
        f"  MLP   k             = {k_mlp:.6f}  |dk| = {abs(k_mlp - k_ref):.6e}  pcm = {abs(k_mlp - k_ref) * 1e5:.1f}\n"
        f"  U-Net rel-L2(phi)   = {rel_unet:.6e}\n"
        f"  MLP   rel-L2(phi)   = {rel_mlp:.6e}\n"
        f"  params(U-Net)       = {unet_params:,}\n"
        f"  params(MLP)         = {mlp_params:,}\n"
        "\n"
        f"{_fmt_split(unet_split_metrics, 'U-Net')}\n"
        "\n"
        f"{_fmt_split(mlp_split_metrics, 'MLP')}\n"
    )
    (cfg.out_dir / "comparison_summary.txt").write_text(summary)
    print(summary)


def main() -> None:
    cfg = TrainConfig()
    unet_model, norm = load_or_train_unet(cfg)
    mlp_model, _, mlp_history = train_mlp(cfg)

    print("\n--- U-Net in-distribution metrics (physical units) ---")
    unet_split_metrics = report_split_metrics(unet_model, cfg, label="UNet")
    print("\n--- MLP in-distribution metrics (physical units) ---")
    mlp_split_metrics = report_split_metrics(mlp_model, cfg, label="MLP ")

    ref_true, unet_pred, k_ref, k_unet = infer_reference_case(unet_model, norm, cfg.data_dir)
    _, mlp_pred, _, k_mlp = infer_reference_case(mlp_model, norm, cfg.data_dir)

    save_comparison(
        cfg,
        ref_true=ref_true,
        unet_pred=unet_pred,
        mlp_pred=mlp_pred,
        k_ref=k_ref,
        k_unet=k_unet,
        k_mlp=k_mlp,
        unet_params=count_params(unet_model),
        mlp_params=count_params(mlp_model),
        mlp_history=mlp_history,
        unet_split_metrics=unet_split_metrics,
        mlp_split_metrics=mlp_split_metrics,
    )
    print(f"saved comparison to: {cfg.out_dir}")


if __name__ == "__main__":
    main()
