"""Train/test a compact U-Net surrogate on generated diffusion data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split


@dataclass
class TrainConfig:
    data_dir: Path = Path(__file__).resolve().parent / "train_data"
    out_dir: Path = Path(__file__).resolve().parent / "results"
    batch_size: int = 32
    epochs: int = 3000
    lr: float = 1.0e-3
    weight_decay: float = 5.0e-4
    dropout_p: float = 0.30
    # Try 10.0 first; compare with 15.0 in a second run.
    keff_weight: float = 10.0
    ch1: int = 24
    ch2: int = 48
    ch3: int = 96
    bottleneck_ch: int = 128
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout_p: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 2,
        dropout_p: float = 0.30,
        ch1: int = 24,
        ch2: int = 48,
        ch3: int = 96,
        bottleneck_ch: int = 128,
    ) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, ch1, dropout_p)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(ch1, ch2, dropout_p)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(ch2, ch3, dropout_p)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(ch3, bottleneck_ch, dropout_p)

        self.up3 = nn.ConvTranspose2d(bottleneck_ch, ch3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(ch3 * 2, ch3, dropout_p)
        self.up2 = nn.ConvTranspose2d(ch3, ch2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(ch2 * 2, ch2, dropout_p)
        self.up1 = nn.ConvTranspose2d(ch2, ch1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(ch1 * 2, ch1, dropout_p)
        self.out_flux = nn.Conv2d(ch1, out_channels, kernel_size=1)

        self.keff_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bottleneck_ch, max(64, bottleneck_ch // 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(max(64, bottleneck_ch // 2), 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = f.pad(d3, (0, e3.shape[3] - d3.shape[3], 0, e3.shape[2] - d3.shape[2]))
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = f.pad(d2, (0, e2.shape[3] - d2.shape[3], 0, e2.shape[2] - d2.shape[2]))
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = f.pad(d1, (0, e1.shape[3] - d1.shape[3], 0, e1.shape[2] - d1.shape[2]))
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        flux = self.out_flux(d1)
        keff = self.keff_head(b).squeeze(1)
        return flux, keff


def load_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.load(data_dir / "inputs.npy")
    y_flux = np.load(data_dir / "targets_flux.npy")
    y_keff = np.load(data_dir / "targets_keff.npy")
    return x, y_flux, y_keff


def build_dataloaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, np.ndarray]]:
    x, y_flux, y_keff = load_dataset(cfg.data_dir)
    x_mean = x.mean(axis=(0, 2, 3), keepdims=True)
    x_std = x.std(axis=(0, 2, 3), keepdims=True) + 1.0e-8
    y_mean = y_flux.mean(axis=(0, 2, 3), keepdims=True)
    y_std = y_flux.std(axis=(0, 2, 3), keepdims=True) + 1.0e-8

    x_n = (x - x_mean) / x_std
    y_n = (y_flux - y_mean) / y_std

    k_mean = y_keff.mean(keepdims=True).astype(np.float32)
    k_std = (y_keff.std(keepdims=True) + 1.0e-8).astype(np.float32)
    y_keff_n = (y_keff - k_mean) / k_std

    dataset = TensorDataset(
        torch.from_numpy(x_n).float(),
        torch.from_numpy(y_n).float(),
        torch.from_numpy(y_keff_n).float(),
    )

    split_train_path = cfg.data_dir / "split_train.npy"
    split_val_path = cfg.data_dir / "split_val.npy"
    split_test_path = cfg.data_dir / "split_test.npy"
    if split_train_path.exists() and split_val_path.exists() and split_test_path.exists():
        train_idx = np.load(split_train_path).tolist()
        val_idx = np.load(split_val_path).tolist()
        test_idx = np.load(split_test_path).tolist()
        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)
        test_ds = Subset(dataset, test_idx)
    else:
        n_total = len(dataset)
        n_test = int(n_total * cfg.test_ratio)
        n_val = int(n_total * cfg.val_ratio)
        n_train = n_total - n_val - n_test
        generator = torch.Generator().manual_seed(cfg.seed)
        train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
    norm = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "k_mean": k_mean,
        "k_std": k_std,
    }
    return train_loader, val_loader, test_loader, norm


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, keff_weight: float = 5.0) -> tuple[float, float, float]:
    model.eval()
    total_loss = total_flux = total_keff = 0.0
    n_batches = 0
    with torch.no_grad():
        for x_b, y_flux_b, y_keff_b in loader:
            x_b, y_flux_b, y_keff_b = x_b.to(device), y_flux_b.to(device), y_keff_b.to(device)
            pred_flux, pred_keff = model(x_b)
            loss_flux = f.mse_loss(pred_flux, y_flux_b)
            loss_keff = f.mse_loss(pred_keff, y_keff_b)
            loss = loss_flux + keff_weight * loss_keff
            total_loss += loss.item()
            total_flux += loss_flux.item()
            total_keff += loss_keff.item()
            n_batches += 1
    return total_loss / n_batches, total_flux / n_batches, total_keff / n_batches


def evaluate_physical(model: nn.Module, loader: DataLoader, device: torch.device, norm: dict[str, np.ndarray]) -> dict[str, float]:
    """Per-sample rel-L2(flux) and |dk_eff| (pcm) after denormalizing to physical units."""
    y_mean = torch.from_numpy(norm["y_mean"]).float().to(device)
    y_std = torch.from_numpy(norm["y_std"]).float().to(device)
    k_mean = float(norm["k_mean"][0])
    k_std = float(norm["k_std"][0])

    rel_l2_list: list[float] = []
    abs_dk_list: list[float] = []
    model.eval()
    with torch.no_grad():
        for x_b, y_flux_n, y_keff_n in loader:
            x_b = x_b.to(device)
            y_flux_n = y_flux_n.to(device)
            y_keff_n = y_keff_n.to(device)
            pred_flux_n, pred_keff_n = model(x_b)

            pred_flux = pred_flux_n * y_std + y_mean
            true_flux = y_flux_n * y_std + y_mean
            diff = (pred_flux - true_flux).reshape(pred_flux.size(0), -1)
            truth = true_flux.reshape(true_flux.size(0), -1)
            sample_rel = diff.norm(dim=1) / (truth.norm(dim=1) + 1.0e-20)
            rel_l2_list.extend(sample_rel.cpu().tolist())

            pred_k = pred_keff_n.cpu().numpy() * k_std + k_mean
            true_k = y_keff_n.cpu().numpy() * k_std + k_mean
            abs_dk_list.extend(np.abs(pred_k - true_k).tolist())

    rel = np.asarray(rel_l2_list)
    dk = np.asarray(abs_dk_list)
    return {
        "n_samples": int(rel.size),
        "flux_rel_l2_mean": float(rel.mean()),
        "flux_rel_l2_median": float(np.median(rel)),
        "flux_rel_l2_max": float(rel.max()),
        "keff_pcm_mean": float(dk.mean() * 1.0e5),
        "keff_pcm_median": float(np.median(dk) * 1.0e5),
        "keff_pcm_max": float(dk.max() * 1.0e5),
    }


def report_split_metrics(model: nn.Module, cfg: "TrainConfig", label: str = "") -> dict[str, dict[str, float]]:
    """Build loaders deterministically, run evaluate_physical on each split, and print a table."""
    train_loader, val_loader, test_loader, norm = build_dataloaders(cfg)
    device = next(model.parameters()).device
    out: dict[str, dict[str, float]] = {}
    prefix = f"[{label}] " if label else ""
    for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        m = evaluate_physical(model, loader, device, norm)
        out[name] = m
        print(
            f"{prefix}{name:5s} n={int(m['n_samples']):4d}  "
            f"flux rel-L2 mean={m['flux_rel_l2_mean']:.4f} med={m['flux_rel_l2_median']:.4f} max={m['flux_rel_l2_max']:.4f}  "
            f"keff pcm mean={m['keff_pcm_mean']:8.1f} med={m['keff_pcm_median']:8.1f} max={m['keff_pcm_max']:8.1f}"
        )
    return out


def train_model(cfg: TrainConfig) -> tuple[TinyUNet, dict[str, np.ndarray], dict[str, list[float]]]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, test_loader, norm = build_dataloaders(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyUNet(
        dropout_p=cfg.dropout_p,
        ch1=cfg.ch1,
        ch2=cfg.ch2,
        ch3=cfg.ch3,
        bottleneck_ch=cfg.bottleneck_ch,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    keff_weight = cfg.keff_weight

    history: dict[str, list[float]] = {"train": [], "val": []}
    best_val = float("inf")
    best_path = cfg.out_dir / "best_model.pt"

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
            print(f"epoch {epoch:03d} | train {train_loss:.5e} | val {val_loss:.5e}")

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_flux, test_keff = evaluate(model, test_loader, device=device, keff_weight=keff_weight)
    print(f"test total={test_loss:.5e} flux={test_flux:.5e} keff={test_keff:.5e}")
    return model, norm, history


def infer_reference_case(model: TinyUNet, norm: dict[str, np.ndarray], data_dir: Path) -> tuple[np.ndarray, np.ndarray, float, float]:
    device = next(model.parameters()).device
    x_ref = np.load(data_dir / "reference_input.npy").astype(np.float32)
    y_ref = np.load(data_dir / "reference_flux.npy").astype(np.float32)
    k_ref = float(np.load(data_dir / "reference_keff.npy"))

    x_ref_n = (x_ref - norm["x_mean"][0]) / norm["x_std"][0]
    x_t = torch.from_numpy(x_ref_n[None, ...]).float().to(device)
    model.eval()
    with torch.no_grad():
        pred_flux_n, pred_keff = model(x_t)
    pred_flux_n = pred_flux_n.cpu().numpy()[0]
    pred_keff = float(pred_keff.cpu().numpy()[0] * norm["k_std"][0] + norm["k_mean"][0])
    pred_flux = pred_flux_n * norm["y_std"][0] + norm["y_mean"][0]
    return y_ref, pred_flux.astype(np.float32), k_ref, pred_keff


def save_plots(cfg: TrainConfig, history: dict[str, list[float]], ref_true: np.ndarray, ref_pred: np.ndarray, k_ref: float, k_pred: float) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    fig1 = plt.figure(figsize=(6, 4))
    plt.plot(history["train"], label="train")
    plt.plot(history["val"], label="val")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training history")
    plt.legend()
    plt.tight_layout()
    fig1.savefig(cfg.out_dir / "training_history.png", dpi=180)
    plt.close(fig1)

    fig2, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    im00 = axes[0, 0].imshow(ref_true[0], cmap="viridis", origin="upper")
    axes[0, 0].set_title("True phi1")
    fig2.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.04)
    im01 = axes[0, 1].imshow(ref_pred[0], cmap="viridis", origin="upper")
    axes[0, 1].set_title("Pred phi1")
    fig2.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.04)
    im10 = axes[1, 0].imshow(ref_true[1], cmap="plasma", origin="upper")
    axes[1, 0].set_title("True phi2")
    fig2.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.04)
    im11 = axes[1, 1].imshow(ref_pred[1], cmap="plasma", origin="upper")
    axes[1, 1].set_title("Pred phi2")
    fig2.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.04)
    fig2.suptitle(
        f"Reference compare | k_ref={k_ref:.6f}, k_pred={k_pred:.6f}, abs={abs(k_ref-k_pred):.6e}",
        fontsize=11,
    )
    fig2.savefig(cfg.out_dir / "reference_comparison.png", dpi=180)
    plt.close(fig2)


def main() -> None:
    cfg = TrainConfig()
    model, norm, history = train_model(cfg)
    print("\n--- in-distribution metrics in physical units ---")
    report_split_metrics(model, cfg, label="UNet")
    ref_true, ref_pred, k_ref, k_pred = infer_reference_case(model, norm, cfg.data_dir)
    rel_l2 = np.linalg.norm(ref_pred - ref_true) / (np.linalg.norm(ref_true) + 1.0e-20)
    print(f"\nreference (OOD) keff true={k_ref:.6f}, pred={k_pred:.6f}, abs={abs(k_pred-k_ref):.6e} ({abs(k_pred-k_ref)*1e5:.1f} pcm)")
    print(f"reference (OOD) flux relative L2 error={rel_l2:.6e}")
    save_plots(cfg, history, ref_true, ref_pred, k_ref, k_pred)
    print(f"saved outputs in: {cfg.out_dir}")


if __name__ == "__main__":
    main()
