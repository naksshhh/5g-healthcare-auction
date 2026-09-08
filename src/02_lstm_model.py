"""
Multi-horizon LSTM for IoMT vital-sign forecasting and risk classification.

Consumes sliding windows from ``01_preprocess.py``:

    X          (N, lookback, F)   normalized vitals + fall flag
    y_vitals   (N, H, 5)          HR, SpO2, SBP, DBP, temperature at +5/+10/+15 min
    y_risk     (N, H)             NEWS2-style class {0, 1, 2}

A shared LSTM encoder produces two heads:
  * vital regression  (Smooth-L1)
  * risk classification (class-weighted CE)

Checkpoints and test predictions are written for ``03_fusion_metric.py``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import load_step

DEFAULT_PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts"
N_RISK_CLASSES = 3


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class LSTMConfig:
    processed_dir: Path = DEFAULT_PROCESSED_DIR
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.25
    bidirectional: bool = True
    batch_size: int = 256
    epochs: int = 25
    lr: float = 1e-3
    weight_decay: float = 1e-4
    risk_loss_weight: float = 0.35
    patience: int = 5
    num_workers: int = 0
    seed: int = 42
    max_train_samples: int | None = None

    def __post_init__(self) -> None:
        self.processed_dir = Path(self.processed_dir)
        self.artifact_dir = Path(self.artifact_dir)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y_vitals: np.ndarray, y_risk: np.ndarray):
        self.x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        self.y_vitals = torch.from_numpy(np.ascontiguousarray(y_vitals, dtype=np.float32))
        self.y_risk = torch.from_numpy(np.ascontiguousarray(y_risk, dtype=np.int64))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.y_vitals[idx], self.y_risk[idx]


def load_split(processed_dir: Path, split: str) -> dict[str, np.ndarray]:
    preprocess = load_step("preprocess")
    return preprocess.load_windows(processed_dir, split)


def make_loader(
    split: dict[str, np.ndarray],
    cfg: LSTMConfig,
    shuffle: bool,
    max_samples: int | None = None,
) -> DataLoader:
    x = split["X"]
    y_v = split["y_vitals"]
    y_r = split["y_risk"]
    if max_samples is not None and len(x) > max_samples:
        rng = np.random.default_rng(cfg.seed)
        idx = rng.choice(len(x), size=max_samples, replace=False)
        x, y_v, y_r = x[idx], y_v[idx], y_r[idx]
    ds = WindowDataset(x, y_v, y_r)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def class_weights(y_risk: np.ndarray, device: torch.device) -> torch.Tensor:
    flat = y_risk.reshape(-1)
    counts = np.bincount(flat, minlength=N_RISK_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (N_RISK_CLASSES * counts)
    return torch.tensor(w, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class VitalLSTM(nn.Module):
    """Shared LSTM encoder with attention pooling and dual forecast heads."""

    def __init__(
        self,
        n_features: int,
        n_vitals: int,
        n_horizons: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.25,
        bidirectional: bool = True,
        n_classes: int = N_RISK_CLASSES,
    ) -> None:
        super().__init__()
        self.n_vitals = n_vitals
        self.n_horizons = n_horizons
        self.n_classes = n_classes
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        enc_dim = hidden_size * (2 if bidirectional else 1)
        self.attn = nn.Sequential(
            nn.Linear(enc_dim, enc_dim // 2),
            nn.Tanh(),
            nn.Linear(enc_dim // 2, 1),
        )
        self.norm = nn.LayerNorm(enc_dim)
        self.drop = nn.Dropout(dropout)
        self.vitals_head = nn.Sequential(
            nn.Linear(enc_dim, enc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(enc_dim, n_horizons * n_vitals),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(enc_dim, enc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(enc_dim, n_horizons * n_classes),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        scores = self.attn(out)
        weights = torch.softmax(scores, dim=1)
        ctx = (weights * out).sum(dim=1)
        return self.drop(self.norm(ctx))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        b = x.size(0)
        vitals = self.vitals_head(h).view(b, self.n_horizons, self.n_vitals)
        risk_logits = self.risk_head(h).view(b, self.n_horizons, self.n_classes)
        return vitals, risk_logits


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
def batch_loss(
    y_v_hat: torch.Tensor,
    y_r_hat: torch.Tensor,
    y_v: torch.Tensor,
    y_r: torch.Tensor,
    vital_crit: nn.Module,
    risk_crit: nn.Module,
    risk_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    l_v = vital_crit(y_v_hat, y_v)
    l_r = risk_crit(y_r_hat.reshape(-1, y_r_hat.size(-1)), y_r.reshape(-1))
    total = l_v + risk_weight * l_r
    with torch.no_grad():
        pred = y_r_hat.argmax(dim=-1)
        acc = (pred == y_r).float().mean().item()
    return total, {"vital": float(l_v.item()), "risk": float(l_r.item()), "acc": acc}


def run_epoch(
    model: VitalLSTM,
    loader: DataLoader,
    device: torch.device,
    vital_crit: nn.Module,
    risk_crit: nn.Module,
    risk_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    meters = {"loss": 0.0, "vital": 0.0, "risk": 0.0, "acc": 0.0}
    n = 0
    for x, y_v, y_r in loader:
        x = x.to(device, non_blocking=True)
        y_v = y_v.to(device, non_blocking=True)
        y_r = y_r.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        y_v_hat, y_r_hat = model(x)
        loss, parts = batch_loss(y_v_hat, y_r_hat, y_v, y_r, vital_crit, risk_crit, risk_weight)
        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        bs = x.size(0)
        meters["loss"] += float(loss.item()) * bs
        meters["vital"] += parts["vital"] * bs
        meters["risk"] += parts["risk"] * bs
        meters["acc"] += parts["acc"] * bs
        n += bs
    return {k: v / max(n, 1) for k, v in meters.items()}


@torch.no_grad()
def predict_loader(
    model: VitalLSTM, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    v_hat, r_hat, r_prob = [], [], []
    for x, _, _ in loader:
        x = x.to(device, non_blocking=True)
        vitals, logits = model(x)
        prob = torch.softmax(logits, dim=-1)
        v_hat.append(vitals.cpu().numpy())
        r_hat.append(logits.argmax(dim=-1).cpu().numpy())
        r_prob.append(prob.cpu().numpy())
    return np.concatenate(v_hat), np.concatenate(r_hat), np.concatenate(r_prob)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def evaluate_predictions(
    y_vitals_true_raw: np.ndarray,
    y_vitals_pred_raw: np.ndarray,
    y_risk_true: np.ndarray,
    y_risk_pred: np.ndarray,
    vital_names: list[str],
    horizons: list[int],
) -> dict:
    report: dict = {"per_horizon": {}, "overall": {}}
    mae_all, rmse_all = [], []
    for h_i, h in enumerate(horizons):
        slot: dict = {"horizon_min": int(h), "vitals": {}, "risk_accuracy": None}
        for v_i, name in enumerate(vital_names):
            yt = y_vitals_true_raw[:, h_i, v_i]
            yp = y_vitals_pred_raw[:, h_i, v_i]
            slot["vitals"][name] = {"mae": mae(yt, yp), "rmse": rmse(yt, yp)}
            mae_all.append(slot["vitals"][name]["mae"])
            rmse_all.append(slot["vitals"][name]["rmse"])
        yt_r = y_risk_true[:, h_i]
        yp_r = y_risk_pred[:, h_i]
        slot["risk_accuracy"] = float((yt_r == yp_r).mean())
        slot["risk_macro_f1"] = _macro_f1(yt_r, yp_r, N_RISK_CLASSES)
        report["per_horizon"][str(h)] = slot
    report["overall"] = {
        "vital_mae": float(np.mean(mae_all)),
        "vital_rmse": float(np.mean(rmse_all)),
        "risk_accuracy": float((y_risk_true == y_risk_pred).mean()),
        "risk_macro_f1": _macro_f1(y_risk_true.reshape(-1), y_risk_pred.reshape(-1), N_RISK_CLASSES),
    }
    return report


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1s.append(2 * prec * rec / (prec + rec + 1e-9))
    return float(np.mean(f1s))


def _print_report(title: str, report: dict, vital_names: list[str], horizons: list[int]) -> None:
    print(f"\n[{title}]")
    o = report["overall"]
    print(
        f"  overall  vital MAE={o['vital_mae']:.3f}  RMSE={o['vital_rmse']:.3f}  "
        f"risk acc={o['risk_accuracy']:.3f}  macro-F1={o['risk_macro_f1']:.3f}"
    )
    header = f"  {'h':>4}  " + "  ".join(f"{n:>10}" for n in vital_names) + "  risk_acc  f1"
    print(header)
    for h in horizons:
        slot = report["per_horizon"][str(h)]
        maes = "  ".join(f"{slot['vitals'][n]['mae']:10.3f}" for n in vital_names)
        print(f"  {h:>4}  {maes}  {slot['risk_accuracy']:8.3f}  {slot['risk_macro_f1']:.3f}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_model(n_features: int, n_vitals: int, n_horizons: int, cfg: LSTMConfig) -> VitalLSTM:
    return VitalLSTM(
        n_features=n_features,
        n_vitals=n_vitals,
        n_horizons=n_horizons,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        bidirectional=cfg.bidirectional,
    )


def save_checkpoint(
    path: Path,
    model: VitalLSTM,
    cfg: LSTMConfig,
    extra: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
        "n_features": model.lstm.input_size,
        "n_vitals": model.n_vitals,
        "n_horizons": model.n_horizons,
        "n_classes": model.n_classes,
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_layers,
        "dropout": cfg.dropout,
        "bidirectional": cfg.bidirectional,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_trained_model(ckpt_path: Path | str, device: torch.device | None = None) -> VitalLSTM:
    device = device or resolve_device()
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    model = VitalLSTM(
        n_features=int(ckpt["n_features"]),
        n_vitals=int(ckpt["n_vitals"]),
        n_horizons=int(ckpt["n_horizons"]),
        hidden_size=int(ckpt["hidden_size"]),
        num_layers=int(ckpt["num_layers"]),
        dropout=float(ckpt["dropout"]),
        bidirectional=bool(ckpt["bidirectional"]),
        n_classes=int(ckpt.get("n_classes", N_RISK_CLASSES)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def export_split_predictions(
    model: VitalLSTM,
    split: dict[str, np.ndarray],
    cfg: LSTMConfig,
    device: torch.device,
    inverse_fn,
) -> dict[str, np.ndarray]:
    loader = make_loader(split, cfg, shuffle=False)
    v_norm, r_hat, r_prob = predict_loader(model, loader, device)
    v_raw = inverse_fn(cfg.processed_dir, v_norm)
    return {
        "patient_id": split["patient_id"],
        "y_vitals_pred_norm": v_norm.astype(np.float32),
        "y_vitals_pred_raw": v_raw.astype(np.float32),
        "y_vitals_true_raw": split["y_vitals_raw"].astype(np.float32),
        "y_vitals_true_norm": split["y_vitals"].astype(np.float32),
        "y_risk_pred": r_hat.astype(np.int64),
        "y_risk_true": split["y_risk"].astype(np.int64),
        "y_risk_proba": r_prob.astype(np.float32),
        "horizons_min": split["horizons_min"],
        "vital_names": split["vital_names"],
    }


def run_lstm(cfg: LSTMConfig | None = None) -> dict[str, Path]:
    cfg = cfg or LSTMConfig()
    if not (cfg.processed_dir / "windows_train.npz").exists():
        raise FileNotFoundError(
            f"Processed windows not found in {cfg.processed_dir}. Run 01_preprocess.py first."
        )
    set_seed(cfg.seed)
    device = resolve_device()
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)

    train = load_split(cfg.processed_dir, "train")
    val = load_split(cfg.processed_dir, "val")
    test = load_split(cfg.processed_dir, "test")

    n_features = int(train["X"].shape[-1])
    n_vitals = int(train["y_vitals"].shape[-1])
    n_horizons = int(train["y_vitals"].shape[1])
    vital_names = [str(x) for x in train["vital_names"]]
    horizons = [int(h) for h in train["horizons_min"]]

    print(
        f"[lstm] device={device}  train={len(train['X'])}  val={len(val['X'])}  "
        f"test={len(test['X'])}  F={n_features}  H={n_horizons}"
    )

    train_loader = make_loader(train, cfg, shuffle=True, max_samples=cfg.max_train_samples)
    val_loader = make_loader(val, cfg, shuffle=False)

    model = build_model(n_features, n_vitals, n_horizons, cfg).to(device)
    weights = class_weights(train["y_risk"], device)
    print(f"[lstm] risk class weights={weights.detach().cpu().tolist()}")
    vital_crit = nn.SmoothL1Loss()
    risk_crit = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val = float("inf")
    best_epoch = -1
    stale = 0
    history: list[dict] = []
    ckpt_path = cfg.artifact_dir / "lstm_best.pt"

    for epoch in range(1, cfg.epochs + 1):
        tr = run_epoch(model, train_loader, device, vital_crit, risk_crit, cfg.risk_loss_weight, optimizer)
        va = run_epoch(model, val_loader, device, vital_crit, risk_crit, cfg.risk_loss_weight, None)
        scheduler.step(va["loss"])
        history.append({"epoch": epoch, "train": tr, "val": va})
        print(
            f"[lstm] epoch {epoch:02d}/{cfg.epochs}  "
            f"train loss={tr['loss']:.4f} vital={tr['vital']:.4f} acc={tr['acc']:.3f}  "
            f"val loss={va['loss']:.4f} vital={va['vital']:.4f} acc={va['acc']:.3f}"
        )
        if va["loss"] + 1e-4 < best_val:
            best_val = va["loss"]
            best_epoch = epoch
            stale = 0
            save_checkpoint(ckpt_path, model, cfg, extra={"best_val_loss": best_val, "best_epoch": best_epoch})
        else:
            stale += 1
            if stale >= cfg.patience:
                print(f"[lstm] early stop at epoch {epoch} (best={best_epoch})")
                break

    model = load_trained_model(ckpt_path, device)
    preprocess = load_step("preprocess")

    artifacts: dict[str, Path] = {"checkpoint": ckpt_path}
    merged_ids, merged_split = [], []
    merged_arrays: dict[str, list] = {}

    reports = {}
    for split_name, split in (("train", train), ("val", val), ("test", test)):
        pred = export_split_predictions(model, split, cfg, device, preprocess.inverse_vitals)
        out_path = cfg.processed_dir / f"lstm_pred_{split_name}.npz"
        np.savez_compressed(out_path, **pred)
        artifacts[f"pred_{split_name}"] = out_path
        report = evaluate_predictions(
            pred["y_vitals_true_raw"],
            pred["y_vitals_pred_raw"],
            pred["y_risk_true"],
            pred["y_risk_pred"],
            vital_names,
            horizons,
        )
        reports[split_name] = report
        _print_report(f"lstm {split_name}", report, vital_names, horizons)
        merged_ids.append(pred["patient_id"])
        merged_split.append(np.full(len(pred["patient_id"]), split_name))
        for key in (
            "y_vitals_pred_norm",
            "y_vitals_pred_raw",
            "y_vitals_true_raw",
            "y_risk_pred",
            "y_risk_true",
            "y_risk_proba",
        ):
            merged_arrays.setdefault(key, []).append(pred[key])

    all_path = cfg.processed_dir / "lstm_predictions.npz"
    np.savez_compressed(
        all_path,
        patient_id=np.concatenate(merged_ids),
        split=np.concatenate(merged_split),
        horizons_min=np.array(horizons, dtype=np.int32),
        vital_names=np.array(vital_names),
        **{k: np.concatenate(v) for k, v in merged_arrays.items()},
    )
    artifacts["predictions"] = all_path

    metrics_path = cfg.artifact_dir / "lstm_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "device": str(device),
                "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
                "reports": reports,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    artifacts["metrics"] = metrics_path
    fig_paths = plot_lstm_results(history, reports, vital_names, horizons, cfg.artifact_dir)
    artifacts.update(fig_paths)
    print(f"[lstm] saved checkpoint {ckpt_path}")
    print(f"[lstm] saved predictions {all_path}")
    return artifacts


def plot_lstm_results(
    history: list[dict],
    reports: dict,
    vital_names: list[str],
    horizons: list[int],
    artifact_dir: Path,
) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    figdir = artifact_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    if history:
        epochs = [h["epoch"] for h in history]
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
        axes[0].plot(epochs, [h["train"]["loss"] for h in history], label="train")
        axes[0].plot(epochs, [h["val"]["loss"] for h in history], label="val")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("loss")
        axes[0].set_title("LSTM training")
        axes[0].legend()
        axes[0].grid(True, linestyle="--", alpha=0.45)
        axes[1].plot(epochs, [h["train"]["acc"] for h in history], label="train acc")
        axes[1].plot(epochs, [h["val"]["acc"] for h in history], label="val acc")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("risk accuracy")
        axes[1].set_title("Risk head")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.45)
        fig.tight_layout()
        p = figdir / "fig_lstm_training.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        out["fig_training"] = p

    if "test" in reports:
        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        x = np.arange(len(vital_names))
        width = 0.25
        for i, h in enumerate(horizons):
            maes = [reports["test"]["per_horizon"][str(h)]["vitals"][n]["mae"] for n in vital_names]
            ax.bar(x + (i - 1) * width, maes, width, label=f"+{h} min")
        ax.set_xticks(x)
        ax.set_xticklabels(vital_names, rotation=20)
        ax.set_ylabel("MAE (raw units)")
        ax.set_title("Test vital MAE by horizon")
        ax.legend()
        ax.grid(True, axis="y", linestyle="--", alpha=0.45)
        fig.tight_layout()
        p = figdir / "fig_lstm_mae.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        out["fig_mae"] = p
    current = ROOT / "report" / "current"
    archive_mimic = ROOT / "report" / "archive" / "mimic"
    dest = archive_mimic if "mimic" in str(artifact_dir).lower() else current
    dest.mkdir(parents=True, exist_ok=True)
    for p in out.values():
        if p.exists() and p.suffix.lower() == ".png":
            (dest / p.name).write_bytes(p.read_bytes())
    return out


def parse_args(argv: list[str] | None = None) -> LSTMConfig:
    p = argparse.ArgumentParser(description="Train LSTM vital-sign / risk model.")
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    p.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-samples", type=int, default=None)
    args = p.parse_args(argv)
    return LSTMConfig(
        processed_dir=args.processed_dir,
        artifact_dir=args.artifact_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
    )


if __name__ == "__main__":
    run_lstm(parse_args())
