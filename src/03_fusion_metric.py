"""
Multi-variable fusion of predicted vitals into a customer criticality score.

Each LSTM horizon (5 / 10 / 15 min) is scored with NEWS2-inspired piecewise
maps on HR, SpO2, SBP, DBP and temperature, then mixed with fall status and
the learned risk-head probability. A recency-weighted + peak fusion produces
a single customer score ``c_n ∈ [0, 1]`` for the double-auction preference
parameter ``rho_wk``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts"

# Clinical sub-score weights (sum to 1). Hypoxemia dominates URLLC urgency.
VITAL_WEIGHTS: dict[str, float] = {
    "heart_rate": 0.16,
    "spo2": 0.26,
    "sbp": 0.18,
    "dbp": 0.10,
    "temperature": 0.08,
}
FALL_WEIGHT = 0.12
RISK_HEAD_WEIGHT = 0.10

# Recency weights over LSTM horizons [5, 10, 15] min.
DEFAULT_HORIZON_WEIGHTS: tuple[float, ...] = (0.50, 0.30, 0.20)
PEAK_MIX = 0.40  # fraction of fused score taken from the worst horizon


@dataclass
class FusionConfig:
    processed_dir: Path = DEFAULT_PROCESSED_DIR
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR
    peak_mix: float = PEAK_MIX
    horizon_weights: tuple[float, ...] = DEFAULT_HORIZON_WEIGHTS
    predictions_name: str = "lstm_predictions.npz"

    def __post_init__(self) -> None:
        self.processed_dir = Path(self.processed_dir)
        self.artifact_dir = Path(self.artifact_dir)
        w = np.asarray(self.horizon_weights, dtype=np.float64)
        if w.ndim != 1 or np.any(w < 0) or not np.isfinite(w).all():
            raise ValueError("horizon_weights must be a finite non-negative vector")
        s = float(w.sum())
        if s <= 0:
            raise ValueError("horizon_weights must sum to a positive value")
        self.horizon_weights = tuple(float(x) / s for x in w)
        self.peak_mix = float(np.clip(self.peak_mix, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Piecewise clinical maps  →  [0, 1]
# ---------------------------------------------------------------------------
def _band_score(
    x: np.ndarray,
    inner_lo: float,
    inner_hi: float,
    outer_lo: float,
    outer_hi: float,
) -> np.ndarray:
    """0 inside ``[inner_lo, inner_hi]``, 1 outside ``[outer_lo, outer_hi]``."""
    x = np.asarray(x, dtype=np.float64)
    low = np.where(
        x >= inner_lo,
        0.0,
        np.clip((inner_lo - x) / max(inner_lo - outer_lo, 1e-6), 0.0, 1.0),
    )
    high = np.where(
        x <= inner_hi,
        0.0,
        np.clip((x - inner_hi) / max(outer_hi - inner_hi, 1e-6), 0.0, 1.0),
    )
    return np.maximum(low, high)


def _low_score(x: np.ndarray, inner: float, outer: float) -> np.ndarray:
    """One-sided: 0 at/above ``inner``, 1 at/below ``outer`` (SpO2)."""
    x = np.asarray(x, dtype=np.float64)
    return np.clip((inner - x) / max(inner - outer, 1e-6), 0.0, 1.0)


def vital_subscores(vitals: np.ndarray, vital_names: list[str]) -> dict[str, np.ndarray]:
    """
    Map raw vitals of shape ``(..., n_vitals)`` to per-channel scores in [0, 1].

    NEWS2-aligned bands:
      HR   51–90 normal; ≤40 or ≥131 critical
      SpO2 ≥96 normal; ≤88 critical
      SBP  111–219 normal; ≤90 or ≥220 critical
      DBP  60–90 normal; ≤40 or ≥120 critical
      Temp 36.1–38.0 normal; ≤35.0 or ≥39.1 critical
    """
    name_to_idx = {n: i for i, n in enumerate(vital_names)}

    def col(name: str) -> np.ndarray:
        if name not in name_to_idx:
            raise KeyError(f"Missing vital '{name}' in {vital_names}")
        return vitals[..., name_to_idx[name]]

    scores: dict[str, np.ndarray] = {}
    if "heart_rate" in name_to_idx:
        scores["heart_rate"] = _band_score(col("heart_rate"), 51.0, 90.0, 40.0, 131.0)
    if "spo2" in name_to_idx:
        scores["spo2"] = _low_score(col("spo2"), 96.0, 88.0)
    if "sbp" in name_to_idx:
        scores["sbp"] = _band_score(col("sbp"), 111.0, 219.0, 90.0, 220.0)
    if "dbp" in name_to_idx:
        scores["dbp"] = _band_score(col("dbp"), 60.0, 90.0, 40.0, 120.0)
    if "temperature" in name_to_idx:
        scores["temperature"] = _band_score(col("temperature"), 36.1, 38.0, 35.0, 39.1)
    return scores


def risk_head_score(proba: np.ndarray) -> np.ndarray:
    """Collapse 3-class softmax to a [0, 1] severity: P(critical) + 0.5 P(moderate)."""
    p = np.asarray(proba, dtype=np.float64)
    if p.shape[-1] < 3:
        raise ValueError(f"Expected 3-class probabilities, got shape {p.shape}")
    return np.clip(p[..., 2] + 0.5 * p[..., 1], 0.0, 1.0)


def fuse_channels(
    subs: dict[str, np.ndarray],
    fall: np.ndarray,
    risk: np.ndarray,
    vital_weights: dict[str, float] | None = None,
) -> np.ndarray:
    """Weighted sum of channel scores, clipped to [0, 1]. Broadcasts over horizons."""
    weights = dict(VITAL_WEIGHTS if vital_weights is None else vital_weights)
    used = {k: v for k, v in weights.items() if k in subs}
    w_sum = sum(used.values()) + FALL_WEIGHT + RISK_HEAD_WEIGHT
    acc = np.zeros_like(risk, dtype=np.float64)
    for name, w in used.items():
        acc = acc + (w / w_sum) * np.asarray(subs[name], dtype=np.float64)
    acc = acc + (FALL_WEIGHT / w_sum) * np.asarray(fall, dtype=np.float64)
    acc = acc + (RISK_HEAD_WEIGHT / w_sum) * np.asarray(risk, dtype=np.float64)
    return np.clip(acc, 0.0, 1.0)


def fuse_horizons(
    horizon_scores: np.ndarray,
    horizon_weights: tuple[float, ...] | None = None,
    peak_mix: float = PEAK_MIX,
) -> np.ndarray:
    """
    Mix recency-weighted mean with the peak horizon so a delayed crash still
    raises the customer score. ``horizon_scores`` has shape ``(N, H)``.
    """
    s = np.asarray(horizon_scores, dtype=np.float64)
    if s.ndim != 2:
        raise ValueError(f"horizon_scores must be (N, H), got {s.shape}")
    n, h = s.shape
    w = np.asarray(
        horizon_weights if horizon_weights is not None else DEFAULT_HORIZON_WEIGHTS[:h],
        dtype=np.float64,
    )
    if w.shape[0] != h:
        if w.shape[0] > h:
            w = w[:h]
        else:
            extra = np.full(h - w.shape[0], w[-1] if w.size else 1.0)
            w = np.concatenate([w, extra])
    w = w / w.sum()
    weighted = s @ w
    peak = s.max(axis=1)
    mix = float(np.clip(peak_mix, 0.0, 1.0))
    return np.clip(mix * peak + (1.0 - mix) * weighted, 0.0, 1.0)


def criticality_from_predictions(
    vitals_raw: np.ndarray,
    vital_names: list[str],
    risk_proba: np.ndarray,
    fall: np.ndarray,
    horizon_weights: tuple[float, ...] | None = None,
    peak_mix: float = PEAK_MIX,
) -> dict[str, np.ndarray]:
    """
    Parameters
    ----------
    vitals_raw : (N, H, V)
    risk_proba : (N, H, 3)
    fall       : (N,) or (N, H) in {0, 1}

    Returns dict with per-channel scores, per-horizon scores, and fused ``c_n``.
    """
    vitals_raw = np.asarray(vitals_raw, dtype=np.float64)
    if vitals_raw.ndim != 3:
        raise ValueError(f"vitals_raw must be (N, H, V), got {vitals_raw.shape}")
    n, h, _ = vitals_raw.shape
    fall_arr = np.asarray(fall, dtype=np.float64)
    if fall_arr.ndim == 1:
        fall_arr = np.broadcast_to(fall_arr[:, None], (n, h)).astype(np.float64, copy=True)
    elif fall_arr.shape != (n, h):
        raise ValueError(f"fall must be (N,) or (N, H), got {fall_arr.shape}")

    subs = vital_subscores(vitals_raw, list(vital_names))
    risk = risk_head_score(risk_proba)
    per_h = fuse_channels(subs, fall_arr, risk)
    fused = fuse_horizons(per_h, horizon_weights=horizon_weights, peak_mix=peak_mix)
    out: dict[str, np.ndarray] = {
        "criticality": fused.astype(np.float32),
        "criticality_horizon": per_h.astype(np.float32),
        "risk_score": risk.astype(np.float32),
        "fall": fall_arr.astype(np.float32),
    }
    for name, arr in subs.items():
        out[f"score_{name}"] = arr.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def _load_predictions(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"LSTM predictions not found: {path}. Run 02_lstm_model.py first.")
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _align_fall(patient_id: np.ndarray, processed_dir: Path, n: int, h: int) -> np.ndarray:
    meta_path = processed_dir / "patients_all.csv"
    snap_path = processed_dir / "patient_snapshot.npz"
    fall = np.zeros(n, dtype=np.float64)
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        lookup = dict(zip(meta["patient_id"].astype(int), meta["fall"].astype(float)))
        fall = np.array([lookup.get(int(p), 0.0) for p in patient_id], dtype=np.float64)
    elif snap_path.exists():
        with np.load(snap_path, allow_pickle=True) as snap:
            lookup = dict(zip(snap["patient_id"].astype(int), snap["fall"].astype(float)))
            fall = np.array([lookup.get(int(p), 0.0) for p in patient_id], dtype=np.float64)
    return fall


def run_fusion(cfg: FusionConfig | None = None) -> dict[str, Path]:
    cfg = cfg or FusionConfig()
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    pred_path = cfg.processed_dir / cfg.predictions_name
    pred = _load_predictions(pred_path)

    patient_id = pred["patient_id"].astype(np.int64)
    vitals = pred["y_vitals_pred_raw"]
    proba = pred["y_risk_proba"]
    vital_names = [str(x) for x in pred["vital_names"]]
    horizons = [int(h) for h in pred["horizons_min"]]
    n, h, _ = vitals.shape
    fall = _align_fall(patient_id, cfg.processed_dir, n, h)

    hw = cfg.horizon_weights[:h] if len(cfg.horizon_weights) >= h else cfg.horizon_weights
    scores = criticality_from_predictions(
        vitals,
        vital_names,
        proba,
        fall,
        horizon_weights=hw,
        peak_mix=cfg.peak_mix,
    )
    c = scores["criticality"]
    if not np.isfinite(c).all():
        raise RuntimeError("Non-finite criticality scores produced.")
    if c.min() < 0.0 - 1e-9 or c.max() > 1.0 + 1e-9:
        raise RuntimeError(f"Criticality escaped [0, 1]: min={c.min()} max={c.max()}")
    scores["criticality"] = np.clip(c, 0.0, 1.0).astype(np.float32)

    split = pred["split"] if "split" in pred else np.array(["unknown"] * n)
    npz_path = cfg.processed_dir / "criticality_scores.npz"
    np.savez_compressed(
        npz_path,
        patient_id=patient_id,
        split=split,
        criticality=scores["criticality"],
        criticality_horizon=scores["criticality_horizon"],
        risk_score=scores["risk_score"],
        fall=scores["fall"],
        horizons_min=np.array(horizons, dtype=np.int32),
        vital_names=np.array(vital_names),
        **{k: v for k, v in scores.items() if k.startswith("score_")},
    )

    table = pd.DataFrame(
        {
            "patient_id": patient_id,
            "split": split,
            "criticality": scores["criticality"],
            **{f"c_h{h_min}": scores["criticality_horizon"][:, i] for i, h_min in enumerate(horizons)},
            "fall": scores["fall"][:, 0] if scores["fall"].ndim == 2 else scores["fall"],
        }
    )
    meta_path = cfg.processed_dir / "patients_all.csv"
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        keep = [c for c in ("patient_id", "disease", "snapshot_risk", "hr_alert", "spo2_alert", "bp_alert", "temp_alert") if c in meta.columns]
        table = table.merge(meta[keep], on="patient_id", how="left")
    csv_path = cfg.processed_dir / "criticality_scores.csv"
    table.to_csv(csv_path, index=False)

    summary = {
        "n": int(n),
        "horizons_min": horizons,
        "range": {"min": float(c.min()), "max": float(c.max()), "mean": float(c.mean()), "std": float(c.std())},
        "percentiles": {str(p): float(np.percentile(c, p)) for p in (5, 25, 50, 75, 95)},
        "in_unit_interval": bool(c.min() >= 0.0 and c.max() <= 1.0),
        "weights": {
            "vitals": dict(VITAL_WEIGHTS),
            "fall": FALL_WEIGHT,
            "risk_head": RISK_HEAD_WEIGHT,
            "horizons": list(hw),
            "peak_mix": cfg.peak_mix,
        },
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
    }
    if "snapshot_risk" in table.columns:
        summary["mean_by_snapshot_risk"] = {
            str(int(k)): float(v)
            for k, v in table.groupby("snapshot_risk")["criticality"].mean().items()
        }
    if "disease" in table.columns:
        summary["mean_by_disease"] = {
            str(k): float(v) for k, v in table.groupby("disease")["criticality"].mean().items()
        }

    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    summary_path = cfg.artifact_dir / "fusion_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"[fusion] n={n}  criticality min={c.min():.3f}  "
        f"p50={np.median(c):.3f}  mean={c.mean():.3f}  max={c.max():.3f}"
    )
    if "mean_by_snapshot_risk" in summary:
        print(f"[fusion] mean by snapshot_risk={summary['mean_by_snapshot_risk']}")
    if "mean_by_disease" in summary:
        print(f"[fusion] mean by disease={summary['mean_by_disease']}")
    print(f"[fusion] wrote {npz_path.name} and {csv_path.name}")
    return {"scores_npz": npz_path, "scores_csv": csv_path, "summary": summary_path}


def parse_args(argv: list[str] | None = None) -> FusionConfig:
    p = argparse.ArgumentParser(description="Fuse predicted vitals into [0, 1] criticality scores.")
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    p.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    p.add_argument("--peak-mix", type=float, default=PEAK_MIX)
    args = p.parse_args(argv)
    return FusionConfig(
        processed_dir=args.processed_dir,
        artifact_dir=args.artifact_dir,
        peak_mix=args.peak_mix,
    )


if __name__ == "__main__":
    run_fusion(parse_args())
