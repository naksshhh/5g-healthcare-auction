"""
eICU-CRD vitalPeriodic preprocessing for LSTM vital-sign forecasting.

Uses *real* bedside-monitor numerics: 5-minute medians of 1-minute samples
from ``vitalPeriodic``, with NIBP filled from ``vitalAperiodic`` when no
arterial line is present.

Default source is the open-access eICU-CRD Demo v2.0.1 (same schema as the
credentialed full eICU-CRD v2.0). Full-corpus files can be dropped in the
same directory after PhysioNet access.

Outputs the same window tensors consumed by ``02_lstm_model.py``.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "eicu"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed"
EICU_DEMO_BASE = "https://physionet.org/files/eicu-crd-demo/2.0.1"
EICU_DEMO_FILES = (
    "vitalPeriodic.csv.gz",
    "vitalAperiodic.csv.gz",
    "patient.csv.gz",
    "diagnosis.csv.gz",
    "admissionDx.csv.gz",
)

VITAL_FEATURES: tuple[str, ...] = (
    "heart_rate",
    "spo2",
    "sbp",
    "dbp",
    "temperature",
)
SEQUENCE_FEATURES: tuple[str, ...] = VITAL_FEATURES + ("fall",)
HORIZONS_MIN: tuple[int, ...] = (5, 10, 15)

PHYSIO_BOUNDS: dict[str, tuple[float, float]] = {
    "heart_rate": (40.0, 180.0),
    "spo2": (80.0, 100.0),
    "sbp": (80.0, 220.0),
    "dbp": (40.0, 140.0),
    "temperature": (34.5, 41.0),
    "fall": (0.0, 1.0),
}

# Implausible raw values are treated as missing (before clip / interpolate).
PLAUSIBLE: dict[str, tuple[float, float]] = {
    "heart_rate": (20.0, 250.0),
    "spo2": (50.0, 100.0),
    "sbp": (40.0, 280.0),
    "dbp": (20.0, 180.0),
    "temperature": (30.0, 45.0),
}

UNIT_TO_DISEASE: dict[str, str] = {
    "ccu-cticu": "Heart Disease",
    "csicu": "Heart Disease",
    "cticu": "Heart Disease",
    "cardiac icu": "Heart Disease",
    "ccu": "Heart Disease",
    "micu": "Respiratory Failure",
    "neuro icu": "Neurologic",
    "med-surg icu": "General",
    "sicu": "General",
}

CARDIAC_KEYS = (
    "cardiovascular",
    "acute coronary",
    "myocardial",
    "heart failure",
    "chf",
    "arrhythmia",
    "cabg",
    "cardiac",
    "ami",
    "nstemi",
    "stemi",
    "hypertension",
    "unstable angina",
)
EMERGENCY_KEYS = (
    "respiratory",
    "pulmonary",
    "sepsis",
    "septic",
    "trauma",
    "shock",
    "asthma",
    "copd",
    "pneumonia",
    "arrest",
    "overdose",
    "neurologic",
    "stroke",
    "hemorrhage",
    "respiratory failure",
)
DIABETES_KEYS = ("diabetes", "dka", "glucose")
HYPERTENSION_KEYS = ("hypertension",)
ASTHMA_KEYS = ("asthma",)


@dataclass
class PreprocessConfig:
    data_path: Path = DEFAULT_DATA_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    sample_period_min: int = 5
    lookback_min: int = 150  # 30 samples on the native 5-min grid
    horizons_min: tuple[int, ...] = HORIZONS_MIN
    window_stride_min: int = 30
    interp_limit_min: int = 15
    nibp_tolerance_min: int = 30
    max_windows_per_stay: int = 48
    min_bp_frac: float = 0.15
    test_size: float = 0.15
    val_size: float = 0.15
    seed: int = 42
    max_patients: int | None = None
    fetch: bool = True

    def __post_init__(self) -> None:
        self.data_path = Path(self.data_path)
        self.output_dir = Path(self.output_dir)
        self.horizons_min = tuple(int(h) for h in self.horizons_min)
        if self.sample_period_min <= 0:
            raise ValueError("sample_period_min must be > 0")
        if self.lookback_min % self.sample_period_min != 0:
            raise ValueError("lookback_min must be a multiple of sample_period_min")
        for h in self.horizons_min:
            if h % self.sample_period_min != 0:
                raise ValueError(f"horizon {h} must be a multiple of sample_period_min")
        if self.window_stride_min % self.sample_period_min != 0:
            raise ValueError("window_stride_min must be a multiple of sample_period_min")

    @property
    def lookback_steps(self) -> int:
        return self.lookback_min // self.sample_period_min

    @property
    def stride_steps(self) -> int:
        return self.window_stride_min // self.sample_period_min

    @property
    def horizon_steps(self) -> tuple[int, ...]:
        return tuple(h // self.sample_period_min for h in self.horizons_min)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def _eicu_file(data_dir: Path, name: str) -> Path:
    direct = data_dir / name
    if direct.exists():
        return direct
    nested = data_dir / "eicu-crd-demo" / name
    if nested.exists():
        return nested
    return direct


def fetch_eicu_demo(data_dir: Path, files: Iterable[str] = EICU_DEMO_FILES) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        dest = data_dir / name
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"[preprocess] have {name} ({dest.stat().st_size} bytes)")
            continue
        url = f"{EICU_DEMO_BASE}/{name}"
        print(f"[preprocess] downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "eicu-lstm-pipeline/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as fh:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                fh.write(chunk)
        print(f"[preprocess] wrote {dest} ({dest.stat().st_size} bytes)")


# ---------------------------------------------------------------------------
# Clinical labels
# ---------------------------------------------------------------------------
def _norm_text(x: object) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip().lower()


def infer_disease(unittype: object, apache: object, diagnosis: object) -> str:
    blob = " ".join((_norm_text(unittype), _norm_text(apache), _norm_text(diagnosis)))
    if any(k in blob for k in ASTHMA_KEYS):
        return "Asthma"
    if any(k in blob for k in HYPERTENSION_KEYS) and "cardiovascular" in blob:
        return "Hypertension"
    if any(k in blob for k in CARDIAC_KEYS):
        return "Heart Disease"
    if any(k in blob for k in DIABETES_KEYS):
        return "Diabetes Mellitus"
    if any(k in blob for k in EMERGENCY_KEYS):
        if "asthma" in blob:
            return "Asthma"
        if "trauma" in blob:
            return "Trauma"
        if "sepsis" in blob or "septic" in blob:
            return "Sepsis"
        if "neuro" in blob or "stroke" in blob:
            return "Neurologic"
        return "Respiratory Failure"
    unit = _norm_text(unittype)
    if unit in UNIT_TO_DISEASE:
        return UNIT_TO_DISEASE[unit]
    return "General"


def vital_risk_class(
    heart_rate: np.ndarray,
    spo2: np.ndarray,
    sbp: np.ndarray,
    dbp: np.ndarray,
    temperature: np.ndarray,
    fall: np.ndarray,
) -> np.ndarray:
    """Vectorized NEWS2-style risk from instantaneous vitals."""
    score = np.zeros(heart_rate.shape, dtype=np.int32)
    score = score + np.where((heart_rate < 50) | (heart_rate > 110), 1, 0)
    score = score + np.where((heart_rate < 40) | (heart_rate > 130), 1, 0)
    score = score + np.where(spo2 < 94, 1, 0)
    score = score + np.where(spo2 < 91, 1, 0)
    score = score + np.where((sbp < 100) | (sbp > 160) | (dbp < 60) | (dbp > 100), 1, 0)
    score = score + np.where((sbp < 90) | (sbp > 180), 1, 0)
    score = score + np.where((temperature < 36.0) | (temperature > 38.0), 1, 0)
    score = score + np.where(fall >= 0.5, 2, 0)
    return np.where(score >= 3, 2, np.where(score >= 1, 1, 0)).astype(np.int64)


# ---------------------------------------------------------------------------
# Loading / cleaning
# ---------------------------------------------------------------------------
def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"eICU table not found: {path}")
    return pd.read_csv(path, **kwargs)


def _celsius(temp: pd.Series) -> pd.Series:
    t = pd.to_numeric(temp, errors="coerce")
    fahrenheit = t > 45.0
    t = t.mask(fahrenheit, (t - 32.0) * 5.0 / 9.0)
    return t


def _mask_implausible(s: pd.Series, lo: float, hi: float) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return x.where((x >= lo) & (x <= hi))


def load_patient_table(data_dir: Path) -> pd.DataFrame:
    patient = _read_csv(
        _eicu_file(data_dir, "patient.csv.gz"),
        usecols=lambda c: c
        in {
            "patientunitstayid",
            "uniquepid",
            "unittype",
            "apacheadmissiondx",
            "hospitaldischargestatus",
            "unitdischargestatus",
            "gender",
            "age",
        },
    )
    diag_path = _eicu_file(data_dir, "diagnosis.csv.gz")
    adm_path = _eicu_file(data_dir, "admissionDx.csv.gz")
    diag_txt = pd.Series("", index=patient.index, dtype=object)
    if diag_path.exists():
        diag = _read_csv(diag_path, usecols=lambda c: c in {"patientunitstayid", "diagnosisstring"})
        if not diag.empty:
            blob = (
                diag.dropna(subset=["diagnosisstring"])
                .groupby("patientunitstayid")["diagnosisstring"]
                .agg(lambda s: " | ".join(str(x) for x in s.head(8)))
            )
            diag_txt = patient["patientunitstayid"].map(blob).fillna("")
    if adm_path.exists():
        adm = _read_csv(adm_path)
        name_col = next(
            (c for c in ("admitdxname", "admitDxName", "admitdxpath") if c in adm.columns),
            None,
        )
        if name_col is not None:
            adm_blob = (
                adm.dropna(subset=[name_col])
                .groupby("patientunitstayid")[name_col]
                .agg(lambda s: " | ".join(str(x) for x in s.head(6)))
            )
            extra = patient["patientunitstayid"].map(adm_blob).fillna("")
            diag_txt = (diag_txt.astype(str) + " | " + extra.astype(str)).str.strip(" |")

    out = pd.DataFrame(
        {
            "patient_id": patient["patientunitstayid"].astype(np.int64),
            "unique_pid": patient["uniquepid"].astype(str) if "uniquepid" in patient.columns else "",
            "unittype": patient["unittype"].astype(str) if "unittype" in patient.columns else "",
            "apacheadmissiondx": (
                patient["apacheadmissiondx"].astype(str) if "apacheadmissiondx" in patient.columns else ""
            ),
            "diagnosis": diag_txt.astype(str),
        }
    )
    out["disease"] = [
        infer_disease(u, a, d)
        for u, a, d in zip(out["unittype"], out["apacheadmissiondx"], out["diagnosis"])
    ]
    return out


def load_vitals(data_dir: Path, cfg: PreprocessConfig) -> pd.DataFrame:
    usecols = [
        "patientunitstayid",
        "observationoffset",
        "temperature",
        "sao2",
        "heartrate",
        "systemicsystolic",
        "systemicdiastolic",
    ]
    vp = _read_csv(_eicu_file(data_dir, "vitalPeriodic.csv.gz"), usecols=usecols)
    vp = vp.rename(columns={"patientunitstayid": "patient_id", "observationoffset": "offset"})
    vp["offset"] = pd.to_numeric(vp["offset"], errors="coerce")
    vp = vp.dropna(subset=["patient_id", "offset"])
    vp["patient_id"] = vp["patient_id"].astype(np.int64)
    vp["offset"] = vp["offset"].astype(np.int64)
    vp = vp[vp["offset"] >= 0].copy()

    vp["heart_rate"] = _mask_implausible(vp["heartrate"], *PLAUSIBLE["heart_rate"])
    vp["spo2"] = _mask_implausible(vp["sao2"], *PLAUSIBLE["spo2"])
    vp["sbp_inv"] = _mask_implausible(vp["systemicsystolic"], *PLAUSIBLE["sbp"])
    vp["dbp_inv"] = _mask_implausible(vp["systemicdiastolic"], *PLAUSIBLE["dbp"])
    vp["temperature"] = _mask_implausible(_celsius(vp["temperature"]), *PLAUSIBLE["temperature"])

    period = int(cfg.sample_period_min)
    vp["t"] = (vp["offset"] / period).round().astype(np.int64) * period
    vp["sbp"] = vp["sbp_inv"]
    vp["dbp"] = vp["dbp_inv"]

    va_path = _eicu_file(data_dir, "vitalAperiodic.csv.gz")
    if va_path.exists():
        va = _read_csv(
            va_path,
            usecols=lambda c: c
            in {
                "patientunitstayid",
                "observationoffset",
                "noninvasivesystolic",
                "noninvasivediastolic",
            },
        )
        va = va.rename(columns={"patientunitstayid": "patient_id", "observationoffset": "offset"})
        va["offset"] = pd.to_numeric(va["offset"], errors="coerce")
        va = va.dropna(subset=["patient_id", "offset"])
        va["patient_id"] = va["patient_id"].astype(np.int64)
        va["t"] = (va["offset"].astype(np.int64) / period).round().astype(np.int64) * period
        va["sbp_ni"] = _mask_implausible(va["noninvasivesystolic"], *PLAUSIBLE["sbp"])
        va["dbp_ni"] = _mask_implausible(va["noninvasivediastolic"], *PLAUSIBLE["dbp"])
        nibp = (
            va.dropna(subset=["sbp_ni", "dbp_ni"], how="all")
            .groupby(["patient_id", "t"], as_index=False)[["sbp_ni", "dbp_ni"]]
            .mean()
        )
        vp = vp.merge(nibp, on=["patient_id", "t"], how="left")
        vp["sbp"] = vp["sbp_inv"].fillna(vp["sbp_ni"])
        vp["dbp"] = vp["dbp_inv"].fillna(vp["dbp_ni"])

    agg = vp.groupby(["patient_id", "t"], as_index=False)[list(VITAL_FEATURES)].mean()
    return agg


def _fill_series(frame: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame | None:
    period = int(cfg.sample_period_min)
    limit = max(1, int(cfg.interp_limit_min) // period)
    t = frame["t"].to_numpy(dtype=np.int64)
    if t.size < cfg.lookback_steps + max(cfg.horizon_steps):
        return None
    t0, t1 = int(t.min()), int(t.max())
    grid = np.arange(t0, t1 + period, period, dtype=np.int64)
    out = pd.DataFrame({"t": grid}).merge(frame, on="t", how="left")
    for col in VITAL_FEATURES:
        out[col] = out[col].interpolate(limit=limit, limit_area="inside")
    # HR / SpO2 must be almost complete; BP / temp may use stay medians.
    hr_ok = out["heart_rate"].notna().mean()
    spo2_ok = out["spo2"].notna().mean()
    bp_ok = out["sbp"].notna().mean()
    if hr_ok < 0.70 or spo2_ok < 0.70 or bp_ok < cfg.min_bp_frac:
        return None
    stay_med = {c: float(out[c].median()) for c in VITAL_FEATURES if out[c].notna().any()}
    defaults = {
        "heart_rate": 80.0,
        "spo2": 97.0,
        "sbp": 120.0,
        "dbp": 70.0,
        "temperature": 37.0,
    }
    for col in VITAL_FEATURES:
        fill = stay_med.get(col, defaults[col])
        out[col] = out[col].ffill(limit=limit).bfill(limit=limit).fillna(fill)
        lo, hi = PHYSIO_BOUNDS[col]
        out[col] = out[col].clip(lo, hi)
    out["fall"] = 0.0
    return out


def build_stay_series(
    vitals: pd.DataFrame,
    meta: pd.DataFrame,
    cfg: PreprocessConfig,
) -> tuple[list[np.ndarray], pd.DataFrame]:
    keep_ids = set(meta["patient_id"].tolist())
    series_list: list[np.ndarray] = []
    rows: list[dict] = []
    grouped = vitals.groupby("patient_id", sort=False)
    n_used = 0
    for pid, g in grouped:
        if pid not in keep_ids:
            continue
        filled = _fill_series(g[["t", *VITAL_FEATURES]], cfg)
        if filled is None:
            continue
        arr = filled.loc[:, list(SEQUENCE_FEATURES)].to_numpy(dtype=np.float64)
        min_len = cfg.lookback_steps + max(cfg.horizon_steps)
        if arr.shape[0] < min_len:
            continue
        series_list.append(arr)
        mrow = meta.loc[meta["patient_id"] == pid].iloc[0]
        last = arr[-1]
        rows.append(
            {
                "patient_id": int(pid),
                "n_samples": int(arr.shape[0]),
                "duration_min": int(arr.shape[0] * cfg.sample_period_min),
                "heart_rate": float(last[0]),
                "spo2": float(last[1]),
                "sbp": float(last[2]),
                "dbp": float(last[3]),
                "temperature": float(last[4]),
                "fall": 0,
                "disease": str(mrow["disease"]),
                "unittype": str(mrow.get("unittype", "")),
                "apacheadmissiondx": str(mrow.get("apacheadmissiondx", "")),
            }
        )
        n_used += 1
        if cfg.max_patients is not None and n_used >= int(cfg.max_patients):
            break
    stay_meta = pd.DataFrame(rows)
    return series_list, stay_meta


# ---------------------------------------------------------------------------
# Sliding windows
# ---------------------------------------------------------------------------
def build_windows_from_stays(
    series_list: list[np.ndarray],
    stay_ids: np.ndarray,
    diseases: np.ndarray,
    lookback: int,
    horizon_steps: tuple[int, ...],
    stride: int,
    max_windows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    max_h = max(horizon_steps)
    n_v = len(VITAL_FEATURES)
    fall_ch = SEQUENCE_FEATURES.index("fall")
    x_parts, y_v_parts, y_r_parts, pid_parts, t0_parts, dis_parts = [], [], [], [], [], []

    for arr, pid, disease in zip(series_list, stay_ids, diseases):
        t_len = arr.shape[0]
        last_start = t_len - lookback - max_h
        if last_start < 0:
            continue
        starts = np.arange(0, last_start + 1, stride, dtype=np.int64)
        if len(starts) > max_windows:
            starts = starts[-max_windows:]
        for t0 in starts:
            window = arr[t0 : t0 + lookback]
            y_h, r_h = [], []
            for h in horizon_steps:
                future = arr[t0 + lookback + h - 1]
                y_h.append(future[:n_v])
                r_h.append(
                    vital_risk_class(
                        np.array([future[0]]),
                        np.array([future[1]]),
                        np.array([future[2]]),
                        np.array([future[3]]),
                        np.array([future[4]]),
                        np.array([future[fall_ch]]),
                    )[0]
                )
            x_parts.append(window.astype(np.float32))
            y_v_parts.append(np.stack(y_h, axis=0).astype(np.float32))
            y_r_parts.append(np.asarray(r_h, dtype=np.int64))
            pid_parts.append(int(pid))
            t0_parts.append(int(t0))
            dis_parts.append(str(disease))

    if not x_parts:
        raise RuntimeError("No valid eICU windows. Check filters / lookback.")
    return (
        np.stack(x_parts, axis=0),
        np.stack(y_v_parts, axis=0),
        np.stack(y_r_parts, axis=0),
        np.asarray(pid_parts, dtype=np.int64),
        np.asarray(t0_parts, dtype=np.int64),
        np.asarray(dis_parts),
    )


def _scale_split(
    scaler: StandardScaler,
    x: np.ndarray,
    y_vitals: np.ndarray,
    fit: bool,
) -> tuple[np.ndarray, np.ndarray]:
    n_v = len(VITAL_FEATURES)
    n, t, _ = x.shape
    x_v = x[:, :, :n_v].reshape(-1, n_v)
    if fit:
        scaler.fit(x_v)
    x_vs = scaler.transform(x_v).reshape(n, t, n_v).astype(np.float32)
    x_out = x.copy()
    x_out[:, :, :n_v] = x_vs
    n_w, h, _ = y_vitals.shape
    y_s = scaler.transform(y_vitals.reshape(-1, n_v)).reshape(n_w, h, n_v).astype(np.float32)
    return x_out, y_s


def snapshot_risk_from_vitals(df: pd.DataFrame) -> np.ndarray:
    return vital_risk_class(
        df["heart_rate"].to_numpy(),
        df["spo2"].to_numpy(),
        df["sbp"].to_numpy(),
        df["dbp"].to_numpy(),
        df["temperature"].to_numpy(),
        df["fall"].to_numpy() if "fall" in df.columns else np.zeros(len(df)),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_preprocess(cfg: PreprocessConfig | None = None) -> dict[str, Path]:
    cfg = cfg or PreprocessConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = cfg.data_path
    if data_dir.is_file():
        data_dir = data_dir.parent
    if cfg.fetch:
        try:
            fetch_eicu_demo(data_dir)
        except Exception as exc:
            if not _eicu_file(data_dir, "vitalPeriodic.csv.gz").exists():
                raise RuntimeError(
                    "Could not download eICU-CRD Demo and no local vitalPeriodic.csv.gz "
                    f"was found in {data_dir}. Full eICU-CRD is credentialed on PhysioNet."
                ) from exc
            print(f"[preprocess] download skipped ({exc}); using local files")

    print(f"[preprocess] loading eICU tables from {data_dir}")
    meta = load_patient_table(data_dir)
    vitals = load_vitals(data_dir, cfg)
    print(
        f"[preprocess] stays_in_patient_table={len(meta)}  "
        f"vital_rows={len(vitals)}  unique_stays_with_vitals={vitals['patient_id'].nunique()}"
    )

    series_list, stay_meta = build_stay_series(vitals, meta, cfg)
    if stay_meta.empty:
        raise RuntimeError("No eICU stays survived quality filters.")
    print(
        f"[preprocess] usable stays={len(stay_meta)}  "
        f"median duration={stay_meta['duration_min'].median():.0f} min  "
        f"diseases={stay_meta['disease'].value_counts().to_dict()}"
    )

    stay_meta["snapshot_risk"] = snapshot_risk_from_vitals(stay_meta)
    idx = np.arange(len(stay_meta))
    strat = stay_meta["snapshot_risk"].to_numpy()
    # Stratify only if every class has at least 2 members in this cut.
    def _can_stratify(labels: np.ndarray) -> bool:
        _, counts = np.unique(labels, return_counts=True)
        return counts.min() >= 2 and len(counts) > 1

    idx_train, idx_tmp = train_test_split(
        idx,
        test_size=cfg.test_size + cfg.val_size,
        random_state=cfg.seed,
        stratify=strat if _can_stratify(strat) else None,
    )
    rel_val = cfg.val_size / (cfg.test_size + cfg.val_size)
    strat_tmp = strat[idx_tmp]
    idx_val, idx_test = train_test_split(
        idx_tmp,
        test_size=1.0 - rel_val,
        random_state=cfg.seed,
        stratify=strat_tmp if _can_stratify(strat_tmp) else None,
    )

    scaler = StandardScaler()
    artifacts: dict[str, Path] = {}
    split_map = {"train": idx_train, "val": idx_val, "test": idx_test}

    for split_name, split_idx in split_map.items():
        sub_series = [series_list[i] for i in split_idx]
        sub_meta = stay_meta.iloc[split_idx]
        x, y_v, y_r, pids, t0, diseases = build_windows_from_stays(
            sub_series,
            sub_meta["patient_id"].to_numpy(),
            sub_meta["disease"].to_numpy(),
            lookback=cfg.lookback_steps,
            horizon_steps=cfg.horizon_steps,
            stride=cfg.stride_steps,
            max_windows=cfg.max_windows_per_stay,
        )
        x_s, y_vs = _scale_split(scaler, x, y_v, fit=(split_name == "train"))
        out_path = cfg.output_dir / f"windows_{split_name}.npz"
        np.savez_compressed(
            out_path,
            X=x_s,
            y_vitals=y_vs,
            y_vitals_raw=y_v,
            y_risk=y_r,
            patient_id=pids,
            window_t0=t0,
            disease=diseases,
            feature_names=np.array(SEQUENCE_FEATURES),
            vital_names=np.array(VITAL_FEATURES),
            horizons_min=np.array(cfg.horizons_min, dtype=np.int32),
            sample_period_min=np.int32(cfg.sample_period_min),
        )
        artifacts[split_name] = out_path
        print(
            f"[preprocess] {split_name:5s}  stays={len(split_idx):5d}  "
            f"windows={x_s.shape[0]:6d}  X={tuple(x_s.shape)}  -> {out_path.name}"
        )
        meta_split = sub_meta.copy()
        meta_split["split"] = split_name
        meta_path = cfg.output_dir / f"patients_{split_name}.csv"
        meta_split.to_csv(meta_path, index=False)
        artifacts[f"meta_{split_name}"] = meta_path

    scaler_path = cfg.output_dir / "scaler.joblib"
    joblib.dump(
        {
            "scaler": scaler,
            "feature_names": list(SEQUENCE_FEATURES),
            "vital_names": list(VITAL_FEATURES),
            "physio_bounds": PHYSIO_BOUNDS,
            "dataset": "eicu-crd-demo",
        },
        scaler_path,
    )
    artifacts["scaler"] = scaler_path

    present = stay_meta.loc[:, list(VITAL_FEATURES)].to_numpy(dtype=np.float64)
    present_norm = scaler.transform(present).astype(np.float32)
    snapshot_path = cfg.output_dir / "patient_snapshot.npz"
    np.savez_compressed(
        snapshot_path,
        patient_id=stay_meta["patient_id"].to_numpy(),
        vitals_raw=present.astype(np.float32),
        vitals_norm=present_norm,
        fall=np.zeros(len(stay_meta), dtype=np.float32),
        disease=stay_meta["disease"].to_numpy(),
        snapshot_risk=stay_meta["snapshot_risk"].to_numpy(),
        vital_names=np.array(VITAL_FEATURES),
    )
    artifacts["snapshot"] = snapshot_path

    meta_all_path = cfg.output_dir / "patients_all.csv"
    stay_meta.to_csv(meta_all_path, index=False)
    artifacts["meta_all"] = meta_all_path

    cfg_path = cfg.output_dir / "preprocess_config.json"
    payload = asdict(cfg)
    payload["data_path"] = str(cfg.data_path)
    payload["output_dir"] = str(cfg.output_dir)
    payload["horizons_min"] = list(cfg.horizons_min)
    payload["n_stays"] = int(len(stay_meta))
    payload["lookback_steps"] = cfg.lookback_steps
    payload["horizon_steps"] = list(cfg.horizon_steps)
    payload["dataset"] = "eICU-CRD Demo v2.0.1 (vitalPeriodic + vitalAperiodic)"
    payload["citation"] = (
        "Pollard et al., Sci Data 2018 (eICU-CRD); "
        "Johnson et al., PhysioNet 2021 (eICU-CRD Demo v2.0.1)"
    )
    payload["disease_counts"] = {str(k): int(v) for k, v in stay_meta["disease"].value_counts().items()}
    cfg_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    artifacts["config"] = cfg_path
    print(f"[preprocess] done. artifacts in {cfg.output_dir}")
    return artifacts


def load_windows(output_dir: Path | str, split: str = "train") -> dict[str, np.ndarray]:
    path = Path(output_dir) / f"windows_{split}.npz"
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def inverse_vitals(output_dir: Path | str, y_norm: np.ndarray) -> np.ndarray:
    bundle = joblib.load(Path(output_dir) / "scaler.joblib")
    scaler: StandardScaler = bundle["scaler"]
    shape = y_norm.shape
    flat = y_norm.reshape(-1, shape[-1])
    return scaler.inverse_transform(flat).reshape(shape)


def parse_args(argv: list[str] | None = None) -> PreprocessConfig:
    p = argparse.ArgumentParser(description="Preprocess eICU vitalPeriodic for LSTM forecasting.")
    p.add_argument("--data-path", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--lookback-min", type=int, default=150)
    p.add_argument("--window-stride-min", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-patients", type=int, default=None, help="Optional cap on ICU stays.")
    p.add_argument("--no-fetch", action="store_true")
    args = p.parse_args(argv)
    return PreprocessConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        lookback_min=args.lookback_min,
        window_stride_min=args.window_stride_min,
        seed=args.seed,
        max_patients=args.max_patients,
        fetch=not args.no_fetch,
        horizons_min=HORIZONS_MIN,
    )


if __name__ == "__main__":
    run_preprocess(parse_args())
