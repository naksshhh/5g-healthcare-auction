"""
MIMIC-III + MIMIC-IV *numeric* vitals → LSTM windows.

Only monitor numerics are fetched:
  * MIMIC-III matched ``*n`` records (HR / SpO2 / NBP-ABP), never ECG/PPG ``.dat``
  * MIMIC-IV Waveform ``*n.csv.gz`` tables, never segment waveform files

Resampled to the same 5-minute grid as eICU-CRD. Writes
``data/processed_mimic`` so ``data/processed`` (eICU) is not overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import load_step

DEFAULT_DATA_DIR = ROOT / "data" / "mimic"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed_mimic"
RECORDS_URL = "https://physionet.org/files/mimic3wdb-matched/1.0/RECORDS-numerics"
PN_DIR = "mimic3wdb-matched/1.0"
NEONATE_URL = "https://physionet.org/files/mimic3wdb/1.0/RECORDS-neonates"
MIMIC4_BASE = "https://physionet.org/files/mimic4wdb/0.1.0"
MIMIC4_RECORDS_URL = f"{MIMIC4_BASE}/RECORDS"
COUNTER_HZ = 999.52  # MIMIC-IV numeric time ticks → seconds

HR_NAMES = ("hr", "heart rate", "pulse", "hr ")
SPO2_NAMES = ("spo2", "sao2", "%spo2", "sp o2")
SBP_PREF = (("abpsys", "abp sys", "abps", "art sys", "abp systolic"), ("nbpsys", "nbp sys", "nbps", "nbp systolic"))
DBP_PREF = (("abpdias", "abp dias", "abpd", "art dias", "abp diastolic"), ("nbpdias", "nbp dias", "nbpd", "nbp diastolic"))
TEMP_NAMES = ("temp", "tblood", "tskin", "tcore", "temperature")


@dataclass
class MimicConfig:
    data_dir: Path = DEFAULT_DATA_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    sample_period_min: int = 5
    lookback_min: int = 150
    horizons_min: tuple[int, ...] = (5, 10, 15)
    window_stride_min: int = 30
    interp_limit_min: int = 15
    max_windows_per_stay: int = 48
    min_bp_frac: float = 0.15
    max_stays: int = 3500
    max_records_scan: int = 8000
    workers: int = 6
    seed: int = 42
    test_size: float = 0.15
    val_size: float = 0.15
    fetch: bool = True
    use_iv: bool = True
    use_iii: bool = True

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.horizons_min = tuple(int(h) for h in self.horizons_min)

    @property
    def lookback_steps(self) -> int:
        return self.lookback_min // self.sample_period_min

    @property
    def stride_steps(self) -> int:
        return self.window_stride_min // self.sample_period_min

    @property
    def horizon_steps(self) -> tuple[int, ...]:
        return tuple(h // self.sample_period_min for h in self.horizons_min)


def _download(url: str, dest: Path, timeout: int = 120) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "mimic-lstm-pipeline/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, dest.open("wb") as fh:
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            fh.write(chunk)
    return dest


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def _find_channel(names: list[str], candidates: tuple[str, ...]) -> int | None:
    norms = [_norm(n) for n in names]
    compact = [n.replace(" ", "") for n in norms]
    for cand in candidates:
        c = _norm(cand)
        cc = c.replace(" ", "")
        if not cc:
            continue
        for i, (n, k) in enumerate(zip(norms, compact)):
            if n == c or k == cc or n.startswith(c) or (len(cc) >= 4 and cc in k):
                return i
    return None


def _pick_bp(names: list[str], groups: tuple[tuple[str, ...], ...]) -> int | None:
    for group in groups:
        hit = _find_channel(names, group)
        if hit is not None:
            return hit
    return None


def subject_from_record(rec: str) -> int:
    m = re.search(r"p(\d{6,8})", rec)
    if m:
        return int(m.group(1))
    digest = hashlib.md5(rec.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 10_000_000


def stay_id_from_record(rec: str) -> int:
    digest = hashlib.md5(rec.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % 2_000_000_000


def _resample_5min(sig: np.ndarray, fs: float, period_min: int) -> np.ndarray:
    """Median-resample (T, C) at ``fs`` Hz to one row per ``period_min`` minutes."""
    if sig.ndim != 2:
        raise ValueError(sig.shape)
    samples = max(1, int(round(fs * 60.0 * period_min)))
    n = (sig.shape[0] // samples) * samples
    if n < samples:
        return np.empty((0, sig.shape[1]), dtype=np.float64)
    block = sig[:n].reshape(-1, samples, sig.shape[1])
    with np.errstate(all="ignore"):
        out = np.nanmedian(block, axis=1)
    return out.astype(np.float64)


def _cache_path(cfg: MimicConfig, rec: str) -> Path:
    return cfg.data_dir / "cache" / f"{stay_id_from_record(rec)}.npz"


def _load_cache(cache: Path, cfg: MimicConfig) -> dict | None:
    if not cache.exists():
        return None
    with np.load(cache, allow_pickle=True) as z:
        arr = z["vitals"]
        if arr.shape[0] < cfg.lookback_steps + max(cfg.horizon_steps):
            return None
        return {
            "patient_id": int(z["patient_id"]),
            "subject_id": int(z["subject_id"]),
            "record": str(z["record"]),
            "source": str(z["source"]) if "source" in z.files else "mimic3",
            "vitals": arr,
        }


def _save_cache(cache: Path, rec: str, grid: np.ndarray, source: str) -> dict:
    cache.parent.mkdir(parents=True, exist_ok=True)
    pid = stay_id_from_record(rec)
    sid = subject_from_record(rec)
    np.savez_compressed(
        cache,
        vitals=grid.astype(np.float32),
        patient_id=np.int64(pid),
        subject_id=np.int64(sid),
        record=np.array(rec),
        source=np.array(source),
    )
    return {"patient_id": pid, "subject_id": sid, "record": rec, "source": source, "vitals": grid}


def _apply_plausible(grid: np.ndarray) -> np.ndarray:
    plausible = (
        (20.0, 250.0),
        (50.0, 100.0),
        (40.0, 280.0),
        (20.0, 180.0),
        (30.0, 45.0),
    )
    out = grid.copy()
    for j, (lo, hi) in enumerate(plausible):
        col = out[:, j]
        out[:, j] = np.where((col >= lo) & (col <= hi), col, np.nan)
    return out


def _map_named_columns(names: list[str]) -> dict[str, int] | None:
    hr_i = _find_channel(names, HR_NAMES)
    spo2_i = _find_channel(names, SPO2_NAMES)
    sbp_i = _pick_bp(names, SBP_PREF)
    dbp_i = _pick_bp(names, DBP_PREF)
    if hr_i is None or spo2_i is None or sbp_i is None or dbp_i is None:
        return None
    out = {"hr": hr_i, "spo2": spo2_i, "sbp": sbp_i, "dbp": dbp_i}
    temp_i = _find_channel(names, TEMP_NAMES)
    if temp_i is not None:
        out["temp"] = temp_i
    return out


def _read_one_record(rec: str, cfg: MimicConfig) -> dict | None:
    """MIMIC-III numerics only. Record path must end with ``n``."""
    import wfdb

    rec = rec.strip().strip("/")
    if not rec or not rec.endswith("n"):
        return None
    cache = _cache_path(cfg, rec)
    hit = _load_cache(cache, cfg)
    if hit is not None:
        return hit

    parts = rec.strip("/").split("/")
    rec_name = parts[-1]
    if not rec_name.endswith("n"):
        return None
    pn_dir = PN_DIR if len(parts) == 1 else f"{PN_DIR}/{'/'.join(parts[:-1])}"
    try:
        header = wfdb.rdheader(rec_name, pn_dir=pn_dir)
    except Exception:
        return None
    names = [str(s) for s in (header.sig_name or [])]
    mapping = _map_named_columns(names)
    if mapping is None:
        return None
    try:
        rec_data = wfdb.rdrecord(rec_name, pn_dir=pn_dir, physical=True)
    except Exception:
        return None
    sig = np.asarray(rec_data.p_signal, dtype=np.float64)
    fs = float(rec_data.fs or header.fs or 1.0)
    if sig.size == 0 or fs <= 0:
        return None
    cols = [mapping["hr"], mapping["spo2"], mapping["sbp"], mapping["dbp"]]
    if "temp" in mapping:
        cols.append(mapping["temp"])
    raw = sig[:, cols]
    raw[~np.isfinite(raw)] = np.nan
    grid = _resample_5min(raw, fs, cfg.sample_period_min)
    if grid.shape[0] < cfg.lookback_steps + max(cfg.horizon_steps):
        return None
    if "temp" not in mapping:
        grid = np.concatenate([grid, np.full((grid.shape[0], 1), np.nan)], axis=1)
    grid = _apply_plausible(grid)
    return _save_cache(cache, rec, grid, "mimic3")


def _iv_numeric_paths(data_dir: Path, workers: int = 8) -> list[tuple[str, str]]:
    """Return (record_key, url) for every MIMIC-IV ``*n.csv.gz`` only."""
    rec_path = _download(MIMIC4_RECORDS_URL, data_dir / "RECORDS-iv")
    subjects = [ln.strip().strip("/") + "/" for ln in rec_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    def _one(subj: str) -> list[tuple[str, str]]:
        try:
            local = _download(
                f"{MIMIC4_BASE}/{subj}RECORDS",
                data_dir / "iv_index" / (subj.replace("/", "_") + "RECORDS"),
            )
        except Exception:
            return []
        found: list[tuple[str, str]] = []
        for rec in local.read_text(encoding="utf-8").splitlines():
            rec = rec.strip().strip("/")
            if not rec:
                continue
            # RECORDS lines are "83411188/83411188" — only the record id, never .dat
            record_id = rec.split("/")[0]
            url = f"{MIMIC4_BASE}/{subj}{record_id}/{record_id}n.csv.gz"
            found.append((f"mimic4/{subj}{record_id}n", url))
        return found

    out: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for chunk in pool.map(_one, subjects):
            out.extend(chunk)
    return out


def _read_iv_numeric(key: str, url: str, cfg: MimicConfig) -> dict | None:
    cache = _cache_path(cfg, key)
    hit = _load_cache(cache, cfg)
    if hit is not None:
        return hit
    dest = cfg.data_dir / "iv_numerics" / (key.replace("/", "_") + ".csv.gz")
    try:
        _download(url, dest, timeout=180)
    except Exception:
        return None
    try:
        df = pd.read_csv(dest, compression="gzip", low_memory=False)
    except Exception:
        return None
    if df.empty or "time" not in df.columns:
        return None
    names = [c for c in df.columns if c != "time"]
    mapping = _map_named_columns(names)
    if mapping is None:
        return None
    t_min = pd.to_numeric(df["time"], errors="coerce") / COUNTER_HZ / 60.0
    period = float(cfg.sample_period_min)
    bucket = (t_min / period).round().astype("Int64")
    cols = {
        "heart_rate": names[mapping["hr"]],
        "spo2": names[mapping["spo2"]],
        "sbp": names[mapping["sbp"]],
        "dbp": names[mapping["dbp"]],
    }
    if "temp" in mapping:
        cols["temperature"] = names[mapping["temp"]]
    work = pd.DataFrame({k: pd.to_numeric(df[v], errors="coerce") for k, v in cols.items()})
    work["bucket"] = bucket
    work = work.dropna(subset=["bucket"])
    agg = work.groupby("bucket", as_index=True).median(numeric_only=True)
    if agg.empty:
        return None
    b0, b1 = int(agg.index.min()), int(agg.index.max())
    agg = agg.reindex(range(b0, b1 + 1))
    grid = np.column_stack(
        [
            agg["heart_rate"].to_numpy(dtype=np.float64) if "heart_rate" in agg else np.full(len(agg), np.nan),
            agg["spo2"].to_numpy(dtype=np.float64) if "spo2" in agg else np.full(len(agg), np.nan),
            agg["sbp"].to_numpy(dtype=np.float64) if "sbp" in agg else np.full(len(agg), np.nan),
            agg["dbp"].to_numpy(dtype=np.float64) if "dbp" in agg else np.full(len(agg), np.nan),
            agg["temperature"].to_numpy(dtype=np.float64) if "temperature" in agg else np.full(len(agg), np.nan),
        ]
    )
    if grid.shape[0] < cfg.lookback_steps + max(cfg.horizon_steps):
        return None
    grid = _apply_plausible(grid)
    return _save_cache(cache, key, grid, "mimic4")


def fetch_record_list(data_dir: Path) -> list[str]:
    alt = data_dir / "RECORDS-numerics-iii"
    dest = data_dir / "RECORDS-numerics"
    if alt.exists() and not dest.exists():
        dest.write_bytes(alt.read_bytes())
    path = _download(RECORDS_URL, dest)
    recs = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    neonate_ids: set[str] = set()
    try:
        npath = _download(NEONATE_URL, data_dir / "RECORDS-neonates")
        neonate_ids = {ln.strip().split("/")[-1] for ln in npath.read_text(encoding="utf-8").splitlines()}
    except Exception:
        pass
    if neonate_ids:
        recs = [r for r in recs if r.split("/")[-1] not in neonate_ids]
    return recs


def _fill_grid(arr: np.ndarray, cfg: MimicConfig) -> np.ndarray | None:
    pp = load_step("preprocess")
    df = pd.DataFrame(arr, columns=list(pp.VITAL_FEATURES))
    period = cfg.sample_period_min
    limit = max(1, cfg.interp_limit_min // period)
    for col in pp.VITAL_FEATURES:
        df[col] = df[col].interpolate(limit=limit, limit_area="inside")
    if df["heart_rate"].notna().mean() < 0.70 or df["spo2"].notna().mean() < 0.70:
        return None
    if df["sbp"].notna().mean() < cfg.min_bp_frac:
        return None
    defaults = {"heart_rate": 80.0, "spo2": 97.0, "sbp": 120.0, "dbp": 70.0, "temperature": 37.0}
    for col in pp.VITAL_FEATURES:
        med = float(df[col].median()) if df[col].notna().any() else defaults[col]
        df[col] = df[col].ffill(limit=limit).bfill(limit=limit).fillna(med)
        lo, hi = pp.PHYSIO_BOUNDS[col]
        df[col] = df[col].clip(lo, hi)
    fall = np.zeros((len(df), 1), dtype=np.float64)
    return np.concatenate([df.to_numpy(dtype=np.float64), fall], axis=1)


def _accept_item(
    item: dict | None,
    cfg: MimicConfig,
    seen_subjects: set[int],
    series: list[np.ndarray],
    rows: list[dict],
) -> bool:
    if item is None or len(series) >= cfg.max_stays:
        return False
    sid = int(item["subject_id"])
    if sid in seen_subjects:
        return False
    filled = _fill_grid(np.asarray(item["vitals"], dtype=np.float64), cfg)
    if filled is None:
        return False
    if filled.shape[0] < cfg.lookback_steps + max(cfg.horizon_steps):
        return False
    seen_subjects.add(sid)
    series.append(filled)
    last = filled[-1]
    rows.append(
        {
            "patient_id": int(item["patient_id"]),
            "subject_id": sid,
            "record": item["record"],
            "source": item.get("source", "mimic3"),
            "n_samples": int(filled.shape[0]),
            "duration_min": int(filled.shape[0] * cfg.sample_period_min),
            "heart_rate": float(last[0]),
            "spo2": float(last[1]),
            "sbp": float(last[2]),
            "dbp": float(last[3]),
            "temperature": float(last[4]),
            "fall": 0,
            "disease": "Unknown",
        }
    )
    if len(series) % 50 == 0:
        print(f"[mimic] kept {len(series)} stays")
    return True


def collect_stays(cfg: MimicConfig) -> tuple[list[np.ndarray], pd.DataFrame]:
    series: list[np.ndarray] = []
    rows: list[dict] = []
    seen_subjects: set[int] = set()

    cache_dir = cfg.data_dir / "cache"
    if cache_dir.exists():
        for path in cache_dir.glob("*.npz"):
            item = _load_cache(path, cfg)
            _accept_item(item, cfg, seen_subjects, series, rows)
        print(f"[mimic] loaded {len(series)} stays from local numeric cache")

    if cfg.use_iv:
        print("[mimic] fetching MIMIC-IV numeric CSVs only (*n.csv.gz, no waveforms)")
        iv_jobs = _iv_numeric_paths(cfg.data_dir, workers=cfg.workers)
    else:
        iv_jobs = []
    print(f"[mimic] MIMIC-IV numeric files listed={len(iv_jobs)}")
    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        futs = {pool.submit(_read_iv_numeric, key, url, cfg): key for key, url in iv_jobs}
        for fut in as_completed(futs):
            try:
                item = fut.result()
            except Exception:
                item = None
            _accept_item(item, cfg, seen_subjects, series, rows)
    n_iv = len(series)
    print(f"[mimic] MIMIC-IV usable stays={n_iv}")

    if not cfg.use_iii:
        recs = []
    else:
        recs = fetch_record_list(cfg.data_dir)
    rng = np.random.default_rng(cfg.seed)
    order = np.arange(len(recs))
    rng.shuffle(order)
    scan = [recs[i] for i in order[: cfg.max_records_scan]]
    print(
        f"[mimic] MIMIC-III numerics listed={len(recs)}  "
        f"scanning={len(scan)} (*n records only)  target_stays={cfg.max_stays}"
    )
    scanned = 0
    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        futs = {pool.submit(_read_one_record, rec, cfg): rec for rec in scan}
        for fut in as_completed(futs):
            if len(series) >= cfg.max_stays:
                break
            scanned += 1
            try:
                item = fut.result()
            except Exception:
                item = None
            _accept_item(item, cfg, seen_subjects, series, rows)
            if scanned % 100 == 0:
                print(f"[mimic] III scanned={scanned} kept={len(series)}")

    meta = pd.DataFrame(rows)
    if not meta.empty:
        print(f"[mimic] source counts={meta['source'].value_counts().to_dict()}")
    return series, meta


def run_mimic_preprocess(cfg: MimicConfig | None = None) -> dict[str, Path]:
    cfg = cfg or MimicConfig()
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    pp = load_step("preprocess")

    series_list, stay_meta = collect_stays(cfg)
    if stay_meta.empty:
        raise RuntimeError("No MIMIC numerics stays survived filters.")
    print(
        f"[mimic] usable stays={len(stay_meta)}  "
        f"median duration={stay_meta['duration_min'].median():.0f} min  "
        f"subjects={stay_meta['subject_id'].nunique()}"
    )
    stay_meta["snapshot_risk"] = pp.snapshot_risk_from_vitals(stay_meta)

    # Split by subject so multiple records from one person cannot leak.
    subjects = stay_meta["subject_id"].drop_duplicates().to_numpy()
    strat_map = stay_meta.groupby("subject_id")["snapshot_risk"].first()
    strat = strat_map.loc[subjects].to_numpy()

    def _can_stratify(labels: np.ndarray) -> bool:
        _, counts = np.unique(labels, return_counts=True)
        return len(counts) > 1 and counts.min() >= 2

    idx_tr, idx_tmp = train_test_split(
        np.arange(len(subjects)),
        test_size=cfg.test_size + cfg.val_size,
        random_state=cfg.seed,
        stratify=strat if _can_stratify(strat) else None,
    )
    rel_val = cfg.val_size / (cfg.test_size + cfg.val_size)
    idx_va, idx_te = train_test_split(
        idx_tmp,
        test_size=1.0 - rel_val,
        random_state=cfg.seed,
        stratify=strat[idx_tmp] if _can_stratify(strat[idx_tmp]) else None,
    )
    split_subjects = {
        "train": set(subjects[idx_tr].tolist()),
        "val": set(subjects[idx_va].tolist()),
        "test": set(subjects[idx_te].tolist()),
    }

    scaler = StandardScaler()
    artifacts: dict[str, Path] = {}
    for split_name in ("train", "val", "test"):
        mask = stay_meta["subject_id"].isin(split_subjects[split_name]).to_numpy()
        sub_series = [series_list[i] for i, keep in enumerate(mask) if keep]
        sub_meta = stay_meta.loc[mask]
        x, y_v, y_r, pids, t0, diseases = pp.build_windows_from_stays(
            sub_series,
            sub_meta["patient_id"].to_numpy(),
            sub_meta["disease"].to_numpy(),
            lookback=cfg.lookback_steps,
            horizon_steps=cfg.horizon_steps,
            stride=cfg.stride_steps,
            max_windows=cfg.max_windows_per_stay,
        )
        x_s, y_vs = pp._scale_split(scaler, x, y_v, fit=(split_name == "train"))
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
            feature_names=np.array(pp.SEQUENCE_FEATURES),
            vital_names=np.array(pp.VITAL_FEATURES),
            horizons_min=np.array(cfg.horizons_min, dtype=np.int32),
            sample_period_min=np.int32(cfg.sample_period_min),
        )
        artifacts[split_name] = out_path
        print(
            f"[mimic] {split_name:5s}  stays={int(mask.sum()):5d}  "
            f"windows={x_s.shape[0]:6d}  X={tuple(x_s.shape)}  -> {out_path.name}"
        )
        meta_split = sub_meta.copy()
        meta_split["split"] = split_name
        meta_split.to_csv(cfg.output_dir / f"patients_{split_name}.csv", index=False)

    joblib.dump(
        {
            "scaler": scaler,
            "feature_names": list(pp.SEQUENCE_FEATURES),
            "vital_names": list(pp.VITAL_FEATURES),
            "physio_bounds": pp.PHYSIO_BOUNDS,
            "dataset": "mimic3wdb-matched-numerics",
        },
        cfg.output_dir / "scaler.joblib",
    )
    present = stay_meta.loc[:, list(pp.VITAL_FEATURES)].to_numpy(dtype=np.float64)
    np.savez_compressed(
        cfg.output_dir / "patient_snapshot.npz",
        patient_id=stay_meta["patient_id"].to_numpy(),
        vitals_raw=present.astype(np.float32),
        vitals_norm=scaler.transform(present).astype(np.float32),
        fall=np.zeros(len(stay_meta), dtype=np.float32),
        disease=stay_meta["disease"].to_numpy(),
        snapshot_risk=stay_meta["snapshot_risk"].to_numpy(),
        vital_names=np.array(pp.VITAL_FEATURES),
    )
    stay_meta.to_csv(cfg.output_dir / "patients_all.csv", index=False)
    payload = asdict(cfg)
    payload["data_dir"] = str(cfg.data_dir)
    payload["output_dir"] = str(cfg.output_dir)
    payload["horizons_min"] = list(cfg.horizons_min)
    payload["n_stays"] = int(len(stay_meta))
    payload["n_subjects"] = int(stay_meta["subject_id"].nunique())
    payload["dataset"] = (
        "MIMIC-III Waveform Matched numerics (*n) + "
        "MIMIC-IV Waveform numerics (*n.csv.gz); no waveform .dat"
    )
    payload["citation"] = (
        "Moody et al., MIMIC-III Waveform Database Matched Subset (PhysioNet); "
        "Johnson et al., Sci Data 2016 (MIMIC-III); "
        "MIMIC-IV Waveform Database v0.1.0 numeric tables (PhysioNet)."
    )
    payload["source_counts"] = (
        {str(k): int(v) for k, v in stay_meta["source"].value_counts().items()}
        if "source" in stay_meta.columns
        else {}
    )
    (cfg.output_dir / "preprocess_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[mimic] done. artifacts in {cfg.output_dir} (eICU processed dir untouched)")
    return artifacts


def parse_args(argv: list[str] | None = None) -> MimicConfig:
    p = argparse.ArgumentParser(description="Preprocess MIMIC-III waveform numerics for LSTM.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max-stays", type=int, default=3500)
    p.add_argument("--max-records-scan", type=int, default=8000)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-iv", action="store_true", help="Skip MIMIC-IV numeric CSVs")
    p.add_argument("--no-iii", action="store_true", help="Skip MIMIC-III numerics")
    args = p.parse_args(argv)
    return MimicConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_stays=args.max_stays,
        max_records_scan=args.max_records_scan,
        workers=args.workers,
        seed=args.seed,
        use_iv=not args.no_iv,
        use_iii=not args.no_iii,
    )


if __name__ == "__main__":
    run_mimic_preprocess(parse_args())
