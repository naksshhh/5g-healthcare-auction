"""
Aggregate customer criticality onto (base station, HSP) links.

Each IoMT customer is placed in a 5G urban-micro cell, associated to the
serving base station (max RSRP / nearest site) and randomly subscribed to
one of W competing HSPs (HSP1, HSP2, ...). Per-link mass

    C_wk = sum_{n : BS(n)=w, HSP(n)=k} c_n

becomes the dynamic preference matrix ``rho_wk`` consumed by the double-auction
KKT solver (OPT1 / OPT2).
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

HSP_NAMES: tuple[str, ...] = ("HSP1", "HSP2", "HSP3")
EPS = 1e-9
FC_GHZ = 3.5  # 5G mid-band
TX_POWER_DBM = 30.0


@dataclass
class AggregatorConfig:
    processed_dir: Path = DEFAULT_PROCESSED_DIR
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR
    n_bs: int = 4
    n_hsp: int = 3
    area_m: float = 2000.0
    seed: int = 42
    scores_name: str = "criticality_scores.csv"

    def __post_init__(self) -> None:
        self.processed_dir = Path(self.processed_dir)
        self.artifact_dir = Path(self.artifact_dir)
        if self.n_bs < 1 or self.n_hsp < 1:
            raise ValueError("n_bs and n_hsp must be >= 1")
        if self.n_hsp > len(HSP_NAMES):
            raise ValueError(f"n_hsp must be <= {len(HSP_NAMES)} ({HSP_NAMES})")


def place_base_stations(n_bs: int, area_m: float) -> np.ndarray:
    """Grid layout inset from the map edge so every site has a coverage footprint."""
    cols = int(np.ceil(np.sqrt(n_bs)))
    rows = int(np.ceil(n_bs / cols))
    xs = np.linspace(area_m * 0.22, area_m * 0.78, cols)
    ys = np.linspace(area_m * 0.22, area_m * 0.78, rows)
    sites = np.array([(x, y) for y in ys for x in xs], dtype=np.float64)[:n_bs]
    return sites


def place_customers(n: int, area_m: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, area_m, size=(n, 2))


def path_loss_db(distance_m: np.ndarray, fc_ghz: float = FC_GHZ) -> np.ndarray:
    """3GPP UMi-street NLOS-style path loss (dB), d clipped at 1 m."""
    d = np.maximum(np.asarray(distance_m, dtype=np.float64), 1.0)
    return 32.4 + 20.0 * np.log10(fc_ghz) + 31.9 * np.log10(d)


def associate_bs(xy: np.ndarray, bs_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Max-RSRP association with equal TX power → nearest site after path loss."""
    delta = xy[:, None, :] - bs_xy[None, :, :]
    dist = np.sqrt((delta**2).sum(axis=-1))
    rsrp = TX_POWER_DBM - path_loss_db(dist)
    bs_id = np.argmax(rsrp, axis=1).astype(np.int64)
    serving_dist = dist[np.arange(len(xy)), bs_id]
    serving_rsrp = rsrp[np.arange(len(xy)), bs_id]
    return bs_id, serving_dist, serving_rsrp


def assign_hsp(n_users: int, n_hsp: int, seed: int) -> np.ndarray:
    """Split users at random across competing HSPs. No disease / specialty map."""
    rng = np.random.default_rng(seed + 7)
    return rng.integers(0, n_hsp, size=int(n_users), dtype=np.int64)


def aggregate_links(
    bs_id: np.ndarray,
    hsp_id: np.ndarray,
    criticality: np.ndarray,
    n_bs: int,
    n_hsp: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return C_wk, N_wk, mean criticality Ā_wk with shape (W, K)."""
    c = np.asarray(criticality, dtype=np.float64)
    c_wk = np.zeros((n_bs, n_hsp), dtype=np.float64)
    n_wk = np.zeros((n_bs, n_hsp), dtype=np.int64)
    np.add.at(c_wk, (bs_id, hsp_id), c)
    np.add.at(n_wk, (bs_id, hsp_id), 1)
    mean_wk = np.divide(c_wk, np.maximum(n_wk, 1), dtype=np.float64)
    mean_wk = np.where(n_wk > 0, mean_wk, 0.0)
    return c_wk, n_wk, mean_wk


def build_rho(
    c_wk: np.ndarray,
    n_wk: np.ndarray,
    mean_wk: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Preference matrices for the double auction.

    rho_wk
        Mass-scaled mean criticality, then max-normalized into (0, 1]:
        ρ_wk = (C_wk / C̄) · (1 + Ā_wk), then ρ ← ρ / max ρ.
        Empty links get ``EPS`` so log-barriers / KKT multipliers stay defined.
    rho_hsp
        Column-stochastic: HSP k's preference distribution over base stations.
    rho_bs
        Row-stochastic: BS w's preference distribution over HSPs.
    """
    c_bar = float(np.mean(c_wk) + EPS)
    rho = (c_wk / c_bar) * (1.0 + mean_wk)
    rho = np.where(n_wk > 0, rho, EPS)
    rho = rho / max(float(np.max(rho)), EPS)
    hsp_den = rho.sum(axis=0, keepdims=True) + EPS
    bs_den = rho.sum(axis=1, keepdims=True) + EPS
    return {
        "rho_wk": rho.astype(np.float64),
        "rho_hsp": (rho / hsp_den).astype(np.float64),
        "rho_bs": (rho / bs_den).astype(np.float64),
    }


def _load_customers(cfg: AggregatorConfig) -> pd.DataFrame:
    csv_path = cfg.processed_dir / cfg.scores_name
    npz_path = cfg.processed_dir / "criticality_scores.npz"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    elif npz_path.exists():
        with np.load(npz_path, allow_pickle=True) as z:
            df = pd.DataFrame(
                {
                    "patient_id": z["patient_id"].astype(np.int64),
                    "criticality": z["criticality"].astype(np.float64),
                    "split": z["split"] if "split" in z.files else "unknown",
                }
            )
        meta = cfg.processed_dir / "patients_all.csv"
        if meta.exists():
            extra = pd.read_csv(meta)
            keep = [c for c in ("patient_id", "disease", "snapshot_risk", "fall") if c in extra.columns]
            df = df.merge(extra[keep], on="patient_id", how="left")
    else:
        raise FileNotFoundError(
            f"Criticality scores not found in {cfg.processed_dir}. Run 03_fusion_metric.py first."
        )
    if "criticality" not in df.columns:
        raise KeyError("criticality column missing from scores table")
    if "disease" not in df.columns:
        df["disease"] = "Unknown"
    df["patient_id"] = df["patient_id"].astype(np.int64)
    df["criticality"] = np.clip(df["criticality"].astype(np.float64), 0.0, 1.0)
    return df.reset_index(drop=True)


def run_aggregator(cfg: AggregatorConfig | None = None) -> dict[str, Path]:
    cfg = cfg or AggregatorConfig()
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)

    customers = _load_customers(cfg)
    n = len(customers)
    bs_xy = place_base_stations(cfg.n_bs, cfg.area_m)
    xy = place_customers(n, cfg.area_m, cfg.seed)
    bs_id, dist_m, rsrp_dbm = associate_bs(xy, bs_xy)
    hsp_id = assign_hsp(n, cfg.n_hsp, cfg.seed)
    c = customers["criticality"].to_numpy(dtype=np.float64)

    c_wk, n_wk, mean_wk = aggregate_links(bs_id, hsp_id, c, cfg.n_bs, cfg.n_hsp)
    prefs = build_rho(c_wk, n_wk, mean_wk)
    rho = prefs["rho_wk"]

    # Mean linear channel gain of customers on each (w, k) link.
    gain = 10.0 ** ((rsrp_dbm - TX_POWER_DBM) / 10.0)
    g_wk = np.zeros((cfg.n_bs, cfg.n_hsp), dtype=np.float64)
    np.add.at(g_wk, (bs_id, hsp_id), gain)
    g_wk = np.divide(g_wk, np.maximum(n_wk, 1))
    g_wk = np.where(n_wk > 0, g_wk, EPS)

    hsp_names = list(HSP_NAMES[: cfg.n_hsp])
    bs_names = [f"BS{w + 1}" for w in range(cfg.n_bs)]

    map_df = customers.copy()
    map_df["x_m"] = xy[:, 0]
    map_df["y_m"] = xy[:, 1]
    map_df["bs_id"] = bs_id
    map_df["hsp_id"] = hsp_id
    map_df["hsp"] = [hsp_names[i] for i in hsp_id]
    map_df["distance_m"] = dist_m
    map_df["rsrp_dbm"] = rsrp_dbm
    map_path = cfg.processed_dir / "customer_assignment.csv"
    map_df.to_csv(map_path, index=False)

    rho_path = cfg.processed_dir / "rho_wk.npz"
    np.savez_compressed(
        rho_path,
        rho_wk=rho,
        rho_hsp=prefs["rho_hsp"],
        rho_bs=prefs["rho_bs"],
        C_wk=c_wk,
        N_wk=n_wk,
        mean_criticality_wk=mean_wk,
        channel_gain_wk=g_wk,
        bs_xy=bs_xy,
        hsp_names=np.array(hsp_names),
        bs_names=np.array(bs_names),
        n_bs=np.int32(cfg.n_bs),
        n_hsp=np.int32(cfg.n_hsp),
        area_m=np.float64(cfg.area_m),
        demand_k=c_wk.sum(axis=0),
        load_w=n_wk.sum(axis=1),
    )

    rho_csv = cfg.processed_dir / "rho_wk.csv"
    pd.DataFrame(rho, index=bs_names, columns=hsp_names).to_csv(rho_csv)
    pd.DataFrame(n_wk, index=bs_names, columns=hsp_names).to_csv(cfg.processed_dir / "N_wk.csv")
    pd.DataFrame(c_wk, index=bs_names, columns=hsp_names).to_csv(cfg.processed_dir / "C_wk.csv")

    summary = {
        "n_customers": int(n),
        "n_bs": cfg.n_bs,
        "n_hsp": cfg.n_hsp,
        "area_m": cfg.area_m,
        "bs_xy": bs_xy.tolist(),
        "hsp_names": hsp_names,
        "load_w": n_wk.sum(axis=1).astype(int).tolist(),
        "users_k": n_wk.sum(axis=0).astype(int).tolist(),
        "demand_k": c_wk.sum(axis=0).tolist(),
        "N_wk": n_wk.tolist(),
        "C_wk": c_wk.tolist(),
        "mean_criticality_wk": mean_wk.tolist(),
        "rho_wk": rho.tolist(),
        "rho_hsp": prefs["rho_hsp"].tolist(),
        "rho_bs": prefs["rho_bs"].tolist(),
        "empty_links": int((n_wk == 0).sum()),
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
    }
    summary_path = cfg.artifact_dir / "aggregator_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[aggregate] customers={n}  BS={cfg.n_bs}  HSP={cfg.n_hsp}  area={cfg.area_m:.0f} m")
    print("[aggregate] N_wk (users)")
    print(pd.DataFrame(n_wk, index=bs_names, columns=hsp_names).to_string())
    print("[aggregate] rho_wk (dynamic preference)")
    print(pd.DataFrame(rho, index=bs_names, columns=hsp_names).round(4).to_string())
    print(f"[aggregate] load per BS={n_wk.sum(axis=1).tolist()}  demand per HSP={np.round(c_wk.sum(axis=0), 2).tolist()}")
    print(f"[aggregate] wrote {rho_path.name}")
    return {
        "rho_npz": rho_path,
        "rho_csv": rho_csv,
        "assignment": map_path,
        "summary": summary_path,
    }


def load_rho(processed_dir: Path | str | None = None) -> dict[str, np.ndarray]:
    path = Path(processed_dir or DEFAULT_PROCESSED_DIR) / "rho_wk.npz"
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def parse_args(argv: list[str] | None = None) -> AggregatorConfig:
    p = argparse.ArgumentParser(description="Build rho_wk from customer criticality.")
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    p.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    p.add_argument("--n-bs", type=int, default=4)
    p.add_argument("--n-hsp", type=int, default=3)
    p.add_argument("--area-m", type=float, default=2000.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    return AggregatorConfig(
        processed_dir=args.processed_dir,
        artifact_dir=args.artifact_dir,
        n_bs=args.n_bs,
        n_hsp=args.n_hsp,
        area_m=args.area_m,
        seed=args.seed,
    )


if __name__ == "__main__":
    run_aggregator(parse_args())
