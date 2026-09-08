"""
Dynamic BS reluctance ω_kw from simulated radio features.

Contract (frozen during the inner dual loop of OPT1/OPT2)
--------------------------------------------------------
    T_kw = ω_kw * (c r_kw + (β/2) r_kw²),   ω_kw ∈ (0, 1)
    ζ_kw = ω_kw * (c/r + β)

Feature vector ξ_kw (all in [0, 1], code layout (BS, HSP)):
    eta       η_k        cell load (simulated PRB occupancy)
    gamma     γ̄_kw      mean link quality (from map RSRP; higher = better)
    backhaul  b_k        backhaul occupancy (simulated, correlated with load)
    q_nat     q_k^nat    native-user QoS degradation (simulated, monotone in load)
    interfer  I_k        inter-cell interference (geometry × neighbour load)

Ground truth
    ω* = σ(aᵀ ξ + b)     logistic; a has + on load/backhaul/q_nat/I and − on γ

GradientBoosting is trained on *simulated slots* of this geometry. Labels are ω*.
Predictor inputs are noisy KPIs a BS could observe. The eICU one-shot market is
an evaluation geometry, not the training set.

n_slots defaults to 1 for the auction; the training table still has many slots
so a later few-slot / RAN-trace experiment can reuse xi[t, bs, hsp, F].
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import load_step

DEFAULT_PROCESSED = ROOT / "data" / "processed"
DEFAULT_OUT = ROOT / "data" / "processed" / "reluctance"
DEFAULT_ARTIFACTS = ROOT / "artifacts" / "reluctance"
EPS = 1e-12
FEATURE_NAMES = ("eta", "gamma", "backhaul", "q_nat", "interfer")
HSP_NAMES = ("HSP1", "HSP2", "HSP3")

# a: +load, −quality, +backhaul, +native-QoS-degradation, +interference
WEIGHT_A = np.array([1.35, -1.55, 0.95, 1.05, 0.90], dtype=np.float64)
BIAS_B = -0.85
STRESS_BS = 0  # BS1 in paper labels


@dataclass
class ReluctanceConfig:
    processed_dir: Path = DEFAULT_PROCESSED
    out_dir: Path = DEFAULT_OUT
    artifact_dir: Path = DEFAULT_ARTIFACTS
    n_train_slots: int = 3000
    n_slots: int = 1  # auction evaluation depth (Phase B/C raise this)
    n_bs: int = 3
    n_hsp: int = 2
    seed: int = 42
    train_frac: float = 0.70
    val_frac: float = 0.15
    stress_bs: int = STRESS_BS
    stress_eta: float = 0.92
    obs_noise: float = 0.045

    def __post_init__(self) -> None:
        self.processed_dir = Path(self.processed_dir)
        self.out_dir = Path(self.out_dir)
        self.artifact_dir = Path(self.artifact_dir)


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(z, dtype=np.float64), -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-z))


def omega_star(xi: np.ndarray, a: np.ndarray = WEIGHT_A, b: float = BIAS_B) -> np.ndarray:
    """Closed-form ground truth. xi shape (..., 5)."""
    return np.clip(sigmoid(xi @ a + b), 1e-3, 1.0 - 1e-3)


def _rsrp_to_gamma(rsrp_dbm: np.ndarray) -> np.ndarray:
    """Map RSRP [dBm] to link quality in [0, 1]. Higher = better."""
    return np.clip((np.asarray(rsrp_dbm, dtype=np.float64) + 110.0) / 50.0, 0.0, 1.0)


def site_coupling(bs_xy: np.ndarray) -> np.ndarray:
    """Neighbour coupling in [0, 1], zero diagonal. Uses the aggregator path-loss."""
    agg = load_step("aggregator")
    n = len(bs_xy)
    delta = bs_xy[:, None, :] - bs_xy[None, :, :]
    dist = np.sqrt((delta**2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    pl = agg.path_loss_db(dist)
    g = 10.0 ** (-pl / 10.0)
    np.fill_diagonal(g, 0.0)
    peak = float(g.max()) if np.isfinite(g).any() else 1.0
    return g / max(peak, EPS)


def interference_from_load(coupling: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """I_k from neighbour loads; clip to [0, 1]. eta (n_bs,) or (n_slots, n_bs)."""
    raw = np.tensordot(eta, coupling, axes=([-1], [1]))
    return np.clip(raw, 0.0, 1.0)


def load_geometry(processed_dir: Path, n_bs: int | None = None, n_hsp: int | None = None) -> dict:
    """Read association map. Shape comes from rho_wk (code layout: BS x HSP)."""
    rho_path = processed_dir / "rho_wk.npz"
    assign_path = processed_dir / "customer_assignment.csv"
    if not rho_path.exists():
        raise FileNotFoundError(f"{rho_path} missing — run aggregator first.")
    with np.load(rho_path, allow_pickle=True) as z:
        bs_xy = np.asarray(z["bs_xy"], dtype=np.float64)
        n_wk = np.asarray(z["N_wk"], dtype=np.float64)
        load_w = np.asarray(z["load_w"], dtype=np.float64)
        hsp_names = [str(x) for x in z["hsp_names"]]
        bs_names = [str(x) for x in z["bs_names"]]
        gain = np.asarray(z["channel_gain_wk"], dtype=np.float64)
    n_bs, n_hsp = int(n_wk.shape[0]), int(n_wk.shape[1])
    bs_xy = bs_xy[:n_bs]

    rsrp_mean = np.full((n_bs, n_hsp), np.nan)
    if assign_path.exists():
        df = pd.read_csv(assign_path, usecols=["bs_id", "hsp_id", "rsrp_dbm"])
        grp = df.groupby(["bs_id", "hsp_id"])["rsrp_dbm"].mean()
        for (bs, hsp), val in grp.items():
            bi, hi = int(bs), int(hsp)
            if 0 <= bi < n_bs and 0 <= hi < n_hsp:
                rsrp_mean[bi, hi] = float(val)
    missing = np.isnan(rsrp_mean)
    if missing.any():
        g = np.maximum(gain, EPS)
        proxy = -90.0 + 10.0 * np.log10(g / np.max(g))
        rsrp_mean = np.where(missing, proxy, rsrp_mean)

    gamma_geom = _rsrp_to_gamma(rsrp_mean)
    return {
        "bs_xy": bs_xy,
        "n_wk": n_wk,
        "load_w": load_w,
        "rsrp_mean": rsrp_mean,
        "gamma_geom": gamma_geom,
        "coupling": site_coupling(bs_xy),
        "hsp_names": hsp_names,
        "bs_names": bs_names,
    }


def _eta_for_slot(rng: np.random.Generator, n_bs: int) -> np.ndarray:
    u = float(rng.random())
    if u < 0.30:
        eta = rng.uniform(0.08, 0.30, size=n_bs)
    elif u < 0.72:
        eta = rng.uniform(0.28, 0.58, size=n_bs)
    else:
        eta = rng.uniform(0.42, 0.68, size=n_bs)
        if rng.random() < 0.65:
            eta[int(rng.integers(n_bs))] = float(rng.uniform(0.78, 0.95))
    return np.clip(eta, 0.02, 0.98)


def simulate_slots(geom: dict, cfg: ReluctanceConfig) -> dict:
    """xi[t, bs, hsp, F], omega_star[t, bs, hsp] on the current BS layout."""
    rng = np.random.default_rng(cfg.seed)
    t, w, k = cfg.n_train_slots, cfg.n_bs, cfg.n_hsp
    eta = np.stack([_eta_for_slot(rng, w) for _ in range(t)], axis=0)
    backhaul = np.clip(0.15 + 0.80 * eta + rng.normal(0.0, 0.06, size=eta.shape), 0.0, 1.0)
    q_nat = np.clip(eta**1.35 + rng.normal(0.0, 0.03, size=eta.shape), 0.0, 1.0)
    interfer = interference_from_load(geom["coupling"], eta)
    gamma = np.broadcast_to(geom["gamma_geom"][None, :, :], (t, w, k)).copy()
    gamma = np.clip(gamma + rng.normal(0.0, 0.035, size=gamma.shape), 0.0, 1.0)

    xi = np.zeros((t, w, k, 5), dtype=np.float64)
    xi[..., 0] = eta[:, :, None]
    xi[..., 1] = gamma
    xi[..., 2] = backhaul[:, :, None]
    xi[..., 3] = q_nat[:, :, None]
    xi[..., 4] = interfer[:, :, None]
    star = omega_star(xi)
    return {"xi": xi, "omega_star": star, "eta": eta, "slot_id": np.arange(t, dtype=np.int32)}


def observe_xi(xi: np.ndarray, rng: np.random.Generator, noise: float) -> np.ndarray:
    """Noisy KPIs a BS could measure. Target ω* stays on the clean ξ."""
    obs = xi + rng.normal(0.0, noise, size=xi.shape)
    return np.clip(obs, 0.0, 1.0)


def _flatten(xi: np.ndarray, y: np.ndarray | None = None):
    n = int(np.prod(xi.shape[:3]))
    x = xi.reshape(n, xi.shape[-1])
    if y is None:
        return x
    return x, y.reshape(n)


def slot_split(n_slots: int, train_frac: float, val_frac: float, seed: int):
    rng = np.random.default_rng(seed + 11)
    perm = rng.permutation(n_slots)
    n_tr = int(train_frac * n_slots)
    n_va = int(val_frac * n_slots)
    return perm[:n_tr], perm[n_tr : n_tr + n_va], perm[n_tr + n_va :]


def train_predictor(xi: np.ndarray, star: np.ndarray, cfg: ReluctanceConfig) -> dict:
    rng = np.random.default_rng(cfg.seed + 3)
    obs = observe_xi(xi, rng, cfg.obs_noise)
    tr, va, te = slot_split(xi.shape[0], cfg.train_frac, cfg.val_frac, cfg.seed)
    model = HistGradientBoostingRegressor(
        max_depth=4,
        learning_rate=0.08,
        max_iter=250,
        l2_regularization=0.05,
        random_state=cfg.seed,
    )
    x_tr, y_tr = _flatten(obs[tr], star[tr])
    model.fit(x_tr, y_tr)
    metrics = {}
    for name, idx in (("train", tr), ("val", va), ("test", te)):
        x, y = _flatten(obs[idx], star[idx])
        pred = np.clip(model.predict(x), 1e-3, 1.0 - 1e-3)
        metrics[name] = {
            "n_slots": int(len(idx)),
            "n_rows": int(len(y)),
            "mae": float(mean_absolute_error(y, pred)),
            "r2": float(r2_score(y, pred)),
        }
    return {"model": model, "metrics": metrics, "split": {"train": tr, "val": va, "test": te}}


def predict_omega(model, xi: np.ndarray) -> np.ndarray:
    shape = xi.shape[:-1]
    pred = model.predict(_flatten(xi))
    return np.clip(pred.reshape(shape), 1e-3, 1.0 - 1e-3)


def eicu_eval_slot(geom: dict, cfg: ReluctanceConfig, stress: bool = False) -> dict:
    """One-shot ξ on the real association map. Optional stress on one cell."""
    rng = np.random.default_rng(cfg.seed + 99)
    w, k = cfg.n_bs, cfg.n_hsp
    load = np.asarray(geom["load_w"], dtype=np.float64)
    load_n = load / max(float(load.mean()), EPS)
    eta = np.clip(0.38 + 0.12 * (load_n - 1.0) + rng.normal(0.0, 0.03, size=w), 0.15, 0.75)
    eta = np.clip(eta + 0.05 * np.linspace(-1.0, 1.0, w), 0.15, 0.85)
    if stress:
        eta[cfg.stress_bs] = cfg.stress_eta
    backhaul = np.clip(0.15 + 0.80 * eta, 0.0, 1.0)
    q_nat = np.clip(eta**1.35, 0.0, 1.0)
    interfer = interference_from_load(geom["coupling"], eta)
    gamma = np.asarray(geom["gamma_geom"], dtype=np.float64)
    xi = np.zeros((w, k, 5), dtype=np.float64)
    xi[..., 0] = eta[:, None]
    xi[..., 1] = gamma
    xi[..., 2] = backhaul[:, None]
    xi[..., 3] = q_nat[:, None]
    xi[..., 4] = interfer[:, None]
    star = omega_star(xi)
    return {"xi": xi[None, ...], "omega_star": star[None, ...], "eta": eta}


def _rows_table(xi: np.ndarray, star: np.ndarray, hat: np.ndarray | None, split_name: np.ndarray) -> pd.DataFrame:
    t, w, k, _ = xi.shape
    recs = []
    for s in range(t):
        for i in range(w):
            for j in range(k):
                row = {
                    "slot_id": int(s),
                    "split": str(split_name[s]),
                    "bs": i,
                    "hsp": j,
                    "hsp_name": HSP_NAMES[j] if j < len(HSP_NAMES) else str(j),
                    "omega_star": float(star[s, i, j]),
                }
                for f, name in enumerate(FEATURE_NAMES):
                    row[name] = float(xi[s, i, j, f])
                if hat is not None:
                    row["omega_hat"] = float(hat[s, i, j])
                recs.append(row)
    return pd.DataFrame.from_records(recs)


def run_reluctance(cfg: ReluctanceConfig | None = None) -> dict[str, Path]:
    cfg = cfg or ReluctanceConfig()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    geom = load_geometry(cfg.processed_dir, cfg.n_bs, cfg.n_hsp)
    cfg.n_bs, cfg.n_hsp = int(geom["n_wk"].shape[0]), int(geom["n_wk"].shape[1])
    sim = simulate_slots(geom, cfg)
    print(
        f"[reluctance] simulated slots={sim['xi'].shape[0]}  "
        f"links={cfg.n_bs}x{cfg.n_hsp}  "
        f"omega* mean={sim['omega_star'].mean():.3f}  "
        f"range=[{sim['omega_star'].min():.3f},{sim['omega_star'].max():.3f}]"
    )
    fit = train_predictor(sim["xi"], sim["omega_star"], cfg)
    print("[reluctance] GBDT slot-split metrics")
    for split, m in fit["metrics"].items():
        print(f"  {split:5s}  n_slots={m['n_slots']:4d}  MAE={m['mae']:.4f}  R²={m['r2']:.3f}")

    hat_train = predict_omega(fit["model"], sim["xi"])
    split_name = np.full(sim["xi"].shape[0], "train", dtype=object)
    split_name[fit["split"]["val"]] = "val"
    split_name[fit["split"]["test"]] = "test"

    eicu = eicu_eval_slot(geom, cfg, stress=False)
    stress = eicu_eval_slot(geom, cfg, stress=True)
    eicu_hat = predict_omega(fit["model"], eicu["xi"])
    stress_hat = predict_omega(fit["model"], stress["xi"])

    joblib.dump(fit["model"], cfg.artifact_dir / "predictor.joblib")
    np.savez_compressed(
        cfg.out_dir / "train_slots.npz",
        xi=sim["xi"],
        omega_star=sim["omega_star"],
        omega_hat=hat_train,
        eta=sim["eta"],
        slot_id=sim["slot_id"],
        feature_names=np.array(FEATURE_NAMES),
        weight_a=WEIGHT_A,
        bias_b=np.float64(BIAS_B),
    )
    table = _rows_table(sim["xi"], sim["omega_star"], hat_train, split_name)
    table_path = cfg.out_dir / "train_table.csv"
    table.to_csv(table_path, index=False)

    def _save_eval(name: str, pack: dict, hat: np.ndarray) -> Path:
        path = cfg.out_dir / f"{name}.npz"
        np.savez_compressed(
            path,
            xi=pack["xi"],
            omega_star=pack["omega_star"],
            omega_hat=hat,
            omega_wk=hat[0],
            eta=pack["eta"],
            feature_names=np.array(FEATURE_NAMES),
            hsp_names=np.array(geom["hsp_names"]),
            bs_names=np.array(geom["bs_names"]),
            n_slots=np.int32(cfg.n_slots),
            stress_bs=np.int32(cfg.stress_bs if name == "stress_slot" else -1),
        )
        return path

    eicu_path = _save_eval("eicu_slot", eicu, eicu_hat)
    stress_path = _save_eval("stress_slot", stress, stress_hat)
    ones = np.ones((cfg.n_bs, cfg.n_hsp), dtype=np.float64)
    np.savez_compressed(cfg.out_dir / "omega_ones.npz", omega_wk=ones, omega_hat=ones[None], omega_star=ones[None])

    summary = {
        "source": "physics-inspired radio simulator on eICU association geometry (not operator RAN traces)",
        "features": list(FEATURE_NAMES),
        "weight_a": WEIGHT_A.tolist(),
        "bias_b": BIAS_B,
        "n_train_slots": cfg.n_train_slots,
        "n_eval_slots": cfg.n_slots,
        "omega_star_train": {
            "mean": float(sim["omega_star"].mean()),
            "min": float(sim["omega_star"].min()),
            "max": float(sim["omega_star"].max()),
        },
        "gbdt": fit["metrics"],
        "eicu_omega_hat": eicu_hat[0].tolist(),
        "eicu_omega_star": eicu["omega_star"][0].tolist(),
        "stress_omega_hat": stress_hat[0].tolist(),
        "stress_bs": cfg.stress_bs,
        "eicu_mae_hat_vs_star": float(mean_absolute_error(eicu["omega_star"].ravel(), eicu_hat.ravel())),
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
    }
    summary_path = cfg.artifact_dir / "reluctance_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[reluctance] eICU omega-hat")
    print(pd.DataFrame(eicu_hat[0], index=geom["bs_names"], columns=geom["hsp_names"]).round(3).to_string())
    print("[reluctance] stress omega-hat (busy cell "
          f"{geom['bs_names'][cfg.stress_bs]})")
    print(pd.DataFrame(stress_hat[0], index=geom["bs_names"], columns=geom["hsp_names"]).round(3).to_string())
    print(f"[reluctance] wrote {cfg.out_dir}")
    return {
        "train_npz": cfg.out_dir / "train_slots.npz",
        "train_csv": table_path,
        "eicu_slot": eicu_path,
        "stress_slot": stress_path,
        "omega_ones": cfg.out_dir / "omega_ones.npz",
        "predictor": cfg.artifact_dir / "predictor.joblib",
        "summary": summary_path,
    }


def write_frozen_omega(processed_dir: Path, out_dir: Path, seed: int = 42) -> Path:
    """One-shot radio ω* on the current association map. No GBDT. Constant for paper 1."""
    processed_dir = Path(processed_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    geom = load_geometry(processed_dir)
    n_bs, n_hsp = int(geom["n_wk"].shape[0]), int(geom["n_wk"].shape[1])
    cfg = ReluctanceConfig(
        processed_dir=processed_dir,
        out_dir=out_dir,
        n_bs=n_bs,
        n_hsp=n_hsp,
        seed=seed,
    )
    pack = eicu_eval_slot(geom, cfg, stress=False)
    omega = np.asarray(pack["omega_star"][0], dtype=np.float64)
    path = out_dir / "omega_frozen.npz"
    np.savez_compressed(
        path,
        omega_wk=omega,
        omega_star=pack["omega_star"],
        omega_hat=pack["omega_star"],
        eta=pack["eta"],
        hsp_names=np.array(geom["hsp_names"]),
        bs_names=np.array(geom["bs_names"]),
    )
    pd.DataFrame(omega, index=geom["bs_names"], columns=geom["hsp_names"]).to_csv(
        out_dir / "omega_wk.csv"
    )
    print("[reluctance] frozen omega (closed-form, one slot)")
    print(pd.DataFrame(omega, index=geom["bs_names"], columns=geom["hsp_names"]).round(3).to_string())
    return path


def parse_args(argv=None) -> tuple[ReluctanceConfig, bool]:
    p = argparse.ArgumentParser(description="Simulate radio features and train omega predictor.")
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACTS)
    p.add_argument("--n-train-slots", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-bs", type=int, default=3)
    p.add_argument("--n-hsp", type=int, default=2)
    p.add_argument(
        "--frozen-only",
        action="store_true",
        help="Write constant omega* from one radio snapshot; skip GBDT training.",
    )
    args = p.parse_args(argv)
    cfg = ReluctanceConfig(
        processed_dir=args.processed_dir,
        out_dir=args.out_dir,
        artifact_dir=args.artifact_dir,
        n_train_slots=args.n_train_slots,
        n_bs=args.n_bs,
        n_hsp=args.n_hsp,
        seed=args.seed,
    )
    return cfg, bool(args.frozen_only)


if __name__ == "__main__":
    cfg, frozen_only = parse_args()
    if frozen_only:
        write_frozen_omega(cfg.processed_dir, cfg.out_dir, seed=cfg.seed)
    else:
        run_reluctance(cfg)
