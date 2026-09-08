"""
Double-auction market clearing via KKT stationarity and dual sub-gradient.

Two convex programs share a price π_wk on every (BS w, HSP k) link:

OPT1 (HSP k, buyer)
    max_{d ≥ 0}  Σ_w ρ_wk log(1 + h_wk d_wk) − π_wk d_wk
    s.t.         Σ_w d_wk ≤ D_k
    KKT:         d_wk* = [ ρ_wk / (π_wk + λ_k) − 1/h_wk ]_+
                 λ_k ≥ 0,  λ_k (Σ_w d_wk − D_k) = 0

OPT2 (BS w, seller)
    max_{r ≥ 0}  Σ_k π_wk r_kw − c_w r_kw − (β_w / 2) r_kw²
    s.t.         Σ_k r_kw ≤ R_w
    KKT:         r_kw* = [ (π_wk − c_w − μ_w) / β_w ]_+
                 μ_w ≥ 0,  μ_w (Σ_k r_kw − R_w) = 0

Prices follow the market-mismatch sub-gradient
    π ← [ π + α_t (d − r) ]_+
until ||d − r|| is small (supply = demand). Payments are π ⊙ d at clearing.
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
EPS = 1e-12


@dataclass
class OptimizerConfig:
    processed_dir: Path = DEFAULT_PROCESSED_DIR
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR
    bs_capacity_mbps: float = 5.0
    scale_capacity_by_load: bool = False
    oversubscribe: float = 1.20
    cost_c: float = 0.03
    congestion_beta: float = 0.015
    snr_scale: float = 10.0
    step0: float = 0.04
    max_iter: int = 600
    tol: float = 1e-3
    pi_min: float = 1e-4
    pi_max: float = 50.0
    bisect_iters: int = 48
    seed: int = 42
    omega_path: Path | None = None
    clearing_dir: Path | None = None

    def __post_init__(self) -> None:
        self.processed_dir = Path(self.processed_dir)
        self.artifact_dir = Path(self.artifact_dir)
        if self.omega_path is not None:
            self.omega_path = Path(self.omega_path)
        if self.clearing_dir is not None:
            self.clearing_dir = Path(self.clearing_dir)
        if self.bs_capacity_mbps <= 0 or self.oversubscribe <= 0:
            raise ValueError("capacities must be positive")
        if self.congestion_beta <= 0:
            raise ValueError("congestion_beta must be positive")


def _as_matrix(x, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(shape, float(arr))
    if arr.shape == shape:
        return arr
    if arr.shape == (shape[0],):
        return np.repeat(arr[:, None], shape[1], axis=1)
    if arr.shape == (shape[1],):
        return np.repeat(arr[None, :], shape[0], axis=0)
    raise ValueError(f"Cannot broadcast {arr.shape} to {shape}")


def normalize_snr(gain_wk: np.ndarray, scale: float) -> np.ndarray:
    g = np.asarray(gain_wk, dtype=np.float64)
    g = np.maximum(g, EPS)
    return scale * (g / np.max(g))


def kkt_demand(
    rho: np.ndarray,
    h: np.ndarray,
    pi: np.ndarray,
    lam: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    """OPT1 stationarity: d_wk* = [ρ / (π + λ) − 1/h]_+."""
    denom = np.maximum(pi + lam[None, :], EPS)
    raw = rho / denom - 1.0 / np.maximum(h, EPS)
    return np.where(active, np.maximum(raw, 0.0), 0.0)


def kkt_supply(
    pi: np.ndarray,
    cost: np.ndarray,
    beta: np.ndarray,
    mu: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    """OPT2 stationarity: r_kw* = [(π − c − μ) / β]_+."""
    raw = (pi - cost - mu[:, None]) / np.maximum(beta, EPS)
    return np.where(active, np.maximum(raw, 0.0), 0.0)


def solve_opt1(
    rho: np.ndarray,
    h: np.ndarray,
    pi: np.ndarray,
    D_k: np.ndarray,
    active: np.ndarray,
    n_bisect: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bisection on λ_k so each HSP meets its rate budget (or λ=0 if slack)."""
    w, k = rho.shape
    lam = np.zeros(k, dtype=np.float64)
    d0 = kkt_demand(rho, h, pi, lam, active)
    slack = d0.sum(axis=0) <= D_k + 1e-9
    hi = np.max(np.where(active, np.maximum(rho * h - pi, 0.0), 0.0), axis=0) + 1.0
    lo = np.zeros(k, dtype=np.float64)
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        s = kkt_demand(rho, h, pi, mid, active).sum(axis=0)
        # d(λ) is decreasing: excess demand ⇒ λ too small (raise lo).
        too_much = s > D_k
        lo = np.where(too_much, mid, lo)
        hi = np.where(too_much, hi, mid)
    lam = np.where(slack, 0.0, 0.5 * (lo + hi))
    d = kkt_demand(rho, h, pi, lam, active)
    # Project onto the simplex-like budget if residual numerical drift remains.
    tot = d.sum(axis=0)
    scale = np.ones(k, dtype=np.float64)
    over = tot > D_k + 1e-8
    scale[over] = D_k[over] / np.maximum(tot[over], EPS)
    d = d * scale[None, :]
    return d, lam


def solve_opt2(
    pi: np.ndarray,
    cost: np.ndarray,
    beta: np.ndarray,
    R_w: np.ndarray,
    active: np.ndarray,
    n_bisect: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bisection on μ_w so each BS meets its capacity (or μ=0 if slack)."""
    w, k = pi.shape
    mu = np.zeros(w, dtype=np.float64)
    r0 = kkt_supply(pi, cost, beta, mu, active)
    slack = r0.sum(axis=1) <= R_w + 1e-9
    hi = np.max(np.where(active, np.maximum(pi - cost, 0.0), 0.0), axis=1) + 1.0
    lo = np.zeros(w, dtype=np.float64)
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        s = kkt_supply(pi, cost, beta, mid, active).sum(axis=1)
        # r(μ) is decreasing: excess supply ⇒ μ too small (raise lo).
        too_much = s > R_w
        lo = np.where(too_much, mid, lo)
        hi = np.where(too_much, hi, mid)
    mu = np.where(slack, 0.0, 0.5 * (lo + hi))
    r = kkt_supply(pi, cost, beta, mu, active)
    tot = r.sum(axis=1)
    scale = np.ones(w, dtype=np.float64)
    over = tot > R_w + 1e-8
    scale[over] = R_w[over] / np.maximum(tot[over], EPS)
    r = r * scale[:, None]
    return r, mu


def welfare(
    d: np.ndarray,
    r: np.ndarray,
    rho: np.ndarray,
    h: np.ndarray,
    cost: np.ndarray,
    beta: np.ndarray,
) -> float:
    util = np.sum(rho * np.log(1.0 + h * d))
    opex = np.sum(cost * r) + 0.5 * np.sum(beta * r**2)
    return float(util - opex)


def market_gap(d: np.ndarray, r: np.ndarray) -> float:
    num = float(np.linalg.norm(d - r))
    den = float(np.linalg.norm(d) + np.linalg.norm(r) + EPS)
    return num / den


def load_omega(path: Path | None, shape: tuple[int, int]) -> np.ndarray:
    """ω_wk. Missing path ⇒ ones (legacy homogeneous reluctance)."""
    if path is None:
        return np.ones(shape, dtype=np.float64)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as z:
        if "omega_wk" in z.files:
            arr = np.asarray(z["omega_wk"], dtype=np.float64)
        elif "omega_hat" in z.files:
            arr = np.asarray(z["omega_hat"], dtype=np.float64)
        else:
            raise KeyError(f"{path} has no omega_wk / omega_hat")
    if arr.ndim == 3:
        arr = arr[0]
    if arr.shape != shape:
        raise ValueError(f"omega shape {arr.shape} != {shape}")
    return np.clip(arr, 1e-3, 1.0)


def allocate_customers(
    d_wk: np.ndarray,
    assignment_path: Path,
    c_wk: np.ndarray,
) -> pd.DataFrame | None:
    """Split each slice rate across customers in proportion to criticality."""
    if not assignment_path.exists():
        return None
    df = pd.read_csv(assignment_path)
    need = {"patient_id", "bs_id", "hsp_id", "criticality"}
    if not need.issubset(df.columns):
        return None
    share = np.zeros(len(df), dtype=np.float64)
    bs = df["bs_id"].to_numpy(dtype=np.int64)
    hs = df["hsp_id"].to_numpy(dtype=np.int64)
    c = np.clip(df["criticality"].to_numpy(dtype=np.float64), 0.0, 1.0)
    mass = np.maximum(c_wk[bs, hs], EPS)
    share = d_wk[bs, hs] * (c / mass)
    out = df[["patient_id", "bs_id", "hsp_id", "criticality"]].copy()
    out["d_n_mbps"] = share
    return out


def run_optimizer(cfg: OptimizerConfig | None = None) -> dict[str, Path]:
    cfg = cfg or OptimizerConfig()
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    rho_path = cfg.processed_dir / "rho_wk.npz"
    if not rho_path.exists():
        raise FileNotFoundError(f"{rho_path} missing. Run 04_aggregator.py first.")

    with np.load(rho_path, allow_pickle=True) as z:
        rho = np.asarray(z["rho_wk"], dtype=np.float64)
        n_wk = np.asarray(z["N_wk"], dtype=np.int64)
        c_wk = np.asarray(z["C_wk"], dtype=np.float64)
        gain = np.asarray(z["channel_gain_wk"], dtype=np.float64)
        demand_k = np.asarray(z["demand_k"], dtype=np.float64)
        load_w = np.asarray(z["load_w"], dtype=np.float64)
        hsp_names = [str(x) for x in z["hsp_names"]]
        bs_names = [str(x) for x in z["bs_names"]]

    w, k = rho.shape
    active = n_wk > 0
    h = normalize_snr(gain, cfg.snr_scale)
    h = np.where(active, h, EPS)

    R_w = np.full(w, float(cfg.bs_capacity_mbps), dtype=np.float64)
    if cfg.scale_capacity_by_load:
        R_w = cfg.bs_capacity_mbps * (load_w / np.mean(load_w))
    D_k = cfg.oversubscribe * float(R_w.sum()) * (demand_k / np.maximum(demand_k.sum(), EPS))
    omega = load_omega(cfg.omega_path, (w, k))
    cost = _as_matrix(cfg.cost_c, (w, k)) * omega
    beta = _as_matrix(cfg.congestion_beta, (w, k)) * omega
    out_dir = cfg.clearing_dir or cfg.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir = cfg.artifact_dir
    art_dir.mkdir(parents=True, exist_ok=True)

    pi_floor = max(cfg.pi_min, 0.5 * cfg.cost_c)
    rng = np.random.default_rng(cfg.seed)
    pi = np.clip(cfg.cost_c + 0.04 * np.maximum(rho, 0.0) + 0.005 * rng.random((w, k)), pi_floor, cfg.pi_max)
    pi = np.where(active, pi, pi_floor)

    d_bar = np.zeros_like(rho)
    r_bar = np.zeros_like(rho)
    history: list[dict] = []
    best_gap = float("inf")
    best_score = -float("inf")
    best = None
    min_volume = 0.05 * float(R_w.sum())

    print(
        f"[opt] W={w} K={k}  supply={R_w.sum():.1f} Mbps  "
        f"demand_cap={D_k.sum():.1f} Mbps  oversub={cfg.oversubscribe}"
    )

    for t in range(1, cfg.max_iter + 1):
        d, lam = solve_opt1(rho, h, pi, D_k, active, cfg.bisect_iters)
        r, mu = solve_opt2(pi, cost, beta, R_w, active, cfg.bisect_iters)
        gap = market_gap(d, r)
        wel = welfare(d, r, rho, h, cost, beta)
        volume = 0.5 * float(d.sum() + r.sum())
        d_bar += d
        r_bar += r
        history.append(
            {
                "iter": t,
                "gap": gap,
                "welfare": wel,
                "mean_pi": float(pi[active].mean()) if active.any() else 0.0,
                "demand_mbps": float(d.sum()),
                "supply_mbps": float(r.sum()),
            }
        )
        score = (wel - 50.0 * gap) if volume >= min_volume else -1e9
        if volume >= min_volume and score >= best_score:
            best_score = score
            best_gap = gap
            best = {
                "d": d.copy(),
                "r": r.copy(),
                "pi": pi.copy(),
                "lam": lam.copy(),
                "mu": mu.copy(),
                "iter": t,
            }

        if t == 1 or t % 50 == 0 or (gap < cfg.tol and volume >= min_volume):
            print(
                f"[opt] iter {t:4d}  gap={gap:.4e}  welfare={wel:.3f}  "
                f"d_sum={d.sum():.1f}  r_sum={r.sum():.1f}  "
                f"lam={np.round(lam, 3)}  mu={np.round(mu, 3)}"
            )
        if gap < cfg.tol and volume >= min_volume:
            print(f"[opt] converged at iter {t}")
            break

        excess = d - r
        step = cfg.step0 / np.sqrt(t)
        scale = 1.0 + np.abs(excess)
        pi = np.clip(pi + step * excess / scale, pi_floor, cfg.pi_max)
        pi = np.where(active, pi, pi_floor)

    if best is None:
        best = {"d": d, "r": r, "pi": pi, "lam": lam, "mu": mu, "iter": t}
        best_gap = market_gap(d, r)
    n_avg = max(len(history), 1)
    d_avg = d_bar / n_avg
    r_avg = r_bar / n_avg
    # Report last-best primal (tightest clearing); keep ergodic as a diagnostic.
    d, r, pi = best["d"], best["r"], best["pi"]
    lam, mu = best["lam"], best["mu"]
    # Cleared rate used for payments: midpoint of last-best d and r.
    x = 0.5 * (d + r)
    pay = pi * x
    wel = welfare(x, x, rho, h, cost, beta)
    util_k = np.sum(rho * np.log(1.0 + h * x), axis=0) - np.sum(pi * x, axis=0)
    profit_w = np.sum(pi * x, axis=1) - np.sum(cost * x, axis=1) - 0.5 * np.sum(beta * x**2, axis=1)

    out_npz = out_dir / "auction_clearing.npz"
    np.savez_compressed(
        out_npz,
        d_wk=d,
        r_kw=r,
        d_r_mid=x,
        d_ergodic=d_avg,
        r_ergodic=r_avg,
        pi_wk=pi,
        payment_wk=pay,
        lambda_k=lam,
        mu_w=mu,
        h_wk=h,
        R_w=R_w,
        D_k=D_k,
        rho_wk=rho,
        omega_wk=omega,
        N_wk=n_wk,
        C_wk=c_wk,
        hsp_names=np.array(hsp_names),
        bs_names=np.array(bs_names),
        welfare=np.float64(wel),
        gap=np.float64(best_gap),
        best_iter=np.int32(best["iter"]),
        utility_k=util_k,
        profit_w=profit_w,
    )

    def _df(mat: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(mat, index=bs_names, columns=hsp_names)

    _df(x).to_csv(out_dir / "d_wk.csv")
    _df(r).to_csv(out_dir / "r_kw.csv")
    _df(pi).to_csv(out_dir / "pi_wk.csv")
    _df(pay).to_csv(out_dir / "payment_wk.csv")
    _df(omega).to_csv(out_dir / "omega_wk.csv")

    cust_path = out_dir / "customer_rates.csv"
    cust = allocate_customers(x, cfg.processed_dir / "customer_assignment.csv", c_wk)
    if cust is not None:
        cust.to_csv(cust_path, index=False)

    summary = {
        "converged": bool(best_gap < cfg.tol),
        "best_iter": int(best["iter"]),
        "n_iters": len(history),
        "gap": best_gap,
        "welfare": wel,
        "total_rate_mbps": float(x.sum()),
        "total_payment": float(pay.sum()),
        "mean_omega": float(omega.mean()),
        "R_w": R_w.tolist(),
        "D_k": D_k.tolist(),
        "lambda_k": lam.tolist(),
        "mu_w": mu.tolist(),
        "d_wk": d.tolist(),
        "r_kw": r.tolist(),
        "d_r_mid": x.tolist(),
        "pi_wk": pi.tolist(),
        "payment_wk": pay.tolist(),
        "omega_wk": omega.tolist(),
        "utility_k": {hsp_names[i]: float(util_k[i]) for i in range(k)},
        "profit_w": {bs_names[i]: float(profit_w[i]) for i in range(w)},
        "bs_names": bs_names,
        "hsp_names": hsp_names,
        "config": {key: (str(v) if isinstance(v, Path) else v) for key, v in asdict(cfg).items()},
        "history_tail": history[-10:],
    }
    summary_path = art_dir / "optimizer_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "optimizer_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    hist_path = art_dir / "optimizer_history.csv"
    pd.DataFrame(history).to_csv(hist_path, index=False)

    print("[opt] cleared rates d_wk ~ r_kw (Mbps, midpoint)")
    print(_df(x).round(3).to_string())
    print("[opt] prices pi_wk")
    print(_df(pi).round(4).to_string())
    print("[opt] payments pi * x")
    print(_df(pay).round(3).to_string())
    print(
        f"[opt] welfare={wel:.3f}  gap={best_gap:.4e}  "
        f"total_rate={x.sum():.2f} Mbps  total_pay={pay.sum():.2f}  "
        f"mean_omega={omega.mean():.3f}"
    )
    print(f"[opt] HSP utility={np.round(util_k, 3).tolist()}  BS profit={np.round(profit_w, 3).tolist()}")
    print(f"[opt] wrote {out_npz.name}")
    artifacts = {
        "clearing": out_npz,
        "summary": summary_path,
        "history": hist_path,
        "d_csv": out_dir / "d_wk.csv",
        "pi_csv": out_dir / "pi_wk.csv",
        "pay_csv": out_dir / "payment_wk.csv",
    }
    if cust is not None:
        artifacts["customer_rates"] = cust_path
    return artifacts


def parse_args(argv: list[str] | None = None) -> OptimizerConfig:
    p = argparse.ArgumentParser(description="KKT / sub-gradient double-auction clearing.")
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    p.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    p.add_argument("--bs-capacity-mbps", type=float, default=5.0)
    p.add_argument("--scale-capacity-by-load", action="store_true")
    p.add_argument("--oversubscribe", type=float, default=1.20)
    p.add_argument("--max-iter", type=int, default=600)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--step0", type=float, default=0.04)
    p.add_argument("--omega-path", type=Path, default=None)
    p.add_argument("--clearing-dir", type=Path, default=None)
    args = p.parse_args(argv)
    return OptimizerConfig(
        processed_dir=args.processed_dir,
        artifact_dir=args.artifact_dir,
        bs_capacity_mbps=args.bs_capacity_mbps,
        scale_capacity_by_load=args.scale_capacity_by_load,
        oversubscribe=args.oversubscribe,
        max_iter=args.max_iter,
        tol=args.tol,
        step0=args.step0,
        omega_path=args.omega_path,
        clearing_dir=args.clearing_dir,
    )


if __name__ == "__main__":
    run_optimizer(parse_args())
