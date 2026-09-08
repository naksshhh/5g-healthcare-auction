"""
Paper-style result figures for the LSTM-preference extension.

Reproduces the three result families from Kumar & Kumar:
  Fig. 2  Convergence of social welfare for step sizes Delta
  Fig. 3  Per-link demand-response gap G_wk = d_wk - r_kw
  Fig. 4  Evolution of BS bids  zeta_kw
  Fig. 5  Evolution of HSP bids varrho_wk

Uses paper utilities instantiated with LSTM-driven rho:
  S_wk = rho log(1 + h d),   T_kw = c r + (beta/2) r^2
  varrho_wk = d * dS/dd = rho h d / (1 + h d)
  zeta_kw   = (1/r) dT/dr = c/r + beta

Array layout in this repo is (BS, HSP); plot labels use paper names
HSP1 / HSP2 (random competing buyers), BS1--BS3.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patheffects import withStroke
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import load_step

DEFAULT_PROCESSED = ROOT / "data" / "processed"
DEFAULT_FIGDIR = ROOT / "artifacts" / "figures"
EPS = 1e-12
DELTAS = (0.02, 0.04, 0.06)
FIG345_DELTA = 0.04  # matches optimizer step0 so traces clear on the 5 Mbps cap
HSP_PAPER = ("HSP1", "HSP2", "HSP3")
BS_PAPER = ("BS1", "BS2", "BS3", "BS4")
LINESTYLES = ("-", "--", "-.", ":")

DELTA_STYLE = {
    0.02: dict(color="#2ca02c", marker="+", linestyle="-", markevery=(1, 8), label=r"$\Delta=0.02$"),
    0.04: dict(color="#8c564b", marker="v", linestyle="--", markevery=(3, 8), label=r"$\Delta=0.04$"),
    0.06: dict(color="#1f77b4", marker="o", linestyle="-.", markevery=(5, 8), label=r"$\Delta=0.06$"),
}
BS_STYLE = (
    dict(color="#2ca02c", marker="+", linestyle="-"),
    dict(color="#d62728", marker="x", linestyle="--"),
    dict(color="#1f77b4", marker="o", linestyle="-."),
    dict(color="#9467bd", marker="s", linestyle=":"),
)
HSP_STYLE = (
    dict(color="#2ca02c", marker="+", linestyle="-"),
    dict(color="#1f77b4", marker="o", linestyle="--"),
    dict(color="#d62728", marker="x", linestyle="-."),
)


def _trace_style(base: dict, index: int) -> dict:
    """Stagger markers and halo the stroke so overlapping series stay identifiable."""
    style = dict(base)
    style.setdefault("linestyle", LINESTYLES[index % len(LINESTYLES)])
    style["markevery"] = (2 + 3 * index, 10)
    style["markersize"] = 5.0
    style["markeredgewidth"] = 1.05
    style["linewidth"] = 1.7
    style["zorder"] = 3 + index
    style["path_effects"] = [withStroke(linewidth=3.35, foreground="white", alpha=0.9)]
    return style


def _pick_trace(traj: dict[float, dict]) -> dict:
    if FIG345_DELTA in traj:
        return traj[FIG345_DELTA]
    return next(iter(traj.values()))


def _ieee_rc() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.45,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "lines.linewidth": 1.45,
            "lines.markersize": 4.2,
        }
    )


def _load_market(processed_dir: Path, opt, omega_path: Path | None = None):
    z = np.load(processed_dir / "rho_wk.npz", allow_pickle=True)
    rho = np.asarray(z["rho_wk"], dtype=np.float64)
    n_wk = np.asarray(z["N_wk"], dtype=np.int64)
    gain = np.asarray(z["channel_gain_wk"], dtype=np.float64)
    demand_k = np.asarray(z["demand_k"], dtype=np.float64)
    hsp_names = [str(x) for x in z["hsp_names"]]
    active = n_wk > 0
    cfg = opt.OptimizerConfig(
        processed_dir=processed_dir,
        omega_path=omega_path,
        bs_capacity_mbps=5.0,
        scale_capacity_by_load=False,
    )
    h = opt.normalize_snr(gain, cfg.snr_scale)
    h = np.where(active, h, EPS)
    n_bs = rho.shape[0]
    r_w = np.full(n_bs, float(cfg.bs_capacity_mbps), dtype=np.float64)
    d_k = cfg.oversubscribe * float(r_w.sum()) * (demand_k / max(float(demand_k.sum()), EPS))
    omega = opt.load_omega(cfg.omega_path, rho.shape)
    cost = opt._as_matrix(cfg.cost_c, rho.shape) * omega
    beta = opt._as_matrix(cfg.congestion_beta, rho.shape) * omega
    return {
        "rho": rho,
        "h": h,
        "active": active,
        "R_w": r_w,
        "D_k": d_k,
        "cost": cost,
        "beta": beta,
        "omega": omega,
        "hsp_names": hsp_names,
        "cfg": cfg,
    }


def hsp_bid(rho: np.ndarray, h: np.ndarray, d: np.ndarray) -> np.ndarray:
    """varrho_wk = d * dS/dd  (paper Eq. 21) for S = rho log(1+h d)."""
    return rho * h * d / np.maximum(1.0 + h * d, EPS)


def bs_bid(
    cost: np.ndarray,
    beta: np.ndarray,
    r: np.ndarray,
    omega: np.ndarray | None = None,
) -> np.ndarray:
    """zeta_kw = (1/r) dT/dr. If cost/beta are unscaled, pass omega to apply T=ω g(r)."""
    z = cost / np.maximum(r, EPS) + beta
    if omega is None:
        return z
    return np.asarray(omega, dtype=np.float64) * z


def run_trajectory(mkt: dict, opt, delta: float, max_iter: int, seed: int) -> dict:
    """Constant-step sub-gradient on pi; log d, r, SW, bids every iteration."""
    rho, h, active = mkt["rho"], mkt["h"], mkt["active"]
    cfg = mkt["cfg"]
    n_bs, n_hsp = rho.shape
    rng = np.random.default_rng(seed)
    pi_floor = max(cfg.pi_min, 0.5 * cfg.cost_c)
    pi = np.clip(
        cfg.cost_c + 0.04 * np.maximum(rho, 0.0) + 0.005 * rng.random(rho.shape),
        pi_floor,
        cfg.pi_max,
    )
    pi = np.where(active, pi, pi_floor)

    rec = {
        "d": np.zeros((max_iter, n_bs, n_hsp)),
        "r": np.zeros((max_iter, n_bs, n_hsp)),
        "sw": np.zeros(max_iter),
        "sw_matched": np.zeros(max_iter),
        "gap": np.zeros(max_iter),
        "varrho": np.zeros((max_iter, n_bs, n_hsp)),
        "zeta": np.zeros((max_iter, n_bs, n_hsp)),
    }
    n_done = 0
    for t in range(max_iter):
        d, _ = opt.solve_opt1(rho, h, pi, mkt["D_k"], active, cfg.bisect_iters)
        r, _ = opt.solve_opt2(pi, mkt["cost"], mkt["beta"], mkt["R_w"], active, cfg.bisect_iters)
        matched = np.minimum(d, r)
        rec["d"][t] = d
        rec["r"][t] = r
        rec["sw"][t] = opt.welfare(d, r, rho, h, mkt["cost"], mkt["beta"])
        rec["sw_matched"][t] = opt.welfare(matched, matched, rho, h, mkt["cost"], mkt["beta"])
        rec["gap"][t] = opt.market_gap(d, r)
        rec["varrho"][t] = hsp_bid(rho, h, d)
        rec["zeta"][t] = bs_bid(mkt["cost"], mkt["beta"], r)
        n_done = t + 1
        excess = d - r
        step = delta / np.sqrt(t + 1.0)
        pi = np.clip(pi + step * excess / (1.0 + np.abs(excess)), pi_floor, cfg.pi_max)
        pi = np.where(active, pi, pi_floor)

    for key, arr in rec.items():
        rec[key] = arr[:n_done]
    rec["n"] = n_done
    rec["delta"] = delta
    return rec


def _save(fig, figdir: Path, stem: str) -> None:
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / f"{stem}.png")
    plt.close(fig)


def fig_convergence(traj: dict[float, dict], figdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    max_sw = max(float(tr["sw_matched"][-1]) for tr in traj.values())
    xmax = max(tr["n"] for tr in traj.values())
    ax.axhline(
        max_sw,
        color="#d62728",
        linestyle=":",
        linewidth=1.2,
        label="Maximum Social welfare",
    )
    for i, delta in enumerate(DELTAS):
        tr = traj[delta]
        it = np.arange(1, tr["n"] + 1)
        ax.plot(it, tr["sw_matched"], **_trace_style(DELTA_STYLE[delta], i))
    ax.set_xlabel("iteration")
    ax.set_ylabel("Social welfare")
    ax.set_xlim(0, max(xmax, 80))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()
    _save(fig, figdir, "fig2_convergence")


def fig_gap(traj: dict, figdir: Path, n_hsp_panels: int = 2) -> None:
    tr = _pick_trace(traj)
    n_bs, n_hsp = tr["d"].shape[1], tr["d"].shape[2]
    n_hsp_panels = min(n_hsp_panels, n_hsp)
    fig, axes = plt.subplots(1, n_hsp_panels, figsize=(5.8, 3.5), sharey=True)
    if n_hsp_panels == 1:
        axes = [axes]
    it = np.arange(1, tr["n"] + 1)
    for j, ax in enumerate(axes):
        for i in range(n_bs):
            g = tr["d"][:, i, j] - tr["r"][:, i, j]
            ax.plot(
                it,
                g,
                label=f"{HSP_PAPER[j]}-{BS_PAPER[i]}",
                **_trace_style(BS_STYLE[i % len(BS_STYLE)], i),
            )
        ax.axhline(0.0, color="k", linewidth=0.7, alpha=0.55)
        ax.set_xlabel("iteration")
        ax.legend(fontsize=6.5, loc="upper right", framealpha=0.9, handlelength=2.6)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_ylabel("Demand and response gap")
    fig.tight_layout()
    _save(fig, figdir, "fig3_demand_response_gap")

    fig, axes = plt.subplots(1, n_hsp, figsize=(8.8, 3.5), sharey=True)
    if n_hsp == 1:
        axes = [axes]
    for j, ax in enumerate(axes):
        for i in range(n_bs):
            g = tr["d"][:, i, j] - tr["r"][:, i, j]
            ax.plot(
                it,
                g,
                label=BS_PAPER[i],
                **_trace_style(BS_STYLE[i % len(BS_STYLE)], i),
            )
        ax.axhline(0.0, color="k", linewidth=0.7, alpha=0.55)
        ax.set_xlabel("iteration")
        ax.legend(fontsize=6.5, loc="upper right", framealpha=0.9, handlelength=2.6)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_ylabel("Demand and response gap")
    fig.tight_layout()
    _save(fig, figdir, "fig3_demand_response_gap_all")


def fig_bs_bids(traj: dict, figdir: Path) -> None:
    tr = _pick_trace(traj)
    n_bs, n_hsp = tr["zeta"].shape[1], tr["zeta"].shape[2]
    it = np.arange(1, tr["n"] + 1)
    fig, axes = plt.subplots(1, n_bs, figsize=(2.9 * n_bs, 3.5), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for i in range(n_bs):
        ax = axes[i]
        for j in range(n_hsp):
            ax.plot(
                it,
                tr["zeta"][:, i, j],
                label=HSP_PAPER[j],
                **_trace_style(HSP_STYLE[j % len(HSP_STYLE)], j),
            )
        ax.set_ylabel("BSs Bids")
        ax.legend(fontsize=7, handlelength=2.6, framealpha=0.9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("iteration")
    fig.tight_layout()
    _save(fig, figdir, "fig4_bs_bids")


def fig_hsp_bids(traj: dict, figdir: Path) -> None:
    tr = _pick_trace(traj)
    n_bs, n_hsp = tr["varrho"].shape[1], tr["varrho"].shape[2]
    it = np.arange(1, tr["n"] + 1)
    fig, axes = plt.subplots(1, n_hsp, figsize=(8.8, 3.5), sharey=False)
    if n_hsp == 1:
        axes = [axes]
    for j, ax in enumerate(axes):
        for i in range(n_bs):
            ax.plot(
                it,
                tr["varrho"][:, i, j],
                label=BS_PAPER[i],
                **_trace_style(BS_STYLE[i % len(BS_STYLE)], i),
            )
        ax.set_xlabel("iteration")
        ax.legend(fontsize=7, handlelength=2.6, framealpha=0.9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_ylabel("HSPs Bids")
    fig.tight_layout()
    _save(fig, figdir, "fig5_hsp_bids")


def fig_omega_const(omega: np.ndarray, figdir: Path, hsp_names: list[str]) -> None:
    n_bs, n_hsp = omega.shape
    x = np.arange(n_hsp)
    width = 0.8 / max(n_bs, 1)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for i in range(n_bs):
        ax.bar(
            x + (i - (n_bs - 1) / 2) * width,
            omega[i],
            width,
            label=BS_PAPER[i],
            color=("#2ca02c", "#d62728", "#1f77b4")[i % 3],
            edgecolor="k",
            linewidth=0.4,
        )
    ax.set_xticks(x, [HSP_PAPER[j] for j in range(n_hsp)])
    ax.set_xlabel("HSP")
    ax.set_ylabel(r"Reluctance $\omega_{kw}$ (constant)")
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=7, framealpha=0.92)
    fig.tight_layout()
    _save(fig, figdir, "fig_omega_const")


def persist_arrays(traj: dict[float, dict], figdir: Path, hsp_names: list[str]) -> None:
    payload = {
        "hsp_map": {
            HSP_PAPER[j]: hsp_names[j] if j < len(hsp_names) else HSP_PAPER[j]
            for j in range(min(len(HSP_PAPER), len(hsp_names)))
        },
        "deltas": [float(x) for x in traj.keys()],
    }
    for delta, tr in traj.items():
        tag = f"{delta:.3f}".replace(".", "p")
        np.savez_compressed(
            figdir / f"trace_delta_{tag}.npz",
            d=tr["d"],
            r=tr["r"],
            sw_matched=tr["sw_matched"],
            sw=tr["sw"],
            gap=tr["gap"],
            varrho=tr["varrho"],
            zeta=tr["zeta"],
        )
        payload[str(delta)] = {
            "n_iter": int(tr["n"]),
            "final_sw": float(tr["sw_matched"][-1]),
            "final_gap": float(tr["gap"][-1]),
        }
    (figdir / "figure_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def generate_figures(
    processed_dir: Path | None = None,
    figdir: Path | None = None,
    max_iter: int = 80,
    seed: int = 42,
    omega_path: Path | None = None,
) -> Path:
    processed_dir = Path(processed_dir or DEFAULT_PROCESSED)
    figdir = Path(figdir or DEFAULT_FIGDIR)
    figdir.mkdir(parents=True, exist_ok=True)
    opt = load_step("optimizer")
    mkt = _load_market(processed_dir, opt, omega_path=omega_path)
    print(
        f"[figures] market  BS x HSP = {mkt['rho'].shape}  "
        f"HSP={mkt['hsp_names']}  supply={mkt['R_w'].sum():.1f} Mbps"
    )
    traj = {}
    for delta in DELTAS:
        tr = run_trajectory(mkt, opt, delta=delta, max_iter=max_iter, seed=seed)
        traj[delta] = tr
        print(
            f"[figures] Delta={delta:.3f}  iters={tr['n']}  "
            f"SW={tr['sw_matched'][-1]:.3f}  gap={tr['gap'][-1]:.3e}"
        )
    _ieee_rc()
    fig_convergence(traj, figdir)
    fig_gap(traj, figdir)
    fig_bs_bids(traj, figdir)
    fig_hsp_bids(traj, figdir)
    fig_omega_const(mkt["omega"], figdir, mkt["hsp_names"])
    persist_arrays(traj, figdir, mkt["hsp_names"])
    dest = ROOT / "report" / "current"
    dest.mkdir(parents=True, exist_ok=True)
    for p in figdir.glob("fig*.png"):
        (dest / p.name).write_bytes(p.read_bytes())
    print(f"[figures] wrote {figdir}")
    return figdir


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate paper-style auction result figures.")
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    p.add_argument("--figdir", type=Path, default=DEFAULT_FIGDIR)
    p.add_argument("--max-iter", type=int, default=80)
    p.add_argument("--omega-path", type=Path, default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    generate_figures(args.processed_dir, args.figdir, args.max_iter, omega_path=args.omega_path)
