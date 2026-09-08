"""
Economic-layer result figures (Figs 6--8).

  Fig. 6  Criticality mass, mean intensity, and preference rho_wk
  Fig. 7  HSP outlay vs BS receipt, plus buyer/seller surplus
  Fig. 8  Preference vs cleared rate and payment; comparative statics in rho

No disease-name HSP labels. Reluctance is a frozen constant on this paper-1 run.
Uses the already-cleared eICU market (CSV / optimizer summary) for Figs 6--7
and the 6-link scatter. Fig. 8(c) re-clears with one HSP's rho scaled.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import load_step

DEFAULT_PROCESSED = ROOT / "data" / "processed"
DEFAULT_FIGDIR = ROOT / "artifacts" / "figures"
EPS = 1e-12
HSP_PAPER = ("HSP1", "HSP2", "HSP3")
BS_PAPER = ("BS1", "BS2", "BS3", "BS4")
ALPHAS = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00)
STEMS = (
    "fig6_preference_criticality",
    "fig7_payments_surplus",
    "fig8_preference_vs_payment",
)

HSP_COLOR = ("#2ca02c", "#1f77b4", "#d62728")
BS_COLOR = ("#2ca02c", "#d62728", "#1f77b4", "#9467bd")
HSP_MARKER = ("+", "o", "x")
BS_MARKER = ("+", "x", "o", "s")
HATCH = ("", "//", "\\\\", "xx")


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
        }
    )


def _save(fig, figdir: Path, stem: str) -> None:
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / f"{stem}.png")
    plt.close(fig)


def _publish(figdir: Path) -> None:
    dest = ROOT / "report" / "current"
    dest.mkdir(parents=True, exist_ok=True)
    for stem in STEMS:
        src = figdir / f"{stem}.png"
        if src.exists():
            (dest / src.name).write_bytes(src.read_bytes())


def _load_cleared(processed_dir: Path, clearing_dir: Path | None = None) -> dict:
    """Load preference anatomy + Walrasian transfers from an existing clearing."""
    processed_dir = Path(processed_dir)
    clearing_dir = Path(clearing_dir) if clearing_dir is not None else processed_dir
    rho_npz = processed_dir / "rho_wk.npz"
    if rho_npz.exists():
        with np.load(rho_npz, allow_pickle=True) as z:
            hsp_names = [str(x) for x in z["hsp_names"]]
            rho = np.asarray(z["rho_wk"], dtype=np.float64)
            c_wk = np.asarray(z["C_wk"], dtype=np.float64)
            n_wk = np.asarray(z["N_wk"], dtype=np.float64)
            mean_c = np.asarray(z["mean_criticality_wk"], dtype=np.float64)
    else:
        agg = json.loads((ROOT / "artifacts" / "aggregator_summary.json").read_text(encoding="utf-8"))
        hsp_names = list(agg["hsp_names"])
        rho = np.asarray(agg["rho_wk"], dtype=np.float64)
        c_wk = np.asarray(agg["C_wk"], dtype=np.float64)
        n_wk = np.asarray(agg["N_wk"], dtype=np.float64)
        mean_c = np.asarray(agg["mean_criticality_wk"], dtype=np.float64)
    opt_path = clearing_dir / "optimizer_summary.json"
    if not opt_path.exists():
        opt_path = processed_dir / "optimizer_summary.json"
    if not opt_path.exists():
        opt_path = ROOT / "artifacts" / "w2k3" / "optimizer_summary.json"
    if not opt_path.exists():
        opt_path = ROOT / "artifacts" / "optimizer_summary.json"
    opt_sum = json.loads(opt_path.read_text(encoding="utf-8"))
    n_hsp = rho.shape[1]
    pay = np.loadtxt(clearing_dir / "payment_wk.csv", delimiter=",", skiprows=1, usecols=range(1, 1 + n_hsp))
    rate = np.loadtxt(clearing_dir / "d_wk.csv", delimiter=",", skiprows=1, usecols=range(1, 1 + n_hsp))
    pi = np.loadtxt(clearing_dir / "pi_wk.csv", delimiter=",", skiprows=1, usecols=range(1, 1 + n_hsp))
    pay = np.atleast_2d(np.asarray(pay, dtype=np.float64))
    rate = np.atleast_2d(np.asarray(rate, dtype=np.float64))
    pi = np.atleast_2d(np.asarray(pi, dtype=np.float64))
    util = np.array([opt_sum["utility_k"][name] for name in hsp_names], dtype=np.float64)
    bs_names = list(opt_sum.get("bs_names", [f"BS{i + 1}" for i in range(rho.shape[0])]))
    profit = np.array([opt_sum["profit_w"][name] for name in bs_names], dtype=np.float64)
    return {
        "hsp_names": hsp_names,
        "rho": rho,
        "c_wk": c_wk,
        "n_wk": n_wk,
        "mean_c": mean_c,
        "pay": pay,
        "rate": rate,
        "pi": pi,
        "util": util,
        "profit": profit,
        "welfare": float(opt_sum["welfare"]),
        "total_payment": float(opt_sum["total_payment"]),
    }


def fig_preference_anatomy(data: dict, figdir: Path) -> None:
    n_bs, n_hsp = data["rho"].shape
    x = np.arange(n_hsp)
    width = 0.18
    panels = (
        (data["c_wk"], "Criticality mass $C_{wk}$"),
        (data["mean_c"], "Mean criticality $\\bar{A}_{wk}$"),
        (data["rho"], r"Preference $\rho_{wk}$"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.45), sharex=True)
    for ax, (mat, ylabel) in zip(axes, panels):
        for i in range(n_bs):
            ax.bar(
                x + (i - (n_bs - 1) / 2) * width,
                mat[i],
                width,
                label=BS_PAPER[i],
                color=BS_COLOR[i],
                edgecolor="k",
                linewidth=0.4,
                hatch=HATCH[i],
            )
        ax.set_xticks(x, HSP_PAPER[:n_hsp])
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=6.5, framealpha=0.92, ncol=2, handlelength=1.4)
        if "Mean" in ylabel:
            lo = float(mat.min())
            hi = float(mat.max())
            pad = 0.15 * max(hi - lo, 1e-3)
            ax.set_ylim(lo - pad, hi + pad)
    axes[0].set_xlabel("HSP")
    axes[1].set_xlabel("HSP")
    axes[2].set_xlabel("HSP")
    fig.tight_layout()
    _save(fig, figdir, STEMS[0])


def fig_payments_surplus(data: dict, figdir: Path) -> None:
    n_bs, n_hsp = data["pay"].shape
    hsp_pay = data["pay"].sum(axis=0)
    bs_pay = data["pay"].sum(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.55))

    ax = axes[0]
    hsp_x = np.arange(n_hsp)
    bs_x = np.arange(n_hsp + 1, n_hsp + 1 + n_bs)
    ax.bar(hsp_x, hsp_pay, color=list(HSP_COLOR[:n_hsp]), edgecolor="k", linewidth=0.4, label="HSP outlay")
    ax.bar(bs_x, bs_pay, color=list(BS_COLOR[:n_bs]), edgecolor="k", linewidth=0.4, hatch="//", label="BS receipt")
    ax.axvline(n_hsp - 0.5 + 0.5, color="0.55", linewidth=0.8, linestyle=":")
    ax.set_xticks(list(hsp_x) + list(bs_x), list(HSP_PAPER[:n_hsp]) + list(BS_PAPER[:n_bs]), rotation=0)
    ax.set_ylabel(r"Payment $\pi_{wk}\,x_{wk}$")
    ax.legend(loc="upper right", framealpha=0.92)
    total = float(hsp_pay.sum())
    ax.text(
        0.02,
        0.95,
        rf"$\sum\Gamma_w=\sum\Upsilon_k={total:.2f}$",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )

    ax = axes[1]
    labels = list(HSP_PAPER[:n_hsp]) + list(BS_PAPER[:n_bs])
    values = np.concatenate([data["util"], data["profit"]])
    colors = list(HSP_COLOR[:n_hsp]) + list(BS_COLOR[:n_bs])
    hatches = [""] * n_hsp + ["//"] * n_bs
    bars = ax.bar(np.arange(len(values)), values, color=colors, edgecolor="k", linewidth=0.4)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
    ax.axhline(0.0, color="k", linewidth=0.7, alpha=0.55)
    ax.set_xticks(np.arange(len(values)), labels)
    ax.set_ylabel("Surplus")
    wel = data["welfare"]
    leftover = float(hsp_pay.sum() - bs_pay.sum())
    ax.text(
        0.02,
        0.95,
        rf"welfare$={wel:.2f}$,  TPE leftover$={leftover:.2e}$",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )
    fig.tight_layout()
    _save(fig, figdir, STEMS[1])


def _trend(ax, xs: np.ndarray, ys: np.ndarray) -> None:
    if xs.size < 2:
        return
    coef = np.polyfit(xs, ys, 1)
    grid = np.linspace(xs.min(), xs.max(), 50)
    ax.plot(grid, np.polyval(coef, grid), color="0.35", linestyle=":", linewidth=1.1, zorder=1)


def _clear_once(opt, mkt: dict, rho: np.ndarray, max_iter: int, seed: int) -> dict:
    cfg = mkt["cfg"]
    h, active = mkt["h"], mkt["active"]
    rng = np.random.default_rng(seed)
    pi_floor = max(cfg.pi_min, 0.5 * cfg.cost_c)
    pi = np.clip(
        cfg.cost_c + 0.04 * np.maximum(rho, 0.0) + 0.005 * rng.random(rho.shape),
        pi_floor,
        cfg.pi_max,
    )
    pi = np.where(active, pi, pi_floor)
    best = None
    best_score = -np.inf
    min_volume = 0.05 * float(mkt["R_w"].sum())
    for t in range(1, max_iter + 1):
        d, _ = opt.solve_opt1(rho, h, pi, mkt["D_k"], active, cfg.bisect_iters)
        r, _ = opt.solve_opt2(pi, mkt["cost"], mkt["beta"], mkt["R_w"], active, cfg.bisect_iters)
        gap = opt.market_gap(d, r)
        wel = opt.welfare(d, r, rho, h, mkt["cost"], mkt["beta"])
        volume = 0.5 * float(d.sum() + r.sum())
        score = (wel - 50.0 * gap) if volume >= min_volume else -1e9
        if score >= best_score:
            best_score = score
            best = (d.copy(), r.copy(), pi.copy(), gap)
        if gap < cfg.tol and volume >= min_volume:
            break
        excess = d - r
        step = cfg.step0 / np.sqrt(t)
        pi = np.clip(pi + step * excess / (1.0 + np.abs(excess)), pi_floor, cfg.pi_max)
        pi = np.where(active, pi, pi_floor)
    d, r, pi, gap = best
    x = 0.5 * (d + r)
    return {"x": x, "pi": pi, "pay": pi * x, "gap": gap}


def _rho_sweep(
    processed_dir: Path,
    max_iter: int,
    seed: int,
    omega_path: Path | None = None,
) -> dict:
    opt = load_step("optimizer")
    figs = load_step("auction_figures")
    mkt = figs._load_market(processed_dir, opt, omega_path=omega_path)
    rho0 = mkt["rho"]
    n_bs, n_hsp = rho0.shape
    pay = np.zeros((len(ALPHAS), n_hsp))
    rate = np.zeros((len(ALPHAS), n_hsp))
    unit = np.zeros((len(ALPHAS), n_hsp))
    for j in range(n_hsp):
        for a_i, alpha in enumerate(ALPHAS):
            rho = rho0.copy()
            rho[:, j] = rho0[:, j] * alpha
            rec = _clear_once(opt, mkt, rho, max_iter=max_iter, seed=seed)
            pay[a_i, j] = rec["pay"][:, j].sum()
            rate[a_i, j] = rec["x"][:, j].sum()
            unit[a_i, j] = rec["pay"][:, j].sum() / max(rec["x"][:, j].sum(), EPS)
            print(
                f"[econ] scale {HSP_PAPER[j]}  alpha={alpha:.2f}  "
                f"pay={pay[a_i, j]:.3f}  rate={rate[a_i, j]:.2f}  gap={rec['gap']:.3e}"
            )
    return {"alpha": np.asarray(ALPHAS), "pay": pay, "rate": rate, "unit": unit}


def fig_preference_payment(data: dict, sweep: dict, figdir: Path) -> None:
    n_bs, n_hsp = data["rho"].shape
    fig = plt.figure(figsize=(8.8, 6.4))
    gs = GridSpec(2, 2, height_ratios=(1.0, 1.05), hspace=0.38, wspace=0.32)

    ax_rate = fig.add_subplot(gs[0, 0])
    ax_pay = fig.add_subplot(gs[0, 1])
    ax_sens = fig.add_subplot(gs[1, :])

    for j in range(n_hsp):
        for i in range(n_bs):
            ax_rate.scatter(
                data["rho"][i, j],
                data["rate"][i, j],
                color=HSP_COLOR[j],
                marker=BS_MARKER[i],
                s=42,
                linewidths=1.05,
                zorder=3,
            )
            ax_pay.scatter(
                data["rho"][i, j],
                data["pay"][i, j],
                color=HSP_COLOR[j],
                marker=BS_MARKER[i],
                s=42,
                linewidths=1.05,
                zorder=3,
            )
    _trend(ax_rate, data["rho"].ravel(), data["rate"].ravel())
    _trend(ax_pay, data["rho"].ravel(), data["pay"].ravel())
    ax_rate.set_xlabel(r"Preference $\rho_{wk}$")
    ax_rate.set_ylabel("Cleared rate $x_{wk}$ (Mbps)")
    ax_pay.set_xlabel(r"Preference $\rho_{wk}$")
    ax_pay.set_ylabel(r"Payment $\pi_{wk} x_{wk}$")
    hsp_handles = [
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=HSP_COLOR[j],
            markeredgecolor=HSP_COLOR[j], markersize=6, label=HSP_PAPER[j],
        )
        for j in range(n_hsp)
    ]
    bs_handles = [
        Line2D(
            [0], [0], marker=BS_MARKER[i], color="0.25", linestyle="none",
            markersize=6, label=BS_PAPER[i],
        )
        for i in range(n_bs)
    ]
    ax_rate.legend(handles=hsp_handles, fontsize=7, framealpha=0.92, loc="upper left")
    ax_pay.legend(handles=bs_handles, fontsize=7, framealpha=0.92, loc="upper left")

    ax_rate2 = ax_sens.twinx()
    ax_rate2.grid(False)
    for j in range(n_hsp):
        ax_sens.plot(
            sweep["alpha"],
            sweep["pay"][:, j],
            color=HSP_COLOR[j],
            marker=HSP_MARKER[j],
            linestyle="-",
            linewidth=1.6,
            markersize=6,
            label=f"{HSP_PAPER[j]} payment",
        )
        ax_rate2.plot(
            sweep["alpha"],
            sweep["rate"][:, j],
            color=HSP_COLOR[j],
            marker=HSP_MARKER[j],
            linestyle="--",
            linewidth=1.2,
            markersize=5,
            alpha=0.85,
            label=f"{HSP_PAPER[j]} rate",
        )
    ax_sens.axvline(1.0, color="0.5", linestyle=":", linewidth=0.9)
    ax_sens.set_xlabel(r"Preference scale $\alpha$  ($\rho_{w\cdot}\leftarrow\alpha\rho_{w\cdot}$, others fixed)")
    ax_sens.set_ylabel("HSP outlay")
    ax_rate2.set_ylabel("HSP rate (Mbps)")
    ax_sens.xaxis.set_major_locator(MaxNLocator(nbins=6))
    h1, l1 = ax_sens.get_legend_handles_labels()
    h2, l2 = ax_rate2.get_legend_handles_labels()
    ax_sens.legend(h1 + h2, l1 + l2, ncol=3, fontsize=7, loc="upper left", framealpha=0.92)

    _save(fig, figdir, STEMS[2])


def generate_economic_figures(
    processed_dir: Path | None = None,
    figdir: Path | None = None,
    max_iter: int = 80,
    seed: int = 42,
    clearing_dir: Path | None = None,
    omega_path: Path | None = None,
) -> Path:
    processed_dir = Path(processed_dir or DEFAULT_PROCESSED)
    figdir = Path(figdir or DEFAULT_FIGDIR)
    figdir.mkdir(parents=True, exist_ok=True)
    data = _load_cleared(processed_dir, clearing_dir=clearing_dir)
    print(
        f"[econ] HSP={data['hsp_names']}  "
        f"pay={data['pay'].sum(axis=0).round(3).tolist()}  "
        f"receipt={data['pay'].sum(axis=1).round(3).tolist()}"
    )
    _ieee_rc()
    fig_preference_anatomy(data, figdir)
    fig_payments_surplus(data, figdir)
    sweep = _rho_sweep(processed_dir, max_iter=max_iter, seed=seed, omega_path=omega_path)
    fig_preference_payment(data, sweep, figdir)
    payload = {
        "hsp_map": {HSP_PAPER[j]: data["hsp_names"][j] for j in range(len(data["hsp_names"]))},
        "hsp_outlay": data["pay"].sum(axis=0).tolist(),
        "bs_receipt": data["pay"].sum(axis=1).tolist(),
        "buyer_surplus": data["util"].tolist(),
        "seller_profit": data["profit"].tolist(),
        "welfare": data["welfare"],
        "tpe_leftover": float(data["pay"].sum(axis=0).sum() - data["pay"].sum(axis=1).sum()),
        "alphas": list(ALPHAS),
        "sweep_payment": sweep["pay"].tolist(),
        "sweep_rate": sweep["rate"].tolist(),
        "sweep_unit_price": sweep["unit"].tolist(),
    }
    (figdir / "economic_figure_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _publish(figdir)
    print(f"[econ] wrote {figdir}")
    return figdir


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate Figs 6--8 (economic layer).")
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    p.add_argument("--figdir", type=Path, default=DEFAULT_FIGDIR)
    p.add_argument("--clearing-dir", type=Path, default=None)
    p.add_argument("--omega-path", type=Path, default=None)
    p.add_argument("--max-iter", type=int, default=80)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    generate_economic_figures(
        args.processed_dir,
        args.figdir,
        args.max_iter,
        clearing_dir=args.clearing_dir,
        omega_path=args.omega_path,
    )
