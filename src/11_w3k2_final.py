"""
W=3 HSP, K=2 BS market plus a preference-vs-patient-split experiment.

Writes every table, figure, and README into ``final result/`` so the
paper-1 W=2 K=3 tree stays untouched.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import load_step

FINAL_DIR = ROOT / "final result"
MARKET_DIR = FINAL_DIR / "market"
SWEEP_DIR = FINAL_DIR / "preference_sweep"
FIG_DIR = FINAL_DIR / "figures"
SCORE_SRC = ROOT / "data" / "processed_w2k3"
SEED = 42
AREA_M = 2000.0
N_BS = 2
N_HSP = 3
HSP_NAMES = ("HSP1", "HSP2", "HSP3")
BS_NAMES = ("BS1", "BS2")
HSP_COLOR = ("#2ca02c", "#1f77b4", "#d62728")
# Example ratios from the user, applied to the FULL cohort (not 10 people).
FULL_RATIOS = (
    (1, 1, 1),
    (4, 3, 3),
    (5, 4, 3),
    (5, 4, 1),
    (7, 2, 1),
    (8, 1, 1),
    (2, 1, 1),
    (3, 2, 1),
    (9, 1, 0),
    (10, 0, 0),
)
SCALE_NS = (10, 50, 100, 500, 1000, 5000, 20000)


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


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def compositions(n: int, parts: int = 3) -> list[tuple[int, ...]]:
    """Ordered non-negative integer compositions of n into ``parts`` bins."""
    if parts == 1:
        return [(n,)]
    out: list[tuple[int, ...]] = []
    for first in range(n + 1):
        for rest in compositions(n - first, parts - 1):
            out.append((first, *rest))
    return out


def counts_from_ratio(n: int, ratio: tuple[int, ...]) -> tuple[int, ...]:
    weights = np.asarray(ratio, dtype=np.float64)
    if float(weights.sum()) <= 0:
        out = [0] * len(ratio)
        out[0] = n
        return tuple(out)
    raw = n * weights / weights.sum()
    counts = np.floor(raw).astype(int)
    leftover = n - int(counts.sum())
    order = np.argsort(-(raw - counts))
    for i in range(leftover):
        counts[order[i % len(counts)]] += 1
    return tuple(int(x) for x in counts)


def assign_counts(n: int, counts: tuple[int, ...], seed: int) -> np.ndarray:
    if sum(counts) != n:
        raise ValueError(f"counts {counts} do not sum to n={n}")
    labels = np.concatenate([np.full(c, i, dtype=np.int64) for i, c in enumerate(counts)])
    rng = np.random.default_rng(seed)
    rng.shuffle(labels)
    return labels


def rho_for_subset(
    agg,
    crit: np.ndarray,
    xy: np.ndarray,
    bs_xy: np.ndarray,
    index: np.ndarray,
    hsp_id: np.ndarray,
) -> dict:
    c = crit[index]
    pts = xy[index]
    bs_id, _, _ = agg.associate_bs(pts, bs_xy)
    c_wk, n_wk, mean_wk = agg.aggregate_links(bs_id, hsp_id, c, N_BS, N_HSP)
    prefs = agg.build_rho(c_wk, n_wk, mean_wk)
    rho = prefs["rho_wk"]
    c_bar = float(np.mean(c_wk) + agg.EPS)
    raw = (c_wk / c_bar) * (1.0 + mean_wk)
    raw = np.where(n_wk > 0, raw, 0.0)
    return {
        "rho": rho,
        "raw": raw,
        "C_wk": c_wk,
        "N_wk": n_wk,
        "mean_c": mean_wk,
        "bs_id": bs_id,
        "hsp_id": hsp_id,
        "mean_rho_hsp": rho.mean(axis=0),
        "mean_raw_hsp": raw.mean(axis=0),
        "mass_hsp": c_wk.sum(axis=0),
    }


def prepare_market_dir() -> None:
    MARKET_DIR.mkdir(parents=True, exist_ok=True)
    scores = SCORE_SRC / "criticality_scores.csv"
    if not scores.exists():
        raise FileNotFoundError(f"Need {scores} (run fusion on eICU first).")
    dest = MARKET_DIR / "criticality_scores.csv"
    if not dest.exists() or dest.stat().st_size != scores.stat().st_size:
        shutil.copy2(scores, dest)
    meta = SCORE_SRC / "patients_all.csv"
    if meta.exists():
        shutil.copy2(meta, MARKET_DIR / "patients_all.csv")


def run_full_market() -> dict:
    prepare_market_dir()
    agg = load_step("aggregator")
    rel = load_step("reluctance")
    opt = load_step("optimizer")
    figs = load_step("auction_figures")
    econ = load_step("economic_figures")

    agg.run_aggregator(
        agg.AggregatorConfig(
            processed_dir=MARKET_DIR,
            artifact_dir=MARKET_DIR,
            n_bs=N_BS,
            n_hsp=N_HSP,
            area_m=AREA_M,
            seed=SEED,
        )
    )
    omega_path = rel.write_frozen_omega(MARKET_DIR, MARKET_DIR / "reluctance", seed=SEED)
    opt.run_optimizer(
        opt.OptimizerConfig(
            processed_dir=MARKET_DIR,
            artifact_dir=MARKET_DIR,
            clearing_dir=MARKET_DIR,
            omega_path=omega_path,
            bs_capacity_mbps=5.0,
            scale_capacity_by_load=False,
            max_iter=1500,
            step0=0.08,
            pi_init=0.32,
        )
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figs.generate_figures(
        processed_dir=MARKET_DIR,
        figdir=FIG_DIR,
        max_iter=450,
        omega_path=omega_path,
        copy_to_report=False,
        deltas=(0.034, 0.040, 0.046),
        pi_init=0.30,
    )
    econ.generate_economic_figures(
        processed_dir=MARKET_DIR,
        figdir=FIG_DIR,
        max_iter=150,
        clearing_dir=MARKET_DIR,
        omega_path=omega_path,
        copy_to_report=False,
    )
    return {"omega_path": omega_path}


def load_pool(agg) -> dict:
    cfg = agg.AggregatorConfig(processed_dir=MARKET_DIR, n_bs=N_BS, n_hsp=N_HSP, seed=SEED)
    customers = agg._load_customers(cfg)
    n = len(customers)
    crit = customers["criticality"].to_numpy(dtype=np.float64)
    xy = agg.place_customers(n, AREA_M, SEED)
    bs_xy = agg.place_base_stations(N_BS, AREA_M)
    bs_id, dist_m, rsrp = agg.associate_bs(xy, bs_xy)
    hsp_id = agg.assign_hsp(n, N_HSP, SEED)
    return {
        "customers": customers,
        "crit": crit,
        "xy": xy,
        "bs_xy": bs_xy,
        "bs_id": bs_id,
        "hsp_id": hsp_id,
        "dist_m": dist_m,
        "rsrp": rsrp,
    }


def _record_pack(n_total: int, counts: tuple[int, ...], pack: dict, ratio=None) -> dict:
    rec = {
        "n_total": n_total,
        "n_hsp1": counts[0],
        "n_hsp2": counts[1],
        "n_hsp3": counts[2],
        "split": f"{counts[0]}-{counts[1]}-{counts[2]}",
        "ratio": ":".join(str(x) for x in ratio) if ratio is not None else "",
        "share_hsp1": counts[0] / n_total,
        "share_hsp2": counts[1] / n_total,
        "share_hsp3": counts[2] / n_total,
        "imbalance": max(counts) / n_total,
    }
    for i, bs in enumerate(BS_NAMES):
        for j, hsp in enumerate(HSP_NAMES):
            rec[f"rho_{bs}_{hsp}"] = float(pack["rho"][i, j])
            rec[f"N_{bs}_{hsp}"] = int(pack["N_wk"][i, j])
    for j, hsp in enumerate(HSP_NAMES):
        rec[f"mean_rho_{hsp}"] = float(pack["mean_rho_hsp"][j])
        rec[f"mean_raw_{hsp}"] = float(pack["mean_raw_hsp"][j])
        rec[f"mass_{hsp}"] = float(pack["mass_hsp"][j])
    return rec


def run_preference_sweep(agg, pool: dict) -> dict:
    """Main experiment uses ALL users. N=10 is kept only as a toy check."""
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    n_all = len(pool["crit"])
    rng = np.random.default_rng(SEED + 3)
    order = rng.permutation(n_all)
    full_index = np.arange(n_all)

    full_rows = []
    for ratio in FULL_RATIOS:
        counts = counts_from_ratio(n_all, ratio)
        hsp_id = assign_counts(n_all, counts, seed=SEED + 21 * sum(ratio) + ratio[0])
        pack = rho_for_subset(agg, pool["crit"], pool["xy"], pool["bs_xy"], full_index, hsp_id)
        full_rows.append(_record_pack(n_all, counts, pack, ratio=ratio))
    full_df = pd.DataFrame(full_rows)
    full_df.to_csv(SWEEP_DIR / "preference_vs_split.csv", index=False)

    share_rows = []
    for share in np.round(np.linspace(0.05, 0.90, 18), 4):
        n1 = int(round(share * n_all))
        rem = n_all - n1
        n2 = rem // 2
        n3 = rem - n2
        counts = (n1, n2, n3)
        hsp_id = assign_counts(n_all, counts, seed=SEED + 33)
        pack = rho_for_subset(agg, pool["crit"], pool["xy"], pool["bs_xy"], full_index, hsp_id)
        share_rows.append(_record_pack(n_all, counts, pack))
    share_df = pd.DataFrame(share_rows)
    share_df.to_csv(SWEEP_DIR / "preference_vs_share.csv", index=False)

    toy_rows = []
    for counts in compositions(10) + [(5, 4, 3)]:
        n_total = sum(counts)
        index = order[:n_total]
        hsp_id = assign_counts(n_total, counts, seed=SEED + 1000 * n_total + counts[0])
        pack = rho_for_subset(agg, pool["crit"], pool["xy"], pool["bs_xy"], index, hsp_id)
        toy_rows.append(_record_pack(n_total, counts, pack))
    toy_df = pd.DataFrame(toy_rows)
    toy_df.to_csv(SWEEP_DIR / "preference_vs_split_N10_toy.csv", index=False)

    scale_rows = []
    for n_total in SCALE_NS + (n_all,):
        index = order[:n_total]
        base, rem = divmod(n_total, N_HSP)
        counts = tuple(base + (1 if j < rem else 0) for j in range(N_HSP))
        hsp_id = assign_counts(n_total, counts, seed=SEED + 9)
        pack = rho_for_subset(agg, pool["crit"], pool["xy"], pool["bs_xy"], index, hsp_id)
        scale_rows.append(_record_pack(n_total, counts, pack))
    scale_df = pd.DataFrame(scale_rows)
    scale_df.to_csv(SWEEP_DIR / "preference_vs_n_balanced.csv", index=False)

    _ieee_rc()
    _fig_full_ratios(full_df)
    _fig_share_sweep(share_df)
    _fig_highlight_splits(toy_df)
    _fig_composition_scatter(toy_df)
    _fig_raw_vs_count(toy_df)
    _fig_scale(scale_df)
    return {"sweep": full_df, "share": share_df, "scale": scale_df, "toy": toy_df}


def _fig_full_ratios(df: pd.DataFrame) -> None:
    labels = [
        f"{row.ratio}\n({row.n_hsp1 // 1000}k-{row.n_hsp2 // 1000}k-{row.n_hsp3 // 1000}k)"
        for row in df.itertuples()
    ]
    x = np.arange(len(df))
    width = 0.22
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    for j, hsp in enumerate(HSP_NAMES):
        ax.bar(
            x + (j - 1) * width,
            df[f"mean_rho_{hsp}"].to_numpy(),
            width,
            label=hsp,
            color=HSP_COLOR[j],
            edgecolor="k",
            linewidth=0.3,
        )
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylabel(r"Mean $\rho$ of that HSP (2 cells)")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Share of all 74,454 users (example ratios, not 10 patients)")
    ax.legend(framealpha=0.92)
    fig.tight_layout()
    _save(fig, SWEEP_DIR / "fig_preference_selected_splits.png")


def _fig_share_sweep(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    ax = axes[0]
    for j, hsp in enumerate(HSP_NAMES):
        ax.plot(
            df["share_hsp1"],
            df[f"mass_{hsp}"],
            marker="o",
            color=HSP_COLOR[j],
            label=hsp,
        )
    ax.set_xlabel("Share of all users given to HSP1")
    ax.set_ylabel("Criticality mass C of that HSP")
    ax.legend(framealpha=0.92)
    ax = axes[1]
    for j, hsp in enumerate(HSP_NAMES):
        ax.plot(
            df["share_hsp1"],
            df[f"mean_rho_{hsp}"],
            marker="o",
            color=HSP_COLOR[j],
            label=hsp,
        )
    ax.set_xlabel("Share of all users given to HSP1")
    ax.set_ylabel(r"Mean $\rho$ (max-normalized)")
    ax.set_ylim(0.0, 1.05)
    ax.legend(framealpha=0.92)
    fig.tight_layout()
    _save(fig, SWEEP_DIR / "fig_preference_vs_share.png")


def _fig_highlight_splits(df: pd.DataFrame) -> None:
    wanted = {"4-3-3", "5-4-1", "7-2-1", "8-1-1", "10-0-0", "5-4-3"}
    sub = df[df["split"].isin(wanted)].drop_duplicates("split")
    order = [s for s in ("4-3-3", "5-4-1", "7-2-1", "8-1-1", "10-0-0", "5-4-3") if s in set(sub["split"])]
    sub = sub.set_index("split").loc[order]
    links = [f"rho_{bs}_{hsp}" for bs in BS_NAMES for hsp in HSP_NAMES]
    x = np.arange(len(order))
    width = 0.13
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    for i, link in enumerate(links):
        ax.bar(
            x + (i - 2.5) * width,
            sub[link].to_numpy(),
            width,
            label=link.replace("rho_", "").replace("_", "–"),
            edgecolor="k",
            linewidth=0.3,
        )
    ax.set_xticks(x, order)
    ax.set_xlabel("Toy split of 10 patients (not the real market)")
    ax.set_ylabel(r"Preference $\rho_{wk}$ (max-normalized)")
    ax.set_ylim(0.0, 1.05)
    ax.legend(ncol=3, fontsize=7, framealpha=0.92)
    fig.tight_layout()
    _save(fig, SWEEP_DIR / "fig_preference_selected_splits_N10_toy.png")


def _fig_composition_scatter(df: pd.DataFrame) -> None:
    ten = df[df["n_total"] == 10]
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.3), sharey=True)
    for j, hsp in enumerate(HSP_NAMES):
        ax = axes[j]
        ax.scatter(
            ten[f"n_{hsp.lower()}"],
            ten[f"mean_rho_{hsp}"],
            s=18,
            c=HSP_COLOR[j],
            edgecolors="k",
            linewidths=0.25,
            alpha=0.75,
        )
        ax.set_xlabel(f"Patients on {hsp} (out of 10)")
        ax.set_title(hsp)
    axes[0].set_ylabel(r"Mean $\rho$ of that HSP across 2 BSs")
    fig.tight_layout()
    _save(fig, SWEEP_DIR / "fig_preference_vs_count_N10.png")

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.scatter(ten["imbalance"], ten["mean_rho_HSP1"], s=16, c=HSP_COLOR[0], label="HSP1", alpha=0.75)
    ax.scatter(ten["imbalance"], ten["mean_rho_HSP2"], s=16, c=HSP_COLOR[1], label="HSP2", alpha=0.75)
    ax.scatter(ten["imbalance"], ten["mean_rho_HSP3"], s=16, c=HSP_COLOR[2], label="HSP3", alpha=0.75)
    ax.set_xlabel(r"Imbalance $\max_w n_w / 10$")
    ax.set_ylabel(r"Mean $\rho$ of each HSP")
    ax.legend(framealpha=0.92)
    fig.tight_layout()
    _save(fig, SWEEP_DIR / "fig_preference_vs_imbalance_N10.png")


def _fig_raw_vs_count(df: pd.DataFrame) -> None:
    ten = df[df["n_total"] == 10]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5))
    ax = axes[0]
    for j, hsp in enumerate(HSP_NAMES):
        ax.scatter(
            ten[f"n_{hsp.lower()}"],
            ten[f"mass_{hsp}"],
            s=18,
            c=HSP_COLOR[j],
            edgecolors="k",
            linewidths=0.25,
            alpha=0.75,
            label=hsp,
        )
    ax.set_xlabel("Patients on that HSP (out of 10)")
    ax.set_ylabel("Criticality mass C of that HSP")
    ax.legend(framealpha=0.92)
    ax = axes[1]
    for j, hsp in enumerate(HSP_NAMES):
        ax.scatter(
            ten[f"n_{hsp.lower()}"],
            ten[f"mean_raw_{hsp}"],
            s=18,
            c=HSP_COLOR[j],
            edgecolors="k",
            linewidths=0.25,
            alpha=0.75,
            label=hsp,
        )
    ax.set_xlabel("Patients on that HSP (out of 10)")
    ax.set_ylabel("Raw preference (before max-normalize)")
    ax.legend(framealpha=0.92)
    fig.tight_layout()
    _save(fig, SWEEP_DIR / "fig_raw_preference_vs_count_N10.png")


def _fig_scale(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    markers = ("o", "s")
    for i, bs in enumerate(BS_NAMES):
        for j, hsp in enumerate(HSP_NAMES):
            ax.plot(
                df["n_total"],
                df[f"rho_{bs}_{hsp}"],
                marker=markers[i],
                color=HSP_COLOR[j],
                linestyle=("-" if i == 0 else "--"),
                label=f"{bs}–{hsp}",
            )
    ax.set_xscale("log")
    ax.set_xlabel("Number of patients (balanced 3-way split)")
    ax.set_ylabel(r"Preference $\rho_{wk}$")
    ax.set_ylim(0.0, 1.05)
    ax.legend(ncol=2, fontsize=7, framealpha=0.92)
    fig.tight_layout()
    _save(fig, SWEEP_DIR / "fig_preference_vs_n_balanced.png")


def fig_allocation_map(pool: dict) -> None:
    _ieee_rc()
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    rng = np.random.default_rng(SEED)
    take = rng.choice(len(pool["xy"]), size=min(1600, len(pool["xy"])), replace=False)
    for j, _hsp in enumerate(HSP_NAMES):
        mask = pool["hsp_id"][take] == j
        ax.scatter(
            pool["xy"][take][mask, 0],
            pool["xy"][take][mask, 1],
            s=3,
            c=HSP_COLOR[j],
            alpha=0.28,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
    bs_xy = np.asarray(pool["bs_xy"], dtype=np.float64)
    if len(bs_xy) == 2:
        mid = 0.5 * (bs_xy[0] + bs_xy[1])
        delta = bs_xy[1] - bs_xy[0]
        nrm = np.array([-delta[1], delta[0]], dtype=np.float64)
        nrm = nrm / max(float(np.linalg.norm(nrm)), 1e-9)
        span = 1.2 * AREA_M
        ax.plot(
            [mid[0] - span * nrm[0], mid[0] + span * nrm[0]],
            [mid[1] - span * nrm[1], mid[1] + span * nrm[1]],
            color="0.35",
            linestyle=":",
            linewidth=1.0,
            zorder=2,
        )
    ax.scatter(
        bs_xy[:, 0],
        bs_xy[:, 1],
        marker="^",
        s=55,
        c="k",
        edgecolors="w",
        linewidths=0.7,
        zorder=5,
    )
    for i, name in enumerate(BS_NAMES):
        ax.annotate(
            name,
            bs_xy[i],
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=8,
            fontweight="bold",
            zorder=6,
        )
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=HSP_COLOR[j],
            markeredgecolor="none",
            markersize=6,
            label=HSP_NAMES[j],
        )
        for j in range(N_HSP)
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor="k",
            markeredgecolor="k",
            markersize=8,
            label="BS",
        )
    )
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=4,
        framealpha=0.95,
        fontsize=8,
        handletextpad=0.35,
        columnspacing=1.1,
        borderpad=0.35,
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(0, AREA_M)
    ax.set_ylim(0, AREA_M)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.yaxis.set_major_locator(MultipleLocator(500))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    _save(fig, SWEEP_DIR / "fig_allocation_map.png")


def write_allocation_tables(pool: dict, agg) -> dict:
    c_wk, n_wk, mean_wk = agg.aggregate_links(
        pool["bs_id"], pool["hsp_id"], pool["crit"], N_BS, N_HSP
    )
    users_hsp = n_wk.sum(axis=0)
    load_bs = n_wk.sum(axis=1)
    alloc = {
        "n_users": int(len(pool["crit"])),
        "n_unique_stays": int(pool["customers"]["patient_id"].nunique())
        if "patient_id" in pool["customers"].columns
        else None,
        "how": (
            "Each eICU vital window is one uplink user. Users are placed uniformly "
            "on a 2 km × 2 km map (seed 42), associated to the stronger of two "
            "cells by max RSRP, and subscribed to HSP1/HSP2/HSP3 by an independent "
            "uniform draw (seed 42+7). No disease-to-hospital map."
        ),
        "users_per_hsp": {HSP_NAMES[j]: int(users_hsp[j]) for j in range(N_HSP)},
        "users_per_bs": {BS_NAMES[i]: int(load_bs[i]) for i in range(N_BS)},
        "N_wk": {
            BS_NAMES[i]: {HSP_NAMES[j]: int(n_wk[i, j]) for j in range(N_HSP)} for i in range(N_BS)
        },
        "C_wk": {
            BS_NAMES[i]: {HSP_NAMES[j]: float(c_wk[i, j]) for j in range(N_HSP)} for i in range(N_BS)
        },
        "share_per_hsp": {HSP_NAMES[j]: float(users_hsp[j] / users_hsp.sum()) for j in range(N_HSP)},
    }
    (FINAL_DIR / "patient_allocation.json").write_text(json.dumps(alloc, indent=2), encoding="utf-8")
    pd.DataFrame(n_wk, index=BS_NAMES, columns=HSP_NAMES).to_csv(FINAL_DIR / "N_wk.csv")
    pd.DataFrame(c_wk, index=BS_NAMES, columns=HSP_NAMES).to_csv(FINAL_DIR / "C_wk.csv")
    return alloc


def write_readmes(alloc: dict, sweep: pd.DataFrame, scale: pd.DataFrame) -> None:
    opt = {}
    opt_path = MARKET_DIR / "optimizer_summary.json"
    if opt_path.exists():
        opt = json.loads(opt_path.read_text(encoding="utf-8"))
    rho_path = MARKET_DIR / "rho_wk.csv"
    rho_txt = rho_path.read_text(encoding="utf-8") if rho_path.exists() else ""
    omega_path = MARKET_DIR / "omega_wk.csv"
    omega_txt = omega_path.read_text(encoding="utf-8") if omega_path.exists() else ""

    examples = []
    for ratio in ("1:1:1", "4:3:3", "5:4:3", "7:2:1", "8:1:1"):
        row = sweep[sweep["ratio"] == ratio] if "ratio" in sweep.columns else sweep.iloc[0:0]
        if row.empty:
            continue
        r = row.iloc[0]
        examples.append(
            f"- **{ratio}** -> {int(r['n_hsp1'])}/{int(r['n_hsp2'])}/{int(r['n_hsp3'])} users: "
            f"mean rho = HSP1 {r['mean_rho_HSP1']:.3f}, "
            f"HSP2 {r['mean_rho_HSP2']:.3f}, HSP3 {r['mean_rho_HSP3']:.3f}"
        )

    users = alloc["users_per_hsp"]
    nwk = alloc["N_wk"]
    (FINAL_DIR / "README.md").write_text(
        f"""# Final result — 3 HSPs × 2 base stations

Topology for this folder: **W = 3 buyers (HSP1, HSP2, HSP3)** and **K = 2 sellers (BS1, BS2)**.
Capacity is **5 Mbps per cell**, reluctance is the frozen one-shot radio table, preference is max-normalized.

Paper-1 files under `data/processed_w2k3/` and `report/` are unchanged (that run is still 2 HSPs × 3 BSs).

## How patients are allocated

{alloc["how"]}

There are **{alloc["n_users"]}** uplink users (eICU Demo vital windows). Unique ICU stays: **{alloc["n_unique_stays"]}**.

### Users per hospital (random 3-way split, seed 42)

| HSP | Users | Share |
| --- | ---: | ---: |
| HSP1 | {users["HSP1"]} | {alloc["share_per_hsp"]["HSP1"]:.1%} |
| HSP2 | {users["HSP2"]} | {alloc["share_per_hsp"]["HSP2"]:.1%} |
| HSP3 | {users["HSP3"]} | {alloc["share_per_hsp"]["HSP3"]:.1%} |

### Users on each (BS, HSP) link

|  | HSP1 | HSP2 | HSP3 | BS total |
| --- | ---: | ---: | ---: | ---: |
| BS1 | {nwk["BS1"]["HSP1"]} | {nwk["BS1"]["HSP2"]} | {nwk["BS1"]["HSP3"]} | {alloc["users_per_bs"]["BS1"]} |
| BS2 | {nwk["BS2"]["HSP1"]} | {nwk["BS2"]["HSP2"]} | {nwk["BS2"]["HSP3"]} | {alloc["users_per_bs"]["BS2"]} |

A user on the left half of the map usually hears **BS1** more strongly; the right half hears **BS2**. HSP membership is independent of location, so each hospital’s users split across the two cells in roughly the same ratio.

Map (1.6k-user sample): `preference_sweep/fig_allocation_map.png`.

## Full-market preference and clearing

Preference matrix (`market/rho_wk.csv`):

```
{rho_txt.strip()}
```

Frozen reluctance (`market/omega_wk.csv`):

```
{omega_txt.strip()}
```

| Metric | Value |
| --- | --- |
| Welfare | {opt.get("welfare", float("nan")):.3f} |
| Total rate (Mbps) | {opt.get("total_rate_mbps", float("nan")):.3f} |
| Total payment | {opt.get("total_payment", float("nan")):.3f} |
| Gap | {opt.get("gap", float("nan")):.3e} |
| Converged | {opt.get("converged")} |

Auction traces: `figures/fig2_convergence.png` … `fig8_preference_vs_payment.png`.

## Preference vs how many patients each HSP has

The **real market uses all {alloc["n_users"]} windows** ({alloc["n_unique_stays"]} ICU stays).
"10 patients as 5-4-3 or 7-2-1" was only an example of *shares*. Those ratios are now applied to the full cohort (5:4:3 -> about 31k / 25k / 19k users).

Preference rho = (C / mean C) * (1 + mean c), then divided by the largest of the six links.
On 74k users the law of large numbers applies: more patients on an HSP raises that HSP's mass and rho smoothly.

Selected full-cohort ratios:

{chr(10).join(examples)}

Main figures:

- `preference_sweep/fig_preference_vs_share.png` -- give HSP1 from 5% to 90% of all 74,454 users
- `preference_sweep/fig_preference_selected_splits.png` -- 5:4:3, 7:2:1, 8:1:1, ... on the full cohort
- `preference_sweep/fig_preference_vs_n_balanced.png` -- grow N with a balanced 3-way split

Toy N=10 files are tagged `_N10_toy` and are **not** the prediction market.

Tables: `preference_sweep/preference_vs_split.csv`, `preference_sweep/preference_vs_share.csv`.

## Folder

```
final result/
  README.md
  patient_allocation.json
  N_wk.csv
  C_wk.csv
  market/                 rho, omega, clearing, assignment
  figures/                auction Figs 2–8
  preference_sweep/       split experiment
```

Re-run:

```bash
python src/11_w3k2_final.py
```
""",
        encoding="utf-8",
    )

    (MARKET_DIR / "README.md").write_text(
        "W=3 HSP x K=2 BS clearing at 5 Mbps with frozen omega.\n"
        "Tables: rho_wk.csv, omega_wk.csv, N_wk.csv, C_wk.csv, d_wk.csv, payment_wk.csv,\n"
        "optimizer_summary.json. Assignment of all 74454 users: customer_assignment.csv.\n",
        encoding="utf-8",
    )
    (SWEEP_DIR / "README.md").write_text(
        """# Preference versus patient split

The prediction market uses **all 74,454 eICU windows** (1,821 stays).
`preference_vs_split.csv` applies the example ratios (5:4:3, 7:2:1, ...) to that full set.
`preference_vs_share.csv` gives HSP1 from 5% to 90% of every user.

N=10 files (`*_N10_toy*`) are a classroom example only.

Columns `rho_BS*_HSP*` are max-normalized preference.
`mass_HSP*` is criticality mass (tracks headcount on the full cohort).
""",
        encoding="utf-8",
    )


def main() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    print("[final] W=3 HSPs, K=2 BSs ->", FINAL_DIR)
    skip_market = "--sweep-only" in sys.argv and (MARKET_DIR / "optimizer_summary.json").exists()
    if skip_market:
        print("[final] skipping market (already cleared)")
    else:
        run_full_market()
    agg = load_step("aggregator")
    pool = load_pool(agg)
    alloc = write_allocation_tables(pool, agg)
    fig_allocation_map(pool)
    sweep_pack = run_preference_sweep(agg, pool)
    write_readmes(alloc, sweep_pack["sweep"], sweep_pack["scale"])
    fig6 = FIG_DIR / "fig6_preference_criticality.png"
    if not fig6.exists():
        econ = load_step("economic_figures")
        data = econ._load_cleared(MARKET_DIR, clearing_dir=MARKET_DIR)
        econ._ieee_rc()
        econ.fig_preference_anatomy(data, FIG_DIR)
    print("[final] users per HSP", alloc["users_per_hsp"])
    print("[final] N_wk", alloc["N_wk"])
    print("[final] wrote", FINAL_DIR)


if __name__ == "__main__":
    main()
