"""
Reluctance comparison figures: one chart per PNG.

Default market for this module is W=2 HSPs and K=3 BSs (paper notation).
Does not overwrite Figs 2--8 of the 4x3 eICU snapshot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import load_step

DEFAULT_PROCESSED = ROOT / "data" / "processed_w2k3"
DEFAULT_REL = ROOT / "data" / "processed_w2k3" / "reluctance"
DEFAULT_FIGDIR = ROOT / "artifacts" / "reluctance_w2k3" / "figures"
HSP_PAPER = ("HSP1", "HSP2", "HSP3")
BS_PAPER = ("BS1", "BS2", "BS3", "BS4")
BS_COLOR = ("#2ca02c", "#d62728", "#1f77b4", "#9467bd")
STEMS = (
    "fig_omega_star",
    "fig_omega_hat",
    "fig_rate_change",
    "fig_bid_change",
    "fig_stress_omega",
    "fig_stress_rate_bs1",
    "fig_stress_rate_per_bs",
    "fig_predictor_fit",
    "fig_welfare_compare",
)


def _ieee() -> None:
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


def _labels(n_bs: int, n_hsp: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return HSP_PAPER[:n_hsp], BS_PAPER[:n_bs]


def _grouped_bars(ax, mat: np.ndarray, hsp_lab, bs_lab) -> None:
    n_bs, n_hsp = mat.shape
    x = np.arange(n_hsp)
    width = 0.8 / max(n_bs, 1)
    for i in range(n_bs):
        ax.bar(
            x + (i - (n_bs - 1) / 2) * width,
            mat[i],
            width,
            label=bs_lab[i],
            color=BS_COLOR[i],
            edgecolor="k",
            linewidth=0.4,
        )
    ax.set_xticks(x, hsp_lab)
    ax.set_xlabel("HSP")
    ax.legend(fontsize=7, framealpha=0.92)


def _load_clearing(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def run_three_clearings(processed_dir: Path, rel_dir: Path, runs_root: Path) -> dict[str, Path]:
    jobs = {
        "baseline": (rel_dir / "omega_ones.npz", runs_root / "baseline"),
        "dynamic": (rel_dir / "eicu_slot.npz", runs_root / "dynamic"),
        "stress": (rel_dir / "stress_slot.npz", runs_root / "stress"),
    }
    existing = {name: dest / "auction_clearing.npz" for name, (_, dest) in jobs.items()}
    if all(p.exists() for p in existing.values()):
        print("[rel-fig] reusing existing clearings")
        return existing
    opt = load_step("optimizer")
    out = {}
    for name, (omega_path, dest) in jobs.items():
        print(f"[rel-fig] clearing {name}  omega={omega_path.name}")
        opt.run_optimizer(
            opt.OptimizerConfig(
                processed_dir=processed_dir,
                artifact_dir=dest,
                omega_path=omega_path,
                clearing_dir=dest,
                bs_capacity_mbps=5.0,
                scale_capacity_by_load=False,
            )
        )
        out[name] = dest / "auction_clearing.npz"
    return out


def fig_omega_matrix(mat: np.ndarray, figdir: Path, stem: str, ylabel: str) -> None:
    n_bs, n_hsp = mat.shape
    hsp_lab, bs_lab = _labels(n_bs, n_hsp)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    _grouped_bars(ax, mat, hsp_lab, bs_lab)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    _save(fig, figdir, stem)


def fig_rate_change(clear: dict[str, dict], figdir: Path) -> None:
    base, dyn = clear["baseline"], clear["dynamic"]
    delta = np.asarray(dyn["d_r_mid"]) - np.asarray(base["d_r_mid"])
    n_bs, n_hsp = delta.shape
    hsp_lab, bs_lab = _labels(n_bs, n_hsp)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    _grouped_bars(ax, delta, hsp_lab, bs_lab)
    ax.axhline(0.0, color="k", linewidth=0.7)
    ax.set_ylabel("Rate change (Mbps)")
    fig.tight_layout()
    _save(fig, figdir, "fig_rate_change")


def fig_bid_change(clear: dict[str, dict], figdir: Path) -> None:
    base, dyn = clear["baseline"], clear["dynamic"]
    zeta_b = np.asarray(base["omega_wk"]) * (
        0.03 / np.maximum(np.asarray(base["d_r_mid"]), 1e-12) + 0.015
    )
    zeta_d = np.asarray(dyn["omega_wk"]) * (
        0.03 / np.maximum(np.asarray(dyn["d_r_mid"]), 1e-12) + 0.015
    )
    delta = zeta_d - zeta_b
    n_bs, n_hsp = delta.shape
    hsp_lab, bs_lab = _labels(n_bs, n_hsp)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    _grouped_bars(ax, delta, hsp_lab, bs_lab)
    ax.axhline(0.0, color="k", linewidth=0.7)
    ax.set_ylabel(r"Bid change $\Delta\zeta$")
    fig.tight_layout()
    _save(fig, figdir, "fig_bid_change")


def fig_stress_omega(clear: dict[str, dict], figdir: Path, stress_bs: int = 0) -> None:
    dyn, st = clear["dynamic"], clear["stress"]
    n_hsp = int(dyn["d_r_mid"].shape[1])
    hsp_lab, _ = _labels(int(dyn["d_r_mid"].shape[0]), n_hsp)
    x = np.arange(n_hsp)
    width = 0.35
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.bar(x - width / 2, dyn["omega_wk"][stress_bs], width, label="Usual load", color="#1f77b4", edgecolor="k", linewidth=0.4)
    ax.bar(x + width / 2, st["omega_wk"][stress_bs], width, label="Stressed BS1", color="#d62728", edgecolor="k", linewidth=0.4)
    ax.set_xticks(x, hsp_lab)
    ax.set_xlabel("HSP")
    ax.set_ylabel(r"Reluctance $\omega$")
    ax.set_ylim(0.0, 1.0)
    ax.legend(framealpha=0.92)
    fig.tight_layout()
    _save(fig, figdir, "fig_stress_omega")


def fig_stress_rate_bs1(clear: dict[str, dict], figdir: Path, stress_bs: int = 0) -> None:
    dyn, st = clear["dynamic"], clear["stress"]
    n_hsp = int(dyn["d_r_mid"].shape[1])
    hsp_lab, _ = _labels(int(dyn["d_r_mid"].shape[0]), n_hsp)
    x = np.arange(n_hsp)
    width = 0.35
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.bar(x - width / 2, dyn["d_r_mid"][stress_bs], width, label="Usual load", color="#1f77b4", edgecolor="k", linewidth=0.4)
    ax.bar(x + width / 2, st["d_r_mid"][stress_bs], width, label="Stressed BS1", color="#d62728", edgecolor="k", linewidth=0.4)
    ax.set_xticks(x, hsp_lab)
    ax.set_xlabel("HSP")
    ax.set_ylabel("Rate (Mbps)")
    ax.legend(framealpha=0.92)
    fig.tight_layout()
    _save(fig, figdir, "fig_stress_rate_bs1")


def fig_stress_rate_per_bs(clear: dict[str, dict], figdir: Path) -> None:
    dyn, st = clear["dynamic"], clear["stress"]
    n_bs = int(dyn["d_r_mid"].shape[0])
    _, bs_lab = _labels(n_bs, int(dyn["d_r_mid"].shape[1]))
    x = np.arange(n_bs)
    width = 0.35
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.bar(x - width / 2, np.asarray(dyn["d_r_mid"]).sum(axis=1), width, label="Usual load", color="#1f77b4", edgecolor="k", linewidth=0.4)
    ax.bar(x + width / 2, np.asarray(st["d_r_mid"]).sum(axis=1), width, label="Stressed BS1", color="#d62728", edgecolor="k", linewidth=0.4)
    ax.set_xticks(x, bs_lab)
    ax.set_xlabel("BS")
    ax.set_ylabel("Total rate (Mbps)")
    ax.legend(framealpha=0.92)
    fig.tight_layout()
    _save(fig, figdir, "fig_stress_rate_per_bs")


def fig_predictor_fit(rel_dir: Path, figdir: Path) -> None:
    df = pd.read_csv(rel_dir / "train_table.csv")
    te = df[df["split"] == "test"]
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.scatter(te["omega_star"], te["omega_hat"], s=8, alpha=0.22, color="#1f77b4", linewidths=0)
    lim = (0.05, 0.95)
    ax.plot(lim, lim, color="#d62728", linestyle=":", linewidth=1.2, label="identity")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r"Ground truth $\omega^\star$")
    ax.set_ylabel(r"Predicted $\hat\omega$")
    ax.legend(framealpha=0.92)
    fig.tight_layout()
    _save(fig, figdir, "fig_predictor_fit")


def fig_welfare(clear: dict[str, dict], figdir: Path) -> None:
    labels = (r"$\omega=1$", r"Dynamic $\hat\omega$", "Stressed BS1")
    vals = (
        float(clear["baseline"]["welfare"]),
        float(clear["dynamic"]["welfare"]),
        float(clear["stress"]["welfare"]),
    )
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.bar(labels, vals, color=("#8c564b", "#1f77b4", "#d62728"), edgecolor="k", linewidth=0.4)
    ax.set_ylabel("Social welfare")
    fig.tight_layout()
    _save(fig, figdir, "fig_welfare_compare")


def generate_reluctance_figures(
    processed_dir: Path | None = None,
    rel_dir: Path | None = None,
    figdir: Path | None = None,
) -> Path:
    processed_dir = Path(processed_dir or DEFAULT_PROCESSED)
    rel_dir = Path(rel_dir or DEFAULT_REL)
    figdir = Path(figdir or DEFAULT_FIGDIR)
    runs_root = figdir.parent / "runs"
    if not (rel_dir / "eicu_slot.npz").exists():
        rel = load_step("reluctance")
        rel.run_reluctance(
            rel.ReluctanceConfig(
                processed_dir=processed_dir,
                out_dir=rel_dir,
                artifact_dir=figdir.parent,
            )
        )
    paths = run_three_clearings(processed_dir, rel_dir, runs_root)
    clear = {k: _load_clearing(p) for k, p in paths.items()}
    _ieee()
    eicu = np.load(rel_dir / "eicu_slot.npz")
    fig_omega_matrix(
        np.asarray(eicu["omega_star"])[0],
        figdir,
        "fig_omega_star",
        r"Reluctance $\omega_{kw}$",
    )
    fig_omega_matrix(
        np.asarray(eicu["omega_hat"])[0],
        figdir,
        "fig_omega_hat",
        r"Reluctance $\omega_{kw}$",
    )
    fig_rate_change(clear, figdir)
    fig_bid_change(clear, figdir)
    fig_stress_omega(clear, figdir)
    fig_stress_rate_bs1(clear, figdir)
    fig_stress_rate_per_bs(clear, figdir)
    fig_predictor_fit(rel_dir, figdir)
    fig_welfare(clear, figdir)
    payload = {
        "market": {"W_HSP": int(clear["dynamic"]["d_r_mid"].shape[1]), "K_BS": int(clear["dynamic"]["d_r_mid"].shape[0])},
        "baseline_welfare": float(clear["baseline"]["welfare"]),
        "dynamic_welfare": float(clear["dynamic"]["welfare"]),
        "stress_welfare": float(clear["stress"]["welfare"]),
        "baseline_gap": float(clear["baseline"]["gap"]),
        "dynamic_gap": float(clear["dynamic"]["gap"]),
        "stress_gap": float(clear["stress"]["gap"]),
        "baseline_rate": float(np.asarray(clear["baseline"]["d_r_mid"]).sum()),
        "dynamic_rate": float(np.asarray(clear["dynamic"]["d_r_mid"]).sum()),
        "stress_rate": float(np.asarray(clear["stress"]["d_r_mid"]).sum()),
        "baseline_pay": float(np.asarray(clear["baseline"]["payment_wk"]).sum()),
        "dynamic_pay": float(np.asarray(clear["dynamic"]["payment_wk"]).sum()),
        "note": "Radio features are simulated on the eICU geometry; not operator RAN traces. One chart per PNG.",
    }
    (figdir.parent / "comparison_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_dest = ROOT / "report" / "current"
    report_dest.mkdir(parents=True, exist_ok=True)
    for stem in STEMS:
        src = figdir / f"{stem}.png"
        if src.exists():
            (report_dest / src.name).write_bytes(src.read_bytes())
    print(f"[rel-fig] wrote {figdir}")
    print(json.dumps(payload, indent=2))
    return figdir


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Single-panel reluctance figures (W=2, K=3).")
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    p.add_argument("--rel-dir", type=Path, default=DEFAULT_REL)
    p.add_argument("--figdir", type=Path, default=DEFAULT_FIGDIR)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    generate_reluctance_figures(args.processed_dir, args.rel_dir, args.figdir)
