"""End-to-end 5G healthcare auction pipeline.

preprocess → LSTM → fusion → aggregator → KKT double-auction clearing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src import load_step

STEPS = ("preprocess", "lstm", "fusion", "aggregate", "optimize")


def main() -> None:
    parser = argparse.ArgumentParser(description="5G healthcare resource allocation pipeline.")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--n-bs", type=int, default=None)
    parser.add_argument("--n-hsp", type=int, default=None)
    parser.add_argument(
        "--from-step",
        choices=list(STEPS),
        default="preprocess",
        help="First pipeline stage to run.",
    )
    parser.add_argument(
        "--to-step",
        choices=list(STEPS),
        default="optimize",
        help="Last pipeline stage to run.",
    )
    args = parser.parse_args()

    start = STEPS.index(args.from_step)
    end = STEPS.index(args.to_step)
    if end < start:
        raise SystemExit("--to-step must be at or after --from-step")
    active = set(STEPS[start : end + 1])
    downstream = ("fusion", "aggregate", "optimize")

    preprocess = load_step("preprocess")
    processed_dir = Path(args.output_dir or preprocess.DEFAULT_OUTPUT_DIR)
    windows_ready = (processed_dir / "windows_train.npz").exists()
    preds_ready = (processed_dir / "lstm_predictions.npz").exists()
    fusion_ready = (processed_dir / "criticality_scores.npz").exists()
    rho_ready = (processed_dir / "rho_wk.npz").exists()

    if "preprocess" in active or not windows_ready:
        cfg_kwargs = {}
        if args.data_path is not None:
            cfg_kwargs["data_path"] = args.data_path
        if args.output_dir is not None:
            cfg_kwargs["output_dir"] = args.output_dir
        if args.max_patients is not None:
            cfg_kwargs["max_patients"] = args.max_patients
        preprocess.run_preprocess(preprocess.PreprocessConfig(**cfg_kwargs))
        preds_ready = False
        fusion_ready = False
        rho_ready = False
    else:
        print(f"[main] skipping preprocess (found {processed_dir / 'windows_train.npz'})")

    need_lstm = "lstm" in active or (any(s in active for s in downstream) and not preds_ready)
    if need_lstm:
        lstm = load_step("lstm_model")
        lstm_kwargs = {"processed_dir": processed_dir}
        if args.epochs is not None:
            lstm_kwargs["epochs"] = args.epochs
        if args.max_train_samples is not None:
            lstm_kwargs["max_train_samples"] = args.max_train_samples
        lstm.run_lstm(lstm.LSTMConfig(**lstm_kwargs))
        fusion_ready = False
        rho_ready = False
    elif any(s in active for s in downstream):
        print(f"[main] skipping lstm (found {processed_dir / 'lstm_predictions.npz'})")

    need_fusion = "fusion" in active or (
        any(s in active for s in ("aggregate", "optimize")) and not fusion_ready
    )
    if need_fusion:
        fusion = load_step("fusion_metric")
        fusion.run_fusion(fusion.FusionConfig(processed_dir=processed_dir))
        rho_ready = False
    elif any(s in active for s in ("aggregate", "optimize")):
        print(f"[main] skipping fusion (found {processed_dir / 'criticality_scores.npz'})")

    need_aggregate = "aggregate" in active or ("optimize" in active and not rho_ready)
    if need_aggregate:
        agg = load_step("aggregator")
        agg_kwargs: dict = {"processed_dir": processed_dir}
        if args.n_bs is not None:
            agg_kwargs["n_bs"] = args.n_bs
        if args.n_hsp is not None:
            agg_kwargs["n_hsp"] = args.n_hsp
        agg.run_aggregator(agg.AggregatorConfig(**agg_kwargs))
    elif "optimize" in active:
        print(f"[main] skipping aggregator (found {processed_dir / 'rho_wk.npz'})")

    if "optimize" in active:
        opt = load_step("optimizer")
        opt.run_optimizer(opt.OptimizerConfig(processed_dir=processed_dir))
        figs = load_step("auction_figures")
        figs.generate_figures(processed_dir=processed_dir)


if __name__ == "__main__":
    main()
