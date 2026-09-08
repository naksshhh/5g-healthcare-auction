"""5G healthcare resource allocation and double-auction pipeline."""

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent

# Numbered step files (01_preprocess.py, ...) are not valid Python identifiers.
_STEP_FILES = {
    "preprocess": "01_preprocess.py",
    "lstm_model": "02_lstm_model.py",
    "fusion_metric": "03_fusion_metric.py",
    "aggregator": "04_aggregator.py",
    "optimizer": "05_optimizer.py",
    "auction_figures": "06_auction_figures.py",
    "mimic_preprocess": "07_mimic_preprocess.py",
    "economic_figures": "08_economic_figures.py",
    "reluctance": "09_reluctance.py",
    "reluctance_figures": "10_reluctance_figures.py",
}


def load_step(name: str):
    """Import a numbered pipeline module by logical name, e.g. ``load_step("preprocess")``."""
    filename = _STEP_FILES.get(name, name)
    path = _SRC_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Pipeline step not found: {path}")
    mod_name = f"src_{path.stem}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module
