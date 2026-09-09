# Data

Raw bedside CSVs are **not** stored in git.

1. Run `python src/01_preprocess.py`. It fetches the open-access [eICU-CRD Demo v2.0.1](https://physionet.org/content/eicu-crd-demo/2.0.1/) into `eicu/` (ignored).
2. Optional: place the full credentialed eICU-CRD files in the same folder.

Tracked here:

- `processed_w2k3/` — paper-1 2 HSP × 3 BS preference, frozen ω, and clearing tables.
