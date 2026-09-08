# Hybrid 5G Healthcare Resource Allocation and Double-Auction Framework

B.Tech project pipeline that couples **LSTM vital-sign forecasting** with a **game-theoretic double auction**. Predicted patient criticality drives the preference matrix ρ_{wk}; KKT stationarity plus dual sub-gradient clearing then allocate uplink data rates between base stations (sellers) and healthcare service providers (buyers).

**Paper 1 (current):** eICU-CRD Demo, **W = 2** unlabeled buyers (HSP1 / HSP2, random split, seed 42), **K = 3** cells, **dynamic preference** (max-normalized into (0, 1]), **constant reluctance** from one radio snapshot, **r_k^max = 5 Mbps** per BS (no load scaling). Clearing: welfare **10.65**, total rate **15.01 Mbps**, payment **4.37**. Draft: `report/main.tex`. Bid-privacy note (memo only): `report/README_bid_privacy_for_mentor.md`.

Tables for that run live in `data/processed_w2k3/`. Do not treat Emergency/Cardiology labels, 120 Mbps cells, or welfare 22.66 as paper-1 numbers.

---

## 1. Problem

IoMT streams produce heterogeneous vital signs. A 5G radio network must give **more uplink to more critical patients**, without a central planner dictating both clinical priority and spectrum price.

This project:

1. Forecasts HR, SpO2, blood pressure and temperature at **5 / 10 / 15 minute** horizons.
2. Fuses those forecasts into a bounded criticality score c_n \in [0, 1].
3. Assigns users at random to HSP1 / HSP2, associates them to cells, and builds **dynamic preferences** ρ_{wk} (then max-normalizes so the largest link is 1).
4. Clears a two-sided market for uplink slice rates d_{wk} ≈ r_{kw} with frozen ω and a 5 Mbps cap per cell.

---



## 2. Pipeline

```
IoMT CSV
   │
   ▼
01  preprocess     clean, synthesize 1-min trajectories, 30-min windows
   │
   ▼
02  LSTM           multi-horizon vitals + 3-class risk
   │
   ▼
03  fusion         NEWS2-style criticality  c_n ∈ [0, 1]
   │
   ▼
04  aggregator     map users → BS / HSP, build ρ_wk
   │
   ▼
05  optimizer      KKT OPT1 / OPT2 + sub-gradient prices
   │
   ▼
cleared rates, prices, payments, per-patient d_n
```


| Step | Module                    | Role                                              |
| ---- | ------------------------- | ------------------------------------------------- |
| 1    | `src/01_preprocess.py`    | Cleaning, normalization, sliding windows          |
| 2    | `src/02_lstm_model.py`    | Bidirectional LSTM, attention pooling, dual heads |
| 3    | `src/03_fusion_metric.py` | Multi-vital + fall + risk-head fusion             |
| 4    | `src/04_aggregator.py`    | 5G layout, association, \rho_{wk}                 |
| 5    | `src/05_optimizer.py`     | Double-auction market clearing                    |
| —    | `main.py`                 | Orchestrator (`--from-step` / `--to-step`)        |


---



## 3. Dataset

Source: `data/Synthetic_patient-HealthCare-Monitoring_dataset.csv` (Kaggle-style synthetic healthcare monitoring).


| Property       | Value                                                                       |
| -------------- | --------------------------------------------------------------------------- |
| Patients       | 60,000 (one snapshot each)                                                  |
| Vitals         | Heart rate, SpO2, SBP, DBP, temperature                                     |
| Labels         | Predicted disease, per-vital alerts, fall detection                         |
| Diseases       | Asthma, Heart Disease, Hypertension, Diabetes Mellitus, Healthy (~12k each) |
| Falls          | 3,074 / 60,000                                                              |
| Missing values | none                                                                        |


The CSV is **cross-sectional** (no timestamps). Preprocessing synthesizes a physiologically plausible **45-minute, 1 sample/min** AR(1) trajectory per patient, pinned to the recorded snapshot at t = 29 (end of the 30-minute lookback). Disease- and alert-conditioned drift plus fall shocks provide the 5/10/15-minute targets.

**Split (by patient, stratified on snapshot risk):** train 42,000 / val 9,000 / test 9,000.

Window tensor: `X ∈ ℝ^{N × 30 × 6}` (five vitals + fall). Targets: `y_vitals ∈ ℝ^{N × 3 × 5}`, `y_risk ∈ {0,1,2}^{N × 3}`.

---



## 4. Step-by-step methods and results



### 4.1 Preprocess (`01_preprocess.py`)

- Median/mode imputation (unused on this file; kept for robustness).
- Column aliasing for the temperature degree-symbol encoding.
- AR(1) synthesis with NEWS2-inspired clip ranges.
- `StandardScaler` fit on **train patients only**.
- Risk classes from a NEWS2-style vital map (0 normal, 1 moderate, 2 critical).

**Outputs:** `data/processed/windows_{train,val,test}.npz`, `scaler.joblib`, `patient_snapshot.npz`, `patients_*.csv`.

---



### 4.2 LSTM (`02_lstm_model.py`)

Shared encoder: 2-layer **bidirectional LSTM** (hidden 128, dropout 0.25) with tanh attention pooling, then:

- Smooth-L1 head → vitals at three horizons  
- Class-weighted CE head → risk `{0,1,2}`  
- Combined loss: L = L_{\text{vital}} + 0.35 L_{\text{risk}}

Trained 25 epochs, AdamW (10^{-3}, weight decay 10^{-4}), batch 256, grad clip 1.0, ReduceLROnPlateau. Device: **CUDA (RTX 3050)**. Best val loss **0.082** at epoch 25.

#### Test set (9,000 held-out patients)

Overall: **vital MAE 1.67**, **RMSE 2.10**, **risk accuracy 89.6%**, **macro-F1 0.893**. Val is essentially identical (MAE 1.68, acc 89.7%) — no overfitting.


| Horizon | HR MAE | SpO2 MAE | SBP MAE | DBP MAE | Temp MAE | Risk acc | Macro-F1 |
| ------- | ------ | -------- | ------- | ------- | -------- | -------- | -------- |
| 5 min   | 3.15   | 0.43     | 2.66    | 1.82    | 0.048    | 90.0%    | 0.897    |
| 10 min  | 3.32   | 0.43     | 2.77    | 1.91    | 0.049    | 89.1%    | 0.888    |
| 15 min  | 3.22   | 0.43     | 2.83    | 1.90    | 0.047    | 89.7%    | 0.895    |


Checkpoint: `artifacts/lstm_best.pt`. Predictions: `data/processed/lstm_predictions.npz`.

---



### 4.3 Fusion metric (`03_fusion_metric.py`)

Each horizon is scored with NEWS2-aligned piecewise maps, then mixed:


| Channel        | Weight |
| -------------- | ------ |
| SpO2           | 0.26   |
| SBP            | 0.18   |
| Heart rate     | 0.16   |
| Fall           | 0.12   |
| DBP            | 0.10   |
| LSTM risk head | 0.10   |
| Temperature    | 0.08   |


Horizons: **40% peak + 60% recency-weighted mean** (weights 0.50 / 0.30 / 0.20 on 5/10/15 min). Output clipped to [0, 1].

#### Criticality on 60,000 patients


| Statistic        | Value          |
| ---------------- | -------------- |
| Range            | [0.000, 0.721] |
| Mean / median    | 0.198 / 0.191  |
| Std              | 0.148          |
| In unit interval | yes            |


Monotone in snapshot risk and clinically ordered by disease:


| Snapshot risk | Mean c_n | Disease       | Mean c_n |
| ------------- | -------- | ------------- | -------- |
| 0 normal      | 0.004    | Healthy       | 0.007    |
| 1 moderate    | 0.216    | Diabetes      | 0.158    |
| 2 critical    | 0.321    | Hypertension  | 0.200    |
|               |          | Heart disease | 0.272    |
|               |          | Asthma        | 0.354    |


Asthma ranks highest because hypoxemia dominates the fusion weights. Healthy patients sit near zero, so they do not compete for URLLC-style uplink.

---



### 4.4 Aggregator (`04_aggregator.py`)

Paper-1 map (`--n-bs 3 --n-hsp 2` into `data/processed_w2k3`):

- **Layout:** 2 km × 2 km urban micro, **3 base stations**.
- **Association:** max RSRP, 3GPP UMi-style path loss at **3.5 GHz**, P_tx = 30 dBm.
- **HSP assignment:** independent uniform draw to **HSP1** or **HSP2** (seed 42). No disease-to-hospital map.

Link mass C_{wk} = sum of c_n on coverage set B_{wk}. Raw preference is mass-scaled intensity, then **max-normalized** so max ρ = 1.

#### Assignment (74,454 users)

|     | HSP1  | HSP2  |
| --- | ----- | ----- |
| BS1 | 9220  | 9379  |
| BS2 | 13902 | 13962 |
| BS3 | 14053 | 13938 |

#### ρ_{wk} (max = 1)

|     | HSP1  | HSP2  |
| --- | ----- | ----- |
| BS1 | 0.665 | 0.660 |
| BS2 | 0.983 | 1.000 |
| BS3 | 0.993 | 0.990 |

Rows are close because the random split is balanced. Columns differ because BS2/BS3 cover more of the map.

---



### 4.5 Optimizer (`05_optimizer.py`)

Two convex programs share a price \pi_{wk}.

**OPT1 (HSP k, buyer)**

\max_{d\ge 0}\ \sum_w \rho_{wk}\log(1+h_{wk}d_{wk})-\pi_{wk}d_{wk}
\quad\text{s.t.}\quad \sum_w d_{wk}\le D_k

KKT: d_{wk}^\star = \big[\rho_{wk}/(\pi_{wk}+\lambda_k)-1/h_{wk}\big]_+.

**OPT2 (BS w, seller)**

\max_{r\ge 0}\ \sum_k \pi_{wk} r_{kw}-c_w r_{kw}-\tfrac{\beta_w}{2} r_{kw}^2
\quad\text{s.t.}\quad \sum_k r_{kw}\le R_w

KKT: r_{kw}^\star = \big[(\pi_{wk}-c_w-\mu_w)/\beta_w\big]_+.

Duals \lambda_k,\mu_w by bisection. Prices: \pi \leftarrow [\pi + \alpha_t (d-r)]_+ with \alpha_t=\alpha_0/\sqrt{t}.

**Paper-1 market parameters:** R_k = **5 Mbps** on every cell (no load scaling), HSP caps from criticality shares with 1.2× oversubscription (total demand envelope **18 Mbps**), c = 0.03, β = 0.015, SNR scale 10, frozen ω from `reluctance/omega_frozen.npz`.

#### Clearing (226 iterations, gap 9.95×10^{-4})

Midpoint rates x_{wk} in Mbps:

|     | HSP1  | HSP2  |
| --- | ----- | ----- |
| BS1 | 2.886 | 2.114 |
| BS2 | 2.917 | 2.084 |
| BS3 | 2.598 | 2.410 |

| Metric             | Value                                     |
| ------------------ | ----------------------------------------- |
| Total cleared rate | **15.01 Mbps** (cap binds on every cell)  |
| Total payments     | **4.37**                                  |
| Social welfare     | **10.65**                                 |
| Frozen ω (BS1–3)   | 0.390 / 0.562 / 0.559 (rows nearly equal) |

| HSP surplus | Value | BS profit | Value |
| ----------- | ----- | --------- | ----- |
| HSP1        | 4.79  | BS1       | 1.01  |
| HSP2        | 1.86  | BS2       | 1.32  |
|             |       | BS3       | 1.67  |

All base stations earn positive profit. Per-patient uplink is d_n = x_{wk} c_n / C_{wk} (`customer_rates.csv`, not tracked in git).

---



## 5. What this shows

1. **Forecasts are usable.** On eICU-CRD Demo, test vital MAE is 2.28 (mixed units) and risk accuracy is 80.7% from 5 to 15 minutes.
2. **Fusion is calibrated.** c_n stays in [0, 1] and tracks a snapshot-risk label that was not an input to fusion.
3. **Preferences are dynamic and bounded.** ρ_{wk} is LSTM-fused criticality mass on each coverage set, then max-normalized into (0, 1].
4. **The 5 Mbps auction clears.** Frozen ω, binding PRB caps, welfare 10.65, positive surplus on both sides.

---



## 6. How to run

```bash
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
# GPU (Windows, CUDA 12.4):
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Full pipeline (skips stages whose artifacts already exist when you start later):

```bash
.\.venv\Scripts\python main.py
.\.venv\Scripts\python main.py --from-step optimize
```

Individual stages:

```bash
.\.venv\Scripts\python src\01_preprocess.py
.\.venv\Scripts\python src\02_lstm_model.py
.\.venv\Scripts\python src\03_fusion_metric.py
.\.venv\Scripts\python src\04_aggregator.py
.\.venv\Scripts\python src\05_optimizer.py
```

Useful flags: `--max-patients`, `--epochs`, `--n-bs`, `--n-hsp`, `--from-step`, `--to-step`.

Paper-1 market tables (eICU already fused; skip LSTM if `processed_w2k3` exists):

```bash
.\.venv\Scripts\python src\04_aggregator.py --processed-dir data\processed_w2k3 --n-bs 3 --n-hsp 2
.\.venv\Scripts\python src\09_reluctance.py --processed-dir data\processed_w2k3 --out-dir data\processed_w2k3\reluctance --frozen-only
.\.venv\Scripts\python src\05_optimizer.py --processed-dir data\processed_w2k3 --omega-path data\processed_w2k3\reluctance\omega_frozen.npz
.\.venv\Scripts\python src\06_auction_figures.py --processed-dir data\processed_w2k3 --figdir artifacts\w2k3\figures --max-iter 250 --omega-path data\processed_w2k3\reluctance\omega_frozen.npz
.\.venv\Scripts\python src\08_economic_figures.py --processed-dir data\processed_w2k3 --figdir artifacts\w2k3\figures --omega-path data\processed_w2k3\reluctance\omega_frozen.npz
```

---



## 7. Repository layout

```
5g-healthcare-auction/
├── data/processed_w2k3/    paper-1 W=2 K=3 tables (ρ, ω, clearing)
├── src/                    preprocess → LSTM → fusion → aggregate → auction
├── report/                 IEEE draft + mentor notes
├── main.py
└── requirements.txt
```

| Artifact | Contents |
| --- | --- |
| `data/processed_w2k3/rho_wk.csv` | Max-normalized preference |
| `data/processed_w2k3/reluctance/omega_frozen.npz` | Constant ω |
| `data/processed_w2k3/optimizer_summary.json` | Rates, payments, duals, welfare 10.65 |
| `report/main.tex` | Paper-1 draft |
| `report/README_bid_privacy_for_mentor.md` | Why PRE cannot hide bids |


---



## 8. Dependencies

`numpy`, `pandas`, `scikit-learn`, `joblib`, `torch` (CUDA optional). LSTM training used **PyTorch 2.6.0+cu124** on an RTX 3050 laptop GPU.

---



## 9. Reproducibility

Global seed **42** for preprocess splits, trajectory noise, LSTM, customer placement, and price initialization. Patient-level split prevents leakage of the same person into train and test.