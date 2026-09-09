# Final result — 3 HSPs × 2 base stations

Topology for this folder: **W = 3 buyers (HSP1, HSP2, HSP3)** and **K = 2 sellers (BS1, BS2)**.
Capacity is **5 Mbps per cell**, reluctance is the frozen one-shot radio table, preference is max-normalized.

Paper-1 files under `data/processed_w2k3/` and `report/` are unchanged (that run is still 2 HSPs × 3 BSs).

## How patients are allocated

Each eICU vital window is one uplink user. Users are placed uniformly on a 2 km × 2 km map (seed 42), associated to the stronger of two cells by max RSRP, and subscribed to HSP1/HSP2/HSP3 by an independent uniform draw (seed 42+7). No disease-to-hospital map.

There are **74454** uplink users (eICU Demo vital windows). Unique ICU stays: **1821**.

### Users per hospital (random 3-way split, seed 42)

| HSP | Users | Share |
| --- | ---: | ---: |
| HSP1 | 24855 | 33.4% |
| HSP2 | 24701 | 33.2% |
| HSP3 | 24898 | 33.4% |

### Users on each (BS, HSP) link

|  | HSP1 | HSP2 | HSP3 | BS total |
| --- | ---: | ---: | ---: | ---: |
| BS1 | 12402 | 12429 | 12486 | 37317 |
| BS2 | 12453 | 12272 | 12412 | 37137 |

A user on the left half of the map usually hears **BS1** more strongly; the right half hears **BS2**. HSP membership is independent of location, so each hospital’s users split across the two cells in roughly the same ratio.

Map (1.6k-user sample): `preference_sweep/fig_allocation_map.png`.

## Full-market preference and clearing

Preference matrix (`market/rho_wk.csv`):

```
,HSP1,HSP2,HSP3
BS1,1.0,0.9941437308528404,0.9983669559810999
BS2,0.986019651267734,0.9910571435523146,0.9848716911490841
```

Frozen reluctance (`market/omega_wk.csv`):

```
,HSP1,HSP2,HSP3
BS1,0.47064554240873346,0.46992468487795774,0.47029294697365487
BS2,0.6140830686839722,0.6144746265679645,0.6145382621654822
```

| Metric | Value |
| --- | --- |
| Welfare | 9.008 |
| Total rate (Mbps) | 10.009 |
| Total payment | 4.057 |
| Gap | 9.945e-04 |
| Converged | True |

Auction traces: `figures/fig2_convergence.png` … `fig8_preference_vs_payment.png`.

## Preference vs how many patients each HSP has

The **real market uses all 74454 windows** (1821 ICU stays).
"10 patients as 5-4-3 or 7-2-1" was only an example of *shares*. Those ratios are now applied to the full cohort (5:4:3 -> about 31k / 25k / 19k users).

Preference rho = (C / mean C) * (1 + mean c), then divided by the largest of the six links.
On 74k users the law of large numbers applies: more patients on an HSP raises that HSP's mass and rho smoothly.

Selected full-cohort ratios:

- **1:1:1** -> 24818/24818/24818 users: mean rho = HSP1 0.988, HSP2 0.991, HSP3 0.993
- **4:3:3** -> 29782/22336/22336 users: mean rho = HSP1 0.994, HSP2 0.749, HSP3 0.749
- **5:4:3** -> 31023/24818/18613 users: mean rho = HSP1 0.994, HSP2 0.779, HSP3 0.583
- **7:2:1** -> 52118/14891/7445 users: mean rho = HSP1 0.992, HSP2 0.282, HSP3 0.144
- **8:1:1** -> 59563/7446/7445 users: mean rho = HSP1 0.992, HSP2 0.126, HSP3 0.124

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
