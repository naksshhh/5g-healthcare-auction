# Paper (LaTeX)

IEEE-style draft: *Efficient Resource Allocation and Payment Scheme for 5G-based Healthcare Networks*.

**Paper 1 claims:** dynamic preference (LSTM + NEWS2 fusion + max-normalized \(\rho_{wk}\)); **constant** reluctance from one radio snapshot; two unlabeled buyers (HSP1, HSP2) split at random; \(r_k^{\max}=5\) Mbps per cell. GBDT / stressed-cell plots in `current/` are leftover experiments, not paper-1 claims.

`main.tex` sets `\graphicspath{{current/}}`. Figures used in the draft:

**Preference / LSTM** (`report/current/`):
- `fig_lstm_training.png`
- `fig_lstm_mae.png`
- `fig6_preference_criticality.png`
- `fig8_preference_vs_payment.png`

**Auction (Figs. 2–7, frozen \(\omega\), 5 Mbps):**
- `fig2_convergence.png`
- `fig3_demand_response_gap.png`
- `fig3_demand_response_gap_all.png`
- `fig4_bs_bids.png`
- `fig5_hsp_bids.png`
- `fig7_payments_surplus.png`
- `fig_omega_const.png`

Older 4×3 / MIMIC / stress–GBDT plots: `report/archive/` or unused files still sitting in `current/`.

Mentor notes (not in the IEEE draft):
- `README_bid_privacy_for_mentor.md` — why PRE cannot hide bids; SOTA ranking
- `README_dynamic_reluctance_for_mentor.md` — **not** paper 1 (time-varying \(\omega\))

```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Requires `IEEEtran.cls` (MiKTeX / TeX Live: package `ieeetran`).
