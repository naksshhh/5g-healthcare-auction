# Preference versus patient split

The prediction market uses **all 74,454 eICU windows** (1,821 stays).
`preference_vs_split.csv` applies the example ratios (5:4:3, 7:2:1, ...) to that full set.
`preference_vs_share.csv` gives HSP1 from 5% to 90% of every user.

N=10 files (`*_N10_toy*`) are a classroom example only.

Columns `rho_BS*_HSP*` are max-normalized preference.
`mass_HSP*` is criticality mass (tracks headcount on the full cohort).
