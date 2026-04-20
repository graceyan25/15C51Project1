# Strategy vs Baseline Comparison

Generated: 2026-04-19T20:21:25

- Best strategy by Sharpe: `momentum` (Sharpe 0.478, Ann Return 10.67%).
- Best strategy by annualized return: `momentum` (10.67%).

## Full Ranking (by Sharpe)

| series_id | series_group | native_frequency | ann_return | ann_vol | sharpe_rf | max_dd | n_obs |
|---|---|---|---|---|---|---|---|
| buy_hold_start | baseline | daily | 13.16% | 16.03% | 0.671 | -33.33% | 2514 |
| dia | baseline | daily | 13.15% | 17.57% | 0.612 | -36.70% | 2513 |
| ew_dow30 | baseline | daily | 12.62% | 17.25% | 0.593 | -34.95% | 2514 |
| momentum | strategy | monthly | 10.67% | 17.43% | 0.478 | -21.68% | 120 |
| mean_reversion | strategy | daily | 3.08% | 21.46% | 0.042 | -37.48% | 2514 |
| t_bill | baseline | daily | 2.17% | 0.12% | NA | 0.00% | 2514 |

## Strategy Excess Performance vs Baselines

| strategy_id | baseline_id | ann_excess_return | tracking_error | information_ratio | excess_hit_rate | n_obs |
|---|---|---|---|---|---|---|
| momentum | ew_dow30 | -1.87% | 10.74% | -0.174 | 49.17% | 120 |
| momentum | t_bill | 8.34% | 17.44% | 0.478 | 50.83% | 120 |
| momentum | dia | -2.33% | 10.33% | -0.226 | 48.33% | 120 |
| momentum | buy_hold_start | -2.10% | 10.26% | -0.205 | 48.33% | 120 |
| mean_reversion | ew_dow30 | -9.67% | 19.36% | -0.499 | 42.50% | 120 |
| mean_reversion | t_bill | 0.90% | 18.79% | 0.048 | 46.67% | 120 |
| mean_reversion | dia | -10.26% | 20.08% | -0.511 | 39.17% | 120 |
| mean_reversion | buy_hold_start | -9.97% | 19.62% | -0.509 | 40.00% | 120 |

## Robustness Caveats

- Results depend on the selected mean-reversion parameter set (fixed in-sample best realistic config).
- DIA comparisons depend on overlap period with supplied DIA CSV.
- Momentum is native-monthly while most baselines are native-daily; metrics are annualized for comparability.
- `dow30_returns.csv` is treated as a price-like total return index matrix, with returns computed via `pct_change`.