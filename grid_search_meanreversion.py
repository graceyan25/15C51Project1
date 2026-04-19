import pandas as pd
import numpy as np
import itertools
from datetime import datetime

# ── LOAD DATA ─────────────────────────────────────────────────
prices  = pd.read_csv("dow30_returns.csv", index_col=0, parse_dates=True)
prices  = prices.sort_index()["2016-01-01":"2025-12-31"]
returns = prices.pct_change()

print(f"Data: {prices.shape[0]} days x {prices.shape[1]} tickers")

# ── PARAMETER GRID ────────────────────────────────────────────
LOOKBACKS         = [20, 60, 126, 252]
ENTRY_THRESHOLDS  = [1.0, 1.5, 2.0, 2.5]
EXIT_THRESHOLDS   = [0.0, 0.5]
STOP_LOSSES       = [3.0, 4.0]
TOP_NS            = [3, 5, 10]
REBAL_FREQS       = [1, 10, 21]       # 1=daily, 10=twice/month, 21=monthly
TRANSACTION_COSTS = [0.0, 0.001, 0.005]  # 0%=no cost, 0.1%=realistic, 0.5%=high

total = (len(LOOKBACKS) * len(ENTRY_THRESHOLDS) * len(EXIT_THRESHOLDS) *
         len(STOP_LOSSES) * len(TOP_NS) * len(REBAL_FREQS) * len(TRANSACTION_COSTS))
print(f"Total combinations: {total}\n")


# ── SIGNAL FUNCTION ───────────────────────────────────────────
def fast_signals(zscore_df, top_n, entry, exit_th, stop, rebal_freq):
    z              = zscore_df.values
    n_days, n_stocks = z.shape
    position       = np.zeros((n_days, n_stocks))

    for i in range(1, n_days):
        z_row    = z[i]
        pos_prev = position[i - 1].copy()
        valid    = ~np.isnan(z_row)

        # Always: close positions for stocks that left the index
        pos_prev[~valid] = 0

        # Always: apply exit and stop-loss rules
        long_mask  = (pos_prev ==  1) & valid
        short_mask = (pos_prev == -1) & valid
        pos_prev[long_mask  & ((z_row >= exit_th) | (z_row <= -stop))] = 0
        pos_prev[short_mask & ((z_row <= exit_th) | (z_row >=  stop))] = 0

        # Only on rebalancing days: open new positions
        if i % rebal_freq == 0:
            flat   = (pos_prev == 0) & valid
            z_flat = np.where(flat, z_row, np.nan)

            long_z = np.where(z_flat < -entry, z_flat, np.nan)
            if not np.all(np.isnan(long_z)):
                long_idx = np.argsort(np.nan_to_num(long_z, nan=np.inf))[:top_n]
                for idx in long_idx:
                    if not np.isnan(long_z[idx]):
                        pos_prev[idx] = 1

            short_z = np.where(z_flat > entry, z_flat, np.nan)
            if not np.all(np.isnan(short_z)):
                short_idx = np.argsort(np.nan_to_num(short_z, nan=-np.inf))[-top_n:]
                for idx in short_idx:
                    if not np.isnan(short_z[idx]):
                        pos_prev[idx] = -1

        position[i] = pos_prev

    return position


# ── METRICS FUNCTION ──────────────────────────────────────────
def compute_metrics(position, ret_values, tc):
    shifted    = np.roll(position, 1, axis=0)
    shifted[0] = 0
    n_active   = np.abs(shifted).sum(axis=1)
    n_active   = np.where(n_active == 0, np.nan, n_active)

    strat_ret  = np.nansum(shifted * ret_values, axis=1) / n_active
    costs      = pd.DataFrame(position).diff().abs().sum(axis=1).values * tc
    strat_net  = np.nan_to_num(strat_ret - costs)

    r = strat_net[strat_net != 0]
    if len(r) < 50:
        return None

    ann_ret  = r.mean() * 252
    ann_vol  = r.std()  * np.sqrt(252)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum      = np.cumprod(1 + r)
    max_dd   = (cum / np.maximum.accumulate(cum) - 1).min()
    win_rate = (r > 0).mean()
    n_trades = int((np.abs(np.diff(position, axis=0)).sum(axis=1) > 0).sum())

    return dict(ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe,
                max_dd=max_dd, win_rate=win_rate, n_trades=n_trades)


# ── PRECOMPUTE Z-SCORES ───────────────────────────────────────
zscore_cache = {}
for lb in LOOKBACKS:
    rm = prices.rolling(window=lb, min_periods=lb).mean()
    rs = prices.rolling(window=lb, min_periods=lb).std()
    zscore_cache[lb] = (prices - rm) / rs
    print(f"Z-scores cached for lookback={lb}d")

# Cache positions per (lb, entry, exit, stop, top_n, rebal) — reuse across tc values
print(f"\nPre-computing positions (shared across transaction cost levels)...")
position_cache = {}
param_combos = list(itertools.product(
    LOOKBACKS, ENTRY_THRESHOLDS, EXIT_THRESHOLDS, STOP_LOSSES, TOP_NS, REBAL_FREQS))

for i, (lb, entry, exit_th, stop, top_n, rebal) in enumerate(param_combos):
    key = (lb, entry, exit_th, stop, top_n, rebal)
    position_cache[key] = fast_signals(zscore_cache[lb], top_n, entry, exit_th, stop, rebal)
    if (i + 1) % 48 == 0:
        print(f"  {i+1}/{len(param_combos)} positions computed...")

ret_values  = returns.values
rebal_label = {1: "Daily", 10: "2x/Month", 21: "Monthly"}
tc_label    = {0.0: "0% (no cost)", 0.001: "0.1% (realistic)", 0.005: "0.5% (high)"}

print(f"\nComputing metrics for all {total} combinations...")
start   = datetime.now()
results = []

for tc in TRANSACTION_COSTS:
    for (lb, entry, exit_th, stop, top_n, rebal), position in position_cache.items():
        m = compute_metrics(position, ret_values, tc)
        if m:
            results.append({
                "lookback":     lb,
                "entry":        entry,
                "exit":         exit_th,
                "stop_loss":    stop,
                "top_n":        top_n,
                "rebal":        rebal_label[rebal],
                "tc":           tc_label[tc],
                **m
            })

elapsed = (datetime.now() - start).seconds
print(f"Done in {elapsed}s")

# ── RESULTS ───────────────────────────────────────────────────
df = pd.DataFrame(results).sort_values("sharpe", ascending=False).reset_index(drop=True)

def fmt(df_in):
    d = df_in.copy()
    d["ann_ret"]  = (d["ann_ret"]  * 100).round(2).astype(str) + "%"
    d["ann_vol"]  = (d["ann_vol"]  * 100).round(2).astype(str) + "%"
    d["sharpe"]   =  d["sharpe"].round(3)
    d["max_dd"]   = (d["max_dd"]   * 100).round(2).astype(str) + "%"
    d["win_rate"] = (d["win_rate"] * 100).round(1).astype(str) + "%"
    d.columns     = ["Lookback", "Entry", "Exit", "StopLoss", "TopN",
                      "Rebal", "TxCost", "Ann.Ret", "Ann.Vol",
                      "Sharpe", "MaxDD", "WinRate", "Trades"]
    return d

print(f"\n{'='*120}")
print("  TOP 20 COMBINATIONS — sorted by Sharpe Ratio")
print(f"{'='*120}")
print(fmt(df.head(20)).to_string(index=True))

# Summary by transaction cost
print(f"\n{'='*70}")
print("  AVERAGE SHARPE BY TRANSACTION COST")
print(f"{'='*70}")
tc_summary = df.groupby("tc")["sharpe"].agg(["mean","max","min"]).round(3)
tc_summary.columns = ["Avg Sharpe", "Best Sharpe", "Worst Sharpe"]
print(tc_summary.to_string())

# Summary by rebalancing frequency
print(f"\n{'='*70}")
print("  AVERAGE SHARPE BY REBALANCING FREQUENCY")
print(f"{'='*70}")
rebal_summary = df.groupby("rebal")["sharpe"].agg(["mean","max","min"]).round(3)
rebal_summary.columns = ["Avg Sharpe", "Best Sharpe", "Worst Sharpe"]
print(rebal_summary.to_string())

# Summary by lookback
print(f"\n{'='*70}")
print("  AVERAGE SHARPE BY LOOKBACK WINDOW")
print(f"{'='*70}")
lb_summary = df.groupby("lookback")["sharpe"].agg(["mean","max","min"]).round(3)
lb_summary.columns = ["Avg Sharpe", "Best Sharpe", "Worst Sharpe"]
print(lb_summary.to_string())

# Save
df.to_csv("grid_search_meanreversion_results.csv", index=False)
print(f"\nFull results saved -> grid_search_meanreversion_results.csv")
print(f"Total time: {(datetime.now() - start).seconds}s")
