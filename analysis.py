import pandas as pd
import numpy as np
from scipy import stats

# Load data
daily_returns = pd.read_csv("dow30_returns.csv", index_col="date", parse_dates=True)
monthly_returns = daily_returns.resample("MS").first()
portfolio = pd.read_csv("portfolio.csv", index_col=0, parse_dates=True)
portfolio = portfolio.astype(float)

# Forward returns: return earned by holding portfolio built at date t
forward_returns = monthly_returns.pct_change().shift(-1)

# Gross strategy monthly returns
strategy_returns = (portfolio * forward_returns).sum(axis=1).dropna()

MONTHS_PER_YEAR = 12
COST_PER_UNIT = 0.001  # 10 bps per unit of turnover; adjust as needed

# Load T-bill rates (annualized %) and convert to monthly decimal rates
tbill = pd.read_csv("MonthlyTBill.csv", index_col="observation_date", parse_dates=True)
tbill_monthly = (1 + tbill["GS1M"] / 100) ** (1 / MONTHS_PER_YEAR) - 1

# Turnover_t = sum_i |w_{i,t} - w_{i,t-1}|
turnover_series = portfolio.diff().abs().sum(axis=1)

# Cost_t = Turnover_t * cost_per_unit
cost_series = turnover_series * COST_PER_UNIT

# Net returns after transaction costs (align index)
net_returns = (strategy_returns - cost_series.reindex(strategy_returns.index).fillna(0))


# ── Performance ──────────────────────────────────────────────────────────────

def annualized_return(returns):
    total = (1 + returns).prod()
    n_years = len(returns) / MONTHS_PER_YEAR
    return total ** (1 / n_years) - 1

def annualized_volatility(returns):
    return returns.std() * np.sqrt(MONTHS_PER_YEAR)

def sharpe_ratio(returns):
    rf = tbill_monthly.reindex(returns.index).ffill()
    excess = returns - rf
    ann_excess = (1 + excess).prod() ** (MONTHS_PER_YEAR / len(excess)) - 1
    ann_vol = annualized_volatility(excess)
    return ann_excess / ann_vol if ann_vol != 0 else np.nan

def max_drawdown(returns):
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    return drawdown.min()

def calmar_ratio(returns):
    ann_ret = annualized_return(returns)
    mdd = max_drawdown(returns)
    return ann_ret / abs(mdd) if mdd != 0 else np.nan


# ── Distributional ────────────────────────────────────────────────────────────

def skewness(returns):
    return stats.skew(returns.dropna())

def excess_kurtosis(returns):
    return stats.kurtosis(returns.dropna())  # Fisher definition: normal = 0

def autocorrelation(returns, lag=1):
    return returns.autocorr(lag=lag)


# ── Trade-level ───────────────────────────────────────────────────────────────

def win_rate(returns):
    return (returns > 0).mean()

def avg_win_loss(returns):
    wins = returns[returns > 0].mean()
    losses = returns[returns < 0].mean()
    return wins, losses

def avg_holding_period(portfolio):
    """Average consecutive months a stock is held."""
    streaks = []
    for col in portfolio.columns:
        held = (portfolio[col] > 0).astype(int)
        streak = 0
        for v in held:
            if v == 1:
                streak += 1
            elif streak > 0:
                streaks.append(streak)
                streak = 0
        if streak > 0:
            streaks.append(streak)
    return np.mean(streaks) if streaks else np.nan


# ── Report ────────────────────────────────────────────────────────────────────

def print_block(label, returns):
    print(f"\n{label}")
    print(f"  Annualized Return  : {annualized_return(returns):.2%}")
    print(f"  Annualized Vol     : {annualized_volatility(returns):.2%}")
    print(f"  Sharpe Ratio       : {sharpe_ratio(returns):.2f}")
    print(f"  Max Drawdown       : {abs(max_drawdown(returns)):.2%}")
    print(f"  Calmar Ratio       : {calmar_ratio(returns):.2f}")

avg_turnover = turnover_series.reindex(strategy_returns.index).mean()
avg_cost     = cost_series.reindex(strategy_returns.index).mean()

print("=" * 40)
print_block("PERFORMANCE (GROSS)", strategy_returns)
print_block("PERFORMANCE (NET)", net_returns)

print("\nDISTRIBUTIONAL PROPERTIES (NET)")
print(f"  Skewness           : {skewness(net_returns):.3f}")
print(f"  Excess Kurtosis    : {excess_kurtosis(net_returns):.3f}")
print(f"  Autocorrelation(1) : {autocorrelation(net_returns):.3f}")

print("\nTRADE-LEVEL STATS")
print(f"  Win Rate           : {win_rate(net_returns):.2%}")
avg_win, avg_loss = avg_win_loss(net_returns)
print(f"  Avg Win            : {avg_win:.2%}")
print(f"  Avg Loss           : {avg_loss:.2%}")
print(f"  Avg Turnover/Month : {avg_turnover:.2%}")
print(f"  Avg Cost/Month     : {avg_cost:.4%}  (cost_per_unit={COST_PER_UNIT:.4f})")
print(f"  Avg Holding Period : {avg_holding_period(portfolio):.1f} months")
print("=" * 40)
