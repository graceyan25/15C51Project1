#!/usr/bin/env python3
"""
Generate strategy and baseline return series for unified comparison.

Outputs:
- returns_daily.csv
- returns_monthly.csv
- series_manifest.csv
- run_metadata.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DIA_CSV = SCRIPT_DIR / "dia.csv"
DEFAULT_OUT_DIR = SCRIPT_DIR / "evaluation_outputs"


MR_DEFAULT_PARAMS = {
    "lookback": 126,
    "entry": 2.5,
    "exit": 0.5,
    "stop_loss": 4.0,
    "top_n": 5,
    "rebal_freq_days": 10,  # 2x/month in existing grid-search code
}


@dataclass
class SeriesSpec:
    series_id: str
    series_group: str
    native_frequency: str
    cost_assumption: str
    params_json: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strategies and baselines and export aligned returns.")
    parser.add_argument(
        "--dia-csv",
        default=str(DEFAULT_DIA_CSV),
        help=(
            "Path to DIA CSV (must include date + adj_close/close). "
            f"Default: {DEFAULT_DIA_CSV}"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help=f"Directory for output artifacts. Default: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument("--start", default="2016-01-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", default="2025-12-31", help="End date (YYYY-MM-DD).")
    parser.add_argument(
        "--mom-tc",
        type=float,
        default=0.001,
        help="Momentum transaction cost per unit turnover (default: 0.001 = 10 bps).",
    )
    parser.add_argument(
        "--mr-tc",
        type=float,
        default=0.001,
        help="Mean-reversion transaction cost per unit turnover (default: 0.001 = 10 bps).",
    )
    return parser.parse_args()


def annualized_pct_to_monthly(rate_annual_pct: pd.Series) -> pd.Series:
    return (1 + rate_annual_pct / 100.0) ** (1.0 / 12.0) - 1.0


def monthly_to_daily_rf(
    monthly_rate: pd.Series,
    trading_calendar: pd.DatetimeIndex,
) -> pd.Series:
    """Infer daily rf so it compounds to each month's monthly rf over trading days in that month."""
    if trading_calendar.empty:
        return pd.Series(dtype=float)

    month_starts = trading_calendar.to_period("M").to_timestamp(how="start")
    per_day = pd.Series(index=trading_calendar, dtype=float)
    grouped = pd.Series(month_starts, index=trading_calendar).groupby(month_starts).groups

    for m_start, month_dates in grouped.items():
        month_dates = pd.DatetimeIndex(month_dates)
        n_days = len(month_dates)
        if n_days == 0:
            continue

        month_rf = monthly_rate.reindex([m_start]).iloc[0]
        if pd.isna(month_rf):
            month_rf = monthly_rate.loc[:m_start].ffill().iloc[-1] if not monthly_rate.loc[:m_start].empty else 0.0
        per_day.loc[month_dates] = (1.0 + month_rf) ** (1.0 / n_days) - 1.0

    return per_day


def load_price_matrix(base_dir: Path, start: str, end: str) -> pd.DataFrame:
    prices = pd.read_csv(base_dir / "dow30_returns.csv", index_col="date", parse_dates=True)
    prices = prices.sort_index().loc[start:end]
    prices = prices[~prices.index.duplicated(keep="first")]
    if prices.empty:
        raise ValueError("Price matrix is empty after date filtering.")
    return prices


def load_tbill_daily(base_dir: Path, trading_calendar: pd.DatetimeIndex) -> pd.Series:
    tbill = pd.read_csv(base_dir / "MonthlyTBill.csv", parse_dates=["observation_date"])
    tbill = tbill.sort_values("observation_date")
    tbill_monthly = annualized_pct_to_monthly(tbill.set_index("observation_date")["GS1M"])
    tbill_daily = monthly_to_daily_rf(tbill_monthly, trading_calendar)
    tbill_daily.name = "t_bill"
    return tbill_daily


def load_dia_daily(path: Path, start: str, end: str) -> pd.Series:
    dia = pd.read_csv(path)
    dia_cols = {c.lower().strip().replace(" ", "_"): c for c in dia.columns}

    if "date" not in dia_cols:
        raise ValueError("DIA CSV must include a date column.")

    price_key = None
    for candidate in ("adj_close", "adjclose", "close"):
        if candidate in dia_cols:
            price_key = candidate
            break

    if price_key is None:
        raise ValueError("DIA CSV must include one of: adj_close, adjclose, close.")

    df = dia[[dia_cols["date"], dia_cols[price_key]]].copy()
    df.columns = ["date", "price"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "price"])
    if df["date"].duplicated().any():
        dup_count = int(df["date"].duplicated().sum())
        raise ValueError(f"DIA CSV contains duplicate dates ({dup_count} duplicates).")
    if not df["date"].is_monotonic_increasing:
        df = df.sort_values("date")
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]

    if df.empty:
        raise ValueError("DIA CSV has no rows in requested date range.")

    dia_ret = df.set_index("date")["price"].pct_change().dropna()
    dia_ret.name = "dia"
    return dia_ret


def build_momentum_monthly(prices_daily: pd.DataFrame, tc: float) -> Tuple[pd.Series, pd.DataFrame]:
    """Replicate existing momentum logic and return net monthly strategy returns + weights."""
    monthly_prices = prices_daily.resample("MS").first()
    momentum_signal = monthly_prices.pct_change(12, fill_method=None).shift(1)

    portfolio = pd.DataFrame(index=monthly_prices.index, columns=monthly_prices.columns, dtype=float)

    for date in momentum_signal.index[12:]:
        signal_row = momentum_signal.loc[date]
        valid_tickers = monthly_prices.columns[monthly_prices.loc[date].notna()]
        signal_row = signal_row[signal_row.index.isin(valid_tickers)]
        top5 = signal_row.nlargest(5).index

        portfolio.loc[date] = 0.0
        if len(top5) > 0:
            portfolio.loc[date, top5] = 1.0 / 5.0

    forward_returns = monthly_prices.pct_change(fill_method=None).shift(-1)
    gross = (portfolio * forward_returns).sum(axis=1)

    turnover = portfolio.diff().abs().sum(axis=1)
    costs = turnover * tc
    net = (gross - costs).dropna()
    net.name = "momentum"

    return net, portfolio


def fast_mr_signals(
    zscore_df: pd.DataFrame,
    top_n: int,
    entry: float,
    exit_th: float,
    stop: float,
    rebal_freq: int,
) -> np.ndarray:
    z = zscore_df.values
    n_days, n_stocks = z.shape
    position = np.zeros((n_days, n_stocks), dtype=float)

    for i in range(1, n_days):
        z_row = z[i]
        pos_prev = position[i - 1].copy()
        valid = ~np.isnan(z_row)

        pos_prev[~valid] = 0

        long_mask = (pos_prev == 1) & valid
        short_mask = (pos_prev == -1) & valid
        pos_prev[long_mask & ((z_row >= exit_th) | (z_row <= -stop))] = 0
        pos_prev[short_mask & ((z_row <= exit_th) | (z_row >= stop))] = 0

        if i % rebal_freq == 0:
            flat = (pos_prev == 0) & valid
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


def build_mean_reversion_daily(prices_daily: pd.DataFrame, tc: float) -> Tuple[pd.Series, np.ndarray]:
    p = MR_DEFAULT_PARAMS
    returns_daily = prices_daily.pct_change(fill_method=None)

    rolling_mean = prices_daily.rolling(window=p["lookback"], min_periods=p["lookback"]).mean()
    rolling_std = prices_daily.rolling(window=p["lookback"], min_periods=p["lookback"]).std()
    zscore = (prices_daily - rolling_mean) / rolling_std

    positions = fast_mr_signals(
        zscore_df=zscore,
        top_n=p["top_n"],
        entry=p["entry"],
        exit_th=p["exit"],
        stop=p["stop_loss"],
        rebal_freq=p["rebal_freq_days"],
    )

    shifted = np.roll(positions, 1, axis=0)
    shifted[0] = 0

    n_active = np.abs(shifted).sum(axis=1)
    n_active = np.where(n_active == 0, np.nan, n_active)

    gross = np.nansum(shifted * returns_daily.values, axis=1) / n_active
    gross = np.nan_to_num(gross, nan=0.0)

    turnover = pd.DataFrame(positions, index=prices_daily.index).diff().abs().sum(axis=1).fillna(0.0)
    costs = turnover.values * tc

    net = gross - costs
    mr_series = pd.Series(net, index=prices_daily.index, name="mean_reversion")

    return mr_series, positions


def build_equal_weight_daily(prices_daily: pd.DataFrame) -> pd.Series:
    r = prices_daily.pct_change(fill_method=None)
    ew = r.mean(axis=1, skipna=True).fillna(0.0)
    ew.name = "ew_dow30"
    return ew


def build_buy_hold_start_daily(prices_daily: pd.DataFrame, rf_daily: pd.Series) -> Tuple[pd.Series, int]:
    dates = prices_daily.index
    start = dates.min()
    start_row = prices_daily.loc[start]
    init_tickers = start_row[start_row.notna()].index.tolist()

    if not init_tickers:
        raise ValueError("No active tickers on start date for buy-and-hold initialization.")

    n_init = len(init_tickers)
    weight0 = 1.0 / n_init

    shares = {t: weight0 / prices_daily.at[start, t] for t in init_tickers}
    last_price = {t: prices_daily.at[start, t] for t in init_tickers}

    cash = 0.0
    total_prev = 1.0
    portfolio_returns = [0.0]

    for date in dates[1:]:
        rf = rf_daily.reindex([date]).fillna(0.0).iloc[0]
        cash *= 1.0 + rf

        stock_value = 0.0
        for t in init_tickers:
            sh = shares[t]
            if sh == 0:
                continue

            px = prices_daily.at[date, t]
            if pd.isna(px):
                cash += sh * last_price[t]
                shares[t] = 0.0
            else:
                last_price[t] = px
                stock_value += sh * px

        total_now = cash + stock_value
        ret = (total_now / total_prev) - 1.0 if total_prev != 0 else 0.0
        portfolio_returns.append(ret)
        total_prev = total_now

    out = pd.Series(portfolio_returns, index=dates, name="buy_hold_start")
    return out, n_init


def daily_to_monthly(daily_returns: pd.Series) -> pd.Series:
    monthly = (1.0 + daily_returns).resample("MS").prod() - 1.0
    return monthly.dropna()


def to_long(df: pd.DataFrame, series_group_map: Dict[str, str]) -> pd.DataFrame:
    stacked = df.stack().rename("return").reset_index()
    stacked.columns = ["date", "series_id", "return"]
    stacked = stacked.dropna(subset=["return"])
    stacked["series_group"] = stacked["series_id"].map(series_group_map)
    return stacked[["date", "series_id", "series_group", "return"]]


def main() -> None:
    args = parse_args()

    base_dir = SCRIPT_DIR
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prices = load_price_matrix(base_dir=base_dir, start=args.start, end=args.end)
    calendar = prices.index

    tbill_daily = load_tbill_daily(base_dir=base_dir, trading_calendar=calendar)

    dia_path = Path(args.dia_csv)
    if not dia_path.exists():
        raise FileNotFoundError(
            f"DIA CSV not found: {dia_path}. "
            f"Place a DIA file at {DEFAULT_DIA_CSV} or pass --dia-csv explicitly."
        )

    dia_daily_raw = load_dia_daily(dia_path, start=args.start, end=args.end)
    overlap = dia_daily_raw.index.intersection(calendar)
    if overlap.empty:
        raise ValueError("No overlapping dates between DIA CSV and Dow30 data.")
    dia_daily = dia_daily_raw.reindex(calendar).dropna()

    momentum_monthly, _ = build_momentum_monthly(prices_daily=prices, tc=args.mom_tc)
    momentum_daily_sparse = momentum_monthly.copy()
    momentum_daily_sparse.name = "momentum"

    mr_daily, _ = build_mean_reversion_daily(prices_daily=prices, tc=args.mr_tc)
    ew_daily = build_equal_weight_daily(prices_daily=prices)
    buy_hold_daily, buy_hold_n = build_buy_hold_start_daily(prices_daily=prices, rf_daily=tbill_daily)

    # Daily panel (native frequency rows; momentum remains monthly-sparse by design)
    daily_wide = pd.concat(
        [
            momentum_daily_sparse,
            mr_daily,
            ew_daily,
            tbill_daily,
            dia_daily,
            buy_hold_daily,
        ],
        sort=False,
        axis=1,
    ).sort_index()

    # Monthly panel for all series (momentum uses native monthly; others compounded from daily)
    monthly_wide = pd.DataFrame(index=prices.resample("MS").first().index)
    monthly_wide["momentum"] = momentum_monthly
    monthly_wide["mean_reversion"] = daily_to_monthly(mr_daily)
    monthly_wide["ew_dow30"] = daily_to_monthly(ew_daily)
    monthly_wide["t_bill"] = daily_to_monthly(tbill_daily)
    monthly_wide["dia"] = daily_to_monthly(dia_daily)
    monthly_wide["buy_hold_start"] = daily_to_monthly(buy_hold_daily)

    series_group = {
        "momentum": "strategy",
        "mean_reversion": "strategy",
        "ew_dow30": "baseline",
        "t_bill": "baseline",
        "dia": "baseline",
        "buy_hold_start": "baseline",
    }

    daily_long = to_long(daily_wide, series_group)
    monthly_long = to_long(monthly_wide, series_group)

    daily_long.to_csv(out_dir / "returns_daily.csv", index=False)
    monthly_long.to_csv(out_dir / "returns_monthly.csv", index=False)

    manifest_rows = [
        SeriesSpec(
            series_id="momentum",
            series_group="strategy",
            native_frequency="monthly",
            cost_assumption=f"turnover_cost_per_unit={args.mom_tc:.6f}",
            params_json=json.dumps(
                {
                    "signal": "12m_momentum_shift_1m",
                    "selection": "top_5_equal_weight",
                    "rebalance": "monthly",
                },
                sort_keys=True,
            ),
        ),
        SeriesSpec(
            series_id="mean_reversion",
            series_group="strategy",
            native_frequency="daily",
            cost_assumption=f"turnover_cost_per_unit={args.mr_tc:.6f}",
            params_json=json.dumps(MR_DEFAULT_PARAMS, sort_keys=True),
        ),
        SeriesSpec(
            series_id="ew_dow30",
            series_group="baseline",
            native_frequency="daily",
            cost_assumption="none",
            params_json=json.dumps({"definition": "daily_equal_weight_active_constituents"}, sort_keys=True),
        ),
        SeriesSpec(
            series_id="t_bill",
            series_group="baseline",
            native_frequency="daily",
            cost_assumption="none",
            params_json=json.dumps({"source": "MonthlyTBill_GS1M", "conversion": "monthly_to_daily_compound"}, sort_keys=True),
        ),
        SeriesSpec(
            series_id="dia",
            series_group="baseline",
            native_frequency="daily",
            cost_assumption="none",
            params_json=json.dumps({"source": str(dia_path)}, sort_keys=True),
        ),
        SeriesSpec(
            series_id="buy_hold_start",
            series_group="baseline",
            native_frequency="daily",
            cost_assumption="none",
            params_json=json.dumps(
                {
                    "definition": "start_of_period_equal_weight_no_rebalance",
                    "start_constituent_count": buy_hold_n,
                    "missing_handling": "liquidate_to_t_bill",
                },
                sort_keys=True,
            ),
        ),
    ]

    manifest_df = pd.DataFrame([m.__dict__ for m in manifest_rows])
    manifest_df.to_csv(out_dir / "series_manifest.csv", index=False)

    metadata = {
        "date_range": {
            "start": str(prices.index.min().date()),
            "end": str(prices.index.max().date()),
        },
        "inputs": {
            "dow30_matrix": str(base_dir / "dow30_returns.csv"),
            "tbill": str(base_dir / "MonthlyTBill.csv"),
            "dia_csv": str(dia_path),
        },
        "settings": {
            "momentum_tc": args.mom_tc,
            "mean_reversion_tc": args.mr_tc,
            "mean_reversion_params": MR_DEFAULT_PARAMS,
            "buy_hold_definition": "start_of_period_equal_weight_no_rebalance_liquidate_to_tbill",
        },
        "row_counts": {
            "returns_daily": int(len(daily_long)),
            "returns_monthly": int(len(monthly_long)),
            "series_manifest": int(len(manifest_df)),
        },
        "sanity": {
            "dia_overlap_days": int(len(overlap)),
            "momentum_non_null_months": int(momentum_monthly.notna().sum()),
            "mean_reversion_non_zero_days": int((mr_daily != 0).sum()),
            "buy_hold_start_constituents": int(buy_hold_n),
        },
    }

    with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote outputs to: {out_dir}")
    print(f" - returns_daily.csv ({len(daily_long)} rows)")
    print(f" - returns_monthly.csv ({len(monthly_long)} rows)")
    print(f" - series_manifest.csv ({len(manifest_df)} rows)")
    print(" - run_metadata.json")


if __name__ == "__main__":
    main()
