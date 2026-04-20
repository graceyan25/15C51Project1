#!/usr/bin/env python3
"""
Compare strategy and baseline efficacy using outputs from run_strategies_and_baselines.py.

Outputs:
- metrics_summary.csv
- relative_metrics.csv
- comparison_table.md
- figures/*.png
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

if "MPLCONFIGDIR" not in os.environ:
    _mpl_dir = Path(tempfile.gettempdir()) / "mplconfig"
    _mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_dir)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "evaluation_outputs"
DEFAULT_OUT_DIR = SCRIPT_DIR / "evaluation_report"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate formatted comparison outputs and plots.")
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help=f"Directory containing Script A outputs. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help=f"Directory for report artifacts. Default: {DEFAULT_OUT_DIR}",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def annualized_return(returns: pd.Series, periods_per_year: int) -> float:
    r = returns.dropna()
    if r.empty:
        return np.nan
    total = (1.0 + r).prod()
    n = len(r)
    return total ** (periods_per_year / n) - 1.0


def annualized_vol(returns: pd.Series, periods_per_year: int) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return np.nan
    return r.std(ddof=1) * np.sqrt(periods_per_year)


def max_drawdown(returns: pd.Series) -> float:
    r = returns.dropna()
    if r.empty:
        return np.nan
    wealth = (1.0 + r).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return dd.min()


def compute_metrics(returns: pd.Series, rf: Optional[pd.Series], periods_per_year: int) -> Dict[str, float]:
    r = returns.dropna()
    if r.empty:
        return {
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "sharpe_rf": np.nan,
            "max_dd": np.nan,
            "calmar": np.nan,
            "win_rate": np.nan,
            "skew": np.nan,
            "ex_kurtosis": np.nan,
            "n_obs": 0,
        }

    ann_ret = annualized_return(r, periods_per_year)
    ann_vol = annualized_vol(r, periods_per_year)
    mdd = max_drawdown(r)

    if rf is None:
        sharpe = np.nan
    else:
        aligned = pd.concat([r, rf], axis=1, join="inner").dropna()
        if len(aligned) < 2:
            sharpe = np.nan
        else:
            ex = aligned.iloc[:, 0] - aligned.iloc[:, 1]
            ex_ann = annualized_return(ex, periods_per_year)
            ex_vol = annualized_vol(ex, periods_per_year)
            sharpe = ex_ann / ex_vol if ex_vol and ex_vol > 0 else np.nan

    calmar = ann_ret / abs(mdd) if pd.notna(mdd) and mdd != 0 else np.nan

    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe_rf": sharpe,
        "max_dd": mdd,
        "calmar": calmar,
        "win_rate": (r > 0).mean(),
        "skew": r.skew(),
        "ex_kurtosis": r.kurt(),
        "n_obs": int(len(r)),
    }


def format_pct(x: float) -> str:
    return "NA" if pd.isna(x) else f"{x:.2%}"


def format_num(x: float) -> str:
    return "NA" if pd.isna(x) else f"{x:.3f}"


def markdown_table(df: pd.DataFrame, columns: List[str]) -> str:
    view = df[columns].copy()
    rows = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in view.iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
    return "\n".join(rows)


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    daily_path = input_dir / "returns_daily.csv"
    monthly_path = input_dir / "returns_monthly.csv"
    manifest_path = input_dir / "series_manifest.csv"

    for p in (daily_path, monthly_path, manifest_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required input file: {p}")

    daily_long = pd.read_csv(daily_path, parse_dates=["date"])
    monthly_long = pd.read_csv(monthly_path, parse_dates=["date"])
    manifest = pd.read_csv(manifest_path)

    require_columns(daily_long, ["date", "series_id", "series_group", "return"], "returns_daily.csv")
    require_columns(monthly_long, ["date", "series_id", "series_group", "return"], "returns_monthly.csv")
    require_columns(
        manifest,
        ["series_id", "series_group", "native_frequency", "cost_assumption", "params_json"],
        "series_manifest.csv",
    )

    daily_wide = daily_long.pivot(index="date", columns="series_id", values="return").sort_index()
    monthly_wide = monthly_long.pivot(index="date", columns="series_id", values="return").sort_index()

    if "t_bill" not in daily_wide.columns or "t_bill" not in monthly_wide.columns:
        raise ValueError("t_bill series is required in both daily and monthly outputs.")

    metrics_rows = []
    for _, row in manifest.iterrows():
        sid = row["series_id"]
        freq = row["native_frequency"]

        if freq == "daily":
            s = daily_wide[sid].dropna() if sid in daily_wide.columns else pd.Series(dtype=float)
            rf = daily_wide["t_bill"].dropna() if sid != "t_bill" else None
            ppy = 252
        elif freq == "monthly":
            s = monthly_wide[sid].dropna() if sid in monthly_wide.columns else pd.Series(dtype=float)
            rf = monthly_wide["t_bill"].dropna() if sid != "t_bill" else None
            ppy = 12
        else:
            raise ValueError(f"Unsupported native_frequency for {sid}: {freq}")

        m = compute_metrics(s, rf, periods_per_year=ppy)
        metrics_rows.append(
            {
                "series_id": sid,
                "series_group": row["series_group"],
                "native_frequency": freq,
                **m,
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_dir / "metrics_summary.csv", index=False)

    strategy_ids = manifest.loc[manifest["series_group"] == "strategy", "series_id"].tolist()
    baseline_ids = manifest.loc[manifest["series_group"] == "baseline", "series_id"].tolist()

    relative_rows = []
    for sid in strategy_ids:
        if sid not in monthly_wide.columns:
            continue
        for bid in baseline_ids:
            if bid not in monthly_wide.columns:
                continue

            aligned = pd.concat([monthly_wide[sid], monthly_wide[bid]], axis=1, join="inner").dropna()
            aligned.columns = ["strategy", "baseline"]
            if aligned.empty:
                continue

            active = aligned["strategy"] - aligned["baseline"]
            ann_excess = annualized_return(active, 12)
            te = annualized_vol(active, 12)
            ir = ann_excess / te if pd.notna(te) and te > 0 else np.nan

            relative_rows.append(
                {
                    "strategy_id": sid,
                    "baseline_id": bid,
                    "ann_excess_return": ann_excess,
                    "tracking_error": te,
                    "information_ratio": ir,
                    "excess_hit_rate": (active > 0).mean(),
                    "n_obs": int(len(active)),
                }
            )

    relative_df = pd.DataFrame(relative_rows)
    relative_df.to_csv(out_dir / "relative_metrics.csv", index=False)

    # -------- Figures --------
    # 1) Cumulative growth of $1 (monthly for all series)
    plt.figure(figsize=(10, 6))
    for sid in monthly_wide.columns:
        s = monthly_wide[sid].dropna()
        if s.empty:
            continue
        wealth = (1.0 + s).cumprod()
        plt.plot(wealth.index, wealth.values, label=sid)
    plt.title("Cumulative Growth of $1 (Monthly)")
    plt.ylabel("Portfolio Value")
    plt.xlabel("Date")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_dir / "cumulative_growth.png", dpi=160)
    plt.close()

    # 2) Drawdown curves
    plt.figure(figsize=(10, 6))
    for sid in monthly_wide.columns:
        s = monthly_wide[sid].dropna()
        if s.empty:
            continue
        wealth = (1.0 + s).cumprod()
        drawdown = wealth / wealth.cummax() - 1.0
        plt.plot(drawdown.index, drawdown.values, label=sid)
    plt.title("Drawdown Curves (Monthly)")
    plt.ylabel("Drawdown")
    plt.xlabel("Date")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_dir / "drawdowns.png", dpi=160)
    plt.close()

    # 3) Risk/return scatter
    plt.figure(figsize=(8, 6))
    color_map = {"strategy": "#1f77b4", "baseline": "#ff7f0e"}
    for _, row in metrics_df.iterrows():
        if pd.isna(row["ann_vol"]) or pd.isna(row["ann_return"]):
            continue
        plt.scatter(
            row["ann_vol"] * 100,
            row["ann_return"] * 100,
            s=80,
            color=color_map.get(row["series_group"], "gray"),
            alpha=0.85,
        )
        plt.annotate(row["series_id"], (row["ann_vol"] * 100, row["ann_return"] * 100), fontsize=8)
    plt.title("Risk / Return Scatter")
    plt.xlabel("Annualized Volatility (%)")
    plt.ylabel("Annualized Return (%)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_dir / "risk_return_scatter.png", dpi=160)
    plt.close()

    # 4) Rolling 12-month Sharpe (monthly, rf-adjusted)
    plt.figure(figsize=(10, 6))
    rf_m = monthly_wide["t_bill"].dropna()
    for sid in monthly_wide.columns:
        if sid == "t_bill":
            continue
        aligned = pd.concat([monthly_wide[sid], rf_m], axis=1, join="inner").dropna()
        if len(aligned) < 13:
            continue
        ex = aligned.iloc[:, 0] - aligned.iloc[:, 1]
        roll_mean = ex.rolling(12).mean() * 12.0
        roll_vol = ex.rolling(12).std() * np.sqrt(12.0)
        roll_sharpe = roll_mean / roll_vol
        plt.plot(roll_sharpe.index, roll_sharpe.values, label=sid)
    plt.title("Rolling 12-Month Sharpe (RF-adjusted)")
    plt.xlabel("Date")
    plt.ylabel("Sharpe")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_dir / "rolling_sharpe_12m.png", dpi=160)
    plt.close()

    # 5) Annualized excess return bars (strategy vs each baseline)
    if not relative_df.empty:
        pivot = relative_df.pivot(index="baseline_id", columns="strategy_id", values="ann_excess_return")
        ax = pivot.plot(kind="bar", figsize=(9, 5))
        ax.set_title("Annualized Excess Return vs Baselines")
        ax.set_ylabel("Annualized Excess Return")
        ax.set_xlabel("Baseline")
        ax.axhline(0, color="black", lw=1)
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(fig_dir / "excess_return_bars.png", dpi=160)
        plt.close()

    # -------- Markdown Summary --------
    ranking = metrics_df.sort_values("sharpe_rf", ascending=False, na_position="last").copy()
    ranking["ann_return"] = ranking["ann_return"].map(format_pct)
    ranking["ann_vol"] = ranking["ann_vol"].map(format_pct)
    ranking["sharpe_rf"] = ranking["sharpe_rf"].map(format_num)
    ranking["max_dd"] = ranking["max_dd"].map(format_pct)

    strategy_metrics = metrics_df[metrics_df["series_group"] == "strategy"].copy()
    best_sharpe_row = strategy_metrics.sort_values("sharpe_rf", ascending=False).head(1)
    best_return_row = strategy_metrics.sort_values("ann_return", ascending=False).head(1)

    lines: List[str] = []
    lines.append("# Strategy vs Baseline Comparison")
    lines.append("")
    lines.append(f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    if not best_sharpe_row.empty:
        r = best_sharpe_row.iloc[0]
        lines.append(
            f"- Best strategy by Sharpe: `{r['series_id']}` (Sharpe {format_num(r['sharpe_rf'])}, Ann Return {format_pct(r['ann_return'])})."
        )
    if not best_return_row.empty:
        r = best_return_row.iloc[0]
        lines.append(
            f"- Best strategy by annualized return: `{r['series_id']}` ({format_pct(r['ann_return'])})."
        )

    lines.append("")
    lines.append("## Full Ranking (by Sharpe)")
    lines.append("")
    lines.append(
        markdown_table(
            ranking,
            ["series_id", "series_group", "native_frequency", "ann_return", "ann_vol", "sharpe_rf", "max_dd", "n_obs"],
        )
    )

    if not relative_df.empty:
        rel_view = relative_df.copy()
        rel_view["ann_excess_return"] = rel_view["ann_excess_return"].map(format_pct)
        rel_view["tracking_error"] = rel_view["tracking_error"].map(format_pct)
        rel_view["information_ratio"] = rel_view["information_ratio"].map(format_num)
        rel_view["excess_hit_rate"] = rel_view["excess_hit_rate"].map(format_pct)

        lines.append("")
        lines.append("## Strategy Excess Performance vs Baselines")
        lines.append("")
        lines.append(
            markdown_table(
                rel_view,
                [
                    "strategy_id",
                    "baseline_id",
                    "ann_excess_return",
                    "tracking_error",
                    "information_ratio",
                    "excess_hit_rate",
                    "n_obs",
                ],
            )
        )

    lines.append("")
    lines.append("## Robustness Caveats")
    lines.append("")
    lines.append("- Results depend on the selected mean-reversion parameter set (fixed in-sample best realistic config).")
    lines.append("- DIA comparisons depend on overlap period with supplied DIA CSV.")
    lines.append("- Momentum is native-monthly while most baselines are native-daily; metrics are annualized for comparability.")
    lines.append("- `dow30_returns.csv` is treated as a price-like total return index matrix, with returns computed via `pct_change`.")

    (out_dir / "comparison_table.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote outputs to: {out_dir}")
    print(" - metrics_summary.csv")
    print(" - relative_metrics.csv")
    print(" - comparison_table.md")
    print(f" - figures/: {', '.join(sorted(p.name for p in fig_dir.glob('*.png')))}")


if __name__ == "__main__":
    main()
