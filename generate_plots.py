import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── LOAD ──────────────────────────────────────────────────────
df = pd.read_csv("/Users/corentinvenet/Desktop/MIT/courses/spring26/fin tech/proj1/grid_search_meanreversion_results.csv")

tc_order     = ["0% (no cost)", "0.1% (realistic)", "0.5% (high)"]
rebal_order  = ["Daily", "2x/Month", "Monthly"]
tc_colors    = {"0% (no cost)": "#2ecc71", "0.1% (realistic)": "#3498db", "0.5% (high)": "#e74c3c"}
rebal_colors = {"Daily": "#e74c3c", "2x/Month": "#f39c12", "Monthly": "#2ecc71"}
lb_colors    = {20: "#e74c3c", 60: "#f39c12", 126: "#3498db", 252: "#2ecc71"}
entry_colors = {1.0: "#e74c3c", 1.5: "#f39c12", 2.0: "#3498db", 2.5: "#2ecc71"}

OUT = "/Users/corentinvenet/Desktop/MIT/courses/spring26/fin tech/proj1/"

# ── PLOT 1 — Sharpe distribution by transaction cost ──────────
fig, ax = plt.subplots(figsize=(7, 4.5))
data_tc = [df[df["tc"] == tc]["sharpe"].values for tc in tc_order]
bp = ax.boxplot(data_tc, patch_artist=True, notch=False,
                medianprops=dict(color="black", lw=2))
for patch, tc in zip(bp["boxes"], tc_order):
    patch.set_facecolor(tc_colors[tc])
    patch.set_alpha(0.75)
ax.set_xticklabels(["0%\n(no cost)", "0.1%\n(realistic)", "0.5%\n(high)"], fontsize=10)
ax.axhline(0, color="black", lw=1, linestyle="--", alpha=0.5)
ax.set_title("Sharpe Ratio Distribution by Transaction Cost", fontsize=12, fontweight="bold")
ax.set_ylabel("Sharpe Ratio")
ax.set_xlabel("Transaction Cost (one-way)")
ax.set_ylim(-10, 1.5)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT + "plot1_sharpe_by_tc.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot 1 saved")

# ── PLOT 2 — % positive Sharpe by transaction cost ───────────
fig, ax = plt.subplots(figsize=(7, 4.5))
pct_pos = df.groupby("tc")["sharpe"].apply(lambda x: (x > 0).mean() * 100).reindex(tc_order)
bars = ax.bar(range(len(tc_order)), pct_pos.values,
              color=[tc_colors[tc] for tc in tc_order], alpha=0.8, edgecolor="white", width=0.5)
for bar, val in zip(bars, pct_pos.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
ax.set_xticks(range(len(tc_order)))
ax.set_xticklabels(["0%\n(no cost)", "0.1%\n(realistic)", "0.5%\n(high)"], fontsize=10)
ax.set_title("Percentage of Parameter Combinations\nwith Positive Sharpe Ratio", fontsize=12, fontweight="bold")
ax.set_ylabel("% of Combinations with Sharpe > 0")
ax.set_xlabel("Transaction Cost (one-way)")
ax.set_ylim(0, 85)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT + "plot2_pct_positive_sharpe.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot 2 saved")

# ── PLOT 3 — Number of trades vs Sharpe ──────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
for tc in tc_order:
    sub = df[df["tc"] == tc]
    ax.scatter(sub["n_trades"], sub["sharpe"], alpha=0.35, s=14,
               color=tc_colors[tc], label=tc)
ax.axhline(0, color="black", lw=1, linestyle="--", alpha=0.5)
# Add correlation annotation
corr = df["n_trades"].corr(df["sharpe"])
ax.text(0.97, 0.95, f"Correlation: {corr:.3f}", transform=ax.transAxes,
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))
ax.set_title("Number of Trades vs Sharpe Ratio", fontsize=12, fontweight="bold")
ax.set_xlabel("Number of Trades (over 10 years)")
ax.set_ylabel("Sharpe Ratio")
ax.legend(fontsize=8, title="Tx Cost", title_fontsize=8)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUT + "plot3_trades_vs_sharpe.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot 3 saved")

# ── PLOT 4 — Heatmap: TC × Rebalancing ───────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
pivot = df.groupby(["tc", "rebal"])["sharpe"].mean().unstack()
pivot = pivot.reindex(index=tc_order, columns=rebal_order)
sns.heatmap(pivot, ax=ax, annot=True, fmt=".3f", cmap="RdYlGn",
            center=0, linewidths=0.8, annot_kws={"size": 11},
            cbar_kws={"label": "Avg Sharpe Ratio"})
ax.set_title("Average Sharpe Ratio:\nTransaction Cost × Rebalancing Frequency", fontsize=12, fontweight="bold")
ax.set_xlabel("Rebalancing Frequency")
ax.set_ylabel("Transaction Cost")
plt.tight_layout()
plt.savefig(OUT + "plot4_heatmap_tc_rebal.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot 4 saved")

# ── PLOT 5 — Sharpe distribution by lookback ─────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
lookbacks = [20, 60, 126, 252]
data_lb = [df[df["lookback"] == lb]["sharpe"].values for lb in lookbacks]
bp = ax.boxplot(data_lb, patch_artist=True, medianprops=dict(color="black", lw=2))
for patch, lb in zip(bp["boxes"], lookbacks):
    patch.set_facecolor(lb_colors[lb])
    patch.set_alpha(0.75)
ax.set_xticklabels([f"{lb} days" for lb in lookbacks], fontsize=10)
ax.axhline(0, color="black", lw=1, linestyle="--", alpha=0.5)
ax.set_title("Sharpe Ratio Distribution by Lookback Window", fontsize=12, fontweight="bold")
ax.set_ylabel("Sharpe Ratio")
ax.set_xlabel("Rolling Window Length")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT + "plot5_sharpe_by_lookback.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot 5 saved")

# ── PLOT 6 — Sharpe distribution by entry threshold ──────────
fig, ax = plt.subplots(figsize=(7, 4.5))
entries = [1.0, 1.5, 2.0, 2.5]
data_entry = [df[df["entry"] == e]["sharpe"].values for e in entries]
bp = ax.boxplot(data_entry, patch_artist=True, medianprops=dict(color="black", lw=2))
for patch, e in zip(bp["boxes"], entries):
    patch.set_facecolor(entry_colors[e])
    patch.set_alpha(0.75)
ax.set_xticklabels([f"±{e}σ" for e in entries], fontsize=11)
ax.axhline(0, color="black", lw=1, linestyle="--", alpha=0.5)
ax.set_title("Sharpe Ratio Distribution by Entry Z-Score Threshold", fontsize=12, fontweight="bold")
ax.set_ylabel("Sharpe Ratio")
ax.set_xlabel("Entry Threshold (z-score)")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT + "plot6_sharpe_by_entry.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot 6 saved")

# ── PLOT 7 — Heatmap: Lookback × Entry (realistic costs) ─────
fig, ax = plt.subplots(figsize=(7, 4))
df_real = df[df["tc"] == "0.1% (realistic)"]
pivot2 = df_real.groupby(["lookback", "entry"])["sharpe"].mean().unstack()
sns.heatmap(pivot2, ax=ax, annot=True, fmt=".3f", cmap="RdYlGn",
            center=0, linewidths=0.8, annot_kws={"size": 11},
            cbar_kws={"label": "Avg Sharpe Ratio"})
ax.set_title("Average Sharpe Ratio: Lookback × Entry Threshold\n(0.1% Transaction Cost)", fontsize=12, fontweight="bold")
ax.set_xlabel("Entry Z-Score Threshold")
ax.set_ylabel("Lookback Window (days)")
plt.tight_layout()
plt.savefig(OUT + "plot7_heatmap_lb_entry.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot 7 saved")

# ── PLOT 8 — Heatmap: Lookback × Rebalancing (realistic) ─────
fig, ax = plt.subplots(figsize=(7, 4))
pivot3 = df_real.groupby(["lookback", "rebal"])["sharpe"].mean().unstack()
pivot3 = pivot3.reindex(columns=rebal_order)
sns.heatmap(pivot3, ax=ax, annot=True, fmt=".3f", cmap="RdYlGn",
            center=0, linewidths=0.8, annot_kws={"size": 11},
            cbar_kws={"label": "Avg Sharpe Ratio"})
ax.set_title("Average Sharpe Ratio: Lookback × Rebalancing Frequency\n(0.1% Transaction Cost)", fontsize=12, fontweight="bold")
ax.set_xlabel("Rebalancing Frequency")
ax.set_ylabel("Lookback Window (days)")
plt.tight_layout()
plt.savefig(OUT + "plot8_heatmap_lb_rebal.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot 8 saved")

# ── PLOT 9 — Top 15 combinations (realistic costs) ───────────
fig, ax = plt.subplots(figsize=(8, 6))
top15 = df[df["tc"] == "0.1% (realistic)"].nlargest(15, "sharpe")
labels = [f"LB={int(r.lookback)}d  E=±{r.entry}σ  {r.rebal}  N={int(r.top_n)}"
          for _, r in top15.iterrows()]
colors_bar = ["#2ecc71" if s > 0 else "#e74c3c" for s in top15["sharpe"]]
ax.barh(range(len(top15)), top15["sharpe"].values, color=colors_bar, alpha=0.85)
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(labels, fontsize=8)
ax.axvline(0, color="black", lw=1)
ax.set_title("Top 15 Parameter Combinations by Sharpe Ratio\n(0.1% Transaction Cost)", fontsize=12, fontweight="bold")
ax.set_xlabel("Sharpe Ratio")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT + "plot9_top15_combos.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot 9 saved")

# ── PLOT 10 — Risk/Return scatter (realistic costs) ───────────
fig, ax = plt.subplots(figsize=(7, 4.5))
sc = ax.scatter(df_real["ann_vol"] * 100, df_real["ann_ret"] * 100,
                c=df_real["sharpe"], cmap="RdYlGn", alpha=0.55, s=25,
                vmin=-2, vmax=0.5)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Sharpe Ratio", fontsize=9)
ax.axhline(0, color="black", lw=1, linestyle="--", alpha=0.5)
best = df_real.nlargest(1, "sharpe").iloc[0]
ax.scatter(best["ann_vol"]*100, best["ann_ret"]*100,
           color="gold", s=200, zorder=5, marker="*",
           edgecolors="black", lw=0.8, label="Best combination")
ax.annotate(f"LB=126d, E=±2.5σ\n2x/Month, TopN=5",
            xy=(best["ann_vol"]*100, best["ann_ret"]*100),
            xytext=(28, 7), fontsize=7.5,
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8))
ax.set_title("Risk/Return Profile of All Combinations\n(0.1% Transaction Cost)", fontsize=12, fontweight="bold")
ax.set_xlabel("Annualized Volatility (%)")
ax.set_ylabel("Annualized Return (%)")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUT + "plot10_risk_return.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot 10 saved")

print("\nAll 10 plots saved successfully.")
