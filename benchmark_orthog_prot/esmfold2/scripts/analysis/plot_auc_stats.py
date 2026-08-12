"""Static summary plots for the auc_stats CSVs written by auc_stats.py: a forest plot of
every metric's AUC with its bootstrap confidence interval, and a bar chart comparing the
top metric's advantage over every other metric. Reads the existing CSVs -- no
recomputation. Runs once per project (cop and dhd)."""

import pandas as pd
import matplotlib.pyplot as plt
from pairing_utils import iter_datasets

MODEL_NAME = "ESMFold2"
SIG_COLOR = "#2a78d6"      # significant (p < 0.05 / CI excludes 0)
NONSIG_COLOR = "#b0afac"   # not significant

for _, csv_stem, project, _, results_dir in iter_datasets():
    stats_dir = results_dir / "auc_stats"

    # --- forest plot: AUC + 95% CI per metric, colored by significance vs random chance ---
    ci_df = pd.read_csv(stats_dir / f"{csv_stem}_auc_confidence_intervals.csv")
    ci_df = ci_df.sort_values("auc")  # ascending so the best metric ends up plotted at the top
    colors = [SIG_COLOR if p < 0.05 else NONSIG_COLOR for p in ci_df["p_value_vs_random"]]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(ci_df))))
    ax.errorbar(
        ci_df["auc"], ci_df["metric"],
        xerr=[ci_df["auc"] - ci_df["ci_low"], ci_df["ci_high"] - ci_df["auc"]],
        fmt="none", ecolor="gray", capsize=3,
    )
    ax.scatter(ci_df["auc"], ci_df["metric"], c=colors, zorder=3)
    ax.axvline(0.5, linestyle="--", color="gray", linewidth=0.8)
    ax.set_xlabel("AUC (95% bootstrap CI)")
    ax.set_title(f"AUC by Metric — {MODEL_NAME} ({project})")
    ax.legend(handles=[
        plt.Line2D([], [], color=SIG_COLOR, marker="o", linestyle="", label="p < 0.05 vs random"),
        plt.Line2D([], [], color=NONSIG_COLOR, marker="o", linestyle="", label="not significant"),
        plt.Line2D([], [], color="gray", linestyle="--", label="Random (AUC = 0.5)"),
    ], loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(stats_dir / f"{csv_stem}_auc_forest.png", dpi=300)
    plt.close()

    # --- top-metric comparison bar chart ---
    cmp_df = pd.read_csv(stats_dir / f"{csv_stem}_top_metric_comparison.csv")
    cmp_df = cmp_df.sort_values("auc_diff")
    top_metric = cmp_df["top_metric"].iloc[0]
    colors = [SIG_COLOR if s else NONSIG_COLOR for s in cmp_df["significant"]]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(cmp_df))))
    ax.barh(
        cmp_df["compared_metric"], cmp_df["auc_diff"], color=colors,
        xerr=[cmp_df["auc_diff"] - cmp_df["ci_low"], cmp_df["ci_high"] - cmp_df["auc_diff"]],
        capsize=3,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"AUC({top_metric}) − AUC(other metric)")
    ax.set_title(f"Top Metric Advantage — {MODEL_NAME} ({project})")
    ax.legend(handles=[
        plt.Line2D([], [], color=SIG_COLOR, marker="o", linestyle="", label="significant"),
        plt.Line2D([], [], color=NONSIG_COLOR, marker="o", linestyle="", label="not significant"),
    ], loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(stats_dir / f"{csv_stem}_top_metric_comparison.png", dpi=300)
    plt.close()
