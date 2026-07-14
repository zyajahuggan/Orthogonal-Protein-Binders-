"""Variation on score_margin.py: instead of each cognate pair's margin against only its
toughest non-cognate competitor, compute the difference against every non-cognate
competitor that shares a protein with it, for every metric. Runs once per project
(cross_docking and dhd)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pairing_utils import orient_scores, UNWANTED_COLUMNS, iter_datasets, find_non_cognate_indices

MODEL_NAME = "Boltz-2"
rng = np.random.default_rng(42)

for df, csv_stem, project, sep, results_dir in iter_datasets():
    non_cognate_indices = find_non_cognate_indices(df, sep=sep)
    assert non_cognate_indices, f"no cognate/non-cognate pairs found for {project}"

    total_differences = {}
    for metric in df.columns:
        if metric.strip() in UNWANTED_COLUMNS:
            continue
        scores = orient_scores(df, metric)
        # standardize so metrics on very different scales (e.g. hbond counts vs pLDDT
        # vs PAE) are comparable in units of standard deviations, not raw magnitude
        std = scores.std()
        if std == 0:
            print(f"Warning: {metric!r} has zero variance in {project} -- skipping")
            continue
        standardized = (scores - scores.mean()) / std
        differences = []
        for cog, non_cog in non_cognate_indices.items():
            cog_score = standardized.loc[cog]
            for idx in non_cog:
                non_cog_score = standardized.loc[idx]
                differences.append(cog_score - non_cog_score)
        total_differences[metric] = differences

    # --- boxplot + jittered points: every individual cognate/non-cognate pairwise
    # difference per metric, standardized so scale doesn't distort the comparison ---
    diff_df = pd.DataFrame(total_differences)

    metric_order = diff_df.median().sort_values(ascending=False).index.tolist()
    diff_df = diff_df[metric_order]

    barplot_dir = results_dir / "score_margin"
    barplot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot([diff_df[metric] for metric in metric_order], tick_labels=metric_order, showfliers=False)
    for i, metric in enumerate(metric_order, start=1):
        y = diff_df[metric].values
        x = rng.normal(i, 0.05, size=len(y))
        ax.scatter(x, y, alpha=0.4, color='black', s=12, zorder=3)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_ylabel("Standardized differential (SD units)")
    ax.set_title(f"All-pairs score differential per metric — {MODEL_NAME} ({project})")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(barplot_dir / f"{csv_stem}_all_pairs_differential_boxplot.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- barplot: total (summed) all-pairs differential per metric, sorted by its
    # own value so the biggest total win reads leftmost ---
    totals = diff_df.sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(totals.index, totals.values, color='steelblue')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_ylabel("Total standardized differential (SD units)")
    ax.set_title(f"Total all-pairs score differential per metric — {MODEL_NAME} ({project})")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(barplot_dir / f"{csv_stem}_all_pairs_differential_barplot.png", dpi=300, bbox_inches='tight')
    plt.close()
