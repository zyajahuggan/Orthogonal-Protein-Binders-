"""I am going to compute the score margin (cognate score minus its toughest non-cognate
competitor) for every cognate pair, for every metric, then sum across all cognate pairs
per metric to see which metrics win by a little vs. win by a lot. Runs once per project
(cross_docking and dhd)."""

import pandas as pd
import matplotlib.pyplot as plt
from pairing_utils import find_non_cognate_indices, orient_scores, UNWANTED_COLUMNS, iter_datasets

ID_COL = "sample"

for df, csv_stem, project, sep, results_dir in iter_datasets():
    non_cognate_indexes = find_non_cognate_indices(df, sep=sep)

    #use row indices to find the toughest non-cognate competitor and the margin against it
    full_margins = {}
    for metric in df.columns:
        if metric.strip() in UNWANTED_COLUMNS:
            continue
        oriented = orient_scores(df, metric)  # same for every cognate pair, so compute once per metric
        margins = {}
        for cog, non_cog in non_cognate_indexes.items():
            margin = oriented.loc[cog] - max(oriented.loc[indx] for indx in non_cog)
            margins[df[ID_COL][cog]] = margin
        full_margins[metric] = margins

    total_margin_per_metric = {metric: sum(margins.values()) for metric, margins in full_margins.items()}
    print(f"--- {project} ---")
    for metric, total in sorted(total_margin_per_metric.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{metric}: {total:.3g}")

    # --- barplot: bar height is the mean margin per metric, error bar is +/- standard
    # deviation of the individual cognate-pair margins that went into that mean ---

    # each metric's margins dict has one entry per cognate sample, so this reshapes cleanly
    # into a table where every row is one cognate sample's margin under every metric
    margin_df = pd.DataFrame(full_margins).sort_index()

    # order metrics best (highest mean margin, i.e. biggest win) to worst, left to right
    metric_order = margin_df.mean().sort_values(ascending=False).index.tolist()
    margin_df = margin_df[metric_order]

    means = margin_df.mean()
    stds = margin_df.std()

    barplot_dir = results_dir / "barplots"
    barplot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(metric_order, means, yerr=stds, capsize=4, color='steelblue')
    ax.set_ylabel("Score margin (cognate - toughest non-cognate)")
    ax.set_title(f"Score margin per metric (mean ± SD) — {project}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(barplot_dir / f"{csv_stem}_margin_barplot.png", dpi=300)
    plt.close()
