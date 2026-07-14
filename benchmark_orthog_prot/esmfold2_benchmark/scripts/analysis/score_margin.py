"""I am going to compute the score margin (cognate score minus its toughest non-cognate
competitor) for every cognate pair, for every metric, then sum across all cognate pairs
per metric to see which metrics win by a little vs. win by a lot. Runs once per project
(cross_docking and dhd)."""

import matplotlib.pyplot as plt
from pairing_utils import find_non_cognate_indices, orient_scores, UNWANTED_COLUMNS, iter_datasets

ID_COL = "job_name"

for df, csv_stem, project, sep, results_dir in iter_datasets():
    non_cognate_indexes = find_non_cognate_indices(df, sep=sep)

    #use row indices to find the toughest non-cognate competitor and the margin against it
    full_margins = {}
    for metric in df.columns:
        if metric.strip() in UNWANTED_COLUMNS:
            continue
        margins = {}
        scores = orient_scores(df, metric)  # same for every cognate pair, so compute once per metric
        # standardize so metrics on very different scales (e.g. hbond counts vs pLDDT
        # vs PAE) are comparable in units of standard deviations, not raw magnitude
        std = scores.std()
        if std == 0:
            print(f"Warning: {metric!r} has zero variance in {project} -- skipping")
            continue
        standardized = (scores - scores.mean()) / std
        for cog, non_cog in non_cognate_indexes.items():
            margin = standardized.loc[cog] - max(standardized.loc[indx] for indx in non_cog)
            margins[df[ID_COL][cog]] = margin
        full_margins[metric] = margins

    total_margin_per_metric = {metric: sum(margins.values()) for metric, margins in full_margins.items()}
    # biggest total win to smallest, reused for both the printout and the barplot below
    ranked_metrics = sorted(total_margin_per_metric.items(), key=lambda kv: kv[1], reverse=True)

    print(f"--- {project} ---")
    for metric, total in ranked_metrics:
        print(f"{metric}: {total:.3g}")

    # --- barplot: bar height is the total margin per metric, summed across all cognate pairs ---
    metric_order = [metric for metric, _ in ranked_metrics]
    totals = [total for _, total in ranked_metrics]

    barplot_dir = results_dir / "score_margin"
    barplot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(metric_order, totals, color='steelblue')
    ax.set_ylabel("Total score margin (standardized, summed across cognate pairs)")
    ax.set_title(f"Total score margin per metric — {project}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(barplot_dir / f"{csv_stem}_margin_barplot.png", dpi=300)
    plt.close()
