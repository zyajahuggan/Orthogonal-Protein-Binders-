"""Take the per-cognate-pair rank computed in topk.py (same matching logic reused here),
reshape it into a (cognate sample x metric) table, and plot the rank distribution per
metric as a boxplot with individual points and a mean +/- standard deviation error bar."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pairing_utils import find_non_cognate_indices, orient_scores, UNWANTED_COLUMNS

metrics_path = Path(__file__).resolve().parent.parent / "cross_docking_metrics_af3.csv"
df = pd.read_csv(metrics_path)
csv_stem = metrics_path.stem
project = "dhd" if "dhd" in csv_stem else "cross_docking"
results_dir = metrics_path.parent.parent / "results" / project

non_cognate_indexes = find_non_cognate_indices(df)

#use row indices to evaluate how each metric ranks the cognate pair out of its non-cognates
full_ranking = {}
for metric in df.columns:
    if metric.strip() in UNWANTED_COLUMNS:
        continue
    ranking = {}
    for cog, non_cog in non_cognate_indexes.items():
        oriented = orient_scores(df, metric)
        cog_score = oriented.loc[cog]
        rank = 1 + sum(oriented.loc[indx] > cog_score for indx in non_cog)
        ranking[df["sample"][cog]] = rank
    full_ranking[metric] = ranking

# each metric's ranking dict has one entry per cognate sample, so this reshapes cleanly
# into a table where every row is one cognate sample ranked by every metric -- that
# row-alignment is what makes the metrics "paired" for the stats below
rank_df = pd.DataFrame(full_ranking).sort_index()

# order metrics best (lowest mean rank) to worst, left to right on the plot
metric_order = rank_df.mean().sort_values().index.tolist()
rank_df = rank_df[metric_order]

# --- boxplot: box+whiskers show the distribution, scattered points show every
# individual cognate sample's rank, black diamonds show mean +/- standard deviation ---
boxplot_dir = results_dir / "boxplots"
boxplot_dir.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 6))
ax.boxplot([rank_df[m].values for m in metric_order], labels=metric_order, showfliers=False)

rng = np.random.default_rng(0)  # fixed seed so the point jitter is reproducible
for i, m in enumerate(metric_order, start=1):
    y = rank_df[m].values
    x = rng.normal(loc=i, scale=0.05, size=len(y))  # jitter x only, for readability
    ax.scatter(x, y, alpha=0.6, color='steelblue', zorder=3)

means = rank_df.mean()
stds = rank_df.std()
ax.errorbar(range(1, len(metric_order) + 1), means, yerr=stds, fmt='D', color='black',
            capsize=4, label='mean ± SD')

ax.set_ylabel("Rank of cognate pair among its non-cognates")
ax.set_title("AF3 Cognate pair ranking per metric")
plt.xticks(rotation=45, ha='right')
ax.legend()
plt.tight_layout()
plt.savefig(boxplot_dir / f"{csv_stem}_rank_boxplot.png", dpi=300)
plt.close()
