"""I am going to compute the score margin (cognate score minus its toughest non-cognate
competitor) for every cognate pair, for every metric, then sum across all cognate pairs
per metric to see which metrics win by a little vs. win by a lot."""

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from pairing_utils import find_non_cognate_indices, orient_scores, UNWANTED_COLUMNS

metrics_path = Path(__file__).resolve().parent.parent / "dhd_metrics_filtered.csv"
df = pd.read_csv(metrics_path)
csv_stem = metrics_path.stem
project = "dhd" if "dhd" in csv_stem else "cross_docking"
results_dir = metrics_path.parent.parent / "results" / project

non_cognate_indexes = find_non_cognate_indices(df)

#use row indices to find the toughest non-cognate competitor and the margin against it
full_margins = {}
for metric in df.columns:
    if metric.strip() in UNWANTED_COLUMNS:
        continue
    margins = {}
    for cog, non_cog in non_cognate_indexes.items():
        oriented = orient_scores(df, metric)
        margin = oriented.loc[cog] - max(oriented.loc[indx] for indx in non_cog)
        margins[df["project"][cog]] = margin
    full_margins[metric] = margins
total_margin_per_metric = {}
for metric, margin_dict in full_margins.items():
    total_margin_per_metric[metric] = f"{sum(margin_dict.values()):.3g}"
print(total_margin_per_metric)

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
ax.set_title("Score margin per metric (mean ± SD)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(barplot_dir / f"{metrics_path.stem}_margin_barplot.png", dpi=300)
plt.close()
