"""Cross-docking is built from orthogonal *design pairs*: for design number N there are two
cognate samples (Na_vs_Nas, Nb_vs_Nbs) and two non-cognate cross samples (Na_vs_Nbs, Nb_vs_Nas).
topk_stats.py already ranks each cognate sample against its own non-cognates individually.
This script asks a stricter, per-design question: for a given metric, do BOTH cognate samples
in a design outscore BOTH non-cognate samples in that design (min(cognate) > max(non-cognate))?
That is the "both of those are scored higher" condition, and the script reports what fraction
of the 14 designs satisfy it for every metric."""

import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pairing_utils import orient_scores, UNWANTED_COLUMNS

metrics_path = Path(__file__).resolve().parent.parent.parent / "metrics" / "cross_docking_metrics_af3.csv"
df = pd.read_csv(metrics_path)
csv_stem = metrics_path.stem
id_col = "sample"
results_dir = metrics_path.parent.parent / "results" / "cross_docking"

# design number is the leading digits of the id, e.g. "12b_vs_12as" -> "12"
df["design"] = df[id_col].apply(lambda s: re.match(r"^\d+", s.split("_vs_")[0]).group())
designs = sorted(df["design"].unique(), key=int)

#for every metric, check per design whether both cognate rows beat both non-cognate rows
success_by_metric = {}
for metric in df.columns:
    if metric.strip() in UNWANTED_COLUMNS or metric == "design":
        continue
    oriented = orient_scores(df, metric)
    successes = {}
    for design in designs:
        rows = df.index[df["design"] == design]
        cog_idx = [i for i in rows if df.loc[i, "cognate_interaction"] == 1]
        non_cog_idx = [i for i in rows if df.loc[i, "cognate_interaction"] == 0]
        successes[design] = min(oriented.loc[cog_idx]) > max(oriented.loc[non_cog_idx])
    success_by_metric[metric] = successes

# designs x metrics table of True/False, plus the fraction of designs each metric got right
success_df = pd.DataFrame(success_by_metric).sort_index(key=lambda idx: idx.astype(int))
win_rate = success_df.mean().sort_values(ascending=False)

print(f"{len(designs)} orthogonal designs total")
for metric, rate in win_rate.items():
    n_win = int(success_df[metric].sum())
    print(f"{metric}: {n_win}/{len(designs)} designs ({rate:.0%}) had both cognate pairs beat both non-cognates")

# --- barplot: one bar per metric, height is the fraction of designs where both cognate
# pairs outscored both non-cognate pairs, ordered best metric to worst ---
barplot_dir = results_dir / "pair_stats"
barplot_dir.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(win_rate.index, win_rate.values, color='steelblue')
ax.set_ylabel("Fraction of designs where both cognate pairs beat both non-cognates")
ax.set_ylim(0, 1)
ax.set_title("AF3 both-cognates-win rate per metric")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(barplot_dir / f"{csv_stem}_both_cognates_winrate.png", dpi=300)
plt.close()
