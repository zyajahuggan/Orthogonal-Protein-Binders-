"""For every combination of 2 cognate pairs that are mutually orthogonal to each other
(i.e. the data contains cross-pairing rows between their two proteins), check whether the
worse-scoring cognate pair still beats the better-scoring cross non-cognate, for every
metric. Runs once per project (cop and dhd).

cop: each numbered design contributes exactly 2 cognate pairs (Na_vs_Nas,
Nb_vs_Nbs), and only those 2 pairs have cross rows between them (Na_vs_Nbs, Nb_vs_Nas) --
cross-design combinations were never run, so they have no cross rows and get skipped. That
recovers the same per-design comparisons as the old cross_docking_pair_stats.py.

dhd: all heterodimers are mutually orthogonal to each other, so every combination of 2
cognate pairs has matching cross rows. dhd's sample names use "__" as the separator
(handled generically via the `sep` iter_datasets() yields per project, not hardcoded here).

Which combinations are "applicable" is discovered directly from which rows exist in the
data, not from parsing sample names -- so the same logic works for both datasets.
"""

import itertools
import pandas as pd
import matplotlib.pyplot as plt
from pairing_utils import orient_scores, UNWANTED_COLUMNS, iter_datasets

MODEL_NAME = "ESMFold2"
ID_COL = "job_name"

for df, csv_stem, project, sep, results_dir in iter_datasets():
    cognate_indices = df.index[df["cognate_interaction"] == 1].tolist()
    chain_sets = {i: set(df.loc[i, ID_COL].split(sep)) for i in cognate_indices}

    # find every combination of 2 cognate pairs that have cross non-cognate rows between
    # them, and cache those cross row indices once -- they're the same for every metric
    combos = []
    for i, j in itertools.combinations(cognate_indices, 2):
        chains_i, chains_j = chain_sets[i], chain_sets[j]
        cross_rows = [
            idx for idx in df.index
            if idx not in (i, j)
            and len(set(df.loc[idx, ID_COL].split(sep)) & chains_i) == 1
            and len(set(df.loc[idx, ID_COL].split(sep)) & chains_j) == 1
        ]
        if cross_rows:
            combos.append((i, j, cross_rows))

    print(f"--- {project}: {len(combos)} applicable combinations out of "
          f"{len(cognate_indices)} cognate pairs ---")

    success_by_metric = {}
    for metric in df.columns:
        if metric.strip() in UNWANTED_COLUMNS:
            continue
        oriented = orient_scores(df, metric)
        successes = []
        for i, j, cross_rows in combos:
            worst_cognate = min(oriented.loc[i], oriented.loc[j])
            best_noncognate = max(oriented.loc[idx] for idx in cross_rows)
            successes.append(worst_cognate > best_noncognate)
        success_by_metric[metric] = successes

    success_df = pd.DataFrame(success_by_metric)
    win_rate = success_df.mean().sort_values(ascending=False)

    for metric, rate in win_rate.items():
        n_win = int(success_df[metric].sum())
        print(f"{metric}: {n_win}/{len(combos)} combinations ({rate:.0%}) had the worse "
              f"cognate pair beat the better non-cognate")

    # --- barplot: one bar per metric, height is the fraction of combinations where the
    # worse-scoring cognate pair still beat the better-scoring non-cognate ---
    barplot_dir = results_dir / "pair_stats"
    barplot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(win_rate.index, win_rate.values, color='steelblue')
    ax.set_ylabel("Fraction of combinations with a clean win")
    ax.set_ylim(0, 1)
    ax.set_title(f"Both-cognates-win rate per metric — {MODEL_NAME} ({project})")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(barplot_dir / f"{csv_stem}_both_cognates_winrate.png", dpi=300, bbox_inches='tight')
    plt.close()
