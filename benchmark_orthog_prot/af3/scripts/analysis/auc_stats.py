"""Statistical analysis of the ROC-AUC leaderboard: a confidence interval per metric, a
significance test against random chance (AUC = 0.5), and a paired comparison of every
metric against the top-ranked metric. Runs once per project (cop and dhd).

Sample sizes here are small (dhd has only 6 cognate pairs), so classical asymptotic AUC
tests (e.g. DeLong's test) aren't reliable -- everything below uses nonparametric
bootstrap/permutation resampling instead. Cognate and non-cognate rows are resampled
independently to keep class balance fixed each draw ("stratified bootstrap"), which is
the standard approach for AUC confidence intervals. Caveat: some rows share a protein
with each other, so they aren't fully statistically independent -- a fully rigorous fix
needs cluster bootstrap, but dhd's overlapping pairing structure (every heterodimer pairs
with every other) makes clean clusters impossible to define, so this accepts the standard
row-level method's small amount of understated uncertainty as a known limitation.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from pairing_utils import orient_scores, UNWANTED_COLUMNS, iter_datasets

MODEL_NAME = "AF3"
N_RESAMPLES = 2000
rng = np.random.default_rng(42)


def auc_score(labels, scores):
    """AUC via the Mann-Whitney rank-sum formula -- mathematically identical to
    sklearn.metrics.roc_auc_score for binary labels, but much cheaper to call thousands
    of times in a loop since it skips sklearn's per-call input validation overhead."""
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    ranks = rankdata(scores)
    sum_ranks_pos = ranks[labels == 1].sum()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def stratified_resample_idx(pos_idx, neg_idx):
    """One bootstrap draw: resample cognate and non-cognate rows separately (with
    replacement) so every draw keeps the same class balance as the real data."""
    resampled_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
    resampled_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
    return np.concatenate([resampled_pos, resampled_neg])


for df, csv_stem, project, sep, results_dir in iter_datasets():
    labels = df['cognate_interaction'].to_numpy()
    pos_idx = np.flatnonzero(labels == 1)
    neg_idx = np.flatnonzero(labels == 0)

    metrics = [m for m in df.columns if m.strip() not in UNWANTED_COLUMNS]
    oriented = {m: orient_scores(df, m).to_numpy() for m in metrics}

    stats_dir = results_dir / "auc_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    # --- per-metric bootstrap confidence interval + permutation test vs random chance ---
    results = []
    for metric in metrics:
        scores = oriented[metric]
        observed_auc = auc_score(labels, scores)

        boot_aucs = np.empty(N_RESAMPLES)
        for i in range(N_RESAMPLES):
            idx = stratified_resample_idx(pos_idx, neg_idx)
            boot_aucs[i] = auc_score(labels[idx], scores[idx])
        ci_low, ci_high = np.percentile(boot_aucs, [2.5, 97.5])

        # permutation test: shuffle the cognate/non-cognate labels so any real signal is
        # destroyed, then see how often a random shuffle "beats" the real AUC
        permuted_aucs = np.empty(N_RESAMPLES)
        for i in range(N_RESAMPLES):
            shuffled_labels = rng.permutation(labels)
            permuted_aucs[i] = auc_score(shuffled_labels, scores)
        p_value = float(np.mean(permuted_aucs >= observed_auc))

        results.append({
            'metric': metric, 'auc': observed_auc,
            'ci_low': ci_low, 'ci_high': ci_high, 'p_value_vs_random': p_value,
        })

    stats_df = pd.DataFrame(results).sort_values('auc', ascending=False)
    stats_df.to_csv(stats_dir / f"{csv_stem}_auc_confidence_intervals.csv", index=False)

    print(f"--- {MODEL_NAME} {project} ---")
    for _, row in stats_df.iterrows():
        print(f"{row['metric']}: AUC={row['auc']:.3f} "
              f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}] "
              f"p_vs_random={row['p_value_vs_random']:.4f}")

    # --- paired comparison of every metric against the single top-ranked metric ---
    top_metric = stats_df.iloc[0]['metric']
    top_scores = oriented[top_metric]

    comparisons = []
    for metric in metrics:
        if metric == top_metric:
            continue
        scores = oriented[metric]
        diffs = np.empty(N_RESAMPLES)
        for i in range(N_RESAMPLES):
            # the SAME resampled rows score both metrics each draw, so shared-protein
            # correlation affects both sides equally and mostly cancels out of the diff
            idx = stratified_resample_idx(pos_idx, neg_idx)
            auc_top = auc_score(labels[idx], top_scores[idx])
            auc_other = auc_score(labels[idx], scores[idx])
            diffs[i] = auc_top - auc_other
        diff_ci_low, diff_ci_high = np.percentile(diffs, [2.5, 97.5])
        comparisons.append({
            'top_metric': top_metric, 'compared_metric': metric,
            'auc_diff': float(diffs.mean()),
            'ci_low': diff_ci_low, 'ci_high': diff_ci_high,
            'significant': bool(diff_ci_low > 0 or diff_ci_high < 0),
        })

    comparison_df = pd.DataFrame(comparisons).sort_values('auc_diff', ascending=False)
    comparison_df.to_csv(stats_dir / f"{csv_stem}_top_metric_comparison.csv", index=False)

    print(f"\ntop metric: {top_metric}")
    for _, row in comparison_df.iterrows():
        flag = "  *significant*" if row['significant'] else ""
        print(f"  vs {row['compared_metric']}: diff={row['auc_diff']:.3f} "
              f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]{flag}")
