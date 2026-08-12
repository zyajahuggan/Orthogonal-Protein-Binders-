"""Compares AUC for the metrics every model computes under the same name (iptm, ptm)
across AF3, Boltz-2, Chai, and ESMFold2, on both cop and dhd.

Each project stores its samples under a different ID column, and for dhd, a different
naming convention (case and separator) -- verified separately that all four projects
agree on both the set of samples and their cognate/non-cognate labels once IDs are
normalized (uppercase, "_vs_" separator). That normalization is what lets rows be paired
across models here; the assertions below re-check it every run in case the underlying
data changes.

Same small-sample-size reasoning as af3/scripts/analysis/auc_stats.py: this
uses paired bootstrap over the shared sample identities rather than a classical test,
since dhd has only 6 cognate pairs.
"""

import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

BASE = Path(__file__).resolve().parent.parent
N_RESAMPLES = 2000
rng = np.random.default_rng(42)

# metrics every project computes under the same name and, per each project's own
# pairing_utils.py, all orient the same way (none list these as lower-is-better), so no
# per-project sign-flipping is needed here
SHARED_METRICS = ["iptm", "ptm"]

# (metrics path, ID column, separator) per project per dataset
PROJECTS = {
    "AF3": {
        "cop": (BASE / "af3/metrics/cop_metrics_af3.csv", "sample", "_vs_"),
        "dhd": (BASE / "af3/metrics/dhd_metrics_af3_filtered.csv", "sample", "_vs_"),
    },
    "Boltz-2": {
        "cop": (BASE / "boltz/metrics/boltz_metrics_cop.csv", "job_name", "_vs_"),
        "dhd": (BASE / "boltz/metrics/boltz_metrics_dhd_filtered.csv", "job_name", "_vs_"),
    },
    "Chai": {
        "cop": (BASE / "chai/metrics/cop_metrics.csv", "project", "_vs_"),
        "dhd": (BASE / "chai/metrics/dhd_metrics_filtered.csv", "project", "_vs_"),
    },
    "ESMFold2": {
        "cop": (BASE / "esmfold2/metrics/cop_combined_metrics.csv", "job_name", "_vs_"),
        "dhd": (BASE / "esmfold2/metrics/dhd_combined_metrics_filtered.csv", "job_name", "__"),
    },
}

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def normalized_id(raw_id, sep):
    """Canonical sample identity so the same protein pair can be matched across
    projects, even though each uses a different ID column, case, and separator."""
    return raw_id.upper().replace(sep, "_VS_")


def auc_score(labels, scores):
    """AUC via the Mann-Whitney rank-sum formula -- see af3's auc_stats.py for
    why this is used instead of sklearn.metrics.roc_auc_score in a tight loop."""
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


for dataset in ["cop", "dhd"]:
    # load every project's data for this dataset, indexed by the normalized sample ID so
    # row i means the same underlying protein pair in every project's dataframe
    aligned = {}
    for model_name, paths_by_dataset in PROJECTS.items():
        path, id_col, sep = paths_by_dataset[dataset]
        df = pd.read_csv(path)
        df["match_id"] = df[id_col].apply(lambda s: normalized_id(s, sep))
        aligned[model_name] = df.set_index("match_id").sort_index()

    # every project must agree on which samples exist and which are cognate -- checked
    # separately before writing this script, but re-checked here since a silent
    # misalignment would corrupt every comparison below
    match_ids = aligned["AF3"].index
    labels = aligned["AF3"]["cognate_interaction"].to_numpy()
    for model_name in aligned:
        aligned[model_name] = aligned[model_name].reindex(match_ids)
        assert (aligned[model_name]["cognate_interaction"].to_numpy() == labels).all(), \
            f"{model_name} cognate labels don't match AF3's for {dataset}"

    pos_idx = np.flatnonzero(labels == 1)
    neg_idx = np.flatnonzero(labels == 0)

    results_dir = RESULTS_DIR / dataset / "auc_stats"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- {dataset} ---")
    comparisons = []
    for metric in SHARED_METRICS:
        model_scores = {name: df[metric].to_numpy() for name, df in aligned.items()}
        model_aucs = {name: auc_score(labels, scores) for name, scores in model_scores.items()}

        for model_a, model_b in itertools.combinations(PROJECTS.keys(), 2):
            scores_a, scores_b = model_scores[model_a], model_scores[model_b]
            diffs = np.empty(N_RESAMPLES)
            for i in range(N_RESAMPLES):
                idx = stratified_resample_idx(pos_idx, neg_idx)
                diffs[i] = auc_score(labels[idx], scores_a[idx]) - auc_score(labels[idx], scores_b[idx])
            ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
            significant = bool(ci_low > 0 or ci_high < 0)

            comparisons.append({
                'metric': metric, 'model_a': model_a, 'model_b': model_b,
                'auc_a': model_aucs[model_a], 'auc_b': model_aucs[model_b],
                'diff': float(diffs.mean()), 'ci_low': ci_low, 'ci_high': ci_high,
                'significant': significant,
            })
            flag = "  *significant*" if significant else ""
            print(f"{metric}: {model_a} ({model_aucs[model_a]:.3f}) vs {model_b} "
                  f"({model_aucs[model_b]:.3f})  diff={diffs.mean():.3f} "
                  f"[{ci_low:.3f}, {ci_high:.3f}]{flag}")

    comparison_df = pd.DataFrame(comparisons)
    comparison_df.to_csv(results_dir / "model_comparison.csv", index=False)
