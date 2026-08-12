"""Compares AUC for every IPSAE metric across AF3, Boltz-2, and ESMFold2, on both cop
and dhd. Chai is excluded -- unlike the other three, Chai's saved outputs never wrote a
full pairwise PAE matrix to disk (only aggregate ptm/iptm), so IPSAE couldn't be computed
for it without re-running predictions through Chai's Python API instead of its CLI.

Structured the same way as compare_models.py (see that file for the normalized_id /
paired-bootstrap reasoning); the only difference is the model set and metric list.
"""

import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

BASE = Path(__file__).resolve().parent.parent
N_RESAMPLES = 2000
rng = np.random.default_rng(42)

# every ipsae_* column IPSAE writes, same set added to all three models' metrics CSVs by
# add_ipsae_metrics.py; all higher-is-better, so no per-project sign-flipping is needed
SHARED_METRICS = [
    "ipsae_score", "ipsae_score_d0chn", "ipsae_score_d0dom",
    "ipsae_iptm_af", "ipsae_iptm_d0chn",
    "ipsae_pdockq", "ipsae_pdockq2", "ipsae_lis",
    "ipsae_n0res", "ipsae_n0chn", "ipsae_n0dom",
    "ipsae_d0res", "ipsae_d0chn", "ipsae_d0dom",
    "ipsae_nres1", "ipsae_nres2", "ipsae_dist1", "ipsae_dist2",
]

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
    aligned = {}
    for model_name, paths_by_dataset in PROJECTS.items():
        path, id_col, sep = paths_by_dataset[dataset]
        df = pd.read_csv(path)
        df["match_id"] = df[id_col].apply(lambda s: normalized_id(s, sep))
        aligned[model_name] = df.set_index("match_id").sort_index()

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
    comparison_df.to_csv(results_dir / "ipsae_model_comparison.csv", index=False)
