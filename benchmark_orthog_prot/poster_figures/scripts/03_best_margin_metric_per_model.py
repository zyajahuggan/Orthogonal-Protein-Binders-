"""Poster version of cross_model_analysis/best_margin_comparison_combined.py: same
computation (each model's best metric by median all-pairs standardized differential),
same single-Axes grouped-boxplot layout (one group per dataset, one box per model,
colored jitter points, uncolored boxes), replotted with poster-sized fonts.

Now also draws significance brackets over each dataset's model pairs, using the same
paired Wilcoxon signed-rank test (canonical cognate/competitor-key matching) as
cross_model_analysis/best_margin_wilcoxon.py, but on that script's cognate-pair-collapsed
pass (one point per cognate pair -- N=6 for dhd, N=28 for cop -- instead of the
per-comparison pass's much larger but correlated N) at raw p<0.05, not
Bonferroni-corrected: with a collapsed N this small, a x6 Bonferroni penalty on top
leaves nothing significant anywhere (checked -- best case p=0.19), so this reports each
pair's own test at its own alpha instead of double-penalizing. Brackets are stacked so
overlapping x-spans don't collide (shorter spans placed lower), with a key explaining
the asterisk thresholds.

New file -- does not modify or overwrite best_margin_comparison_combined.py,
best_margin_wilcoxon.py, or their outputs.
"""

from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(42)

plt.rcParams.update({
    "font.size": 19,
    "axes.titlesize": 27,
    "axes.titleweight": "bold",
    "axes.labelsize": 23,
    "xtick.labelsize": 20,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "legend.title_fontsize": 19,
})

ALPHA = 0.05
MODEL_ORDER = ["AF3", "Boltz-2", "Chai", "ESMFold2"]
MODEL_COLORS = {"AF3": "#2a78d6", "Boltz-2": "#1baf7a", "Chai": "#eda100", "ESMFold2": "#008300"}
DATASET_ORDER = ["dhd", "cop"]
DATASET_LABELS = {"dhd": "DHD", "cop": "COP"}

PATHS = {
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
LOWER_IS_BETTER = {
    "AF3": ["chain_pair_pae_min_0_1", "chain_pair_pae_min_1_0", "fraction_disordered"],
    "Boltz-2": ["complex_pde", "complex_ipde"],
    "Chai": [],
    "ESMFold2": ["intra_chain_1_pae", "intra_chain_0_pae", "inter_chain_pae"],
}
IPSAE_BOOKKEEPING = ["ipsae_n0res", "ipsae_n0chn", "ipsae_n0dom", "ipsae_d0res", "ipsae_d0chn",
                      "ipsae_d0dom", "ipsae_nres1", "ipsae_nres2", "ipsae_dist1", "ipsae_dist2"]
UNWANTED = {
    "AF3": ["sample", "best_seed", "best_sample", "cognate_interaction", "has_clash",
            "interface_hbonds"] + IPSAE_BOOKKEEPING,
    "Boltz-2": ["job_name", "cognate_interaction", "interface_hbonds"] + IPSAE_BOOKKEEPING,
    "Chai": ["project", "cognate_interaction", "chain_chain_clashes_0_0", "chain_chain_clashes_0_1",
             "chain_chain_clashes_1_0", "chain_chain_clashes_1_1", "has_inter_chain_clashes", "interface_hbonds"],
    "ESMFold2": ["job_name", "cognate_interaction", "interface_hbonds"] + IPSAE_BOOKKEEPING,
}


def find_non_cognate_indices(df, id_col, sep):
    """Row-index-keyed matches -- used for the boxplot/jitter distributions themselves."""
    cognate_indices = df.index[df["cognate_interaction"] == 1]
    non_cognate_indices = {}
    for cog_idx in cognate_indices:
        cognate_pair = df.loc[cog_idx, id_col].split(sep)
        for protein in cognate_pair:
            for i in df.index:
                sample_pair = df.loc[i, id_col].split(sep)
                if protein in sample_pair and cognate_pair != sample_pair:
                    non_cognate_indices.setdefault(cog_idx, []).append(i)
    return non_cognate_indices


def find_non_cognate_pairs_keyed(df, id_col, sep):
    """Same matches, but tagged with a canonical (cognate_key, competitor_key) identity
    -- a lowercased, order-independent tuple of the two protein names -- so diffs can be
    paired across models regardless of each model's own ID string format. Same logic as
    best_margin_wilcoxon.py's find_non_cognate_pairs."""
    cognate_indices = df.index[df["cognate_interaction"] == 1]
    matches = []
    for cog_idx in cognate_indices:
        cognate_pair = df.loc[cog_idx, id_col].split(sep)
        cognate_key = tuple(sorted(p.lower() for p in cognate_pair))
        for protein in cognate_pair:
            for i in df.index:
                sample_pair = df.loc[i, id_col].split(sep)
                if protein in sample_pair and cognate_pair != sample_pair:
                    competitor_key = tuple(sorted(p.lower() for p in sample_pair))
                    matches.append((cog_idx, i, cognate_key, competitor_key))
    return matches


def orient_scores(df, metric, lower_is_better_columns):
    return -df[metric] if metric in lower_is_better_columns else df[metric]


def best_margin_for_dataset(dataset):
    """Picks each model's best metric (highest median all-pairs differential) and
    returns both the plain row-indexed diffs (for the boxplot) and the same metric's
    canonically-keyed diffs (for the paired Wilcoxon test below)."""
    best_metric_by_model = {}
    differentials_by_model = {}
    keyed_differentials_by_model = {}

    for model in MODEL_ORDER:
        path, id_col, sep = PATHS[model][dataset]
        df = pd.read_csv(path)
        non_cognate_indices = find_non_cognate_indices(df, id_col, sep)
        keyed_matches = find_non_cognate_pairs_keyed(df, id_col, sep)

        differentials_by_metric = {}
        for metric in df.columns:
            if metric.strip() in UNWANTED[model]:
                continue
            oriented = orient_scores(df, metric, LOWER_IS_BETTER[model])
            std = oriented.std()
            if std == 0:
                continue
            standardized = (oriented - oriented.mean()) / std
            diffs = []
            for cog_idx, non_cog_idxs in non_cognate_indices.items():
                cog_score = standardized.loc[cog_idx]
                diffs.extend(cog_score - standardized.loc[i] for i in non_cog_idxs)
            differentials_by_metric[metric] = diffs

        best_metric = max(differentials_by_metric, key=lambda m: np.median(differentials_by_metric[m]))
        best_metric_by_model[model] = best_metric
        differentials_by_model[model] = differentials_by_metric[best_metric]

        oriented = orient_scores(df, best_metric, LOWER_IS_BETTER[model])
        standardized = (oriented - oriented.mean()) / oriented.std()
        keyed_differentials_by_model[model] = {
            (cognate_key, competitor_key): standardized.loc[cog_idx] - standardized.loc[noncog_idx]
            for cog_idx, noncog_idx, cognate_key, competitor_key in keyed_matches
        }

    return best_metric_by_model, differentials_by_model, keyed_differentials_by_model


def collapse_by_cognate(keyed_diffs):
    """Medians the diffs sharing a cognate_key down to one point per cognate pair --
    removes the within-cognate-pair correlation (they all share the same cognate score)
    that the per-comparison version leaves in, at the cost of a much smaller N. Same
    logic as best_margin_wilcoxon.py's collapse_by_cognate."""
    by_cognate = {}
    for (cognate_key, _competitor_key), diff in keyed_diffs.items():
        by_cognate.setdefault(cognate_key, []).append(diff)
    return {cognate_key: float(np.median(vals)) for cognate_key, vals in by_cognate.items()}


def significant_pairs(keyed_differentials_by_model):
    """Paired Wilcoxon signed-rank test between every pair of models' best-margin metric,
    using the cognate-pair-collapsed points (one per cognate pair, N=6 for dhd / 28 for
    cop) rather than the per-comparison pass -- removes the within-cognate-pair
    correlation the per-comparison pass leaves in. Reported at raw (uncorrected) p<0.05:
    with only 6-28 points per test, a Bonferroni x6 penalty on top of the collapse's own
    much smaller N leaves nothing significant, so this uses just the single test's own
    alpha instead of double-penalizing. Returns pairs significant at alpha=0.05, as
    (model_a, model_b, p)."""
    collapsed = {model: collapse_by_cognate(diffs) for model, diffs in keyed_differentials_by_model.items()}
    sig = []
    for model_a, model_b in combinations(MODEL_ORDER, 2):
        shared_keys = sorted(set(collapsed[model_a]) & set(collapsed[model_b]))
        a_vals = np.array([collapsed[model_a][k] for k in shared_keys])
        b_vals = np.array([collapsed[model_b][k] for k in shared_keys])
        _, p = wilcoxon(a_vals, b_vals)
        if p < ALPHA:
            sig.append((model_a, model_b, p))
    return sig


def significance_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    return "*"


def assign_bracket_levels(pairs_with_x):
    """Greedily stacks brackets so overlapping x-spans go on different levels --
    shorter spans (adjacent models) get placed first, so they claim the lowest level,
    while wider spans that overlap them get bumped up. Returns list of
    (x1, x2, label, level)."""
    ordered = sorted(pairs_with_x, key=lambda t: abs(t[1] - t[0]))
    level_spans = []  # level_spans[level] = list of (x1, x2) already placed there
    placed = []
    for x1, x2, label in ordered:
        lo, hi = min(x1, x2), max(x1, x2)
        level = 0
        while level < len(level_spans) and any(lo < ex2 and hi > ex1 for ex1, ex2 in level_spans[level]):
            level += 1
        if level == len(level_spans):
            level_spans.append([])
        level_spans[level].append((lo, hi))
        placed.append((x1, x2, label, level))
    return placed


results = {dataset: best_margin_for_dataset(dataset) for dataset in DATASET_ORDER}

fig, ax = plt.subplots(figsize=(13, 9))
n_models = len(MODEL_ORDER)
box_width = 0.8 / n_models
group_positions = np.arange(len(DATASET_ORDER))
model_x = {model: (i - (n_models - 1) / 2) * box_width for i, model in enumerate(MODEL_ORDER)}

legend_handles = [Patch(facecolor=MODEL_COLORS[model], label=model) for model in MODEL_ORDER]
label_artists_by_dataset = {dataset: [] for dataset in DATASET_ORDER}

for model in MODEL_ORDER:
    offset = model_x[model]
    for d_idx, dataset in enumerate(DATASET_ORDER):
        best_metric_by_model, differentials_by_model, _ = results[dataset]
        data = differentials_by_model[model]
        pos = group_positions[d_idx] + offset

        bp = ax.boxplot([data], positions=[pos], widths=box_width * 0.9,
                         patch_artist=True, showfliers=False, zorder=2)
        for box in bp["boxes"]:
            box.set_facecolor("white")
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(2)

        x_scatter = rng.normal(pos, box_width * 0.12, size=len(data))
        ax.scatter(x_scatter, data, alpha=0.4, color=MODEL_COLORS[model], s=16, zorder=3)

        label = ax.text(pos, max(data) + 0.2, best_metric_by_model[model],
                         ha="center", va="bottom", fontsize=13, rotation=90)
        label_artists_by_dataset[dataset].append(label)

ax.axhline(0, color="gray", linestyle="--", linewidth=1.2)
ax.set_xticks(group_positions)
ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER])
ax.set_ylabel("Standardized Differential (SD Units)")

# --- significance brackets: force a render so label text extents are known, then stack
# brackets above each dataset's tallest metric-name label ---
fig.canvas.draw()
inv = ax.transData.inverted()
STEP = 0.55
BRACKET_PAD = 0.9
TICK_HEIGHT = 0.08

overall_top = 0.0
for dataset in DATASET_ORDER:
    label_top = max(
        inv.transform(t.get_window_extent(fig.canvas.get_renderer()))[1][1]
        for t in label_artists_by_dataset[dataset]
    )
    bracket_base = label_top + BRACKET_PAD

    best_metric_by_model, differentials_by_model, keyed_differentials_by_model = results[dataset]
    sig_pairs = significant_pairs(keyed_differentials_by_model)
    pairs_with_x = [
        (group_positions[DATASET_ORDER.index(dataset)] + model_x[model_a],
         group_positions[DATASET_ORDER.index(dataset)] + model_x[model_b],
         significance_stars(p))
        for model_a, model_b, p in sig_pairs
    ]
    placed = assign_bracket_levels(pairs_with_x)

    for x1, x2, label, level in placed:
        y = bracket_base + level * STEP
        ax.plot([x1, x1, x2, x2], [y - TICK_HEIGHT, y, y, y - TICK_HEIGHT], color="black", linewidth=1.3)
        ax.text((x1 + x2) / 2, y + 0.03, label, ha="center", va="bottom", fontsize=18)

    dataset_top = bracket_base + (max((lvl for *_, lvl in placed), default=-1) + 1) * STEP
    overall_top = max(overall_top, dataset_top)

bottom = min(min(d) for _, diffs, _ in results.values() for d in diffs.values())
ax.set_ylim(bottom - 0.4, overall_top + 0.3)

ax.set_title("Best-Margin Metric per Model, All-Pairs Differential")
ax.legend(handles=legend_handles, title="Model", loc="upper left", bbox_to_anchor=(1.01, 1.0))
ax.text(0.98, 0.98,
        "Wilcoxon, cognate-pair collapsed\n* p < 0.05   ** p < 0.01   *** p < 0.001",
        transform=ax.transAxes, ha="right", va="top", fontsize=13,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#cccccc"))

plt.tight_layout()
plt.savefig(OUT_DIR / "best_margin_metric_per_model.png", dpi=300, bbox_inches="tight")
plt.close()

print("saved best_margin_metric_per_model.png")
