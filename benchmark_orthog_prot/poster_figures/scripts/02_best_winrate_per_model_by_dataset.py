"""Poster version of cross_model_analysis/best_winrate_all_datasets.py: same
computation (each model's single best metric by both-cognates-win rate, per dataset),
replotted with poster-sized fonts. Per request, "both cognates" is dropped from the
title -- it stays "Best Win Rate per Model by Dataset" instead of naming the win-rate
definition inline.

Also draws significance brackets over each dataset's model pairs, using an exact
McNemar test on paired win/loss outcomes -- not Wilcoxon, since win/loss here is binary,
not continuous. Every model is scored on the *same* combos (same two cognate pairs,
same cross non-cognate rows -- just measured through each model's own predictions and
own best metric), matched across models by canonical (lowercased, order-independent)
protein identity the same way best_margin_wilcoxon.py matches diffs, so the paired test
compares like-for-like matchups. Reported at raw p<0.05 (not Bonferroni-corrected), same
reasoning as best_margin_metric_per_model.py: with only 14-15 combos per dataset, a x6
correction leaves essentially nothing standing.

New file -- does not modify or overwrite best_winrate_all_datasets.py or its outputs.
"""

import itertools
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binomtest

BASE = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent.parent

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


def orient_scores(df, metric, lower_is_better_columns):
    return -df[metric] if metric in lower_is_better_columns else df[metric]


def find_combos(df, id_col, sep):
    cognate_indices = df.index[df["cognate_interaction"] == 1].tolist()
    chain_sets = {i: set(df.loc[i, id_col].split(sep)) for i in cognate_indices}

    combos = []
    for i, j in itertools.combinations(cognate_indices, 2):
        chains_i, chains_j = chain_sets[i], chain_sets[j]
        cross_rows = [
            idx for idx in df.index
            if idx not in (i, j)
            and len(set(df.loc[idx, id_col].split(sep)) & chains_i) == 1
            and len(set(df.loc[idx, id_col].split(sep)) & chains_j) == 1
        ]
        if cross_rows:
            combos.append((i, j, cross_rows))
    return combos


def canonical_combo_key(df, id_col, sep, i, j):
    """Canonical, model-independent identity for a combo -- a frozenset of the two
    cognate pairs' own lowercased, order-independent protein-name tuples -- so the same
    physical matchup can be matched across models despite each model's own ID column and
    separator. Same lowercasing convention as best_margin_wilcoxon.py."""
    key_i = tuple(sorted(p.lower() for p in df.loc[i, id_col].split(sep)))
    key_j = tuple(sorted(p.lower() for p in df.loc[j, id_col].split(sep)))
    return frozenset({key_i, key_j})


def best_winrate_metric(model, dataset):
    path, id_col, sep = PATHS[model][dataset]
    df = pd.read_csv(path)
    combos = find_combos(df, id_col, sep)

    win_rates = {}
    wins_by_metric = {}
    for metric in df.columns:
        if metric.strip() in UNWANTED[model]:
            continue
        oriented = orient_scores(df, metric, LOWER_IS_BETTER[model])
        wins = [
            min(oriented.loc[i], oriented.loc[j]) > max(oriented.loc[idx] for idx in cross_rows)
            for i, j, cross_rows in combos
        ]
        win_rates[metric] = sum(wins)
        wins_by_metric[metric] = wins

    best_metric = max(win_rates, key=win_rates.get)
    wins_by_combo_key = {
        canonical_combo_key(df, id_col, sep, i, j): win
        for (i, j, _cross_rows), win in zip(combos, wins_by_metric[best_metric])
    }
    return best_metric, win_rates[best_metric], len(combos), wins_by_combo_key


def significant_pairs(wins_by_model):
    """Exact McNemar test (via the equivalent exact binomial sign test on discordant
    pairs) between every pair of models' win/loss outcomes over the same combos. Returns
    pairs significant at raw alpha=0.05, as (model_a, model_b, p)."""
    sig = []
    for model_a, model_b in combinations(MODEL_ORDER, 2):
        shared_keys = sorted(set(wins_by_model[model_a]) & set(wins_by_model[model_b]), key=str)
        a_wins = [wins_by_model[model_a][k] for k in shared_keys]
        b_wins = [wins_by_model[model_b][k] for k in shared_keys]
        b = sum(a and not b_ for a, b_ in zip(a_wins, b_wins))
        c = sum(b_ and not a for a, b_ in zip(a_wins, b_wins))
        n_discordant = b + c
        if n_discordant == 0:
            continue
        p = binomtest(min(b, c), n_discordant, 0.5).pvalue
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
    """Greedily stacks brackets so overlapping x-spans go on different levels -- shorter
    spans (adjacent models) get placed first, so they claim the lowest level, while wider
    spans that overlap them get bumped up. Returns list of (x1, x2, label, level)."""
    ordered = sorted(pairs_with_x, key=lambda t: abs(t[1] - t[0]))
    level_spans = []
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


rows = []
wins_by_combo_key = {}  # (dataset, model) -> {combo_key: bool}
for dataset in DATASET_ORDER:
    for model in MODEL_ORDER:
        best_metric, n_win, n_combos, wins = best_winrate_metric(model, dataset)
        rows.append({
            "dataset": DATASET_LABELS[dataset], "model": model, "best_metric": best_metric,
            "win_rate": n_win / n_combos, "n_win": n_win, "n_combos": n_combos,
        })
        wins_by_combo_key[(dataset, model)] = wins

combined = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(12, 9))
n_models = len(MODEL_ORDER)
bar_width = 0.8 / n_models
group_positions = np.arange(len(DATASET_ORDER))
model_x = {model: (i - (n_models - 1) / 2) * bar_width for i, model in enumerate(MODEL_ORDER)}

label_artists_by_dataset = {dataset: [] for dataset in DATASET_ORDER}

for model in MODEL_ORDER:
    offset = model_x[model]
    heights, labels = [], []
    for dataset in DATASET_ORDER:
        row = combined[(combined["dataset"] == DATASET_LABELS[dataset]) & (combined["model"] == model)].iloc[0]
        heights.append(row["win_rate"])
        labels.append(f"{row['best_metric']}\n{row['n_win']}/{row['n_combos']}")
    bars = ax.bar(group_positions + offset, heights, width=bar_width * 0.9,
                   color=MODEL_COLORS[model], label=model)
    for bar, label, dataset in zip(bars, labels, DATASET_ORDER):
        text = ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                        label, ha="center", va="bottom", fontsize=14, rotation=90)
        label_artists_by_dataset[dataset].append(text)

ax.axhline(1 / 6, linestyle="--", color="gray", linewidth=1.2, label="Random (1/6)")
ax.set_ylabel("Win Rate")
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xticks(group_positions)
ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER])
ax.set_title("Best Win Rate per Model by Dataset")

# --- significance brackets: force a render so label text extents are known, then stack
# brackets above each dataset's tallest bar label ---
fig.canvas.draw()
inv = ax.transData.inverted()
STEP = 0.16
BRACKET_PAD = 0.25
TICK_HEIGHT = 0.025

overall_top = 0.0
for d_idx, dataset in enumerate(DATASET_ORDER):
    label_top = max(
        inv.transform(t.get_window_extent(fig.canvas.get_renderer()))[1][1]
        for t in label_artists_by_dataset[dataset]
    )
    bracket_base = label_top + BRACKET_PAD

    wins_by_model = {model: wins_by_combo_key[(dataset, model)] for model in MODEL_ORDER}
    sig_pairs = significant_pairs(wins_by_model)
    pairs_with_x = [
        (group_positions[d_idx] + model_x[model_a],
         group_positions[d_idx] + model_x[model_b],
         significance_stars(p))
        for model_a, model_b, p in sig_pairs
    ]
    placed = assign_bracket_levels(pairs_with_x)

    for x1, x2, label, level in placed:
        y = bracket_base + level * STEP
        ax.plot([x1, x1, x2, x2], [y - TICK_HEIGHT, y, y, y - TICK_HEIGHT], color="black", linewidth=1.3)
        ax.text((x1 + x2) / 2, y + 0.01, label, ha="center", va="bottom", fontsize=18)

    dataset_top = bracket_base + (max((lvl for *_, lvl in placed), default=-1) + 1) * STEP
    overall_top = max(overall_top, dataset_top)

ax.set_ylim(0, max(overall_top + 0.1, 1.1))
ax.legend(title="Model", loc="upper left", bbox_to_anchor=(1.01, 1.0))
ax.text(0.98, 0.02,
        "Exact McNemar test, paired win/loss\n* p < 0.05   ** p < 0.01   *** p < 0.001",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=15,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#cccccc"))

plt.tight_layout()
plt.savefig(OUT_DIR / "best_winrate_per_model_by_dataset.png", dpi=300, bbox_inches="tight")
plt.close()

print(combined.to_string(index=False))
