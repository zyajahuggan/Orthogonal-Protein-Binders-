import pandas as pd

LOWER_IS_BETTER_COLUMNS = ['chain_pair_pae_min_0_1', 'chain_pair_pae_min_1_0', 'fraction_disordered']
UNWANTED_COLUMNS = ["sample", "best_seed", "best_sample", "cognate_interaction", "has_clash"]


def find_non_cognate_indices(df: pd.DataFrame) -> dict[int, list[int]]:
    """For every cognate row, find the row indices of every non-cognate row
    that shares one of its two proteins."""
    cognate_indices = df.index[df["cognate_interaction"] == 1]

    non_cognate_indices = {}
    for cog_idx in cognate_indices:
        cognate_pair = df.loc[cog_idx, "sample"].split("_vs_")
        for protein in cognate_pair:
            for i in df.index:
                sample_pair = df.loc[i, "sample"].split("_vs_")
                if protein in sample_pair and cognate_pair != sample_pair:
                    non_cognate_indices.setdefault(cog_idx, []).append(i)
    return non_cognate_indices


def orient_scores(df: pd.DataFrame, metric: str) -> pd.Series:
    """Flip sign for error-like metrics (PAE, fraction_disordered) so that,
    for every metric, higher always means 'stronger cognate signal'."""
    return -df[metric] if metric in LOWER_IS_BETTER_COLUMNS else df[metric]
