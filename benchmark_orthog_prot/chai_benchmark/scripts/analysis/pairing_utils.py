import pandas as pd

LOWER_IS_BETTER_COLUMNS = []
# clash counts/flags are excluded from scoring entirely (same choice af3 made for has_clash),
# not oriented as lower-is-better metrics
UNWANTED_COLUMNS = [
    "project", "cognate_interaction",
    "chain_chain_clashes_0_0", "chain_chain_clashes_0_1",
    "chain_chain_clashes_1_0", "chain_chain_clashes_1_1",
    "has_inter_chain_clashes",
]


def find_non_cognate_indices(df: pd.DataFrame) -> dict[int, list[int]]:
    """For every cognate row, find the row indices of every non-cognate row
    that shares one of its two proteins."""
    cognate_indices = df.index[df["cognate_interaction"] == 1]

    non_cognate_indices = {}
    for cog_idx in cognate_indices:
        cognate_pair = df.loc[cog_idx, "project"].split("_vs_")
        for protein in cognate_pair:
            for i in df.index:
                sample_pair = df.loc[i, "project"].split("_vs_")
                if protein in sample_pair and cognate_pair != sample_pair:
                    non_cognate_indices.setdefault(cog_idx, []).append(i)
    return non_cognate_indices


def orient_scores(df: pd.DataFrame, metric: str) -> pd.Series:
    """Flip sign for error-like metrics so that, for every metric, higher always means
    'stronger cognate signal'."""
    return -df[metric] if metric in LOWER_IS_BETTER_COLUMNS else df[metric]
