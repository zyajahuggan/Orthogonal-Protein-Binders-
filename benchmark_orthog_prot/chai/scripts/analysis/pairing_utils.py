import pandas as pd
from pathlib import Path
from typing import NamedTuple

LOWER_IS_BETTER_COLUMNS = []
# clash counts/flags are excluded from scoring entirely (same choice af3 made for has_clash),
# not oriented as lower-is-better metrics
UNWANTED_COLUMNS = [
    "project", "cognate_interaction",
    "chain_chain_clashes_0_0", "chain_chain_clashes_0_1",
    "chain_chain_clashes_1_0", "chain_chain_clashes_1_1",
    "has_inter_chain_clashes", "interface_hbonds",
]


def find_non_cognate_indices(df: pd.DataFrame, sep: str = "_vs_") -> dict[int, list[int]]:
    """For every cognate row, find the row indices of every non-cognate row
    that shares one of its two proteins."""
    cognate_indices = df.index[df["cognate_interaction"] == 1]

    non_cognate_indices = {}
    for cog_idx in cognate_indices:
        cognate_pair = df.loc[cog_idx, "project"].split(sep)
        for protein in cognate_pair:
            for i in df.index:
                sample_pair = df.loc[i, "project"].split(sep)
                if protein in sample_pair and cognate_pair != sample_pair:
                    non_cognate_indices.setdefault(cog_idx, []).append(i)
    return non_cognate_indices


def orient_scores(df: pd.DataFrame, metric: str) -> pd.Series:
    """Flip sign for error-like metrics so that, for every metric, higher always means
    'stronger cognate signal'."""
    return -df[metric] if metric in LOWER_IS_BETTER_COLUMNS else df[metric]


class ProjectConfig(NamedTuple):
    metrics_filename: str
    name: str
    sep: str = "_vs_"


PROJECTS = [
    ProjectConfig("cop_metrics.csv", "cop"),
    ProjectConfig("dhd_metrics_filtered.csv", "dhd"),
]

BENCHMARK_DIR = Path(__file__).resolve().parent.parent.parent
METRICS_DIR = BENCHMARK_DIR / "metrics"


def iter_datasets():
    """Yield (df, csv_stem, project_name, sep, results_dir) once per project in
    PROJECTS, creating results_dir if it doesn't exist yet."""
    for cfg in PROJECTS:
        metrics_path = METRICS_DIR / cfg.metrics_filename
        df = pd.read_csv(metrics_path)
        results_dir = BENCHMARK_DIR / "results" / cfg.name
        results_dir.mkdir(parents=True, exist_ok=True)
        yield df, metrics_path.stem, cfg.name, cfg.sep, results_dir
