import pandas as pd
from pathlib import Path
from typing import NamedTuple

LOWER_IS_BETTER_COLUMNS = [
    'binder_score',
    'surface_hydrophobicity',
    'interface_dG',
    'interface_dG_SASA_ratio',
    'interface_delta_unsat_hbonds',
    'interface_delta_unsat_hbonds_percentage',
]
UNWANTED_COLUMNS = ["protein_pair","cognate_status"]


def find_non_cognate_indices(df: pd.DataFrame, sep: str = "_vs_") -> dict[int, list[int]]:
    """For every cognate row, find the row indices of every non-cognate row
    that shares one of its two proteins."""
    cognate_indices = df.index[df["cognate_status"] == 1]

    non_cognate_indices = {}
    for cog_idx in cognate_indices:
        cognate_pair = df.loc[cog_idx, "protein_pair"].split(sep)
        for protein in cognate_pair:
            for i in df.index:
                sample_pair = df.loc[i, "protein_pair"].split(sep)
                if protein in sample_pair and cognate_pair != sample_pair:
                    non_cognate_indices.setdefault(cog_idx, []).append(i)
    return non_cognate_indices


def orient_scores(df: pd.DataFrame, metric: str) -> pd.Series:
    """Flip sign for error-like metrics (PAE, fraction_disordered) so that,
    for every metric, higher always means 'stronger cognate signal'."""
    return -df[metric] if metric in LOWER_IS_BETTER_COLUMNS else df[metric]



class ProjectConfig(NamedTuple):
    metrics_filename: str
    metrics_path: Path
    project: str
    spm: str
    dataset: str
    sep: str = "_vs_"


BENCHMARK_DIR = Path(__file__).resolve().parent.parent.parent
METRICS_DIR = BENCHMARK_DIR / "metrics" / "metrics"

projects = ["riam", "fast_relax_riam"]
spms = ["af3", "esmfold2", "boltz", "chai"]
datasets = ["dhd", "cop"]

PROJECTS = []
for project in projects:
    for spm in spms:
        for dataset in datasets:
            filename = f"{project}_{spm}_{dataset}_combined.csv"
            PROJECTS.append(
                ProjectConfig(
                    metrics_filename=filename,
                    metrics_path=METRICS_DIR / filename,
                    project=project,
                    spm=spm,
                    dataset=dataset,
                )
            )


def iter_datasets():
    """Yield (df, csv_stem, project_name, sep, results_dir) once per project in
    PROJECTS, creating results_dir if it doesn't exist yet."""
    for cfg in PROJECTS:
        df = pd.read_csv(cfg.metrics_path)
        label = f"{cfg.project}_{cfg.spm}_{cfg.dataset}"
        results_dir = BENCHMARK_DIR / "results" / cfg.project / cfg.spm / cfg.dataset
        results_dir.mkdir(parents=True, exist_ok=True)
        yield df, cfg.metrics_path.stem, label, cfg.sep, results_dir
