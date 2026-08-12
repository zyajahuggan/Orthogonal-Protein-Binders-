from pathlib import Path
from typing import NamedTuple
import pymol2
import pandas as pd

BENCHMARK_DIR = Path(__file__).resolve().parent.parent.parent
HBOND_CUTOFF = 3.5
HBOND_ANGLE = 45


def count_interface_hbonds(cmd, cif_path):
    cmd.reinitialize()
    cmd.load(str(cif_path), "structure")
    cmd.h_add("structure")

    chains = cmd.get_chains("structure")
    assert len(chains) == 2, f"expected exactly 2 chains in {cif_path}, found {chains}"
    chain_a, chain_b = chains

    pairs_a_donor = cmd.find_pairs(
        f"(structure and chain {chain_a} and donor)",
        f"(structure and chain {chain_b} and acceptor)",
        cutoff=HBOND_CUTOFF,
        angle=HBOND_ANGLE,
    )
    pairs_b_donor = cmd.find_pairs(
        f"(structure and chain {chain_b} and donor)",
        f"(structure and chain {chain_a} and acceptor)",
        cutoff=HBOND_CUTOFF,
        angle=HBOND_ANGLE,
    )
    return len(pairs_a_donor) + len(pairs_b_donor)


class HbondProject(NamedTuple):
    name: str
    metrics_path: Path
    output_dir: Path


# esmfold2 only ever produces one structure per job, so there's no "best of N" pick
PROJECTS = [
    HbondProject("dhd", BENCHMARK_DIR / "metrics" / "dhd_combined_metrics_filtered.csv", BENCHMARK_DIR / "outputs" / "dhd_outputs"),
    HbondProject("cop", BENCHMARK_DIR / "metrics" / "cop_combined_metrics.csv", BENCHMARK_DIR / "outputs" / "cross_docking_outputs"),
]

with pymol2.PyMOL() as p:
    cmd = p.cmd

    for project in PROJECTS:
        metrics_df = pd.read_csv(project.metrics_path)
        hbond_counts = {}

        for job_name in metrics_df["job_name"]:
            cif_path = project.output_dir / job_name / f"{job_name}.cif"
            if not cif_path.exists():
                print(f"Warning: no structure found for {job_name} ({project.name})")
                continue
            count = count_interface_hbonds(cmd, cif_path)
            hbond_counts[job_name] = count
            print(f"[{project.name}] {job_name}: {count} interface h-bonds")

        metrics_df["interface_hbonds"] = metrics_df["job_name"].map(hbond_counts)
        unmatched = metrics_df["interface_hbonds"].isna().sum()
        if unmatched:
            print(f"Warning: {unmatched} rows in {project.metrics_path.name} had no matching h-bond count")
        metrics_df.to_csv(project.metrics_path, index=False)

        results_dir = BENCHMARK_DIR / "results" / project.name / "hbond_analysis"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_csv_path = results_dir / f"{project.name}_esmfold2_interface_hbond_results.csv"
        pd.DataFrame(hbond_counts.items(), columns=["job_name", "interface_hbonds"]).to_csv(results_csv_path, index=False)

        print(f"Saved {project.name} h-bond results -> {results_csv_path}")
        print(f"Added interface_hbonds column to {project.metrics_path}")
