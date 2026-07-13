from pathlib import Path
import pymol2
import pandas as pd

BENCHMARK_DIR = Path(__file__).resolve().parent.parent.parent
HBOND_CUTOFF = 3.5
HBOND_ANGLE = 45


def count_interface_hbonds(cmd, cif_path):
    cmd.reinitialize()
    cmd.load(str(cif_path), "structure")
    cmd.h_add("structure")

    chain_a, chain_b = cmd.get_chains("structure")[:2]

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


# chai ranks its 5 sampled structures per job; index 0 is the top-ranked one
# (same file confidence_to_csv.py reads scores from: scores.model_idx_0.npz)
PROJECTS = [
    {
        "name": "dhd",
        "metrics_path": BENCHMARK_DIR / "metrics" / "dhd_metrics_filtered.csv",
        "output_dir": BENCHMARK_DIR / "outputs" / "dhd",
    },
    {
        "name": "cross_docking",
        "metrics_path": BENCHMARK_DIR / "metrics" / "cross_docking_metrics.csv",
        "output_dir": BENCHMARK_DIR / "outputs" / "cross_docking",
    },
]

with pymol2.PyMOL() as p:
    cmd = p.cmd

    for project in PROJECTS:
        metrics_df = pd.read_csv(project["metrics_path"])
        hbond_counts = {}

        for sample in metrics_df["project"]:
            cif_path = project["output_dir"] / sample / "pred.model_idx_0.cif"
            if not cif_path.exists():
                print(f"Warning: no structure found for {sample} ({project['name']})")
                continue
            count = count_interface_hbonds(cmd, cif_path)
            hbond_counts[sample] = count
            print(f"[{project['name']}] {sample}: {count} interface h-bonds")

        metrics_df["interface_hbonds"] = metrics_df["project"].map(hbond_counts)
        unmatched = metrics_df["interface_hbonds"].isna().sum()
        if unmatched:
            print(f"Warning: {unmatched} rows in {project['metrics_path'].name} had no matching h-bond count")
        metrics_df.to_csv(project["metrics_path"], index=False)

        results_dir = BENCHMARK_DIR / "results" / project["name"] / "hbond_analysis"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_csv_path = results_dir / f"{project['name']}_chai_interface_hbond_results.csv"
        pd.DataFrame(hbond_counts.items(), columns=["project", "interface_hbonds"]).to_csv(results_csv_path, index=False)

        print(f"Saved {project['name']} h-bond results -> {results_csv_path}")
        print(f"Added interface_hbonds column to {project['metrics_path']}")
