import os
from pathlib import Path

import pandas as pd
from pyrosetta import init

from pyrosetta_utils import (
    global_align_pdbs,
    global_unaligned_rmsd,
    score_interface_ensemble,
    get_protein_name,
    get_cognate_status_dhd,
    get_cognate_status_cross_docking,
)

project = "fast_relax_riam"
spms = ["af3", "boltz", "esmfold2", "chai"]
datasets = ["dhd", "cross_docking"]


def gather_jobs(base_dir):
    """Build the flat, deterministically-ordered list of jobs the SLURM array
    indexes into: one entry per design, across every model and dataset.

    This only does cheap filesystem/text lookups (no PyRosetta) so it's fine
    for every array task to rebuild it independently and index into it.
    """
    jobs = []
    for dataset in datasets:
        for model in spms:
            relaxed_root = base_dir / "outputs" / project / model / dataset / "per_task" / "relaxed_pdbs"
            if not relaxed_root.exists():
                continue

            spm_input_list = base_dir / "inputs" / model / f"{model}_cif_files.txt"
            with open(spm_input_list) as f:
                spm_paths = [line.strip() for line in f if line.strip()]

            for job_dir in sorted(relaxed_root.iterdir()):
                design_name = job_dir.name

                # find the raw spm-produced structure for this same design
                spm_pdb_path = None
                for line in spm_paths:
                    if get_protein_name(Path(line)) == design_name:
                        spm_pdb_path = line
                        break

                if spm_pdb_path is None:
                    print(f"Warning: no {model} input structure found for {design_name}; skipping.")
                    continue

                jobs.append((model, dataset, design_name, job_dir, spm_pdb_path))

    return jobs


def main():
    init()

    base_dir = Path(__file__).resolve().parent.parent.parent

    jobs = gather_jobs(base_dir)
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    model, dataset, design_name, relaxed_job_dir, spm_pdb_path = jobs[task_id]

    relaxed_pdb_paths = [str(p) for p in relaxed_job_dir.glob(f"{design_name}_relaxed_*.pdb")]
    if not relaxed_pdb_paths:
        raise FileNotFoundError(f"No relaxed pdbs found in {relaxed_job_dir}")

    # score_mode="best" returns the best relaxed pdb path (fixed in pyrosetta_utils.py)
    best_relaxed_pdb_path = score_interface_ensemble(
        relaxed_pdb_paths, binder_chain="B", target_chain="A", score_mode="best"
    )

    # relaxed pdb was generated directly from spm_pdb_path, so chain letters
    # already match (binder=B, target=A) -- no need for auto_match.
    # Overwrites spm_pdb_path in place with realigned coordinates.
    global_align_pdbs(
        best_relaxed_pdb_path, spm_pdb_path,
        reference_chain_id="A,B", align_chain_id="A,B",
    )

    rmsd = global_unaligned_rmsd(
        best_relaxed_pdb_path, spm_pdb_path,
        reference_chain_id="A,B", align_chain_id="A,B",
    )

    if dataset == "dhd":
        cognate_status = get_cognate_status_dhd(design_name)
    else:
        cognate_status = get_cognate_status_cross_docking(design_name)

    row = {
        "protein_pair": design_name,
        "model": model,
        "rmsd": rmsd,
        "cognate_status": cognate_status,
    }

    per_task_csv_dir = base_dir / "outputs" / project / model / dataset / "per_task" / "align_rmsd_csvs"
    per_task_csv_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(per_task_csv_dir / f"{design_name}.csv", index=False)


if __name__ == "__main__":
    main()
