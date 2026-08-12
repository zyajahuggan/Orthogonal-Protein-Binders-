import os
from pathlib import Path

import pandas as pd
from pyrosetta import init

from align_rmsd import gather_jobs
from pyrosetta_utils import (
    global_align_pdbs,
    score_interface_ensemble,
    get_cognate_status_dhd,
    get_cognate_status_cross_docking,
)
from interface_rmsd_utils import interface_unaligned_rmsd

project = "fast_relax_riam"
INTERFACE_CUTOFF = 10.0  # Angstroms; same default hotspot_residues uses elsewhere in this codebase


def main():
    init()

    base_dir = Path(__file__).resolve().parent.parent.parent

    # same flat, deterministically-ordered job list align_rmsd.py's SLURM array indexes into
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

    # same whole-complex alignment align_rmsd.py uses (binder=B, target=A already
    # match since the relaxed pdb was generated directly from spm_pdb_path);
    # interface RMSD only restricts which residues get scored, not how they're aligned
    global_align_pdbs(
        best_relaxed_pdb_path, spm_pdb_path,
        reference_chain_id="A,B", align_chain_id="A,B",
    )

    interface_rmsd = interface_unaligned_rmsd(
        best_relaxed_pdb_path, spm_pdb_path,
        reference_chain_id="A,B", align_chain_id="A,B",
        cutoff=INTERFACE_CUTOFF,
    )

    if dataset == "dhd":
        cognate_status = get_cognate_status_dhd(design_name)
    else:
        cognate_status = get_cognate_status_cross_docking(design_name)

    row = {
        "protein_pair": design_name,
        "model": model,
        "interface_rmsd": interface_rmsd,
        "cognate_status": cognate_status,
    }

    per_task_csv_dir = base_dir / "outputs" / project / model / dataset / "per_task" / "interface_rmsd_csvs"
    per_task_csv_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(per_task_csv_dir / f"{design_name}.csv", index=False)


if __name__ == "__main__":
    main()
