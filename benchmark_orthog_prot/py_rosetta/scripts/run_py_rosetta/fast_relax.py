from pyrosetta_utils import score_interface, score_interface_ensemble, pr_relax_parallel, get_protein_name, get_cognate_status_cross_docking, get_cognate_status_dhd
from pyrosetta import init
import pandas as pd
from pathlib import Path
import os 




def main():
    init()
    spm = "chai"

    input_files_path = Path(__file__).resolve().parent.parent.parent / "inputs" / spm / f"{spm}_cif_files.txt"
    with open(input_files_path) as f:
        paths = [line.strip() for line in f if line.strip()]

    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"]) #figure out how to replace with path
    job = Path(paths[task_id])

    design_name = get_protein_name(job)

    project = None 
    if "dhd" in design_name.lower():
        project = "dhd"
    else:
        project = "cross_docking"

    relaxed_pdb_path_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "fast_relax_riam" / f"{spm}" / f"{project}" / "per_task" / "relaxed_pdbs" / f"{design_name}"
    relaxed_pdb_path_dir.mkdir(parents=True, exist_ok=True)
    

    #for practice n_relax = 2 for speed, but for hpc make sure to change it to 5!!!!
    relaxed_pdb_paths = pr_relax_parallel(str(job), relaxed_pdb_path_dir, design_name, n_relax=5)

    single_best_score_metrics = score_interface_ensemble(relaxed_pdb_paths, binder_chain="B", target_chain="A", score_mode="best")

    single_best_score_metrics["protein_pair"] = design_name 
    
    if project == "dhd":
        single_best_score_metrics["cognate_status"] = get_cognate_status_dhd(design_name)
    else:
        single_best_score_metrics["cognate_status"] = get_cognate_status_cross_docking(design_name)
    
    df = pd.DataFrame([single_best_score_metrics])
    per_task_csv_dir = relaxed_pdb_path_dir.parent.parent / "per_task_csvs" 
    per_task_csv_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(per_task_csv_dir / f"{design_name}.csv", index=False)

if __name__ == "__main__":
    main()
