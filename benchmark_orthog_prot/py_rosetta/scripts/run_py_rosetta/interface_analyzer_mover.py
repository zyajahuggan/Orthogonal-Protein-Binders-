import os
from pyrosetta_utils import score_interface
from pyrosetta import init
import pandas as pd
from pathlib import Path


init()

spm = "chai"
input_files_path = Path(__file__).resolve().parent.parent.parent / "inputs" / spm / f"{spm}_cif_files.txt"

with open(input_files_path) as f:
    paths = [line.strip() for line in f if line.strip()]

task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
sample = Path(paths[task_id])
protein_pair = sample.parent.name

if "dhd" in protein_pair.lower():
    project = "dhd"
else:
    project = "cross_docking"

# Each task writes to its own file -- no shared file, no race condition
output_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / spm / project / "per_task"
output_dir.mkdir(parents=True, exist_ok=True)

try:
    metrics = score_interface(str(sample))
    metrics["protein_pair"] = protein_pair
    df = pd.DataFrame([metrics])
    df.to_csv(output_dir / f"{protein_pair}.csv", index=False)
except Exception as e:
    print(f"Task {task_id} failed on {sample}: {e}")