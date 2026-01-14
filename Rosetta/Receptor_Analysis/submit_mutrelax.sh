#!/bin/bash
#SBATCH --job-name=receptor_mut_relax_11:24
#SBATCH --output=/scratch4/jgray21/zhuggan1/logs/%x_%j.out
#SBATCH --error=/scratch4/jgray21/zhuggan1/logs/%x_%j.err
#SBATCH --account=jgray21
#SBATCH --partition=parallel
#SBATCH --nodes=1                       # Number of nodes (max 2 on Rockfish)
#SBATCH --ntasks=1                      # One MPI process per node (satisfies ppr 1:node)
#SBATCH --cpus-per-task=48              # Use all 48 cores via threads
#SBATCH --time=3-00:00:00                  # Adjust time limit as necessary 72 hrs is max on rockfish
#SBATCH --mail-user=zhuggan1@jh.edu
#SBATCH --mail-type=ALL


# 1) Define and create your temp-dir (per array task). Prefer SLURM_TMPDIR if provided.
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
export TMPDIR=${SLURM_TMPDIR:-/scratch4/jgray21/$USER/tmp/${SLURM_JOB_ID}_${TASK_ID}}
mkdir -p "$TMPDIR"
# sanity check writability; if not writable, fall back to a private path under /tmp
if ! ( : >"$TMPDIR/.write_test" && rm -f "$TMPDIR/.write_test" ); then
  echo "[WARN] TMPDIR=$TMPDIR not writable; falling back to /tmp" >&2
  export TMPDIR=/tmp/${USER}/${SLURM_JOB_ID}_${TASK_ID}
  mkdir -p "$TMPDIR"
fi
echo "Using TMPDIR=$TMPDIR"

# 2) Robust cleanup on normal exit and common termination signals (timeout, cancel, node drain)
cleanup() {
  echo "[$(date)] Cleaning up scratch tempdir $TMPDIR"
  rm -rf "$TMPDIR" || true
  }
 # Pre-timeout notice from Slurm (USR1): log and prepare; do NOT delete TMP yet
trap 'echo "[$(date)] [JOB $SLURM_JOB_ID] pre-timeout USR1"' USR1
# Final cleanup on normal exit or termination
trap 'cleanup; exit 0' EXIT TERM

source /scratch4/jgray21/zhuggan1/miniconda3/etc/profile.d/conda.sh
conda activate pyrosetta

# Create output directory if it doesn't exist
mkdir -p logs || { echo "Failed to create directories"; exit 1; }

# Run the job
python3 mutant_relax.py
