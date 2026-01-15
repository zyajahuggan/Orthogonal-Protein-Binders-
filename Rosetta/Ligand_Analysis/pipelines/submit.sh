#!/bin/bash
#SBATCH --job-name=10test_relaxer
#SBATCH --output=../../logs/%x_%j.out
#SBATCH --error=../../logs/%x_%j.err
#SBATCH --account=jgray21
#SBATCH --partition=parallel
#SBATCH --nodes=1                       # Number of nodes (max 2 on Rockfish)
#SBATCH --ntasks=1                      # One MPI process per node (satisfies ppr 1:node)
#SBATCH --cpus-per-task=48              # Use all 48 cores via threads
#SBATCH --time=1-00:00:00                  # Adjust time limit as necessary 72 hrs is max on rockfish
#SBATCH --mail-user=zhuggan1@jh.edu
#SBATCH --mail-type=ALL


source /scratch4/jgray21/zhuggan1/miniconda3/etc/profile.d/conda.sh
conda activate /scratch4/jgray21/zhuggan1/envs/envs/pyrosetta

if [ ! -d "logs" ]; then
    mkdir logs || { echo "Failed to create directories"; exit 1; }
else

    echo "Using existing logs directory."
fi

python3 mutant_relax.py #reference_structures.py