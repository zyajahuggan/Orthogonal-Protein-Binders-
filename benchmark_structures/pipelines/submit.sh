#!/bin/bash

#SBATCH --account=jgray21
#SBATCH --partition=cpu
#SBATCH --job-name=global_sconly_crossbenchmark
#SBATCH --nodes=1     
#SBATCH --ntasks=1                   
#SBATCH --cpus-per-task=100  # adjust to max   
#SBATCH --mem-per-cpu=1G      
#SBATCH --time=1-00:00:00 
#SBATCH --output=/scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/benchmark_structures/pipelines/logs/%x_%A.out
#SBATCH --error=/scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/benchmark_structures/pipelines/logs/%x_%A.err
#SBATCH --mail-user=zhuggan1@jh.edu
#SBATCH --mail-type=ALL


source /home/zyhuggan/miniconda3/etc/profile.d/conda.sh
conda activate pyrosetta 

if [ ! -d "logs" ]; then
    mkdir logs || { echo "Failed to create directories"; exit 1; }
else

    echo "Using existing logs directory."
fi

python3 reference_structures.py --debug 