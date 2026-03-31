#!/bin/bash

#SBATCH --account=jgray21
#SBATCH --partition=cpu
#SBATCH --job-name=B12_screfine_wt
#SBATCH --nodes=1     
#SBATCH --ntasks=1                   
#SBATCH --cpus-per-task=5  # adjust to max   
#SBATCH --mem-per-cpu=1G      
#SBATCH --time=8:00:00 
#SBATCH --output=/scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/Experimental_Structures/pipelines/logs/%x_%A.out
#SBATCH --error=/scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/Experimental_Structures/pipelines/logs/%x_%A.err
#SBATCH --signal=B:USR1@120
#SBATCH --signal=R:USR1@120
#SBATCH --mail-user=zhuggan1@jh.edu
#SBATCH --mail-type=ALL


source ~/.bashrc
conda activate pyrosetta 

if [ ! -d "logs" ]; then
    mkdir logs || { echo "Failed to create directories"; exit 1; }
else

    echo "Using existing logs directory."
fi

python3 mutant_relax.py --debug #reference_structures.py