#!/bin/bash -l
#SBATCH --job-name=r1_relax_array
#SBATCH --array=0-7  # adjust to number of pdbs - 1
#SBATCH --partition=cpu
#SBATCH --nodes=1     
#SBATCH --ntasks=1                   
#SBATCH --cpus-per-task=5  # adjust to max   
#SBATCH --mem-per-cpu=1G      
#SBATCH --time=1-00:00:00 
#SBATCH --output=/scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/pos_neg_pmpnn_design/OrthoMPNN/enrichment_task/log/%x_%A_%a.out
#SBATCH --error=/scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/pos_neg_pmpnn_design/OrthoMPNN/enrichment_task/log/%x_%A_%a.err
#SBATCH --signal=B:USR1@120
#SBATCH --signal=R:USR1@120
#SBATCH --mail-user=zhuggan1@jh.edu
#SBATCH --mail-type=ALL

source ~/.bashrc
conda activate pyrosetta

PDB_DIR="/scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/pos_neg_pmpnn_design/OrthoMPNN/enrichment_task/rosetta_inputs"
OUT_DIR="/scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/pos_neg_pmpnn_design/OrthoMPNN/enrichment_task/rosetta_outputs/r1"

# Pick the Nth pdb based on array index
PDB=$(ls $PDB_DIR/*.pdb | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
BASENAME=$(basename $PDB .pdb)

python reference_structures.py --debug \
    --ortho_pdb $PDB \
    --out_ortho $OUT_DIR/$BASENAME \
    --nstruct 5