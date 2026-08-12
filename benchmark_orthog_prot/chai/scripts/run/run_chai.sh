#!/bin/bash

#SBATCH --job-name=chai_dhd
#SBATCH --output=logs/chai_%A_%a.out
#SBATCH --error=logs/chai_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --array=1-136%10
#SBATCH --account=jgray21
#SBATCH --partition=h100
#SBATCH --qos=h200_4
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=cbufford354@mail.snu.edu
#SBATCH --mail-type=ALL

# Load the module system software
module load anaconda3/2024.02-1
source /apps/software/spack/gcc/8.5.0/anaconda3/2024.02-1-jbzsrx3q6jyqoh2f3wy6n7oxhdav64ml/etc/profile.d/conda.sh
conda activate /weka/scratch/jgray21/envs/chai

# Redirect caches to scratch to avoid home quota issues
export MPLCONFIGDIR=/weka/scratch/jgray21/cbufford1/tmp/matplotlib
export FONTCONFIG_PATH=/weka/scratch/jgray21/cbufford1/tmp/fontconfig
mkdir -p $MPLCONFIGDIR $FONTCONFIG_PATH

FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" cross_docking_list.txt)
base=$(basename "$FILE" .fasta)

echo "Running: $FILE"

chai-lab fold --use-msa-server --use-templates-server "$FILE" "/weka/scratch/jgray21/cbufford1/projects/orthogonal_protein_benchmark/Orthogonal-Protein-Binders-/benchmark_orthog_prot/chai_benchmark/outputs/cross_docking/$base"