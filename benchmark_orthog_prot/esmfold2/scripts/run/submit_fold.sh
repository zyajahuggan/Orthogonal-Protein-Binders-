#!/bin/bash

#SBATCH --job-name=esmfold2_dhd
#SBATCH --output=logs/up2_af3_%A_%a.out
#SBATCH --error=logs/up2_af3_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-55%10       
#SBATCH --account=jgray21
#SBATCH --partition=h200
#SBATCH --qos=h200_4
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=cbufford354@mail.snu.edu
#SBATCH --mail-type=ALL


# Load the module system software
module load anaconda3/2024.02-1

# Activate your environment
conda activate esmfold2


export BIOHUB_TOKEN="3M6wd4f4n31UoOvXLSQzRr"

mkdir -p outputs logs

export HF_HOME=/weka/scratch/jgray21/cbufford1/huggingface_cache
export TRANSFORMERS_CACHE=/weka/scratch/jgray21/cbufford1/huggingface_cache
mkdir -p $HF_HOME

python /home/cbufffor1/scratchjgray21/cbufford1/projects/orthogonal_protein_benchmark/Orthogonal-Protein-Binders-/benchmark_orthog_prot/esmfold2_benchmark/scripts/run/fold_one.py


