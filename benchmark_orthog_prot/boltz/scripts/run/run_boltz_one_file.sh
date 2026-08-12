#!/bin/bash

#SBATCH --job-name=boltz_dhd
#SBATCH --output=logs/boltz_%A_%a.out
#SBATCH --error=logs/boltz_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --array=1      
#SBATCH --account=jgray21
#SBATCH --partition=h200      
#SBATCH --qos=h200_4
#SBATCH --cpus-per-task=8
#SBATCH --exclude=c001,c005,c006,c007,c008,c012,c013
#SBATCH --mail-user=cbufford354@mail.snu.edu
#SBATCH --mail-type=ALL


module load anaconda3/2024.02-1
source /apps/software/spack/gcc/8.5.0/anaconda3/2024.02-1-jbzsrx3q6jyqoh2f3wy6n7oxhdav64ml/etc/profile.d/conda.sh
conda activate boltz




echo "Running: $FILE"

mkdir -p "./outputs/$base"
boltz predict "/home/cbufffor1/scratchjgray21/cbufford1/projects/orthogonal_protein_benchmark/Orthogonal-Protein-Binders-/benchmark_orthog_prot/boltz_benchmark/inputs/DHD13_XAAAa_vs_DHD13_XAAAa.yaml" --use_msa_server --out_dir "./outputs/$base" --model boltz2

echo "Finished: $base"

