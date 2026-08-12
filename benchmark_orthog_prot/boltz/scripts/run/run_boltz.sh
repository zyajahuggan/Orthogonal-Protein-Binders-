#!/bin/bash

#SBATCH --job-name=boltz_dhd
#SBATCH --output=logs/boltz_%A_%a.out
#SBATCH --error=logs/boltz_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --array=1-56      
#SBATCH --account=jgray21
#SBATCH --partition=h200      
#SBATCH --qos=h200_4
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=cbufford354@mail.snu.edu
#SBATCH --mail-type=ALL


module load anaconda3/2024.02-1
source /apps/software/spack/gcc/8.5.0/anaconda3/2024.02-1-jbzsrx3q6jyqoh2f3wy6n7oxhdav64ml/etc/profile.d/conda.sh
conda activate boltz


FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" yaml_list_cross_docking.txt)
base=$(basename "$FILE" .yaml)

echo "Running: $FILE"

boltz predict "$FILE" --use_msa_server --out_dir "./outputs/cross_docking_outputs/$base" --model boltz2

echo "Finished: $base"

