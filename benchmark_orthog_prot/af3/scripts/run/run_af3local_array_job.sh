#!/bin/bash

#SBATCH --job-name=af3_orthogonal_benchmark_crossdocking
#SBATCH --output=logs/up2_af3_%A_%a.out
#SBATCH --error=logs/up2_af3_%A_%a.err
#SBATCH --partition=a100
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-135%20
#SBATCH --account=jgray21	        
#SBATCH --qos=32gpu
#SBATCH --mail-user=cbufford354@mail.snu.edu
#SBATCH --mail-type=ALL

INPUT_DIR=$1
OUTPUT_DIR=$2

JSON_FILES=($(ls ${INPUT_DIR}/*.json))

INPUT_FILE=${JSON_FILES[$SLURM_ARRAY_TASK_ID]}

echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running AF3 on: $INPUT_FILE"

/weka/apps/software/extern/singularity/alphafold/run_alphafold3.sh \
--input_dir=$INPUT_DIR \
--input_file=$(basename $INPUT_FILE) \
--output_dir=$OUTPUT_DIR \
--models_dir=/apps/software/extern/singularity/alphafold/models \
--db_dir=/apps/software/extern/singularity/alphafold/databases
