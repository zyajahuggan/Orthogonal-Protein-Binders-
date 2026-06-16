#!/bin/bash

#SBATCH --job-name=my_job_af3
#SBATCH --partition=a100
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-21
#SBATCH --exclude=c012,c013
#SBATCH -o out/localaf3_%A_%a.out
#SBATCH -e error/localaf3_%A_%a.err

## Load AlphaFold3 module
module load alphafold/3

INPUT_DIR=$1
OUTPUT_DIR=$2

JSON_FILES=($(basename -a ${INPUT_DIR}/*.json))

INPUT_FILE=${JSON_FILES[$SLURM_ARRAY_TASK_ID]}

echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running AF3 on: $INPUT_FILE"

## Running Alphafold3 with multiple inputs on bigmem partition
run_alphafold3.sh \
--input_dir=$INPUT_DIR \
--input_file=$INPUT_FILE \
--output_dir=$OUTPUT_DIR \
--models_dir=/weka/scratch/jgray21/datasets/alphafold3_models \
--db_dir=/weka/scratch/jgray21/datasets/alphafold3 \
--cpu_partition=cpu \
--cpus=10 \
# --account=jgray21_bigmem\
