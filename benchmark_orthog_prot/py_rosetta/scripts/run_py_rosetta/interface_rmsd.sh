#!/bin/bash

#SBATCH --job-name=interface_rmsd
#SBATCH --output=/home/cbufffor1/scratchjgray21/cbufford1/projects/orthogonal_protein_benchmark/Orthogonal-Protein-Binders-/benchmark_orthog_prot/py_rosetta/logs/interface_rmsd_%A_%a.out
#SBATCH --error=/home/cbufffor1/scratchjgray21/cbufford1/projects/orthogonal_protein_benchmark/Orthogonal-Protein-Binders-/benchmark_orthog_prot/py_rosetta/logs/interface_rmsd_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-535%20
#SBATCH --account=jgray21
#SBATCH --mail-user=cbufford354@mail.snu.edu
#SBATCH --mail-type=ALL

mkdir -p /home/cbufffor1/scratchjgray21/cbufford1/projects/orthogonal_protein_benchmark/Orthogonal-Protein-Binders-/benchmark_orthog_prot/py_rosetta/logs

module load anaconda3/2024.02-1
source /apps/software/spack/gcc/8.5.0/anaconda3/2024.02-1-jbzsrx3q6jyqoh2f3wy6n7oxhdav64ml/etc/profile.d/conda.sh
conda activate pyrosetta

python interface_rmsd.py
