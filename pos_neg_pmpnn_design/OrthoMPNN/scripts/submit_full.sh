#!/bin/bash -l
#SBATCH --job-name=enrichment_pipeline
#SBATCH --output=/scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/pos_neg_pmpnn_design/OrthoMPNN/enrichment_task/logs/pipeline_%j.out
#SBATCH --error=/scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/pos_neg_pmpnn_design/OrthoMPNN/enrichment_task/logs/pipeline_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem-per-cpu=2G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhuggan1@jh.edu

# ---- Environment ----
source ~/.bashrc
conda activate mpnn

which python
python -c "import torch; print('Torch OK:', torch.__version__)"

# ---- Make log dir ----
mkdir -p /scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/pos_neg_pmpnn_design/OrthoMPNN/enrichment_task/logs

# ---- Run pipeline ----
python /scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/pos_neg_pmpnn_design/OrthoMPNN/enrichment_task/enrichment_pipeline.py