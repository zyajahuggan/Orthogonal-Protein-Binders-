#!/bin/bash -l
#SBATCH --job-name=r1_enrich_test
#SBATCH --output=logs/r1_enrich_test.out
#SBATCH --error=logs/r1_enrich_test.err
#SBATCH --time=2:00:00
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --mail-user=zhuggan1@jh.edu

source ~/.bashrc
conda activate mpnn

which python

PY=$CONDA_PREFIX/bin/python
echo "Using PY=$PY"
$PY -c "import torch; print('Torch OK:', torch.__version__)"

python -c "import torch; print('Torch loaded:', torch.__version__)"

folder_with_pdbs="/scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/pos_neg_pmpnn_design/OrthoMPNN/enrichment_task/input"

output_dir="/scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/pos_neg_pmpnn_design/OrthoMPNN/enrichment_task/scripts/r1"
if [ ! -d $output_dir ]
then
    mkdir -p $output_dir
fi

path_for_parsed_chains=$output_dir"/parsed_pdb.jsonl"
path_for_assigned_chains=$output_dir"/assigned_chains.jsonl"
path_for_tied_positions=$output_dir"/tied_pos.jsonl"
path_for_fixed_positions=$output_dir"/fixed_positions.jsonl"
chains_to_design="A B"

$PY /scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/pos_neg_pmpnn_design/ProteinMPNN/helper_scripts/parse_multiple_chains.py --input_path=$folder_with_pdbs --output_path=$path_for_parsed_chains

$PY /scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/pos_neg_pmpnn_design/ProteinMPNN/helper_scripts/assign_fixed_chains.py --input_path=$path_for_parsed_chains --output_path=$path_for_assigned_chains --chain_list "$chains_to_design"

$PY /scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/pos_neg_pmpnn_design/OrthoMPNN/enrichment_task/input/make_pos_neg_tie_json.py --input_path=$path_for_parsed_chains --output_path=$path_for_tied_positions

$PY /scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/pos_neg_pmpnn_design/OrthoMPNN/enrichment_task/input/fixed_positions.py --input_path=$path_for_parsed_chains --output_path=$path_for_fixed_positions

$PY /scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/pos_neg_pmpnn_design/ProteinMPNN/protein_mpnn_run.py \
    --jsonl_path $path_for_parsed_chains \
    --chain_id_jsonl $path_for_assigned_chains \
    --fixed_positions_jsonl $path_for_fixed_positions \
    --out_folder $output_dir \
    --tied_positions_jsonl $path_for_tied_positions \
    --conditional_probs_only 0 \
    --num_seq_per_target 100 \
    --sampling_temp 0.1 \
    --interface_residues_json /scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/pos_neg_pmpnn_design/OrthoMPNN/enrichment_task/input/interface_residues.json \
    --chain_map_json /scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/pos_neg_pmpnn_design/OrthoMPNN/enrichment_task/input/chain_map.json \
    --tied_pairs_json /scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/pos_neg_pmpnn_design/OrthoMPNN/enrichment_task/input/orthogonality_pairs.json \
    --batch_size 1
