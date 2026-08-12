import pandas as pd
import os

def excell_to_fasta(absolute_excell_path, output_dir="/home/cbufffor1/scratchjgray21/cbufford1/projects/orthogonal_protein_benchmark/Orthogonal-Protein-Binders-/benchmark_orthog_prot/chai_benchmark/inputs"):
    data = pd.ExcelFile(absolute_excell_path)
    dhd_sheet = pd.read_excel(data, sheet_name=2)
    names = dhd_sheet["name "]
    chains = dhd_sheet["chain"]
    seqs = dhd_sheet["seq"]
    seq_ID = {}
    seq_ID_lst = []

    for name, chain, seq in zip(names, chains, seqs):
        seq_ID[name + chain] = seq
        seq_ID_lst.append([name + chain, seq])
    
    k = 0

    for id, seq in seq_ID.items():
        for i in range(k, len(seq_ID_lst)):
            other_id = seq_ID_lst[i][0]
            other_seq = seq_ID_lst[i][1]

            # If the two chain names are the same (homodimer), make them unique
            if id == other_id:
                id1 = f"{id}_1"
                id2 = f"{other_id}_2"
            else:
                id1 = id
                id2 = other_id

            content = f""">protein | {id1}
    {seq}
>protein | {id2}
    {other_seq}
"""
            filepath = os.path.join(output_dir, f"{id}_vs_{other_id}.fasta")
            with open(filepath, "w") as f:
                f.write(content)
        k += 1


print(excell_to_fasta("/weka/scratch/jgray21/cbufford1/orthogonal_data.xlsx"))