import pandas as pd
import os
def excell_to_fasta(absolute_excell_path):
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
    os.makedirs("outputs", exist_ok=True)
    k = 0
    with open("outputs/result.txt", "w") as f:
        for id, seq in seq_ID.items():
            for i in range(k, len(seq_ID_lst)):
                title = f">{id}__{seq_ID_lst[i][0]}"
                seqs = f"{seq}:{seq_ID_lst[i][1]}"
                f.write(f"{title}\n{seqs}\n")
            k += 1


print(excell_to_fasta("/Users/colterbufford/Desktop/orthogonal_protein_design/orthogonal_data.xlsx"))