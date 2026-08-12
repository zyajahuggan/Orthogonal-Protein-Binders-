import pandas as pd
import os
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs"

def excell_to_fasta(absolute_excell_path):
    data = pd.ExcelFile(absolute_excell_path)
    dhd_sheet = pd.read_excel(data, sheet_name=2)
    names = dhd_sheet["name "]
    chains = dhd_sheet["chain"]
    seqs = dhd_sheet["seq"]
    seq_ID = {}
    seq_ID_lst = []

    for name, chain, seq in zip(names, chains, seqs):
        key = name + chain
        # seq_ID (dict) and seq_ID_lst (list) are indexed together below, so a duplicate
        # key would silently overwrite seq_ID while seq_ID_lst keeps both -- desyncing the
        # two and quietly generating wrong pairs. Fail loudly instead.
        assert key not in seq_ID, f"Duplicate name+chain '{key}' in sheet — check for repeated rows"
        seq_ID[key] = seq
        seq_ID_lst.append([key, seq])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    k = 0


    for id, seq in seq_ID.items():
        for i in range(k, len(seq_ID_lst)):
            content = f"""sequences:
    - protein:
        id: A
        sequence: {seq}

    - protein:
        id: B
        sequence: {seq_ID_lst[i][1]}
    """
            filepath = os.path.join(OUTPUT_DIR, f"{id}_vs_{seq_ID_lst[i][0]}.yaml")
            with open(filepath, "w") as f:
                f.write(f"{content}")
        k += 1


print(excell_to_fasta("/weka/scratch/jgray21/cbufford1/orthogonal_data.xlsx"))