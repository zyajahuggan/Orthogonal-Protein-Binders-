import os
from collections import defaultdict

# ---- USER PARAMETER ----
RESIDUE_OFFSET = 5   # amount to add to residue numbering

def parse_sequences(file_path, offset=0):
    sequences = []

    # Check if the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path) as f:
        next(f)  # Skip the first header line

        for line in f:
            line = line.strip()
            if line and not line.startswith('>'):
                until_slash = line.split('/')[0]  # Keep only the sequence before the first slash
                sequences.append(until_slash)

    if not sequences:
        raise ValueError("No valid sequences found in the file.")

    wt_seq = sequences[0]  # First sequence is wild-type
    mutation_profiles = defaultdict(list)

    # Compare mutant sequences against WT
    for idx, mut_seq in enumerate(sequences[1:], start=1):
        mutations = []

        for i, (wt_char, mut_char) in enumerate(zip(wt_seq, mut_seq)):
            if wt_char != mut_char:
                # Apply offset here
                pos = i + 1 + offset
                mutations.append(f"{wt_char}{pos}{mut_char}")

        if mutations:
            mutation_key = ', '.join(mutations)
            mutation_profiles[mutation_key].append(idx)

    mutation_results = []
    for mutation_profile, indices in mutation_profiles.items():
        indices_str = ', '.join(map(str, indices))
        mutation_results.append((mutation_profile, indices_str))

    return mutation_results


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == "__main__":
    import pandas as pd

    file = "/scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/pos_neg_pmpnn_design/OrthoMPNN/pdgf/output/r1_enrich_ortho100seqs/seqs/triple_renumbered.fa"

    try:
        mutations = parse_sequences(file, offset=RESIDUE_OFFSET)

        df = pd.DataFrame(mutations, columns=["Mutation Profile","Mutant Indices"])

        out_csv = "/scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/pos_neg_pmpnn_design/OrthoMPNN/pdgf/output/csvs/r1_enrich_ortho100seqs.csv"
        df.to_csv(out_csv, index=False)

        print(f"Saved mutation CSV to: {out_csv}")

    except Exception as e:
        print(f"Error: {e}")