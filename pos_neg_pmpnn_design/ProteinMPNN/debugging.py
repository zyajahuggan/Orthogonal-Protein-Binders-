import json
import torch
import pandas as pd
from conditional_analyzer import ConditionalProbabilitiesAnalyzer

# ---- LOAD PARSED PDB ----
with open(
    "/scratch4/jgray21/zhuggan1/projects/orthosystems/tryingggg/pos_neg_design/OrthoMPNN/pdgf/output/parsed_pdb.jsonl"
) as f:
    entry = json.loads(f.readline())

# ---- LOAD INTERFACE RESIDUES (LOCAL INDEXING) ----
with open(
    "/scratch4/jgray21/zhuggan1/projects/orthosystems/tryingggg/pos_neg_design/OrthoMPNN/pdgf/input/interface_residues.json"
) as f:
    interface_residues = json.load(f)

# ---- CHAIN SETUP ----
chain_order = ["A", "B", "E", "F"]
chain_letter_to_id = {ch: i for i, ch in enumerate(chain_order)}

chain_ids = []
residue_idx = []

for ch in chain_order:
    seq = entry.get(f"seq_chain_{ch}")
    if seq is None:
        continue
    cid = chain_letter_to_id[ch]
    for i in range(len(seq)):
        chain_ids.append(cid)
        residue_idx.append(i + 1)

chain_ids = torch.tensor(chain_ids)
residue_idx = torch.tensor(residue_idx)

# ---- ANALYZER (NO MODEL NEEDED) ----
analyzer = ConditionalProbabilitiesAnalyzer(
    model=None,
    neg_weight=-0.5,
    num_repeats=1,
    interface_residues=interface_residues,
    chain_letter_to_id=chain_letter_to_id,
)

# ---- COMPUTE LOCAL INTERFACE ----
intf_local = analyzer.compute_local_interfaces(chain_ids)

# ======================================================
# 1) LOCAL → GLOBAL INDEXING CSV
# ======================================================
rows = []
for ch, cid in analyzer.chain_letter_to_id.items():
    L = int((chain_ids == cid).sum())
    for l in range(1, L + 1):
        g = analyzer.chain_local_to_global(chain_ids, ch, l)
        rows.append({
            "chain": ch,
            "local_idx": l,
            "global_idx": g,
            "is_interface": int(l in intf_local.get(ch, [])),
        })

pd.DataFrame(rows).to_csv(
    "/scratch4/jgray21/zhuggan1/projects/orthosystems/tryingggg/pos_neg_design/ProteinMPNN/debugging_plots/local_to_global_indexing.csv",
    index=False
)

print("[INFO] Wrote local_to_global_indexing.csv")

# ======================================================
# 2) TIED POSITIONS + WEIGHTS CSV
# ======================================================
tied_groups = analyzer.build_tied_groups(
    chain_ids=chain_ids,
    intf_resi=intf_local,
    tied_pairs=[(["A","B"],["E","F"])],
)

rows = []
for group in tied_groups:
    for ch, payload in group.items():
        rows.append({
            "chain": ch,
            "local_idx": payload[0][0],
            "weight": payload[1][0],
        })

pd.DataFrame(rows).to_csv(
    "/scratch4/jgray21/zhuggan1/projects/orthosystems/tryingggg/pos_neg_design/ProteinMPNN/debugging_plots/tied_positions_weights.csv",
    index=False
)

print("[INFO] Wrote tied_positions_weights.csv")


rows = []
for ch, locals_ in intf_local.items():
    for l in locals_:
        rows.append({
            "chain": ch,
            "local_idx": l,
        })

pd.DataFrame(rows).to_csv(
    "/scratch4/jgray21/zhuggan1/projects/orthosystems/tryingggg/pos_neg_design/ProteinMPNN/debugging_plots/interface_positions_local.csv",
    index=False
)

print("[INFO] Wrote interface_positions_local.csv")


rows = []
for ch, cid in analyzer.chain_letter_to_id.items():
    L = int((chain_ids == cid).sum())
    interface_set = set(intf_local.get(ch, []))

    for l in range(1, L + 1):
        g = analyzer.chain_local_to_global(chain_ids, ch, l)
        rows.append({
            "chain": ch,
            "local_idx": l,
            "global_idx": g,
            "is_interface": int(l in interface_set),
        })

pd.DataFrame(rows).to_csv(
    "/scratch4/jgray21/zhuggan1/projects/orthosystems/tryingggg/pos_neg_design/ProteinMPNN/debugging_plots/local_global_interface_map.csv",
    index=False
)
print("[INFO] Wrote local_global_interface_map.csv")
