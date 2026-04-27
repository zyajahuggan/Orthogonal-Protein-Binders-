import torch
import pandas as pd
from typing import Dict, List
import json

class ResidueMapping:
    """Anaylyze Conditional Probailities from PMPNN """
    def __init__(
        self,
        neg_weight: float, 
        pos_weight: float,
        interface_residues: Dict[str, List[int]],  # {'A': [22, 23, ...]}
        chain_letter_to_id: Dict[str, int],        # {"A": 1, "B": 2, ...} 
    ):
        self.interface_residues = interface_residues
        self.chain_letter_to_id = chain_letter_to_id
        self.neg_weight = neg_weight
        self.pos_weight = pos_weight 

    # Indexing helpers
    def chain_local_to_global(
        self, 
        chain_ids, 
        chain_letter, 
        local_idx: int
    ) -> int:
        """Convert local to global index"""
        cid = self.chain_letter_to_id[chain_letter]
        # first global position belonging to this chain
        start = torch.where(chain_ids == cid)[0].min()
        return int(start + (local_idx - 1))

    # Interface mapping
    def compute_local_interfaces(
        self, 
        chain_ids):
        """Verify interface residues are in the chain"""
        intf_resi = {}

        chain_lengths = {}

        for ch, cid in self.chain_letter_to_id.items():
            length = int((chain_ids == cid).sum().item())
            chain_lengths[ch] = length

        for ch, locals_ in self.interface_residues.items():
            L = chain_lengths.get(ch, 0)
            if L == 0:
                continue
            # keep only valid positions
            intf_resi[ch] = [i for i in locals_ if 1 <= i <= L]
        return intf_resi
    
    def find_interface_globals(
        self, 
        intf_resi, 
        chain_ids):
        """Compute global indices for all interface residues"""
        interface_globals = set()
        for ch, locals_ in intf_resi.items():
            for loc in locals_:
                g = self.chain_local_to_global(chain_ids, ch, loc)
                interface_globals.add(g)

        return interface_globals

    def build_tied_groups(
        self,
        chain_ids: torch.Tensor,
        interface_local: Dict[str, List[int]],
        tied_pairs: List,
    ) -> List[Dict[str, List[List[float]]]]:
        """
        Build ProteinMPNN-style tied position groups with weights.
        """
        tied_groups = []

        chain_lengths = {
            ch: int((chain_ids == cid).sum().item())
            for ch, cid in self.chain_letter_to_id.items()
        }

        for positive_chains, negative_chains in tied_pairs:
            reference_chain = positive_chains[0]
            length = chain_lengths[reference_chain]

            for pos in range(1, length + 1):
                is_interface = pos in interface_local.get(reference_chain, [])

                group = {
                    ch: [[pos], [self.pos_weight]] for ch in positive_chains
                }

                for ch in negative_chains:
                    weight = self.neg_weight if is_interface else 1.0
                    group[ch] = [[pos], [weight]]

                tied_groups.append(group)

        return tied_groups
    

class ConditionalProbabilitiesAnalyzer:
    def __init__(
        self,
        model, # log_p = self.model.conditional_probs: computes log probabilities 
        neg_weight: float, 
        pos_weight: float,
        num_repeats: int,
        interface_residues: Dict[str, List[int]],  # {'A': [22, 23, ...]}
        chain_letter_to_id: Dict[str, int],        # {"A": 1, "B": 2, ...} 
        alphabet: str = "ACDEFGHIKLMNPQRSTVWYX", # X for unknown resiudes 
    ):
        self.model = model
        self.neg_weight = neg_weight
        self.pos_weight = pos_weight
        self.num_repeats = num_repeats
        self.interface_residues = interface_residues
        self.chain_letter_to_id = chain_letter_to_id
        self.alphabet = alphabet

        self.instance_mapping = ResidueMapping(
            interface_residues=self.interface_residues,
            chain_letter_to_id=self.chain_letter_to_id,
            neg_weight=self.neg_weight,
            pos_weight=self.pos_weight
            )
    
    def top_mutations_for_view(
        self,
        log_p,
        S,
        chain_ids,
        intf_resi,
        tied_groups,
        primary_chain,
        K=5,
    ):

        rows_for_csv = []

        for group in tied_groups:
            if primary_chain not in group:
                continue

            lp = group[primary_chain][0][0]
            if lp not in intf_resi.get(primary_chain, []):
                continue

            # Global indices
            gA = self.instance_mapping.chain_local_to_global(chain_ids, "A", lp)
            gB = self.instance_mapping.chain_local_to_global(chain_ids, "B", lp)
            gE = self.instance_mapping.chain_local_to_global(chain_ids, "E", lp)
            gF = self.instance_mapping.chain_local_to_global(chain_ids, "F", lp)

            wt_idx = int(S[0, gA].item())
            wt = self.alphabet[wt_idx]

            # ---- WT raw log-probs ----
            logp_A_wt = log_p[gA][wt_idx]
            logp_B_wt = log_p[gB][wt_idx]
            logp_E_wt = log_p[gE][wt_idx]
            logp_F_wt = log_p[gF][wt_idx]

            # ---- WT combined score ----
            combined_wt = (
                self.pos_weight * (logp_A_wt + logp_B_wt)
                + self.neg_weight * (logp_E_wt + logp_F_wt)
            )

            for aa_i, aa in enumerate(self.alphabet):
                if aa_i == wt_idx or aa == "X":
                    continue

                # ---- Mut raw log-probs ----
                logp_A_mut = log_p[gA][aa_i]
                logp_B_mut = log_p[gB][aa_i]
                logp_E_mut = log_p[gE][aa_i]
                logp_F_mut = log_p[gF][aa_i]

                # ---- Mut combined score ----
                combined_mut = (
                    self.pos_weight * (logp_A_mut + logp_B_mut)
                    + self.neg_weight * (logp_E_mut + logp_F_mut)
                )

                # ---- Delta LAST ----
                delta_logp = combined_mut - combined_wt

                rows_for_csv.append({
                    "chain": primary_chain,
                    "pdb_resi": lp,
                    "wt": wt,
                    "mut": aa,

                    # WT logs
                    "logp_A_wt": float(logp_A_wt),
                    "logp_B_wt": float(logp_B_wt),
                    "logp_E_wt": float(logp_E_wt),
                    "logp_F_wt": float(logp_F_wt),
                    "combined_logp_wt": float(combined_wt),

                    # Mut logs
                    "logp_A_mut": float(logp_A_mut),
                    "logp_B_mut": float(logp_B_mut),
                    "logp_E_mut": float(logp_E_mut),
                    "logp_F_mut": float(logp_F_mut),
                    "combined_logp_mut": float(combined_mut),

                    # Final score
                    "delta_logp": float(delta_logp),
                    "neg_weight": self.neg_weight,
                    "pos_weight": self.pos_weight
                })

        df = pd.DataFrame(rows_for_csv)
        if df.empty:
            return [], rows_for_csv

        top_df = (
            df.sort_values("delta_logp", ascending=False)
            .groupby(["chain", "pdb_resi"], as_index=False)
            .first()
            .sort_values("delta_logp", ascending=False)
            .head(K)
        )
        print(top_df)

        top_records = [
            {
                "mutation": f"{r.chain}:{r.wt}{r.pdb_resi}{r.mut}",
                "delta": r.delta_logp,
                "combined_logp_mut": r.combined_logp_mut,
                "combined_logp_wt": r.combined_logp_wt,
            }
            for _, r in top_df.iterrows()
        ]

        return top_records, rows_for_csv

    # Main driver

    def enable_interface_design(self,chain_M, chain_ids, interface_residues, chain_letter_to_id):
        """
        Set chain_M=1 for interface residues on specified chains.
        """
        chain_M = chain_M.clone()

        for ch, local_positions in interface_residues.items():
            cid = chain_letter_to_id[ch]
            global_idxs = torch.where(chain_ids == cid)[0]

            for lp in local_positions:
                g = global_idxs[lp - 1]  # local → global
                chain_M[:, g] = 1

        return chain_M

    def run(
        self, 
        X, # Backbone coordinates
        S, #[batch size, WT index]
        mask, # binary exist mask  
        chain_M, # binary design mask 
        residue_idx, # Internal PMNN numbering 
        chain_encoding_all, # Chain to residue encoding 
        conditional_backbone, 
        tied_pairs, 
        top_k: int # Best mutations 
    ):

        AB_rows = []

        chain_ids = chain_encoding_all[0]

        intf_resi = self.instance_mapping.compute_local_interfaces(chain_ids)
        interface_local = intf_resi

        tied_groups = self.instance_mapping.build_tied_groups(
            chain_ids=chain_ids,
            interface_local=interface_local,
            tied_pairs=tied_pairs,
        )

        for rep in range(self.num_repeats):
            randn_1 = torch.randn(chain_M.shape, device=X.device)
            print(randn_1)
            #randn_1 = torch.zeros(chain_M.shape, device=X.device)

            chain_ids = chain_encoding_all[0]

            chain_M_eff = self.enable_interface_design(
                chain_M=chain_M,
                chain_ids=chain_ids,
                interface_residues=self.interface_residues,
                chain_letter_to_id=self.chain_letter_to_id,
            )

            log_p = self.model.conditional_probs(
                X, S, mask,
                chain_M_eff,
                residue_idx,
                chain_encoding_all,
                randn_1,
                conditional_backbone,
            )[0]

            topA,rowsA = self.top_mutations_for_view(
                log_p=log_p,
                S=S,
                chain_ids=chain_ids,
                intf_resi=intf_resi,
                tied_groups=tied_groups,
                primary_chain="A",
                K=top_k,
            )
        
            AB_rows.extend(rowsA)


        AB_df = pd.DataFrame(AB_rows)

        AB_df.sort_values(
            ["chain", "pdb_resi", "delta_logp"],
            inplace=True
        )

        AB_df.to_csv(
            "/scratch4/jgray21/zhuggan1/projects/orthosystems/tryingggg/pos_neg_design/ProteinMPNN/debugging_plots/posAB_full_mutation_table0.8.csv",
            index=False
        )
        return AB_df