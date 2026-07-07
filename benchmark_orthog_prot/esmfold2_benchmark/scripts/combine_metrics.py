import pandas as pd

pae_df = pd.read_csv("/home/cbufffor1/scratchjgray21/cbufford1/projects/orthogonal_protein_benchmark/Orthogonal-Protein-Binders-/benchmark_orthog_prot/esmfold2_benchmark/metrics/cross_docking_pae_summary.csv")
confidence_df = pd.read_csv("/home/cbufffor1/scratchjgray21/cbufford1/projects/orthogonal_protein_benchmark/Orthogonal-Protein-Binders-/benchmark_orthog_prot/esmfold2_benchmark/metrics/cross_docking_metrics.csv")

print("PAE columns:", pae_df.columns.tolist())
print("Confidence columns:", confidence_df.columns.tolist())