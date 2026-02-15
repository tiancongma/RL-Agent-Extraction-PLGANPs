import pandas as pd

path = r"data\results\run_20260201_0927_bb13267_sample20\formulations_consensus_weak.tsv"
df = pd.read_csv(path, sep="\t", dtype=str).fillna("")

for col, k in [
    ("emul_method_main", 50),
    ("emul_type_main", 20),
    ("organic_solvent_main", 20),
]:
    print("\n==", col, "top", k, "==")
    vc = (
        df[col]
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .value_counts(dropna=True)
        .head(k)
    )
    for v, n in vc.items():
        print(f"{n}\t{v}")
