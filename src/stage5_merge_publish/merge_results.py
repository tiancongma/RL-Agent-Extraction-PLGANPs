"""
merge_results.py

Purpose:
  Merge the prefilter outcome and the LLM outcome into one master CSV.
  - Rows NOT in prefiltered -> Tag = "Prefilter_Irrelevant"
  - Rows in prefiltered -> use LLM AI_Tag ("Relevant"/"Irrelevant"/"Error")

Input:
  ../Data/wos_all.csv
  ../Data/wos_prefiltered.csv
  ../Data/wos_llm_tagged.csv

Output:
  ../Data/wos_master_tagged.csv   (columns: Title/Abstract/... + Master_Tag)
"""

import pandas as pd
from pathlib import Path

ALL_PATH   = Path("../Data/wos_all.csv")
PREF_PATH  = Path("../Data/wos_prefiltered.csv")
LLM_PATH   = Path("../Data/wos_llm_tagged.csv")
OUT_PATH   = Path("../Data/wos_master_tagged.csv")

def main():
    df_all  = pd.read_csv(ALL_PATH,  encoding="utf-8-sig")
    df_pref = pd.read_csv(PREF_PATH, encoding="utf-8-sig")
    df_llm  = pd.read_csv(LLM_PATH,  encoding="utf-8-sig")

    # To join reliably, use a deterministic key.
    # If Zotero export contains "Key" or "DOI", prefer that.
    key_cols = [c for c in ["Key", "key", "DOI", "doi", "Url", "url", "Title"] if c in df_all.columns]

    if not key_cols:
        raise ValueError("No common key found. Please include at least 'DOI' or 'Key' in your export.")

    key = key_cols[0]
    df_all["_k"]  = df_all[key].astype(str)
    df_pref["_k"] = df_pref[key].astype(str)
    df_llm["_k"]  = df_llm[key].astype(str)

    in_pref = set(df_pref["_k"].tolist())

    # Default tag for all: not in prefilter => Prefilter_Irrelevant
    df_all["Master_Tag"] = df_all["_k"].apply(lambda k: "Prefilter_Irrelevant" if k not in in_pref else "")

    # Map LLM tags for those in prefilter set
    llm_map = dict(zip(df_llm["_k"], df_llm["AI_Tag"]))
    def pick_tag(k, current):
        if current:  # already Prefilter_Irrelevant
            return current
        # else use LLM result
        t = llm_map.get(k, "")
        return t if t else "Error"

    df_all["Master_Tag"] = df_all.apply(lambda r: pick_tag(r["_k"], r["Master_Tag"]), axis=1)
    df_all.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"[Merge] Done → {OUT_PATH}")
    print(df_all["Master_Tag"].value_counts(dropna=False))

if __name__ == "__main__":
    main()
