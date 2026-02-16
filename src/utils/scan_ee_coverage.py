# src/utils/scan_ee_coverage.py
from __future__ import annotations

import argparse
import re
from pathlib import Path
import pandas as pd


DEFAULT_PATTERNS = [
    r"encapsulation\s+efficienc(?:y|ies)",
    r"entrapment\s+efficienc(?:y|ies)",
    r"\bEE\s?%",
    r"\bE\.E\.\s?%",  # rare
    r"encapsulation\s+efficienc(?:y|ies)\s*\(",
]


def read_key2txt(path: Path) -> pd.DataFrame:
    # Expect 2 columns: zotero_key \t txt_path
    df = pd.read_csv(path, sep="\t", header=None, names=["zotero_key", "txt_path"])
    # normalize to Path
    df["txt_path"] = df["txt_path"].astype(str)
    return df


def scan_one(text: str, patterns: list[str]) -> dict[str, int]:
    hits = {}
    for p in patterns:
        hits[p] = len(re.findall(p, text, flags=re.IGNORECASE))
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key2txt", required=True, help="Path to data/cleaned/index/key2txt.tsv")
    ap.add_argument("--out", required=True, help="Output TSV listing hits per doc")
    ap.add_argument("--summary", required=True, help="Output summary TSV")
    ap.add_argument("--patterns", default=None, help="Optional path to patterns txt (one regex per line)")
    args = ap.parse_args()

    key2txt = Path(args.key2txt)
    out_path = Path(args.out)
    summary_path = Path(args.summary)

    if args.patterns:
        patterns = [ln.strip() for ln in Path(args.patterns).read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        patterns = DEFAULT_PATTERNS

    df = read_key2txt(key2txt)

    rows = []
    per_pattern_total = {p: 0 for p in patterns}
    n_total = 0
    n_has_ee = 0

    for _, r in df.iterrows():
        n_total += 1
        k = r["zotero_key"]
        p = Path(r["txt_path"])
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            rows.append({"zotero_key": k, "txt_path": str(p), "has_ee": 0, "error": str(e)})
            continue

        hits = scan_one(text, patterns)
        has_ee = 1 if sum(hits.values()) > 0 else 0
        if has_ee:
            n_has_ee += 1

        for pat, c in hits.items():
            per_pattern_total[pat] += c

        row = {
            "zotero_key": k,
            "txt_path": str(p),
            "has_ee": has_ee,
            "error": "",
        }
        # add per-pattern counts (wide format)
        for pat, c in hits.items():
            row[f"hit__{pat}"] = c
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, sep="\t", index=False)

    summary_df = pd.DataFrame([{
        "n_total_docs": n_total,
        "n_docs_with_EE": n_has_ee,
        "pct_docs_with_EE": round((n_has_ee / n_total * 100.0), 2) if n_total else 0.0,
        **{f"total_hits__{k}": v for k, v in per_pattern_total.items()},
    }])
    summary_df.to_csv(summary_path, sep="\t", index=False)

    print(f"[OK] wrote per-doc hits: {out_path}")
    print(f"[OK] wrote summary: {summary_path}")
    print(f"[INFO] docs_total={n_total} | docs_with_EE={n_has_ee} | pct={summary_df.loc[0,'pct_docs_with_EE']}%")


if __name__ == "__main__":
    main()
