"""
Evaluate overlap between our extracted formulations and the Goren 2025 benchmark.

Usage examples:
    python src/stage5_benchmark/evaluate_against_goren.py \
        --our-tsv "data/results/run_20260218_1334_4e58c55_sample20_evidence_v2/weak_labels__gemini.tsv" \
        --our-jsonl "data/results/run_20260218_1334_4e58c55_sample20_evidence_v2/weak_labels__gemini.jsonl" \
        --goren-csv "data/benchmark/goren_2025/NP_dataset_formulations.csv" \
        --out-dir "data/benchmark/goren_2025/overlap_eval"

    python src/stage5_benchmark/evaluate_against_goren.py \
        --our-tsv "data/results/run_xxx/weak_labels__gemini.tsv" \
        --goren-csv "data/benchmark/goren_2025/NP_dataset_formulations.csv" \
        --out-dir "data/benchmark/goren_2025/overlap_eval" \
        --field-map-json "configs/benchmark_field_map.json"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SCHEMA_REPORT = Path("data/benchmark/goren_2025/schema_report__weak_labels.txt")
DOI_NAME_HINTS = ("doi", "reference", "url")

CANONICAL_FIELD_ORDER = [
    "solvent",
    "surfactant_name",
    "small_molecule_name",
    "polymer_MW",
    "LA/GA",
    "drug/polymer",
    "surfactant_concentration",
    "EE",
]

DEFAULT_FIELD_MAP = {
    "solvent": {"our": "organic_solvent", "goren": "solvent", "type": "categorical"},
    "surfactant_name": {"our": "surfactant_name", "goren": "surfactant_name", "type": "categorical"},
    "small_molecule_name": {"our": "drug_name", "goren": "small_molecule_name", "type": "categorical"},
    "polymer_MW": {"our": "plga_mw_kDa", "goren": "polymer_MW", "type": "numeric"},
    "LA/GA": {"our": "la_ga_ratio", "goren": "LA/GA", "type": "numeric_ratio"},
    "drug/polymer": {"our": "drug_feed_amount_text", "goren": "drug/polymer", "type": "numeric"},
    "surfactant_concentration": {"our": "pva_conc_percent", "goren": "surfactant_concentration", "type": "numeric"},
    "EE": {
        "our": "encapsulation_efficiency_percent",
        "goren": "EE",
        "type": "numeric",
    },
}

MATCH_NUMERIC_FIELDS = [
    "polymer_MW",
    "LA/GA",
    "drug/polymer",
    "surfactant_concentration",
    "EE",
]
MATCH_FALLBACK_CATEGORICAL = ["solvent", "surfactant_name"]
METRIC_CATEGORICAL_FIELDS = ["solvent", "surfactant_name", "small_molecule_name"]
METRIC_NUMERIC_FIELDS = ["polymer_MW", "LA/GA", "drug/polymer", "surfactant_concentration", "EE"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare our extracted formulation records against Goren 2025 benchmark."
    )
    parser.add_argument("--our-tsv", required=True, help="Path to our extracted TSV.")
    parser.add_argument(
        "--our-jsonl",
        default=None,
        help="Optional path to our JSONL for supplemental DOI extraction.",
    )
    parser.add_argument("--goren-csv", required=True, help="Path to Goren benchmark CSV.")
    parser.add_argument("--out-dir", required=True, help="Output directory for overlap evaluation files.")
    parser.add_argument(
        "--field-map-json",
        default=None,
        help="Optional JSON config for field/DOI mapping overrides.",
    )
    return parser.parse_args()


def load_overrides(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    override_path = Path(path)
    with override_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_schema_report_doi_candidates(report_path: Path) -> list[str]:
    if not report_path.exists():
        return []
    lines = report_path.read_text(encoding="utf-8").splitlines()
    start = None
    end = None
    for idx, line in enumerate(lines):
        if line.strip() == "[TSV] Likely DOI/Reference field candidates":
            start = idx + 1
            continue
        if start is not None and line.strip().startswith("[") and idx > start:
            end = idx
            break
    if start is None:
        return []
    section = lines[start:end] if end is not None else lines[start:]
    candidates: list[str] = []
    for row in section:
        row = row.strip()
        if not row.startswith("- "):
            continue
        value = row[2:].strip()
        if value and value.lower() != "none":
            candidates.append(value)
    return candidates


def infer_doi_field_from_columns(columns: list[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for preferred in ("doi", "reference", "url"):
        for lc, original in lowered.items():
            if preferred in lc:
                return original
    return None


def normalize_doi(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[)\],.;]+$", "", text)
    return text or None


def extract_first_number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def parse_la_ga(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(" ", "")
    if ":" in text:
        parts = text.split(":")
        if len(parts) == 2:
            try:
                a = float(parts[0])
                b = float(parts[1])
            except ValueError:
                return None
            total = a + b
            if total == 0:
                return None
            return a / total
    return extract_first_number(text)


def normalize_categorical(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    return text if text else None


def extract_doi_map_from_jsonl(jsonl_path: Path) -> dict[str, str]:
    if not jsonl_path.exists():
        return {}
    doi_map: dict[str, str] = {}
    doi_regex = re.compile(r"10\.\d{4,9}/\s*[^\s\"'<>]+", re.IGNORECASE)
    with jsonl_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = obj.get("key")
            if key is None:
                continue
            matches = doi_regex.findall(raw_line)
            cleaned = []
            for m in matches:
                c = normalize_doi(m)
                if c:
                    cleaned.append(c)
            unique = sorted(set(cleaned))
            if len(unique) == 1:
                doi_map[str(key)] = unique[0]
    return doi_map


def resolve_mapping(
    our_df: pd.DataFrame,
    goren_df: pd.DataFrame,
    overrides: dict[str, Any],
    schema_report_path: Path,
    our_jsonl: str | None,
) -> tuple[str | None, str | None, dict[str, dict[str, str]], str]:
    map_override = overrides.get("field_map") or overrides.get("fields") or {}
    field_map: dict[str, dict[str, str]] = {}
    for canonical in CANONICAL_FIELD_ORDER:
        base = dict(DEFAULT_FIELD_MAP.get(canonical, {}))
        user_item = map_override.get(canonical, {})
        if isinstance(user_item, dict):
            base.update(user_item)
        field_map[canonical] = {
            "our": str(base.get("our", "")),
            "goren": str(base.get("goren", "")),
            "type": str(base.get("type", "categorical")),
        }

    our_doi_override = (
        overrides.get("our_doi_field")
        or (overrides.get("doi_fields") or {}).get("our")
        or (overrides.get("doi_field_map") or {}).get("our")
    )
    goren_doi_override = (
        overrides.get("goren_doi_field")
        or (overrides.get("doi_fields") or {}).get("goren")
        or (overrides.get("doi_field_map") or {}).get("goren")
    )

    schema_candidates = parse_schema_report_doi_candidates(schema_report_path)
    our_doi_field = None
    for c in schema_candidates:
        if c in our_df.columns:
            our_doi_field = c
            break
    if our_doi_override and our_doi_override in our_df.columns:
        our_doi_field = our_doi_override
    if our_doi_field is None:
        our_doi_field = infer_doi_field_from_columns(list(our_df.columns))

    goren_doi_field = None
    if goren_doi_override and goren_doi_override in goren_df.columns:
        goren_doi_field = goren_doi_override
    if goren_doi_field is None:
        goren_doi_field = infer_doi_field_from_columns(list(goren_df.columns))

    doi_strategy = "field_based"
    if our_doi_field is None and our_jsonl:
        doi_map = extract_doi_map_from_jsonl(Path(our_jsonl))
        if doi_map and "key" in our_df.columns:
            our_df["doi_from_jsonl"] = our_df["key"].astype(str).map(doi_map)
            our_doi_field = "doi_from_jsonl"
            doi_strategy = "jsonl_key_to_doi"

    return our_doi_field, goren_doi_field, field_map, doi_strategy


def prepare_canonical_dataframe(
    df: pd.DataFrame,
    side: str,
    field_map: dict[str, dict[str, str]],
    doi_field: str | None,
) -> pd.DataFrame:
    out = df.copy()
    out[f"{side}__row_id"] = out.index.astype(str)
    out[f"{side}__doi"] = out[doi_field].map(normalize_doi) if doi_field in out.columns else None
    for canonical, spec in field_map.items():
        source = spec[side]
        target = f"{side}__{canonical}"
        if source not in out.columns:
            out[target] = None
            continue
        if spec["type"] == "categorical":
            out[target] = out[source].map(normalize_categorical)
        elif spec["type"] == "numeric_ratio":
            out[target] = out[source].map(parse_la_ga)
        else:
            out[target] = out[source].map(extract_first_number)
    return out


def normalized_distance(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-6)
    return abs(a - b) / denom


def pair_cost(our_row: pd.Series, goren_row: pd.Series) -> tuple[float, str]:
    numeric_distances: list[float] = []
    for field in MATCH_NUMERIC_FIELDS:
        o = our_row.get(f"our__{field}")
        g = goren_row.get(f"goren__{field}")
        if pd.notna(o) and pd.notna(g):
            numeric_distances.append(normalized_distance(float(o), float(g)))
    if numeric_distances:
        return (float(sum(numeric_distances)) / float(len(numeric_distances)), "numeric")

    matches = 0
    available = 0
    for field in MATCH_FALLBACK_CATEGORICAL:
        o = our_row.get(f"our__{field}")
        g = goren_row.get(f"goren__{field}")
        if o is not None and g is not None and pd.notna(o) and pd.notna(g):
            available += 1
            if str(o) == str(g):
                matches += 1
    if available == 0:
        return (1.0, "fallback_no_fields")
    return (1.0 - (matches / available), "fallback_exact")


def greedy_match_within_doi(
    our_group: pd.DataFrame,
    goren_group: pd.DataFrame,
    doi: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pairs: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []

    if len(our_group) == 1 and len(goren_group) == 1:
        o = our_group.iloc[0]
        g = goren_group.iloc[0]
        cost, mode = pair_cost(o, g)
        pairs.append(
            {
                "doi": doi,
                "our_row_id": o["our__row_id"],
                "goren_row_id": g["goren__row_id"],
                "pair_cost": cost,
                "match_mode": mode,
            }
        )
        return pairs, unmatched, mismatches

    candidates: list[tuple[float, str, str, str]] = []
    for _, o in our_group.iterrows():
        for _, g in goren_group.iterrows():
            cost, mode = pair_cost(o, g)
            candidates.append((cost, str(o["our__row_id"]), str(g["goren__row_id"]), mode))

    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    used_our: set[str] = set()
    used_goren: set[str] = set()
    mode_lookup: dict[tuple[str, str], str] = {}
    for cost, o_id, g_id, mode in candidates:
        mode_lookup[(o_id, g_id)] = mode
        if o_id in used_our or g_id in used_goren:
            continue
        used_our.add(o_id)
        used_goren.add(g_id)
        pairs.append(
            {
                "doi": doi,
                "our_row_id": o_id,
                "goren_row_id": g_id,
                "pair_cost": cost,
                "match_mode": mode,
            }
        )

    for _, o in our_group.iterrows():
        o_id = str(o["our__row_id"])
        if o_id not in used_our:
            unmatched.append(
                {
                    "doi": doi,
                    "source": "our",
                    "row_id": o_id,
                    "reason": "unpaired_in_greedy_matching",
                }
            )
    for _, g in goren_group.iterrows():
        g_id = str(g["goren__row_id"])
        if g_id not in used_goren:
            unmatched.append(
                {
                    "doi": doi,
                    "source": "goren",
                    "row_id": g_id,
                    "reason": "unpaired_in_greedy_matching",
                }
            )
    return pairs, unmatched, mismatches


def build_pair_enriched_table(
    pairs_df: pd.DataFrame,
    our_prepped: pd.DataFrame,
    goren_prepped: pd.DataFrame,
) -> pd.DataFrame:
    if pairs_df.empty:
        return pairs_df.copy()

    our_cols = [c for c in our_prepped.columns if c.startswith("our__")]
    goren_cols = [c for c in goren_prepped.columns if c.startswith("goren__")]
    our_small = our_prepped[our_cols].rename(columns={"our__row_id": "our_row_id"})
    goren_small = goren_prepped[goren_cols].rename(columns={"goren__row_id": "goren_row_id"})

    merged = pairs_df.merge(our_small, on="our_row_id", how="left")
    merged = merged.merge(goren_small, on="goren_row_id", how="left")
    return merged


def compute_metrics(pairs_enriched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    mismatch_rows: list[dict[str, Any]] = []

    if pairs_enriched.empty:
        return pd.DataFrame(metric_rows), pd.DataFrame(mismatch_rows)

    for field in METRIC_CATEGORICAL_FIELDS:
        our_col = f"our__{field}"
        goren_col = f"goren__{field}"
        if our_col not in pairs_enriched.columns or goren_col not in pairs_enriched.columns:
            continue
        sub = pairs_enriched[[our_col, goren_col]].dropna()
        if sub.empty:
            continue
        exact = (sub[our_col] == sub[goren_col]).mean()
        metric_rows.append(
            {
                "field": field,
                "metric": "exact_match_rate",
                "value": float(exact),
                "n": int(len(sub)),
            }
        )

    for field in METRIC_NUMERIC_FIELDS:
        our_col = f"our__{field}"
        goren_col = f"goren__{field}"
        if our_col not in pairs_enriched.columns or goren_col not in pairs_enriched.columns:
            continue
        sub = pairs_enriched[[our_col, goren_col]].dropna()
        if sub.empty:
            continue
        abs_err = (sub[our_col] - sub[goren_col]).abs()
        metric_rows.append(
            {
                "field": field,
                "metric": "mae",
                "value": float(abs_err.mean()),
                "n": int(len(sub)),
            }
        )
        metric_rows.append(
            {
                "field": field,
                "metric": "median_ae",
                "value": float(abs_err.median()),
                "n": int(len(sub)),
            }
        )
        if field == "EE":
            rel = abs_err / sub[goren_col].abs().clip(lower=1e-6)
            metric_rows.append(
                {
                    "field": "EE",
                    "metric": "mean_relative_error",
                    "value": float(rel.mean()),
                    "n": int(len(sub)),
                }
            )
            metric_rows.append(
                {
                    "field": "EE",
                    "metric": "median_relative_error",
                    "value": float(rel.median()),
                    "n": int(len(sub)),
                }
            )

    for _, row in pairs_enriched.iterrows():
        mismatch_fields: list[str] = []
        for field in METRIC_CATEGORICAL_FIELDS:
            o = row.get(f"our__{field}")
            g = row.get(f"goren__{field}")
            if pd.notna(o) and pd.notna(g) and o != g:
                mismatch_fields.append(field)
        for field in METRIC_NUMERIC_FIELDS:
            o = row.get(f"our__{field}")
            g = row.get(f"goren__{field}")
            if pd.notna(o) and pd.notna(g) and float(o) != float(g):
                mismatch_fields.append(field)
        if mismatch_fields:
            mismatch_rows.append(
                {
                    "doi": row.get("doi"),
                    "our_row_id": row.get("our_row_id"),
                    "goren_row_id": row.get("goren_row_id"),
                    "mismatch_fields": ",".join(sorted(set(mismatch_fields))),
                }
            )

    return pd.DataFrame(metric_rows), pd.DataFrame(mismatch_rows)


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    our_df = pd.read_csv(args.our_tsv, sep="\t")
    goren_df = pd.read_csv(args.goren_csv)
    overrides = load_overrides(args.field_map_json)

    our_doi_field, goren_doi_field, field_map, doi_strategy = resolve_mapping(
        our_df=our_df,
        goren_df=goren_df,
        overrides=overrides,
        schema_report_path=DEFAULT_SCHEMA_REPORT,
        our_jsonl=args.our_jsonl,
    )

    our_prepped = prepare_canonical_dataframe(our_df, "our", field_map, our_doi_field)
    goren_prepped = prepare_canonical_dataframe(goren_df, "goren", field_map, goren_doi_field)

    our_with_doi = our_prepped[our_prepped["our__doi"].notna()].copy()
    goren_with_doi = goren_prepped[goren_prepped["goren__doi"].notna()].copy()
    overlap_dois = sorted(set(our_with_doi["our__doi"]).intersection(set(goren_with_doi["goren__doi"])))

    pairs: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    for doi in overlap_dois:
        our_group = our_with_doi[our_with_doi["our__doi"] == doi]
        goren_group = goren_with_doi[goren_with_doi["goren__doi"] == doi]
        p, u, _ = greedy_match_within_doi(our_group, goren_group, doi)
        pairs.extend(p)
        unmatched.extend(u)

    # Records under non-overlap DOIs are unmatched by design.
    non_overlap_our = our_with_doi[~our_with_doi["our__doi"].isin(overlap_dois)]
    non_overlap_goren = goren_with_doi[~goren_with_doi["goren__doi"].isin(overlap_dois)]
    for _, row in non_overlap_our.iterrows():
        unmatched.append(
            {
                "doi": row.get("our__doi"),
                "source": "our",
                "row_id": row.get("our__row_id"),
                "reason": "doi_not_in_overlap",
            }
        )
    for _, row in non_overlap_goren.iterrows():
        unmatched.append(
            {
                "doi": row.get("goren__doi"),
                "source": "goren",
                "row_id": row.get("goren__row_id"),
                "reason": "doi_not_in_overlap",
            }
        )

    # Records missing DOI are unmatched and explicitly reported.
    missing_doi_our = our_prepped[our_prepped["our__doi"].isna()]
    missing_doi_goren = goren_prepped[goren_prepped["goren__doi"].isna()]
    for _, row in missing_doi_our.iterrows():
        unmatched.append(
            {
                "doi": None,
                "source": "our",
                "row_id": row.get("our__row_id"),
                "reason": "missing_doi",
            }
        )
    for _, row in missing_doi_goren.iterrows():
        unmatched.append(
            {
                "doi": None,
                "source": "goren",
                "row_id": row.get("goren__row_id"),
                "reason": "missing_doi",
            }
        )

    pairs_df = pd.DataFrame(pairs)
    unmatched_df = pd.DataFrame(unmatched)
    pairs_enriched = build_pair_enriched_table(pairs_df, our_prepped, goren_prepped)
    metrics_df, mismatches_df = compute_metrics(pairs_enriched)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "our_rows_total": int(len(our_df)),
        "goren_rows_total": int(len(goren_df)),
        "our_doi_field": our_doi_field,
        "goren_doi_field": goren_doi_field,
        "doi_resolution_strategy": doi_strategy,
        "our_rows_with_doi": int(len(our_with_doi)),
        "goren_rows_with_doi": int(len(goren_with_doi)),
        "overlap_doi_count": int(len(overlap_dois)),
        "overlap_dois": overlap_dois,
        "matched_pair_count": int(len(pairs_df)),
        "unmatched_count": int(len(unmatched_df)),
    }

    (out_dir / "overlap_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    metrics_df.to_csv(out_dir / "field_metrics.tsv", sep="\t", index=False)
    pairs_enriched.to_csv(out_dir / "matched_pairs.tsv", sep="\t", index=False)
    mismatches_df.to_csv(out_dir / "mismatches.tsv", sep="\t", index=False)
    unmatched_df.to_csv(out_dir / "unmatched.tsv", sep="\t", index=False)

    return summary


def main() -> None:
    args = parse_args()
    summary = run_evaluation(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
