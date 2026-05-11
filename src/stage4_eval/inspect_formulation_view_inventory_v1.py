#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


REPO_ROOT = Path.cwd()
SEARCH_ROOTS = [
    REPO_ROOT / "data/results",
    REPO_ROOT / "data/db",
    REPO_ROOT / "data/benchmark/goren_2025",
    REPO_ROOT / "data/cleaned/labels/manual",
]

DISCOVERY_PATTERNS = [
    "extracted_formulation_level*.tsv",
    "formulation_core.tsv",
    "measurements.tsv",
    "modeling_ready*.tsv",
    "*_formulation_view*.xlsx",
    "instance_assignment*.tsv",
]


@dataclass
class Candidate:
    path: Path
    size_bytes: int
    mtime: float
    kind: str


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str).fillna("")


def _first_non_empty(values: Iterable[object]) -> str:
    for v in values:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s:
            return s
    return ""


def _normalize_space(v: object) -> str:
    if v is None or pd.isna(v):
        return ""
    return " ".join(str(v).split()).strip()


def _missing_mask(s: pd.Series) -> pd.Series:
    return s.isna() | s.astype(str).map(lambda x: x.strip() == "")


def discover_candidates() -> List[Candidate]:
    out: List[Candidate] = []
    seen: set[Path] = set()
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for pattern in DISCOVERY_PATTERNS:
            for path in root.rglob(pattern):
                rp = path.resolve()
                if rp in seen or not rp.is_file():
                    continue
                seen.add(rp)
                stat = rp.stat()
                kind = "other"
                lname = rp.name.lower()
                if lname.startswith("extracted_formulation_level"):
                    kind = "wide_extracted"
                elif lname == "formulation_core.tsv":
                    kind = "formulation_core"
                elif lname == "measurements.tsv":
                    kind = "measurements"
                elif lname.startswith("modeling_ready"):
                    kind = "modeling_ready"
                elif "formulation_view" in lname and lname.endswith(".xlsx"):
                    kind = "xlsx_view"
                elif "instance_assignment" in lname:
                    kind = "instance_assignment"
                out.append(Candidate(rp, stat.st_size, stat.st_mtime, kind))
    return sorted(out, key=lambda c: c.mtime, reverse=True)


def find_core_measure_pairs(candidates: Sequence[Candidate]) -> List[Tuple[Path, Path]]:
    cores: Dict[Path, Path] = {}
    meas: Dict[Path, Path] = {}
    for c in candidates:
        parent = c.path.parent
        if c.path.name == "formulation_core.tsv":
            cores[parent] = c.path
        elif c.path.name == "measurements.tsv":
            meas[parent] = c.path
    pairs: List[Tuple[Path, Path]] = []
    for d in sorted(set(cores) & set(meas)):
        pairs.append((cores[d], meas[d]))
    return pairs


def choose_primary_source(candidates: Sequence[Candidate]) -> Tuple[str, Dict[str, object]]:
    pairs = find_core_measure_pairs(candidates)
    best_pair: Optional[Tuple[Path, Path]] = None
    best_score = -10**9
    best_reason = ""

    for core_path, meas_path in pairs:
        score = 0
        ptxt = str(core_path).lower()
        if "\\data\\db\\" in ptxt:
            score += 150
        if "\\schema_v3\\" in ptxt:
            score += 60
        elif "\\schema_v2\\" in ptxt:
            score += 40
        elif "\\schema_v1\\" in ptxt:
            score += 20
        score += int(max(core_path.stat().st_mtime, meas_path.stat().st_mtime) // 3600)
        score += int(core_path.stat().st_size / 5000)
        score += int(meas_path.stat().st_size / 5000)

        reason = "paired formulation_core + measurements"
        if "\\data\\db\\" in ptxt:
            reason += "; db_v1 treated as canonical complete export"
        elif "\\schema_v3\\" in ptxt:
            reason += "; latest schema version in run outputs"
        elif "\\schema_v2\\" in ptxt:
            reason += "; recent run output pair"

        if score > best_score:
            best_score = score
            best_pair = (core_path, meas_path)
            best_reason = reason

    if best_pair is not None:
        return (
            "core_measure_pair",
            {
                "formulation_core": best_pair[0],
                "measurements": best_pair[1],
                "reason": best_reason,
            },
        )

    wide = [c for c in candidates if c.kind in {"wide_extracted", "modeling_ready"} and c.path.suffix.lower() == ".tsv"]
    if wide:
        chosen = wide[0]
        return (
            "wide_tsv",
            {
                "wide_tsv": chosen.path,
                "reason": "newest available wide TSV candidate",
            },
        )

    weak = sorted((REPO_ROOT / "data/results").glob("run_*/**/weak_labels__gemini.tsv"))
    if weak:
        weak_path = weak[-1].resolve()
        ia = [c.path for c in candidates if c.kind == "instance_assignment"]
        if ia:
            return (
                "weak_labels_with_assignment",
                {
                    "weak_labels_tsv": weak_path,
                    "instance_assignment_tsv": sorted(ia, key=lambda p: p.stat().st_mtime, reverse=True)[0],
                    "reason": "only weak labels available; using instance assignment to aggregate",
                },
            )
        return (
            "weak_labels_missing_assignment",
            {
                "weak_labels_tsv": weak_path,
                "reason": "weak labels found but no instance assignment TSV found",
            },
        )

    raise RuntimeError("No candidate formulation-level source found.")


def build_from_core_measure(core_path: Path, meas_path: Path) -> pd.DataFrame:
    core = _read_tsv(core_path)
    meas = _read_tsv(meas_path)

    id_col = "formulation_core_id" if "formulation_core_id" in core.columns else "grouping_id"
    if id_col not in core.columns:
        raise RuntimeError(f"Could not find stable ID in {core_path}")

    if "formulation_core_id" not in meas.columns:
        raise RuntimeError(f"measurements table missing formulation_core_id: {meas_path}")

    meas["measurement_type_norm"] = meas.get("measurement_type", "").map(lambda x: str(x).strip().lower())
    meas["measurement_value_num"] = pd.to_numeric(meas.get("measurement_value", ""), errors="coerce")

    text_agg = (
        meas.groupby(["formulation_core_id", "measurement_type_norm"], dropna=False)["measurement_value"]
        .agg(_first_non_empty)
        .unstack(fill_value="")
    )
    text_agg.columns = [f"measurement__{c}" for c in text_agg.columns]

    num_agg = (
        meas.groupby(["formulation_core_id", "measurement_type_norm"], dropna=False)["measurement_value_num"]
        .mean()
        .unstack()
    )
    num_agg.columns = [f"measurement_num__{c}" for c in num_agg.columns]

    wide_meas = text_agg.join(num_agg, how="outer").reset_index()

    ee_rows = meas[meas["measurement_type_norm"].str.contains("ee|encapsulation", na=False)].copy()
    if not ee_rows.empty:
        ee_meta = (
            ee_rows.groupby("formulation_core_id", dropna=False)
            .agg(
                ee_evidence_source=("value_source", _first_non_empty),
                ee_evidence_pointer=("trace_pointer", _first_non_empty),
                ee_evidence_excerpt=("evidence_excerpt", _first_non_empty),
                ee_rule_id=("rule_id", _first_non_empty),
            )
            .reset_index()
        )
        wide_meas = wide_meas.merge(ee_meta, on="formulation_core_id", how="left")

    df = core.merge(wide_meas, left_on=id_col, right_on="formulation_core_id", how="left")

    if "doi" not in df.columns:
        if "reference_normalized_doi" in df.columns:
            df["doi"] = df["reference_normalized_doi"].map(_normalize_space)
        elif "reference" in df.columns:
            df["doi"] = df["reference"].map(_normalize_space)
        else:
            df["doi"] = ""

    df["grouping_id"] = df[id_col].astype(str)
    return add_canonical_fields(df)


def build_from_wide(path: Path) -> pd.DataFrame:
    df = _read_tsv(path)
    if "doi" not in df.columns:
        for alt in ("doi_norm", "reference_normalized_doi", "reference"):
            if alt in df.columns:
                df["doi"] = df[alt].map(_normalize_space)
                break
    if "doi" not in df.columns:
        df["doi"] = ""

    if "grouping_id" not in df.columns:
        for alt in ("formulation_id", "formulation_core_id", "merged_instance_key"):
            if alt in df.columns:
                df["grouping_id"] = df[alt].astype(str)
                break
    if "grouping_id" not in df.columns:
        raise RuntimeError(f"Could not infer stable instance/grouping id from {path}")

    return add_canonical_fields(df)


def build_from_weak_and_assignment(weak_path: Path, ia_path: Path) -> pd.DataFrame:
    weak = _read_tsv(weak_path)
    ia = _read_tsv(ia_path)

    key_col = "key" if "key" in weak.columns else "zotero_key" if "zotero_key" in weak.columns else None
    if key_col is None:
        raise RuntimeError("weak_labels TSV missing key/zotero_key column.")
    if "formulation_id" not in weak.columns:
        raise RuntimeError("weak_labels TSV missing formulation_id column.")

    ia_key_col = "doc_key" if "doc_key" in ia.columns else "zotero_key" if "zotero_key" in ia.columns else "key" if "key" in ia.columns else None
    if ia_key_col is None or "formulation_id" not in ia.columns:
        raise RuntimeError("instance_assignment TSV missing key/formulation_id columns.")

    ia = ia.rename(columns={ia_key_col: "key"})
    weak = weak.rename(columns={key_col: "key"})

    group_id_col = "formulation_core_id" if "formulation_core_id" in ia.columns else "grouping_id" if "grouping_id" in ia.columns else None
    if group_id_col is None:
        raise RuntimeError("instance_assignment TSV missing formulation_core_id/grouping_id.")
    ia = ia[["key", "formulation_id", group_id_col]].drop_duplicates(["key", "formulation_id"], keep="first")
    ia = ia.rename(columns={group_id_col: "grouping_id"})

    merged = weak.merge(ia, on=["key", "formulation_id"], how="left")
    merged["grouping_id"] = merged["grouping_id"].astype(str).str.strip()
    merged = merged[merged["grouping_id"] != ""].copy()
    if merged.empty:
        raise RuntimeError("No rows mapped to grouping_id from instance_assignment TSV.")

    keep_first_cols = ["doi", "doi_norm", "drug_name", "plga_mw_kDa", "la_ga_ratio", "organic_solvent", "surfactant_name", "surfactant_concentration_text", "drug_feed_amount_text"]
    agg_spec = {c: _first_non_empty for c in keep_first_cols if c in merged.columns}
    if "encapsulation_efficiency_percent" in merged.columns:
        agg_spec["encapsulation_efficiency_percent"] = _first_non_empty
    if "size_nm" in merged.columns:
        agg_spec["size_nm"] = _first_non_empty
    if "pdi" in merged.columns:
        agg_spec["pdi"] = _first_non_empty

    grouped = merged.groupby("grouping_id", dropna=False).agg(agg_spec).reset_index()
    if "doi" not in grouped.columns and "doi_norm" in grouped.columns:
        grouped["doi"] = grouped["doi_norm"]
    if "doi" not in grouped.columns:
        grouped["doi"] = ""
    return add_canonical_fields(grouped)


def _resolve_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _resolve_measurement_by_suffix(df: pd.DataFrame, suffixes: Sequence[str], numeric_preferred: bool = True) -> Optional[str]:
    order: List[str] = []
    if numeric_preferred:
        order.extend([c for c in df.columns if c.startswith("measurement_num__")])
        order.extend([c for c in df.columns if c.startswith("measurement__")])
    else:
        order.extend([c for c in df.columns if c.startswith("measurement__")])
        order.extend([c for c in df.columns if c.startswith("measurement_num__")])
    for suf in suffixes:
        for c in order:
            if c.endswith(suf):
                return c
    return None


def add_canonical_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mapping = {
        "drug_name": ["drug_name", "drug_name_normalized", "sig__drug_name"],
        "polymer_identity": ["polymer_identity", "polymer_type_normalized", "polymer_type"],
        "plga_mw": ["plga_mw", "plga_mw_kDa", "polymer_mw_kDa", "sig__polymer_MW"],
        "la_ga_ratio": ["la_ga_ratio", "sig__LA/GA"],
        "drug_polymer_ratio": ["drug_polymer_ratio", "drug_to_polymer_mass_ratio", "sig__drug/polymer"],
        "surfactant_type": ["surfactant_type", "surfactant_type_normalized", "surfactant_name", "sig__surfactant_name"],
        "surfactant_conc": ["surfactant_conc", "surfactant_conc_percent", "surfactant_concentration_text", "sig__surfactant_concentration", "pva_conc_percent"],
        "organic_solvent": ["organic_solvent", "organic_solvent_normalized", "sig__solvent"],
        "size_nm": ["size_nm"],
        "pdi": ["pdi"],
        "ee_value": ["ee_value", "group_mean_EE", "encapsulation_efficiency_percent", "EE"],
    }

    for target, choices in mapping.items():
        if target in out.columns:
            continue
        found = _resolve_column(out, choices)
        if found is not None:
            out[target] = out[found]

    if "ee_value" not in out.columns:
        ee_col = _resolve_measurement_by_suffix(out, ["ee_percent", "encapsulation_efficiency_percent", "ee"], numeric_preferred=True)
        if ee_col is not None:
            out["ee_value"] = out[ee_col]
    if "size_nm" not in out.columns:
        size_col = _resolve_measurement_by_suffix(out, ["size_nm", "size"], numeric_preferred=True)
        if size_col is not None:
            out["size_nm"] = out[size_col]
    if "pdi" not in out.columns:
        pdi_col = _resolve_measurement_by_suffix(out, ["pdi"], numeric_preferred=True)
        if pdi_col is not None:
            out["pdi"] = out[pdi_col]

    return out


def classify_column(col: str) -> str:
    c = col.lower()
    if any(k in c for k in ["evidence", "qc", "support", "fail", "reason", "pointer", "trace"]):
        return "Evidence/QC"
    if any(k in c for k in ["_id", "doi", "key", "created", "schema", "signature", "audit", "version"]):
        return "Identity/Audit"
    if c.startswith("measurement__") or c.startswith("measurement_num__") or any(k in c for k in ["ee", "size", "pdi", "zeta", "release", "loading"]):
        return "Measurements"
    if any(k in c for k in ["solvent", "surfactant", "pva", "phase", "emul", "ph"]):
        return "Process"
    if any(k in c for k in ["drug", "polymer", "plga", "la_ga", "la_", "ga_"]):
        return "Core Formulation"
    return "Other"


def print_candidate_summary(candidates: Sequence[Candidate], chosen: Dict[str, object]) -> None:
    print("STEP 1: Candidate formulation-level tables")
    top5 = list(candidates[:5])
    print("Top 5 candidates by modified time:")
    for c in top5:
        ts = datetime.fromtimestamp(c.mtime).isoformat(timespec="seconds")
        print(f"- {c.path} | size={c.size_bytes} bytes | modified={ts} | kind={c.kind}")

    print("Chosen primary table path(s):")
    for k in ("formulation_core", "measurements", "wide_tsv", "weak_labels_tsv", "instance_assignment_tsv"):
        if k in chosen:
            print(f"- {k}: {chosen[k]}")
    print(f"Why chosen: {chosen.get('reason', '')}")
    print("")


def print_column_inventory(df: pd.DataFrame) -> None:
    print("STEP 3: Column inventory + missingness")
    n_rows = len(df)
    doi_nonempty = df["doi"] if "doi" in df.columns else pd.Series([""] * n_rows)
    n_dois = int(doi_nonempty.astype(str).str.strip().replace("", pd.NA).dropna().nunique())
    print(f"- total_rows: {n_rows}")
    print(f"- unique_dois: {n_dois}")

    buckets: Dict[str, List[str]] = {
        "Identity/Audit": [],
        "Core Formulation": [],
        "Process": [],
        "Measurements": [],
        "Evidence/QC": [],
        "Other": [],
    }
    for c in df.columns:
        buckets[classify_column(c)].append(c)

    print("Columns by section:")
    for section in ["Identity/Audit", "Core Formulation", "Process", "Measurements", "Evidence/QC", "Other"]:
        cols = sorted(buckets[section])
        print(f"- {section} ({len(cols)}):")
        print("  " + (", ".join(cols) if cols else "<none>"))

    stats = []
    for c in df.columns:
        miss = int(_missing_mask(df[c]).sum())
        rate = miss / n_rows if n_rows else 0.0
        stats.append((c, miss, n_rows - miss, rate))
    miss_df = pd.DataFrame(stats, columns=["column", "missing_count", "non_null_count", "missing_rate"]).sort_values(
        ["missing_rate", "column"], ascending=[False, True]
    )
    print("Top 30 worst missing rate:")
    print(miss_df.head(30).to_string(index=False))
    print("Top 30 best missing rate:")
    print(miss_df.tail(30).sort_values(["missing_rate", "column"], ascending=[True, True]).to_string(index=False))

    key_fields = [
        "drug_name",
        "polymer_identity",
        "plga_mw",
        "la_ga_ratio",
        "drug_polymer_ratio",
        "surfactant_type",
        "surfactant_conc",
        "organic_solvent",
        "ee_value",
        "size_nm",
        "pdi",
    ]
    print("Key modeling field coverage:")
    for f in key_fields:
        if f not in df.columns:
            print(f"- {f}: column_missing")
            continue
        non_null = int((~_missing_mask(df[f])).sum())
        rate = (non_null / n_rows) if n_rows else 0.0
        print(f"- {f}: non_null_rate={rate:.3f} ({non_null}/{n_rows})")
    print("")


def print_plga_check(df: pd.DataFrame) -> None:
    print("STEP 4: PLGA-only check")
    if "polymer_identity" not in df.columns:
        print("- WARNING: cannot enforce PLGA-only filter at formulation-level (polymer_identity missing)")
        print("")
        return

    s = df["polymer_identity"].astype(str).str.strip()
    s_nonempty = s[s != ""]
    vc = s_nonempty.value_counts(dropna=False)
    print("- polymer_identity unique values and counts:")
    if vc.empty:
        print("  <no non-empty values>")
    else:
        for k, v in vc.items():
            print(f"  {k}: {int(v)}")
    non_plga = int(s_nonempty.map(lambda x: "plga" not in x.lower()).sum())
    print(f"- non_PLGA_rows: {non_plga}")
    print("")


def print_evidence_qc_check(df: pd.DataFrame) -> None:
    print("STEP 5: Evidence/QC presence check (EE)")
    required = ["ee_evidence_source", "ee_evidence_pointer", "ee_support_level", "ee_fail_reason"]
    missing = []
    for c in required:
        exists = c in df.columns
        print(f"- {c}: {'present' if exists else 'missing'}")
        if not exists:
            missing.append(c)

    if missing:
        alt = [c for c in df.columns if any(k in c.lower() for k in ["evidence", "support", "fail", "reason", "pointer", "trace"])]
        print("- available evidence/qc-like columns:")
        if alt:
            print("  " + ", ".join(sorted(alt)))
        else:
            print("  <none>")
    print("")


def print_example_rows(df: pd.DataFrame) -> None:
    print("STEP 6: Example rows")
    fields = [
        "drug_name",
        "polymer_identity",
        "plga_mw",
        "la_ga_ratio",
        "drug_polymer_ratio",
        "surfactant_type",
        "surfactant_conc",
        "organic_solvent",
        "ee_value",
        "size_nm",
        "pdi",
    ]
    present = [c for c in fields if c in df.columns]
    id_cols = [c for c in ["doi", "grouping_id", "formulation_core_id", "formulation_id"] if c in df.columns]
    show_cols = id_cols + present
    if not show_cols:
        print("- No displayable columns for examples.")
        print("")
        return

    work = df.copy()
    for c in present:
        work[f"__nn__{c}"] = (~_missing_mask(work[c])).astype(int)
    work["completeness_score"] = work[[f"__nn__{c}" for c in present]].sum(axis=1) if present else 0

    top_complete = work.sort_values(["completeness_score"], ascending=[False]).head(5)
    print("- 5 most complete rows (by key modeling non-null count):")
    print(top_complete[show_cols + ["completeness_score"]].to_string(index=False))

    input_cols = [c for c in ["drug_name", "plga_mw", "la_ga_ratio", "drug_polymer_ratio", "surfactant_type", "surfactant_conc", "organic_solvent"] if c in work.columns]
    ee_col = "ee_value" if "ee_value" in work.columns else None
    if ee_col is None:
        print("- 5 high-risk rows: cannot compute (ee_value missing).")
        print("")
        return

    work["ee_present"] = (~_missing_mask(work[ee_col])).astype(int)
    work["inputs_present"] = work[[f"__nn__{c}" for c in input_cols]].sum(axis=1) if input_cols else 0
    work["inputs_missing"] = len(input_cols) - work["inputs_present"]
    risk_mask = ((work["ee_present"] == 1) & (work["inputs_missing"] >= 1)) | (
        (work["ee_present"] == 0) & (work["inputs_present"] >= max(1, min(4, len(input_cols))))
    )
    risk = work[risk_mask].copy()
    risk["risk_score"] = risk["inputs_missing"] + (1 - risk["ee_present"]) * 2
    print("- 5 high-risk rows (EE present with missing inputs OR inputs present with EE missing):")
    if risk.empty:
        print("  <none>")
    else:
        risk_top = risk.sort_values(["risk_score", "completeness_score"], ascending=[False, False]).head(5)
        print(risk_top[show_cols + ["inputs_present", "inputs_missing", "ee_present", "risk_score"]].to_string(index=False))
    print("")


def main() -> None:
    candidates = discover_candidates()
    if not candidates:
        raise RuntimeError("No candidate files found in required roots.")

    source_kind, chosen = choose_primary_source(candidates)
    print_candidate_summary(candidates, chosen)

    print("STEP 2: Build wide formulation view dataframe")
    if source_kind == "core_measure_pair":
        core_path = Path(str(chosen["formulation_core"]))
        meas_path = Path(str(chosen["measurements"]))
        view_df = build_from_core_measure(core_path, meas_path)
        print("- build_mode: formulation_core + measurements join on formulation_core_id")
    elif source_kind == "wide_tsv":
        view_df = build_from_wide(Path(str(chosen["wide_tsv"])))
        print("- build_mode: existing wide formulation-level TSV")
    elif source_kind == "weak_labels_with_assignment":
        view_df = build_from_weak_and_assignment(
            Path(str(chosen["weak_labels_tsv"])),
            Path(str(chosen["instance_assignment_tsv"])),
        )
        print("- build_mode: weak_labels aggregated by instance_assignment grouping_id")
    elif source_kind == "weak_labels_missing_assignment":
        print("- ERROR: weak_labels__gemini.tsv exists but no instance_assignment TSV found.")
        print("- Missing required file for aggregation: instance_assignment*.tsv")
        return
    else:
        raise RuntimeError(f"Unhandled source kind: {source_kind}")

    if "doi" not in view_df.columns:
        print("- WARNING: DOI column still missing after build.")
        view_df["doi"] = ""
    if "grouping_id" not in view_df.columns:
        raise RuntimeError("No stable instance/grouping id in final dataframe.")

    print(f"- final_rows: {len(view_df)}")
    print(f"- final_columns: {len(view_df.columns)}")
    print("")

    print_column_inventory(view_df)
    print_plga_check(view_df)
    print_evidence_qc_check(view_df)
    print_example_rows(view_df)


if __name__ == "__main__":
    main()
