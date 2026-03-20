#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd


CORE_SIGNATURE_FIELDS = [
    "polymer_type_canon",
    "la_ga_ratio_canon",
    "polymer_mw_kda_canon_or_iv",
    "vendor_product_code_canon",
    "drug_name_canon",
    "cargo_type_canon",
    "feed_anchor_canon",
    "organic_solvent_canon",
    "surfactant_name_canon",
]

EXTENDED_FIELDS = [
    "surfactant_concentration_canon",
    "phase_volumes_canon",
    "process_variables_canon",
    "ph_temp_canon",
]

CRITICAL_MISSING_FIELDS = [
    "missing_drug",
    "missing_polymer_identity",
    "missing_solvent",
    "missing_surfactant",
    "missing_feed_anchor",
]


def normalize_text(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_key_token(v: Any) -> str:
    s = normalize_text(v)
    if not s:
        return ""
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def normalize_doi(v: Any) -> str:
    s = normalize_text(v)
    s = re.sub(r"^doi\s*:\s*", "", s)
    s = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi\.org/", "", s)
    return s.strip()


def short_hash(s: str, n: int = 12) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def normalize_la_ga_ratio(v: Any) -> str:
    s = normalize_text(v)
    if not s:
        return ""
    s_no_space = s.replace(" ", "")
    m = re.match(r"^(\d{1,3})[:/](\d{1,3})$", s_no_space)
    if m:
        return f"{int(m.group(1))}:{int(m.group(2))}"
    m = re.match(r"^(\d{2})(\d{2})$", s_no_space)
    if m:
        return f"{int(m.group(1))}:{int(m.group(2))}"
    m = re.search(r"la\s*[:/=-]?\s*(\d{1,3}).*ga\s*[:/=-]?\s*(\d{1,3})", s, re.IGNORECASE)
    if m:
        return f"{int(m.group(1))}:{int(m.group(2))}"
    return s_no_space


def parse_numeric_first(v: Any) -> float | None:
    s = normalize_text(v)
    if not s:
        return None
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def format_num(v: float | None) -> str:
    if v is None:
        return ""
    if float(v).is_integer():
        return str(int(v))
    return f"{v:.6g}"


def normalize_mw_to_kda(v: Any) -> str:
    s = normalize_text(v)
    if not s:
        return ""
    num = parse_numeric_first(s)
    if num is None:
        return ""
    if "mda" in s:
        return format_num(num * 1000.0)
    if "kda" in s:
        return format_num(num)
    if re.search(r"\bda\b", s):
        return format_num(num / 1000.0)
    if re.search(r"\bg/mol\b|\bmol wt\b|\bmolecular weight\b", s):
        return format_num(num / 1000.0)
    if num > 10000:
        return format_num(num / 1000.0)
    return format_num(num)


def get_polymer_mw_raw(row: pd.Series) -> Any:
    for column in ["polymer_mw_kDa", "plga_mw_kDa"]:
        if column in row.index:
            value = row.get(column, "")
            if str(value).strip():
                return value
    return ""


def extract_intrinsic_viscosity(v: Any) -> str:
    s = normalize_text(v)
    if not s:
        return ""
    m = re.search(r"\b(?:iv|intrinsic viscosity)\b[^0-9]*([-+]?\d+(?:\.\d+)?)", s, re.IGNORECASE)
    if not m:
        return ""
    return format_num(float(m.group(1)))


def canonical_solvent(v: Any) -> Tuple[str, bool]:
    s = normalize_text(v)
    if not s:
        return "", False
    s_compact = s.replace(" ", "")
    mapping = [
        (r"\bdcm\b|dichloromethane|methylene chloride", "DCM"),
        (r"\bchloroform\b", "CHLOROFORM"),
        (r"\bacetone\b", "ACETONE"),
        (r"\bethyl acetate\b|\bea\b", "ETHYL_ACETATE"),
        (r"\bacetonitrile\b", "ACETONITRILE"),
        (r"\bdms[o0]\b|dimethyl sulfoxide", "DMSO"),
    ]
    for pattern, label in mapping:
        if re.search(pattern, s, flags=re.IGNORECASE):
            return label, False
    conflict = any(x in s_compact for x in ["/", "+", ",", ";"]) and "mixture" not in s
    return s.upper().replace(" ", "_"), conflict


def build_doc_solvent_consensus(df: pd.DataFrame) -> Dict[str, str]:
    values_by_doc: Dict[str, set[str]] = {}
    doc_has_conflict: Dict[str, bool] = {}
    for _, row in df.iterrows():
        doc_key = normalize_text(row.get("key", "") or row.get("zotero_key", ""))
        if not doc_key:
            continue
        canon, conflict = canonical_solvent(row.get("organic_solvent", ""))
        if conflict:
            doc_has_conflict[doc_key] = True
        if canon:
            values_by_doc.setdefault(doc_key, set()).add(canon)
    out: Dict[str, str] = {}
    for doc_key, vals in values_by_doc.items():
        if doc_has_conflict.get(doc_key, False):
            continue
        if len(vals) == 1:
            out[doc_key] = sorted(vals)[0]
    return out


def canonical_surfactant(name: Any, notes: Any, evidence: Any, pva_conc_percent: Any) -> Tuple[str, bool]:
    raw = " ".join([normalize_text(name), normalize_text(notes), normalize_text(evidence)])
    if not raw and normalize_text(pva_conc_percent):
        return "PVA", False
    mapping = [
        (r"\bpva\b|poly\s*\(?vinyl\)?\s*alcohol|polyvinyl alcohol", "PVA"),
        (r"poloxamer\s*188|pluronic\s*f[-\s]?68|\bf68\b", "POLOXAMER_188"),
        (r"poloxamer\s*407|pluronic\s*f[-\s]?127|\bf127\b", "POLOXAMER_407"),
        (r"tween\s*80|polysorbate\s*80", "TWEEN_80"),
        (r"tween\s*20|polysorbate\s*20", "TWEEN_20"),
        (r"\bsds\b|sodium dodecyl sulfate", "SDS"),
        (r"span\s*80", "SPAN_80"),
    ]
    labels: List[str] = []
    for pattern, label in mapping:
        if re.search(pattern, raw, flags=re.IGNORECASE):
            labels.append(label)
    labels = sorted(set(labels))
    if not labels:
        return "", False
    if len(labels) > 1 and "mixture" not in raw:
        return "+".join(labels), True
    return "+".join(labels), False


def canonical_polymer_type(v: Any, notes: Any, evidence: Any) -> str:
    raw = " ".join([normalize_text(v), normalize_text(notes), normalize_text(evidence)])
    if not raw:
        return "PLGA"
    if re.search(r"plga[\s\-]*peg|peg[\s\-]*plga", raw):
        return "PLGA-PEG"
    if re.search(
        r"\bplga\b"
        r"|poly\s*\(\s*d\s*,\s*l\s*-\s*lactide\s*-\s*co\s*-\s*glycolide\s*\)"
        r"|poly\s*\(\s*dl\s*-\s*lactide\s*-\s*co\s*-\s*glycolide\s*\)"
        r"|polylactide\s*-\s*co\s*-\s*glycolide"
        r"|poly\s*\(\s*lactic\s*-\s*co\s*-\s*glycolic\s*acid\s*\)",
        raw,
        flags=re.IGNORECASE,
    ):
        return "PLGA"
    return normalize_key_token(v).upper() if normalize_key_token(v) else ""


def canonical_drug_name(v: Any) -> str:
    s = normalize_text(v)
    if not s:
        return ""
    s = re.sub(r"[^a-z0-9\-\s/]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.upper().replace(" ", "_")


def parse_mass_from_text_mg(v: Any) -> str:
    s = normalize_text(v)
    if not s:
        return ""
    m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(mg|g|ug|μg)?", s)
    if not m:
        return ""
    val = float(m.group(1))
    unit = m.group(2) or "mg"
    unit = unit.lower().replace("μ", "u")
    if unit == "g":
        val *= 1000.0
    elif unit == "ug":
        val /= 1000.0
    return format_num(val)


def build_feed_anchor(row: pd.Series) -> str:
    ratio_candidates = [
        row.get("drug_to_polymer_ratio"),
        row.get("drug_to_polymer_mass_ratio"),
        row.get("drug/polymer"),
        row.get("drug_polymer_ratio"),
    ]
    for cand in ratio_candidates:
        val = normalize_text(cand)
        if val:
            return f"ratio:{val.replace(' ', '')}"

    drug_mass = parse_mass_from_text_mg(row.get("drug_feed_mass_mg", ""))
    if not drug_mass:
        drug_mass = parse_mass_from_text_mg(row.get("drug_feed_amount_text", ""))
    polymer_mass = parse_mass_from_text_mg(row.get("polymer_mass_mg", ""))
    if not polymer_mass:
        polymer_mass = parse_mass_from_text_mg(row.get("plga_mass_mg", ""))

    if drug_mass and polymer_mass:
        return f"mass_pair:{drug_mass}:{polymer_mass}"
    if drug_mass:
        return f"drug_mass:{drug_mass}"
    if polymer_mass:
        return f"polymer_mass:{polymer_mass}"
    return ""


def build_extended_signature(row: pd.Series) -> Dict[str, str]:
    surf_conc = normalize_text(
        row.get("surfactant_concentration_text", "")
        or row.get("surfactant_concentration", "")
        or row.get("pva_conc_percent", "")
    )
    phase = " ".join(
        [
            normalize_text(row.get("aqueous_phase_volume", "")),
            normalize_text(row.get("organic_phase_volume", "")),
            normalize_text(row.get("phase_ratio_w1_o", "")),
            normalize_text(row.get("phase_ratio_total_water_o", "")),
        ]
    ).strip()
    process = " ".join(
        [
            normalize_text(row.get("emul_method", "")),
            normalize_text(row.get("sonication_time", "")),
            normalize_text(row.get("stirring_speed", "")),
        ]
    ).strip()
    ph_temp = " ".join(
        [
            normalize_text(row.get("aqueous_phase_pH", "")),
            normalize_text(row.get("temperature_c", "")),
        ]
    ).strip()
    return {
        "surfactant_concentration_canon": surf_conc,
        "phase_volumes_canon": phase,
        "process_variables_canon": process,
        "ph_temp_canon": ph_temp,
    }


def explicit_formulation_label(v: Any) -> str:
    token = normalize_key_token(v).upper()
    if not token:
        return ""
    if re.match(r"^(F|NP|FORM|FORMULATION|BATCH|RUN)[_-]?\d+$", token):
        return token
    return ""


def detect_anchor_from_table_row(row: pd.Series) -> str:
    table_cols = ["table_id", "evidence_table_id", "source_table_id"]
    row_cols = ["row_index", "evidence_row_index", "source_row_index", "table_row_index"]
    table_id = ""
    row_idx = ""
    for c in table_cols:
        val = normalize_text(row.get(c, ""))
        if val:
            table_id = val
            break
    for c in row_cols:
        val = normalize_text(row.get(c, ""))
        if val:
            row_idx = val
            break
    if table_id and row_idx:
        return f"{table_id}::{row_idx}"
    return ""


def evidence_ref(row: pd.Series) -> str:
    parts = [
        normalize_text(row.get("evidence_section", "")),
        normalize_text(row.get("evidence_span_start", "")),
        normalize_text(row.get("evidence_span_end", "")),
        normalize_text(row.get("evidence_method", "")),
        normalize_text(row.get("evidence_quality", "")),
    ]
    joined = "|".join(parts).strip("|")
    return joined if joined else ""


@dataclass
class BuildOutputs:
    core_df: pd.DataFrame
    assignment_df: pd.DataFrame
    trace_df: pd.DataFrame
    build_log: Dict[str, Any]


def _parse_doe_factor_field_name(field_name: Any) -> Tuple[str, str]:
    s = str(field_name or "").strip()
    if not s:
        return "", ""
    lower = s.lower()
    if "doe" not in lower or "factor" not in lower:
        return "", ""
    m = re.match(r"^doe_factor::(.+?)::(coded|decoded)$", lower)
    if not m:
        m = re.match(r"^doe_factor[:_]{1,2}(.+?)[:_]{1,2}(coded|decoded)$", lower)
    if not m:
        return "", ""
    factor = normalize_key_token(m.group(1))
    kind = m.group(2)
    return factor, kind


def _build_doe_signature_maps(derived_values_df: pd.DataFrame | None) -> Dict[str, Dict[str, str]]:
    if derived_values_df is None or derived_values_df.empty:
        return {}
    dfx = derived_values_df.copy().fillna("")
    required = {"field_name", "value"}
    if not required.issubset(set(dfx.columns)):
        return {}

    out: Dict[str, Dict[str, str]] = {}
    for _, row in dfx.iterrows():
        factor, kind = _parse_doe_factor_field_name(row.get("field_name", ""))
        if not factor or not kind:
            continue
        group_key = normalize_text(row.get("group_key", ""))
        if not group_key:
            key = normalize_text(row.get("key", ""))
            formulation_id = normalize_text(row.get("formulation_id", ""))
            if key:
                group_key = f"{key}::{formulation_id}"
        if not group_key:
            continue

        val = str(row.get("value", "")).strip().replace("−", "-")
        if not val:
            continue

        slot = out.setdefault(group_key, {"coded": {}, "decoded": {}})
        slot[kind][factor] = val

    sig_map: Dict[str, Dict[str, str]] = {}
    for group_key, payload in out.items():
        coded: Dict[str, str] = payload.get("coded", {})  # type: ignore[assignment]
        decoded: Dict[str, str] = payload.get("decoded", {})  # type: ignore[assignment]
        merged: Dict[str, str] = {}
        for name in sorted(set(list(coded.keys()) + list(decoded.keys()))):
            if name in coded:
                merged[name] = str(coded[name])
            elif name in decoded:
                merged[name] = f"DEC:{decoded[name]}"
        if not merged:
            continue
        sig = "|".join([f"{k}={merged[k]}" for k in sorted(merged.keys())])
        sig_map[group_key] = {
            "doe_signature_canon": sig,
            "doe_signature_source": "derived_doe_decode",
        }
    return sig_map


def build_formulation_core_signature_v1(
    df: pd.DataFrame,
    run_id: str,
    input_tsv: str,
    derived_values_df: pd.DataFrame | None = None,
    derived_values_path: str = "",
) -> BuildOutputs:
    if df.empty:
        empty_cols = [
            "formulation_core_id",
            "signature_string",
            "signature_hash",
            "signature_quality",
            "merge_risk_level",
        ] + CORE_SIGNATURE_FIELDS + [
            "doe_signature_canon",
            "doe_signature_source",
        ] + EXTENDED_FIELDS + [
            "canonical_components_json",
            "provenance_map_json",
            "risk_reasons",
            "n_instances",
            "unresolved_group",
        ]
        return BuildOutputs(
            core_df=pd.DataFrame(columns=empty_cols),
            assignment_df=pd.DataFrame(),
            trace_df=pd.DataFrame(),
            build_log={
                "run_id": run_id,
                "input_tsv": input_tsv,
                "n_instances": 0,
                "n_cores": 0,
                "auto_merged_count": 0,
                "unresolved_count": 0,
                "count_rows_with_doe_signature": 0,
                "count_cores_with_doe_signature": 0,
                "doe_trace_enabled": False,
                "derived_values_path": derived_values_path,
                "top_risk_reasons": [],
            },
        )

    dfx = df.copy().fillna("")
    if "key" not in dfx.columns:
        if "zotero_key" in dfx.columns:
            dfx["key"] = dfx["zotero_key"]
        else:
            raise ValueError("input TSV must contain `key` or `zotero_key`")
    if "formulation_id" not in dfx.columns:
        dfx["formulation_id"] = ""
    dfx["instance_id"] = [f"inst_{i+1:06d}" for i in range(len(dfx))]
    doc_solvent_consensus = build_doc_solvent_consensus(dfx)
    doe_sig_map = _build_doe_signature_maps(derived_values_df=derived_values_df)

    prep_rows: List[Dict[str, Any]] = []
    unknown_solvents: Dict[str, int] = {}
    unknown_surfactants: Dict[str, int] = {}
    for _, row in dfx.iterrows():
        notes = row.get("notes", "")
        evidence = row.get("evidence_span_text", "")
        doc_key_norm = normalize_text(row.get("key", "") or row.get("zotero_key", ""))
        solvent_raw = row.get("organic_solvent", "")
        solvent_canon, solvent_conflict = canonical_solvent(solvent_raw)
        solvent_inherited_from_doc = 0
        if not solvent_canon and doc_key_norm:
            inherited = doc_solvent_consensus.get(doc_key_norm, "")
            if inherited:
                solvent_canon = inherited
                solvent_inherited_from_doc = 1
        if solvent_raw and solvent_canon and solvent_canon == normalize_text(solvent_raw).upper().replace(" ", "_"):
            unknown_solvents[solvent_canon] = unknown_solvents.get(solvent_canon, 0) + 1

        surf_raw = row.get("surfactant_name", "")
        surf_canon, surf_conflict = canonical_surfactant(surf_raw, notes, evidence, row.get("pva_conc_percent", ""))
        if surf_raw and not surf_canon:
            key = normalize_text(surf_raw)
            unknown_surfactants[key] = unknown_surfactants.get(key, 0) + 1

        polymer_type_canon = canonical_polymer_type(row.get("polymer_type", ""), notes, evidence)
        la_ga_ratio_canon = normalize_la_ga_ratio(row.get("la_ga_ratio", ""))
        mw_kda = normalize_mw_to_kda(get_polymer_mw_raw(row))
        iv = extract_intrinsic_viscosity(row.get("intrinsic_viscosity", ""))
        mw_or_iv = mw_kda if mw_kda else (f"IV:{iv}" if iv else "")
        vendor_code = normalize_key_token(row.get("vendor_product_code", "")).upper()
        drug_canon = canonical_drug_name(row.get("drug_name", ""))
        cargo_type = normalize_key_token(row.get("cargo_type", "")).upper()
        feed_anchor = build_feed_anchor(row)
        group_key = f"{normalize_text(row.get('key', ''))}::{normalize_text(row.get('formulation_id', ''))}"
        doe_payload = doe_sig_map.get(group_key, {})
        doe_signature_canon = str(doe_payload.get("doe_signature_canon", ""))
        doe_signature_source = str(doe_payload.get("doe_signature_source", "")) if doe_signature_canon else ""

        core_components = {
            "polymer_type_canon": polymer_type_canon,
            "la_ga_ratio_canon": la_ga_ratio_canon,
            "polymer_mw_kda_canon_or_iv": mw_or_iv,
            "vendor_product_code_canon": vendor_code,
            "drug_name_canon": drug_canon,
            "cargo_type_canon": cargo_type,
            "feed_anchor_canon": feed_anchor,
            "organic_solvent_canon": solvent_canon,
            "surfactant_name_canon": surf_canon,
            "doe_signature_canon": doe_signature_canon,
        }
        extended_components = build_extended_signature(row)
        all_components = {**core_components, **extended_components}

        signature_parts = [f"{k}={all_components.get(k, '')}" for k in CORE_SIGNATURE_FIELDS]
        if doe_signature_canon:
            signature_parts.append(f"doe_signature_canon={doe_signature_canon}")
        signature_string = " | ".join(signature_parts)
        signature_hash = short_hash(signature_string, 12)

        missing_drug = not bool(drug_canon)
        missing_polymer_identity = not bool(polymer_type_canon and (la_ga_ratio_canon or mw_or_iv or vendor_code))
        missing_solvent = not bool(solvent_canon)
        missing_surfactant = not bool(surf_canon)
        missing_feed_anchor = not bool(feed_anchor)
        critical_missing = {
            "missing_drug": missing_drug,
            "missing_polymer_identity": missing_polymer_identity,
            "missing_solvent": missing_solvent,
            "missing_surfactant": missing_surfactant,
            "missing_feed_anchor": missing_feed_anchor,
        }
        core_fields_present_count = sum(1 for k in CORE_SIGNATURE_FIELDS if all_components.get(k, ""))
        critical_missing_count = sum(1 for v in critical_missing.values() if v)
        complete_enough = critical_missing_count == 0

        extended_conflicts = []
        if solvent_conflict:
            extended_conflicts.append("multi_solvent_token_conflict")
        if surf_conflict:
            extended_conflicts.append("multi_surfactant_token_conflict")

        if critical_missing_count >= 3 or extended_conflicts:
            risk = "high"
        elif critical_missing_count >= 1:
            risk = "medium"
        else:
            risk = "low"

        anchor_label = explicit_formulation_label(row.get("formulation_id", ""))
        table_anchor = detect_anchor_from_table_row(row)
        doc_anchor_base = normalize_key_token(row.get("key", "")) or normalize_key_token(row.get("zotero_key", ""))
        gate_a_anchor = ""
        if anchor_label:
            gate_a_anchor = f"label::{doc_anchor_base}::{anchor_label}"
        elif table_anchor:
            gate_a_anchor = f"tablerow::{doc_anchor_base}::{table_anchor}"

        prep_rows.append(
            {
                "instance_id": row["instance_id"],
                "doc_key": row.get("key", ""),
                "formulation_id": row.get("formulation_id", ""),
                "model": row.get("model", ""),
                "signature_string": signature_string,
                "signature_hash": signature_hash,
                "core_fields_present_count": core_fields_present_count,
                "critical_missing_count": critical_missing_count,
                "complete_enough": int(complete_enough),
                "merge_risk_level": risk,
                "critical_missing_json": json.dumps(critical_missing, ensure_ascii=False),
                "extended_conflicts": "|".join(extended_conflicts),
                "gate_a_anchor": gate_a_anchor,
                "evidence_ref": evidence_ref(row),
                "canonical_components_json": json.dumps(all_components, ensure_ascii=False, sort_keys=True),
                "doe_signature_canon": doe_signature_canon,
                "doe_signature_source": doe_signature_source,
                "solvent_inherited_from_doc": int(solvent_inherited_from_doc),
                **all_components,
                **critical_missing,
            }
        )

    prep = pd.DataFrame(prep_rows).sort_values(["doc_key", "instance_id"]).reset_index(drop=True)

    core_id_counter = 0
    core_by_gate_a: Dict[str, str] = {}
    core_by_gate_b: Dict[str, str] = {}
    assignments: List[Dict[str, Any]] = []

    for _, row in prep.iterrows():
        gate_used = "C"
        merge_reason = "no_auto_merge_incomplete_or_no_anchor"
        unresolved_group = 1
        doc = normalize_key_token(row["doc_key"])
        core_id = ""

        gate_a_anchor = row.get("gate_a_anchor", "")
        if gate_a_anchor:
            if gate_a_anchor not in core_by_gate_a:
                core_id_counter += 1
                core_by_gate_a[gate_a_anchor] = f"FC1_{core_id_counter:05d}"
            core_id = core_by_gate_a[gate_a_anchor]
            gate_used = "A"
            merge_reason = "anchor_merge_by_formulation_label_or_table_row"
            unresolved_group = 0
        elif int(row.get("complete_enough", 0)) == 1:
            gate_b_key = f"{doc}::{row['signature_hash']}"
            if gate_b_key not in core_by_gate_b:
                core_id_counter += 1
                core_by_gate_b[gate_b_key] = f"FC1_{core_id_counter:05d}"
            core_id = core_by_gate_b[gate_b_key]
            gate_used = "B"
            merge_reason = "strict_merge_exact_core_signature"
            unresolved_group = 0
        else:
            core_id_counter += 1
            core_id = f"FC1_{core_id_counter:05d}"

        assignments.append(
            {
                "instance_id": row["instance_id"],
                "doc_key": row["doc_key"],
                "formulation_id": row["formulation_id"],
                "model": row["model"],
                "formulation_core_id": core_id,
                "gate_used": gate_used,
                "merge_reason": merge_reason,
                "signature_hash": row["signature_hash"],
                "signature_string": row["signature_string"],
                "merge_risk_level": row["merge_risk_level"],
                "unresolved_group": unresolved_group,
                "critical_missing_json": row["critical_missing_json"],
                "extended_conflicts": row["extended_conflicts"],
                "core_fields_present_count": int(row["core_fields_present_count"]),
                "critical_missing_count": int(row["critical_missing_count"]),
                "gate_a_anchor": row["gate_a_anchor"],
                "evidence_ref": row["evidence_ref"],
                "canonical_components_json": row["canonical_components_json"],
                "doe_signature_canon": row["doe_signature_canon"],
                "doe_signature_source": row["doe_signature_source"],
            }
        )

    assign_df = pd.DataFrame(assignments)
    merged = assign_df.merge(prep, on="instance_id", how="left", suffixes=("", "_prep"))

    core_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    risk_reason_counter: Dict[str, int] = {}

    for core_id, gdf in merged.groupby("formulation_core_id", sort=True):
        first = gdf.iloc[0]
        # Core-level provenance map: field -> evidence refs contributing to that field.
        provenance_map: Dict[str, List[str]] = {}
        for field in CORE_SIGNATURE_FIELDS:
            refs = sorted(set([r for r in gdf["evidence_ref"].astype(str).tolist() if r]))
            provenance_map[field] = refs

        # Extended conflicts across instances inside a core.
        core_extended_conflicts: List[str] = []
        for ef in EXTENDED_FIELDS:
            vals = sorted(set([normalize_text(v) for v in gdf[ef].tolist() if normalize_text(v)]))
            if len(vals) > 1:
                core_extended_conflicts.append(f"{ef}_multi_value")

        critical_missing_counts = {
            k: int(gdf[k].astype(bool).sum())
            for k in CRITICAL_MISSING_FIELDS
        }
        missing_ratio = sum(critical_missing_counts.values())
        unresolved_core = int(gdf["unresolved_group"].max())

        risk_reasons: List[str] = []
        if unresolved_core:
            risk_reasons.append("unresolved_group")
        for k in CRITICAL_MISSING_FIELDS:
            if critical_missing_counts[k] > 0:
                risk_reasons.append(k)
        risk_reasons.extend(core_extended_conflicts)
        if not risk_reasons:
            risk_reasons.append("none")

        for reason in risk_reasons:
            risk_reason_counter[reason] = risk_reason_counter.get(reason, 0) + 1

        if unresolved_core or missing_ratio >= 2 or core_extended_conflicts:
            core_risk = "high"
        elif missing_ratio == 1:
            core_risk = "medium"
        else:
            core_risk = "low"

        quality_payload = {
            "core_fields_present_count": int(first["core_fields_present_count"]),
            "critical_missing": {k: bool(first[k]) for k in CRITICAL_MISSING_FIELDS},
            "critical_missing_count": int(first["critical_missing_count"]),
            "extended_conflicts": core_extended_conflicts,
        }

        core_rows.append(
            {
                "formulation_core_id": core_id,
                "signature_string": first["signature_string"],
                "signature_hash": first["signature_hash"],
                "signature_quality": json.dumps(quality_payload, ensure_ascii=False, sort_keys=True),
                "merge_risk_level": core_risk,
                **{k: first[k] for k in CORE_SIGNATURE_FIELDS},
                "doe_signature_canon": first["doe_signature_canon"],
                "doe_signature_source": first["doe_signature_source"],
                **{k: first[k] for k in EXTENDED_FIELDS},
                "canonical_components_json": first["canonical_components_json"],
                "provenance_map_json": json.dumps(provenance_map, ensure_ascii=False, sort_keys=True),
                "risk_reasons": "|".join(sorted(set(risk_reasons))),
                "n_instances": int(len(gdf)),
                "unresolved_group": unresolved_core,
            }
        )

        for _, r in gdf.sort_values("instance_id").iterrows():
            trace_rows.append(
                {
                    "instance_id": r["instance_id"],
                    "formulation_core_id": core_id,
                    "gate_used": r["gate_used"],
                    "merge_reason": r["merge_reason"],
                    "doc_key": r["doc_key"],
                    "formulation_id": r["formulation_id"],
                    "signature_hash": r["signature_hash"],
                    "signature_string": r["signature_string"],
                    "merge_risk_level": r["merge_risk_level"],
                    "critical_missing_json": r["critical_missing_json"],
                    "extended_conflicts": r["extended_conflicts"],
                    "gate_a_anchor": r["gate_a_anchor"],
                    "evidence_ref": r["evidence_ref"],
                    "canonical_components_json": r["canonical_components_json"],
                    "doe_signature_canon": r["doe_signature_canon"],
                    "doe_signature_source": r["doe_signature_source"],
                }
            )

    core_df = pd.DataFrame(core_rows).sort_values("formulation_core_id").reset_index(drop=True)
    trace_df = pd.DataFrame(trace_rows).sort_values(["formulation_core_id", "instance_id"]).reset_index(drop=True)
    assignment_df = assign_df.sort_values("instance_id").reset_index(drop=True)

    auto_merged_count = int((assignment_df["gate_used"].isin(["A", "B"])).sum())
    unresolved_count = int((assignment_df["unresolved_group"] == 1).sum())
    count_rows_with_doe_signature = int(assignment_df["doe_signature_canon"].astype(str).str.strip().ne("").sum()) if "doe_signature_canon" in assignment_df.columns else 0
    count_cores_with_doe_signature = int(core_df["doe_signature_canon"].astype(str).str.strip().ne("").sum()) if "doe_signature_canon" in core_df.columns else 0
    top_risk_reasons = sorted(risk_reason_counter.items(), key=lambda x: (-x[1], x[0]))[:10]

    build_log = {
        "run_id": run_id,
        "input_tsv": input_tsv,
        "n_instances": int(len(assignment_df)),
        "n_cores": int(len(core_df)),
        "auto_merged_count": auto_merged_count,
        "unresolved_count": unresolved_count,
        "count_rows_with_doe_signature": count_rows_with_doe_signature,
        "count_cores_with_doe_signature": count_cores_with_doe_signature,
        "doe_trace_enabled": bool(derived_values_df is not None and not derived_values_df.empty),
        "derived_values_path": derived_values_path,
        "gate_counts": assignment_df["gate_used"].value_counts().sort_index().to_dict(),
        "risk_counts": assignment_df["merge_risk_level"].value_counts().sort_index().to_dict(),
        "top_risk_reasons": [{"reason": r, "count": int(c)} for r, c in top_risk_reasons],
        "unknown_solvent_tokens_top10": sorted(unknown_solvents.items(), key=lambda x: (-x[1], x[0]))[:10],
        "unknown_surfactant_tokens_top10": sorted(unknown_surfactants.items(), key=lambda x: (-x[1], x[0]))[:10],
    }

    return BuildOutputs(
        core_df=core_df,
        assignment_df=assignment_df,
        trace_df=trace_df,
        build_log=build_log,
    )


def _self_test() -> None:
    rows = [
        {
            "key": "DOC1",
            "formulation_id": "F1",
            "la_ga_ratio": "50/50",
            "polymer_mw_kDa": "30 kDa",
            "organic_solvent": "dichloromethane",
            "drug_name": "Doxorubicin",
            "drug_feed_amount_text": "5 mg",
            "plga_mass_mg": "50",
            "pva_conc_percent": "1.0",
            "notes": "poly(vinyl alcohol)",
            "evidence_span_text": "F1 uses DCM and PVA",
        },
        {
            "key": "DOC1",
            "formulation_id": "F1",
            "la_ga_ratio": "50:50",
            "polymer_mw_kDa": "30000 Da",
            "organic_solvent": "DCM",
            "drug_name": "doxorubicin",
            "drug_feed_amount_text": "5mg",
            "plga_mass_mg": "50 mg",
            "pva_conc_percent": "1",
            "notes": "PVA",
            "evidence_span_text": "same formulation",
        },
        {
            "key": "DOC1",
            "formulation_id": "",
            "la_ga_ratio": "",
            "polymer_mw_kDa": "",
            "organic_solvent": "",
            "drug_name": "",
            "notes": "",
            "evidence_span_text": "",
        },
    ]
    out = build_formulation_core_signature_v1(pd.DataFrame(rows), "run_test", "in.tsv")
    assert len(out.assignment_df) == 3
    assert out.assignment_df["formulation_core_id"].nunique() == 2
    assert int(out.build_log["unresolved_count"]) == 1
