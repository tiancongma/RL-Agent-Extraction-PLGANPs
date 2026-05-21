#!/usr/bin/env python3
from __future__ import annotations

"""
Build Stage5 Evidence Binding Packs.

The builder consumes frozen authority-resolved artifacts and emits audit sidecar
packs. It does not create rows, create values, replace the final table, assign
risk levels, or render workbooks.
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from src.utils.paths import PROJECT_ROOT
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.paths import PROJECT_ROOT


BINDING_STATUSES = {
    "direct_supported",
    "relation_supported",
    "derived_supported",
    "identity_only_match",
    "value_only_match",
    "ambiguous_assignment",
    "unresolved_table",
    "unsupported_text",
    "blank_value",
    "conflict",
    "normalization_pending",
    "relation_path_missing",
    "missing_evidence_anchor",
    "missing_exact_value_evidence",
    "derived_without_direct_text",
    "raw_value_supported_normalization_pending",
    "coded_value_supported_decode_pending",
    "role_tolerant_supported",
    "source_surface_missing",
    "authority_alias_conflict",
}

ASSIGNMENT_PATHS = {
    "direct_same_table_row",
    "stage3_relation_resolved",
    "parent_inheritance",
    "shared_method_context",
    "selection_marker",
    "doe_factor_decode",
    "sequential_optimization_link",
    "derived_from_row_values",
    "blank_value",
    "unresolved",
}

DEFAULT_EXCLUDE_FIELDS = {
    "model",
    "doi",
    "key",
    "zotero_key",
    "review_needed",
}

IDENTITY_FIELDS = {
    "final_formulation_id",
    "representative_source_formulation_id",
    "representative_source_raw_formulation_label",
    "formulation_id",
    "raw_formulation_label",
    "local_instance_id",
    "family_id",
    "parent_core_row_id",
    "variant_role",
    "payload_state",
    "benchmark_default_include",
}

DERIVED_HINT_FIELDS = {
    "derived_mass_provenance_json",
    "polymer_mass_mg_source",
    "drug_mass_mg_source",
}

SUMMARY_FIELDS = ["group_key", "binding_status", "assignment_path", "count"]

VALUE_EXACT_EVIDENCE_FIELD_HINTS = {
    "size_nm",
    "pdi",
    "zeta_mV",
    "encapsulation_efficiency_percent",
    "loading_content_percent",
    "dl_percent",
    "la_ga_ratio",
    "polymer_mw_kDa",
    "plga_mass_mg",
    "organic_phase_volume_mL",
    "external_aqueous_phase_volume_mL",
    "surfactant_concentration_value",
}


try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:  # pragma: no cover - platform dependent
    csv.field_size_limit(10_000_000)


def _norm(value: Any) -> str:
    return str(value or "").strip()


def _first_nonempty(row: dict[str, str], fields: list[str]) -> str:
    for field in fields:
        value = _norm(row.get(field))
        if value:
            return value
    return ""


def _contains_token_surface(haystack: str, needle: str) -> bool:
    haystack_norm = haystack.lower()
    needle_norm = needle.lower()
    return bool(needle_norm and (needle_norm in haystack_norm or haystack_norm in needle_norm))


def base_field_name(field_name: str) -> str:
    for suffix in (
        "_value",
        "_value_text",
        "_scope",
        "_membership_confidence",
        "_evidence_region_type",
        "_missing_reason",
    ):
        if field_name.endswith(suffix):
            return field_name[: -len(suffix)]
    return field_name


def companion_field(row: dict[str, str], field_name: str, suffix: str) -> str:
    base = base_field_name(field_name)
    return _norm(row.get(f"{base}_{suffix}"))


def first_supporting_ref(row: dict[str, str]) -> dict[str, Any]:
    refs = _norm(row.get("supporting_evidence_refs"))
    if not refs:
        return {}
    try:
        payload = json.loads(refs)
    except json.JSONDecodeError:
        return {}
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0]
    if isinstance(payload, dict):
        return payload
    return {}


def has_field_locator(row: dict[str, str], field_name: str) -> bool:
    return bool(
        companion_field(row, field_name, "value_text")
        or companion_field(row, field_name, "evidence_region_type")
        or companion_field(row, field_name, "scope")
    )


def field_scope_assignment_path(row: dict[str, str], field_name: str) -> str:
    scope_text = " ".join(
        [
            companion_field(row, field_name, "scope"),
            companion_field(row, field_name, "evidence_region_type"),
        ]
    ).lower()
    if any(token in scope_text for token in ("global", "shared", "inherited", "preparation", "method")):
        return "shared_method_context"
    return "direct_same_table_row"


def source_surface_type(row: dict[str, str], field_name: str, relation: dict[str, str] | None) -> str:
    if relation is not None:
        relation_type = _norm(relation.get("evidence_source_type")) or _norm(relation.get("source_type"))
        if relation_type:
            return relation_type
        if _norm(relation.get("source_table_id")) or _norm(relation.get("table_id")):
            return "table"
        return "stage3_relation"
    region = companion_field(row, field_name, "evidence_region_type")
    if region:
        return region
    if _norm(row.get("table_id")) or _norm(row.get("table_row_id")):
        return "table"
    if _norm(row.get("evidence_span_text")):
        return "text"
    return "unknown"


def source_label(row: dict[str, str], relation: dict[str, str] | None) -> str:
    if relation is not None:
        return _first_nonempty(
            relation,
            ["evidence_section", "source_table_id", "table_id", "source_section", "evidence_source_id"],
        )
    return _first_nonempty(row, ["table_id", "evidence_section", "instance_evidence_region_type"])


def source_locator_text(row: dict[str, str], field_name: str, relation: dict[str, str] | None) -> str:
    if relation is not None:
        pieces = [
            _first_nonempty(relation, ["source_table_id", "table_id", "evidence_section"]),
            _first_nonempty(relation, ["source_table_row_id", "table_row_id", "source_row_id"]),
            _first_nonempty(relation, ["field_name", "target_field_name"]),
            _first_nonempty(relation, ["source_relation_row_ids", "relation_record_ids"]),
        ]
    else:
        pieces = [
            _first_nonempty(row, ["table_id", "evidence_section"]),
            _first_nonempty(row, ["table_row_id"]),
            base_field_name(field_name),
        ]
        ref = first_supporting_ref(row)
        if not pieces[0] and _norm(ref.get("source_region_type")):
            pieces[0] = _norm(ref.get("source_region_type"))
        if not pieces[1] and _norm(ref.get("source_locator_text")):
            pieces[1] = _norm(ref.get("source_locator_text"))
    return " | ".join(piece for piece in pieces if piece)


def value_evidence_text(row: dict[str, str], field_name: str, relation: dict[str, str] | None, value: str) -> str:
    if relation is not None:
        return _first_nonempty(
            relation,
            ["evidence_text", "evidence_span_text", "source_text", "source_cell_text", "field_value"],
        )
    companion_text = companion_field(row, field_name, "value_text")
    if companion_text:
        return companion_text
    span = _norm(row.get("evidence_span_text"))
    if span:
        return span[:500]
    ref = first_supporting_ref(row)
    snippet = _norm(ref.get("supporting_snippet"))
    if snippet:
        return snippet[:500]
    return ""


def row_identity_evidence_text(row: dict[str, str], relation: dict[str, str] | None) -> str:
    if relation is not None:
        return _first_nonempty(
            relation,
            ["row_identity_evidence_text", "source_row_text", "source_cell_text", "evidence_text", "evidence_span_text"],
        )[:500]
    for field in ("row_identity_evidence_text", "instance_evidence_text", "evidence_span_text"):
        value = _norm(row.get(field))
        if value:
            return value[:500]
    ref = first_supporting_ref(row)
    return _norm(ref.get("supporting_snippet"))[:500]


def source_cell_text(row: dict[str, str], relation: dict[str, str] | None) -> str:
    if relation is not None:
        return _first_nonempty(relation, ["source_cell_text", "field_value", "evidence_text"])[:500]
    return _first_nonempty(row, ["source_cell_text", "cell_text", "evidence_span_text"])[:500]


def source_row_label(row: dict[str, str], relation: dict[str, str] | None) -> str:
    if relation is not None:
        return _first_nonempty(relation, ["source_table_row_id", "table_row_id", "source_row_id", "source_row_label"])
    return _first_nonempty(row, ["source_table_row_id", "table_row_id", "source_row_id", "raw_formulation_label", "local_instance_id"])


def source_column_label(row: dict[str, str], field_name: str, relation: dict[str, str] | None) -> str:
    if relation is not None:
        return _first_nonempty(relation, ["source_metric_header", "source_column_header", "field_name", "target_field_name"])
    return _first_nonempty(row, [f"{base_field_name(field_name)}_column_header", "source_column_header"]) or base_field_name(field_name)


def normalized_value_tokens(value: str) -> set[str]:
    text = _norm(value)
    if not text:
        return set()
    tokens = {text.lower()}
    compact = recompact_numeric(text)
    if compact:
        tokens.add(compact)
    for number in re.findall(r"[-+]?\d+(?:\.\d+)?", text):
        tokens.add(number)
    return {token for token in tokens if token}


def recompact_numeric(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum() or ch in ".+-")


def evidence_contains_value(value: str, evidence_text: str) -> bool:
    evidence = _norm(evidence_text)
    if not _norm(value):
        return False
    if not evidence:
        return False
    evidence_lower = evidence.lower()
    evidence_compact = recompact_numeric(evidence)
    for token in normalized_value_tokens(value):
        if token in evidence_lower or token in evidence_compact:
            return True
    return False


def requires_exact_value_evidence(field_name: str, value: str) -> bool:
    if not value:
        return False
    if field_name in IDENTITY_FIELDS:
        return False
    if field_name in DERIVED_HINT_FIELDS or field_name.endswith("_derived"):
        return False
    base = base_field_name(field_name)
    if base in VALUE_EXACT_EVIDENCE_FIELD_HINTS:
        return True
    return bool(re.search(r"\d", value))


def target_match_basis(row: dict[str, str], relation: dict[str, str] | None) -> str:
    if relation is not None:
        basis = _first_nonempty(
            relation,
            ["target_match_basis", "match_basis", "resolution_rule", "relation_resolution_rule"],
        )
        if basis:
            return basis
    pieces = []
    for field in ("raw_formulation_label", "local_instance_id", "drug_name_value", "polymer_identity_final"):
        value = _norm(row.get(field))
        if value:
            pieces.append(f"{field}={value}")
    return ";".join(pieces)


def metric_match_basis(row: dict[str, str], field_name: str, relation: dict[str, str] | None) -> str:
    if relation is not None:
        basis = _first_nonempty(
            relation,
            ["metric_match_basis", "source_metric_header", "source_column_header", "field_name"],
        )
        if basis:
            return basis
    base = base_field_name(field_name)
    if base in {"size_nm", "pdi", "zeta_mV", "encapsulation_efficiency_percent", "loading_content_percent", "dl_percent"}:
        return base
    return ""


def binding_strength(status: str, assignment_path: str, relation: dict[str, str] | None, refs_class: str) -> str:
    if status == "blank_value":
        return "blank"
    if assignment_path in {"shared_method_context", "parent_inheritance"}:
        return "shared_context"
    if status == "direct_supported":
        return "direct_row"
    if assignment_path == "direct_same_table_row":
        return "direct_row"
    if assignment_path == "stage3_relation_resolved" and relation is not None:
        source_type = source_surface_type({}, "", relation)
        if "table" in source_type or "grid" in source_type:
            return "scoped_table"
        return "relation"
    if assignment_path in {"shared_method_context", "parent_inheritance"}:
        return "shared_context"
    if refs_class == "broad_anchor":
        return "broad_anchor"
    if status == "value_only_match":
        return "value_only"
    if status in {"missing_evidence_anchor", "missing_exact_value_evidence", "relation_path_missing", "ambiguous_assignment"}:
        return "weak_or_missing"
    return "unknown"


def review_display_text(
    *,
    field_name: str,
    value: str,
    assignment_path: str,
    source_label_value: str,
    locator: str,
    evidence_text: str,
) -> str:
    parts = [
        f"field={field_name}",
        f"value={value or '<blank>'}",
        f"path={assignment_path}",
    ]
    if source_label_value:
        parts.append(f"source={source_label_value}")
    if locator:
        parts.append(f"locator={locator}")
    if evidence_text:
        parts.append(f"evidence={evidence_text[:240]}")
    return " | ".join(parts)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [dict(row) for row in reader]


def load_authority_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "artifacts" not in payload:
        raise ValueError(f"Authority manifest missing artifacts: {path}")
    return payload


def manifest_path(manifest: dict[str, Any], semantic_name: str, *, required: bool = True) -> Path | None:
    for row in manifest.get("artifacts", []):
        if row.get("semantic_name") == semantic_name:
            value = _norm(row.get("selected_path"))
            if not value:
                if required:
                    raise FileNotFoundError(f"Authority manifest has empty path for {semantic_name}")
                return None
            path = Path(value)
            if required and not path.exists():
                raise FileNotFoundError(f"Authority path for {semantic_name} not found: {path}")
            return path
    if required:
        raise FileNotFoundError(f"Authority manifest lacks semantic artifact {semantic_name}")
    return None


def relation_key_candidates(row: dict[str, str]) -> list[str]:
    candidates = [
        _norm(row.get("formulation_candidate_id")),
        _norm(row.get("final_formulation_id")),
        _norm(row.get("formulation_id")),
        _norm(row.get("representative_source_formulation_id")),
    ]
    return [value for value in candidates if value]


def build_relation_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    index: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        field = _norm(row.get("field_name"))
        if not field:
            continue
        for key in relation_key_candidates(row):
            index[(key, field)] = row
    return index


def paper_key_for_row(row: dict[str, str]) -> str:
    for field in ("paper_key", "zotero_key", "key"):
        value = _norm(row.get(field))
        if value:
            return value
    final_id = _norm(row.get("final_formulation_id"))
    if "__" in final_id:
        return final_id.split("__", 1)[0]
    rep = _norm(row.get("representative_source_formulation_id"))
    if "__" in rep:
        return rep.split("__", 1)[0]
    return ""


def relation_for_field(row: dict[str, str], field_name: str, relation_index: dict[tuple[str, str], dict[str, str]]) -> dict[str, str] | None:
    field_candidates = [field_name]
    base = base_field_name(field_name)
    if base != field_name:
        field_candidates.append(base)
    for key in [
        _norm(row.get("final_formulation_id")),
        _norm(row.get("representative_source_formulation_id")),
        _norm(row.get("formulation_id")),
    ]:
        for candidate in field_candidates:
            if key and (key, candidate) in relation_index:
                return relation_index[(key, candidate)]
    return None


def supporting_refs_class(row: dict[str, str]) -> str:
    refs = _norm(row.get("supporting_evidence_refs"))
    if not refs:
        return "none"
    return "broad_anchor"


def detect_coded_value_pending(field_name: str, value: str, row: dict[str, str]) -> bool:
    field_text = field_name.lower()
    value_text = value.lower()
    coded_field_hint = any(token in field_text for token in ("coded", "doe_factor", "factor_level"))
    coded_value_hint = any(token in value_text for token in ("coded", "level"))
    compact_factor_code = bool(value) and any(ch.isalpha() for ch in value) and any(ch.isdigit() for ch in value) and len(value) <= 8
    return (coded_field_hint or coded_value_hint) and compact_factor_code


def build_field_pack(
    row: dict[str, str],
    field_name: str,
    *,
    relation_index: dict[tuple[str, str], dict[str, str]],
) -> dict[str, Any]:
    value = _norm(row.get(field_name))
    final_id = _norm(row.get("final_formulation_id")) or _norm(row.get("formulation_id"))
    paper_key = paper_key_for_row(row)
    refs_class = supporting_refs_class(row)
    relation = relation_for_field(row, field_name, relation_index)

    if not value:
        status = "blank_value"
        assignment_path = "blank_value"
    elif field_name in IDENTITY_FIELDS:
        status = "identity_only_match"
        assignment_path = "selection_marker"
    elif relation is not None:
        assignment_path = "stage3_relation_resolved"
        if _norm(relation.get("source_relation_row_ids")) or _norm(relation.get("relation_record_ids")):
            status = "relation_supported"
        else:
            status = "relation_path_missing"
    elif has_field_locator(row, field_name):
        status = "direct_supported"
        assignment_path = field_scope_assignment_path(row, field_name)
    elif field_name == "polymer_identity_final" and any(
        _contains_token_surface(_norm(row.get(source_field)), value)
        for source_field in ("polymer_name_raw", "polymer_identity", "raw_formulation_label")
    ):
        status = "direct_supported"
        assignment_path = "selection_marker"
    elif field_name in DERIVED_HINT_FIELDS or field_name.endswith("_derived"):
        status = "derived_supported"
        assignment_path = "derived_from_row_values"
    elif detect_coded_value_pending(field_name, value, row):
        status = "coded_value_supported_decode_pending"
        assignment_path = "doe_factor_decode"
    else:
        status = "missing_evidence_anchor"
        assignment_path = "unresolved"

    surface_type = source_surface_type(row, field_name, relation)
    source_label_value = source_label(row, relation)
    locator = source_locator_text(row, field_name, relation)
    evidence_text = value_evidence_text(row, field_name, relation, value)
    cell_text = source_cell_text(row, relation)
    exact_value_required = requires_exact_value_evidence(field_name, value)
    exact_value_supported = evidence_contains_value(value, evidence_text) or evidence_contains_value(value, cell_text)
    if exact_value_required and status in {
        "direct_supported",
        "relation_supported",
        "role_tolerant_supported",
        "raw_value_supported_normalization_pending",
    } and not exact_value_supported:
        status = "missing_exact_value_evidence"
    value_evidence_for_pack = evidence_text
    if exact_value_required and not exact_value_supported:
        value_evidence_for_pack = ""
    strength = binding_strength(status, assignment_path, relation, refs_class)

    pack = {
        "paper_key": paper_key,
        "final_formulation_id": final_id,
        "field_name": field_name,
        "frozen_value": value,
        "binding_status": status,
        "assignment_path": assignment_path,
        "source_surface_type": surface_type,
        "source_label": source_label_value,
        "source_locator_text": locator,
        "value_evidence_text": value_evidence_for_pack,
        "row_identity_evidence_text": row_identity_evidence_text(row, relation),
        "source_cell_text": cell_text,
        "source_row_label": source_row_label(row, relation),
        "source_column_label": source_column_label(row, field_name, relation),
        "evidence_exact_value_required": "yes" if exact_value_required else "no",
        "evidence_contains_exact_value": "yes" if exact_value_supported else "no",
        "target_match_basis": target_match_basis(row, relation),
        "metric_match_basis": metric_match_basis(row, field_name, relation),
        "binding_strength": strength,
        "ambiguity_count": "0",
        "conflict_notes": "",
        "review_display_text": review_display_text(
            field_name=field_name,
            value=value,
            assignment_path=assignment_path,
            source_label_value=source_label_value,
            locator=locator,
            evidence_text=value_evidence_for_pack,
        ),
        "supporting_refs_class": refs_class,
        "source_row_identity": {
            "representative_source_formulation_id": _norm(row.get("representative_source_formulation_id")),
            "formulation_id": _norm(row.get("formulation_id")),
            "local_instance_id": _norm(row.get("local_instance_id")),
        },
        "relation_path": {},
        "notes": "",
    }
    if relation is not None:
        pack["relation_path"] = {
            "resolution_rule": _norm(relation.get("resolution_rule")) or _norm(relation.get("relation_resolution_rule")),
            "source_relation_row_ids": _norm(relation.get("source_relation_row_ids")) or _norm(relation.get("relation_record_ids")),
            "deterministic_confidence": _norm(relation.get("deterministic_confidence")) or _norm(relation.get("relation_resolution_confidence")),
        }
    if status == "missing_evidence_anchor" and refs_class == "broad_anchor":
        pack["notes"] = "Row-level supporting_evidence_refs are preserved as anchors only; not promoted to direct field support."
    return pack


def default_pack_fields(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return []
    fields = list(rows[0].keys())
    return [field for field in fields if field not in DEFAULT_EXCLUDE_FIELDS]


def write_tsv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_pack_outputs(
    *,
    rows: list[dict[str, str]],
    fields: list[str],
    relation_index: dict[tuple[str, str], dict[str, str]],
    out_dir: Path,
    source_manifest: dict[str, Any],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    packs: list[dict[str, Any]] = []
    jsonl_path = out_dir / "evidence_binding_packs_v1.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            for field in fields:
                pack = build_field_pack(row, field, relation_index=relation_index)
                packs.append(pack)
                handle.write(json.dumps(pack, sort_keys=True, ensure_ascii=False) + "\n")

    field_counter: Counter[tuple[str, str, str]] = Counter(
        (pack["field_name"], pack["binding_status"], pack["assignment_path"]) for pack in packs
    )
    field_summary = [
        {
            "group_key": field,
            "binding_status": status,
            "assignment_path": path,
            "count": str(count),
        }
        for (field, status, path), count in sorted(field_counter.items())
    ]
    formulation_counter: Counter[tuple[str, str, str]] = Counter(
        (pack["final_formulation_id"], pack["binding_status"], pack["assignment_path"]) for pack in packs
    )
    formulation_summary = [
        {
            "group_key": formulation,
            "binding_status": status,
            "assignment_path": path,
            "count": str(count),
        }
        for (formulation, status, path), count in sorted(formulation_counter.items())
    ]
    write_tsv(out_dir / "evidence_binding_field_summary_v1.tsv", field_summary, SUMMARY_FIELDS)
    write_tsv(out_dir / "evidence_binding_formulation_summary_v1.tsv", formulation_summary, SUMMARY_FIELDS)
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "src/stage5_benchmark/build_evidence_binding_packs_v1.py",
        "benchmark_valid": "no",
        "active_run_id": source_manifest.get("active_run_id", ""),
        "active_run_dir": source_manifest.get("active_run_dir", ""),
        "pack_count": len(packs),
        "row_count": len(rows),
        "field_count": len(fields),
        "status_distribution": dict(Counter(pack["binding_status"] for pack in packs)),
        "assignment_path_distribution": dict(Counter(pack["assignment_path"] for pack in packs)),
    }
    (out_dir / "evidence_binding_pack_metadata_v1.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_run_context(out_dir / "RUN_CONTEXT.md", metadata=metadata, source_manifest=source_manifest)
    return metadata


def write_run_context(path: Path, *, metadata: dict[str, Any], source_manifest: dict[str, Any]) -> None:
    lines = [
        "# RUN_CONTEXT",
        "",
        "## Run purpose",
        "Build diagnostic Evidence Binding Packs for frozen Stage5 final-table rows/fields.",
        "",
        "## Run type",
        "Diagnostic-only audit sidecar. Not benchmark-valid final output.",
        "",
        "## Boundary",
        "This run does not create rows, create values, replace the final table, assign risk, or render a workbook.",
        "",
        "## Authority source",
        f"- active_run_id: `{source_manifest.get('active_run_id', '')}`",
        f"- active_run_dir: `{source_manifest.get('active_run_dir', '')}`",
        f"- pointer_path: `{source_manifest.get('pointer_path', '')}`",
        "",
        "## Outputs",
        "- `evidence_binding_packs_v1.jsonl`",
        "- `evidence_binding_field_summary_v1.tsv`",
        "- `evidence_binding_formulation_summary_v1.tsv`",
        "- `evidence_binding_pack_metadata_v1.json`",
        "",
        "## Counts",
        f"- row_count: {metadata['row_count']}",
        f"- field_count: {metadata['field_count']}",
        f"- pack_count: {metadata['pack_count']}",
        "",
        "## Status distribution",
    ]
    for key, value in sorted(metadata["status_distribution"].items()):
        lines.append(f"- {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_from_authority_manifest(
    *,
    authority_manifest_path: Path,
    out_dir: Path,
    paper_keys: set[str] | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    manifest = load_authority_manifest(authority_manifest_path)
    final_path = manifest_path(manifest, "frozen_final_table", required=True)
    relation_path = manifest_path(manifest, "stage3_resolved_relation_fields", required=False)
    final_rows = read_tsv(final_path) if final_path else []
    if paper_keys:
        final_rows = [row for row in final_rows if paper_key_for_row(row) in paper_keys]
    if max_rows is not None:
        final_rows = final_rows[:max_rows]
    relation_rows = read_tsv(relation_path) if relation_path and relation_path.exists() else []
    relation_index = build_relation_index(relation_rows)
    fields = default_pack_fields(final_rows)
    return write_pack_outputs(
        rows=final_rows,
        fields=fields,
        relation_index=relation_index,
        out_dir=out_dir,
        source_manifest=manifest,
    )


def parse_paper_keys(value: str) -> set[str] | None:
    if not value.strip():
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build diagnostic Evidence Binding Packs.")
    parser.add_argument("--authority-manifest-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--paper-keys", default="", help="Comma-separated bounded paper keys.")
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args(argv)
    metadata = build_from_authority_manifest(
        authority_manifest_path=args.authority_manifest_json,
        out_dir=args.out_dir,
        paper_keys=parse_paper_keys(args.paper_keys),
        max_rows=args.max_rows or None,
    )
    print(f"pack_count={metadata['pack_count']}")
    print(f"row_count={metadata['row_count']}")
    for key, value in sorted(metadata["status_distribution"].items()):
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
