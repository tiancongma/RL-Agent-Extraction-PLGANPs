#!/usr/bin/env python3
from __future__ import annotations

"""
Build a deterministic table-row binding surface for frozen Step 1 formulation rows.

Purpose:
- resolve row-local ownership between existing frozen formulation identities and
  source table rows that carry numeric values
- prepare lawful row-local support for the existing Step 2 helper
- keep formulation membership unchanged

This helper does not fill values into the final dataset by itself.
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CANDIDATES_NAME = "table_row_binding_candidates_v1.tsv"
RESOLVED_NAME = "table_row_binding_resolved_v1.tsv"
SUMMARY_NAME = "table_row_binding_summary_v1.md"
TABLE_LIKE_EVIDENCE_TYPES = {"table_row", "table_cell", "table_header", "table_block"}


@dataclass(frozen=True)
class FieldSpec:
    field_name: str
    final_value_column: str
    final_value_text_column: str
    final_evidence_region_column: str
    header_keywords: tuple[str, ...]


FIELD_SPECS = [
    FieldSpec(
        "encapsulation_efficiency_percent",
        "encapsulation_efficiency_percent_value",
        "encapsulation_efficiency_percent_value_text",
        "encapsulation_efficiency_percent_evidence_region_type",
        ("encapsulation efficiency", "entrapment efficiency", "%ee", "ee (%)", "ee", "e.e.%", "e.e."),
    ),
    FieldSpec(
        "particle_size_nm",
        "size_nm_value",
        "size_nm_value_text",
        "size_nm_evidence_region_type",
        ("particle size", "mean size", "size (nm)", "mean size (nm)", "size"),
    ),
    FieldSpec(
        "pdi",
        "pdi_value",
        "pdi_value_text",
        "pdi_evidence_region_type",
        ("pdi", "polidispersity index", "polydispersity index", "polidispersity", "p. i.", "p i"),
    ),
    FieldSpec(
        "zeta_potential_mV",
        "zeta_mV_value",
        "zeta_mV_value_text",
        "zeta_mV_evidence_region_type",
        ("zeta potential", "zeta", "zeta potential (mv)", "zeta potential (mV)"),
    ),
    FieldSpec(
        "loading_capacity_percent",
        "loading_content_percent_value",
        "loading_content_percent_value_text",
        "loading_content_percent_evidence_region_type",
        ("loading content", "drug loading", "% dl", "%dl", "lc", "d.c.%", "d.c."),
    ),
    FieldSpec(
        "polymer_amount",
        "plga_mass_mg_value",
        "plga_mass_mg_value_text",
        "plga_mass_mg_evidence_region_type",
        ("plga (mg)", "polymer (mg)", "polymer amount", "plga amount", "plga", "polymer"),
    ),
    FieldSpec(
        "drug_feed_amount",
        "drug_feed_amount_text_value",
        "drug_feed_amount_text_value_text",
        "drug_feed_amount_text_evidence_region_type",
        ("drug (mg)", "drug amount", "artemether (mg)", "payload amount", "drug"),
    ),
    FieldSpec(
        "surfactant_concentration",
        "surfactant_concentration_text_value",
        "surfactant_concentration_text_value_text",
        "surfactant_concentration_text_evidence_region_type",
        (
            "pva (mg)",
            "pva (mg/ml)",
            "p188 (mg/ml)",
            "poloxamer",
            "surfactant concentration",
            "stabilizer concentration",
            "pva",
        ),
    ),
]

FIELD_SPEC_BY_NAME = {spec.field_name: spec for spec in FIELD_SPECS}
PRIORITY_FIELDS = [spec.field_name for spec in FIELD_SPECS]


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def normalize_table_text(value: Any) -> str:
    text = normalize_text(value)
    text = (
        text.replace("±", " +/- ")
        .replace("卤", " +/- ")
        .replace("−", "-")
        .replace("鈭?", "-")
        .replace("\xa0", " ")
    )
    return re.sub(r"\s+", " ", text).strip()


def normalize_token(value: Any) -> str:
    text = normalize_table_text(value).lower()
    text = re.sub(r"[^a-z0-9%./+\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_number_token(value: Any) -> str:
    text = normalize_table_text(value)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return ""
    try:
        number = float(match.group(0))
    except ValueError:
        return ""
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.6g}"


def read_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: str(row.get(field, "")) for field in fieldnames})


def parse_json_object_list(value: Any) -> list[dict[str, str]]:
    text = normalize_text(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [{str(key): normalize_text(item_value) for key, item_value in item.items()} for item in parsed if isinstance(item, dict)]


def repo_rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")


def locator_to_path(locator: str) -> Path | None:
    text = normalize_text(locator)
    if not text:
        return None
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    if candidate.exists():
        return candidate
    return None


def discover_candidate_tables(row: dict[str, str]) -> list[Path]:
    tables: list[Path] = []
    seen: set[str] = set()
    refs = parse_json_object_list(row.get("supporting_evidence_refs"))
    for ref in refs:
        locator = ref.get("source_locator_text", "")
        path = locator_to_path(locator)
        if path is None or path.suffix.lower() != ".csv":
            continue
        token = str(path).lower()
        if token in seen:
            continue
        seen.add(token)
        tables.append(path)
    return tables


def probable_data_row(row: list[str]) -> bool:
    nonempty = [normalize_table_text(cell) for cell in row if normalize_table_text(cell)]
    if len(nonempty) < 2:
        return False
    digit_cells = sum(1 for cell in nonempty if re.search(r"\d", cell))
    if digit_cells < 2:
        return False
    return True


def find_first_data_row_index(rows: list[list[str]]) -> int:
    for index, row in enumerate(rows):
        if probable_data_row(row):
            return index
    return 0


def build_header_map(rows: list[list[str]]) -> dict[int, str]:
    first_data_index = find_first_data_row_index(rows)
    header_rows = rows[:first_data_index] if first_data_index > 0 else rows[:1]
    width = max((len(row) for row in rows), default=0)
    header_map: dict[int, str] = {}
    for column_index in range(width):
        parts: list[str] = []
        for row in header_rows:
            if column_index < len(row):
                cell = normalize_table_text(row[column_index])
                if cell:
                    parts.append(cell)
        header_map[column_index] = normalize_table_text(" ".join(parts))
    return header_map


def find_field_column(header_map: dict[int, str], spec: FieldSpec) -> int | None:
    best_index: int | None = None
    best_score = -1
    for index, header in header_map.items():
        token = normalize_token(header)
        if not token:
            continue
        score = 0
        for keyword in spec.header_keywords:
            keyword_token = normalize_token(keyword)
            if keyword_token and keyword_token in token:
                score = max(score, len(keyword_token))
        if score > best_score:
            best_score = score
            best_index = index
    return best_index if best_score > 0 else None


def load_csv_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        return [[normalize_table_text(cell) for cell in row] for row in reader]


def row_text(row_cells: list[str]) -> str:
    return " | ".join(cell for cell in row_cells if normalize_table_text(cell))


def row_label_candidates(row: list[str]) -> set[str]:
    candidates: set[str] = set()
    for cell in row[:2]:
        token = normalize_token(cell)
        if token:
            candidates.add(token)
    return candidates


@dataclass
class TableRowCandidate:
    source_table_path: Path
    source_table_id: str
    source_table_row_id: str
    physical_row_index: int
    row_cells: list[str]
    joined_text: str
    label_candidates: set[str]
    field_values: dict[str, str]
    field_normalized: dict[str, str]


@dataclass
class TableCache:
    header_map: dict[int, str]
    field_columns: dict[str, int | None]
    candidates: list[TableRowCandidate]


def build_table_cache(path: Path) -> TableCache:
    rows = load_csv_rows(path)
    header_map = build_header_map(rows)
    first_data_index = find_first_data_row_index(rows)
    table_id = path.stem
    field_columns = {spec.field_name: find_field_column(header_map, spec) for spec in FIELD_SPECS}
    candidates: list[TableRowCandidate] = []
    for physical_index, row in enumerate(rows[first_data_index:], start=first_data_index + 1):
        if not probable_data_row(row):
            continue
        joined = row_text(row)
        if not joined:
            continue
        values: dict[str, str] = {}
        normalized: dict[str, str] = {}
        for spec in FIELD_SPECS:
            column_index = field_columns.get(spec.field_name)
            value = normalize_table_text(row[column_index]) if column_index is not None and column_index < len(row) else ""
            values[spec.field_name] = value
            normalized[spec.field_name] = normalize_number_token(value)
        candidates.append(
            TableRowCandidate(
                source_table_path=path,
                source_table_id=table_id,
                source_table_row_id=f"{table_id}::row_{physical_index:02d}",
                physical_row_index=physical_index,
                row_cells=row,
                joined_text=joined,
                label_candidates=row_label_candidates(row),
                field_values=values,
                field_normalized=normalized,
            )
        )
    return TableCache(header_map=header_map, field_columns=field_columns, candidates=candidates)


def target_field_value(row: dict[str, str], spec: FieldSpec) -> str:
    return normalize_table_text(row.get(spec.final_value_text_column) or row.get(spec.final_value_column))


def target_field_value_norm(row: dict[str, str], spec: FieldSpec) -> str:
    return normalize_number_token(target_field_value(row, spec))


def target_field_requires_binding(row: dict[str, str], spec: FieldSpec) -> bool:
    if normalize_text(row.get("table_row_id")):
        return False
    field_value = target_field_value(row, spec)
    if not field_value:
        return False
    region = normalize_text(row.get(spec.final_evidence_region_column)).lower()
    return region in TABLE_LIKE_EVIDENCE_TYPES


def instance_row_refs(row: dict[str, str], table_path: Path) -> list[dict[str, str]]:
    refs = parse_json_object_list(row.get("supporting_evidence_refs"))
    out: list[dict[str, str]] = []
    for ref in refs:
        if normalize_text(ref.get("target_field_name")) != "instance":
            continue
        locator_path = locator_to_path(ref.get("source_locator_text", ""))
        if locator_path is None or locator_path.resolve() != table_path.resolve():
            continue
        if normalize_text(ref.get("source_region_type")).lower() not in TABLE_LIKE_EVIDENCE_TYPES:
            continue
        out.append(ref)
    return out


def exact_label_matches(row: dict[str, str], cache: TableCache) -> list[TableRowCandidate]:
    label = normalize_token(row.get("representative_source_raw_formulation_label"))
    if not label:
        representative = normalize_text(row.get("representative_source_formulation_id"))
        if "__" in representative:
            label = normalize_token(representative.split("__")[-1])
        elif "_" in representative:
            label = normalize_token(representative.split("_")[-1])
    if not label:
        return []
    matches = [candidate for candidate in cache.candidates if label in candidate.label_candidates]
    return matches


def exact_snippet_matches(row: dict[str, str], table_path: Path, cache: TableCache) -> list[TableRowCandidate]:
    refs = instance_row_refs(row, table_path)
    matches: list[TableRowCandidate] = []
    for ref in refs:
        snippet = normalize_token(ref.get("supporting_snippet", ""))
        if not snippet:
            continue
        local_matches = [
            candidate
            for candidate in cache.candidates
            if snippet == normalize_token(candidate.joined_text)
            or snippet in normalize_token(candidate.joined_text)
            or normalize_token(candidate.joined_text) in snippet
        ]
        for candidate in local_matches:
            if candidate not in matches:
                matches.append(candidate)
    return matches


def numeric_signature_matches(row: dict[str, str], cache: TableCache) -> list[TableRowCandidate]:
    target_fields = [
        spec.field_name
        for spec in FIELD_SPECS
        if target_field_value_norm(row, spec)
    ]
    if not target_fields:
        return []
    scored: list[tuple[int, TableRowCandidate]] = []
    for candidate in cache.candidates:
        score = 0
        for field_name in target_fields:
            spec = FIELD_SPEC_BY_NAME[field_name]
            if target_field_value_norm(row, spec) and target_field_value_norm(row, spec) == candidate.field_normalized.get(field_name, ""):
                score += 1
        if score:
            scored.append((score, candidate))
    if not scored:
        return []
    best_score = max(score for score, _ in scored)
    if best_score < 2:
        return []
    return [candidate for score, candidate in scored if score == best_score]


def select_bound_row(row: dict[str, str], table_path: Path, cache: TableCache) -> tuple[TableRowCandidate | None, str, list[TableRowCandidate]]:
    snippet_matches = exact_snippet_matches(row, table_path, cache)
    if len(snippet_matches) == 1:
        return snippet_matches[0], "supporting_snippet_exact", snippet_matches
    label_matches = exact_label_matches(row, cache)
    if len(label_matches) == 1:
        return label_matches[0], "article_label_exact", label_matches
    if snippet_matches and label_matches:
        overlap = [candidate for candidate in snippet_matches if candidate in label_matches]
        if len(overlap) == 1:
            return overlap[0], "snippet_and_label_overlap", overlap
    signature_matches = numeric_signature_matches(row, cache)
    if len(signature_matches) == 1:
        return signature_matches[0], "unique_numeric_signature", signature_matches
    combined: list[TableRowCandidate] = []
    for group in (snippet_matches, label_matches, signature_matches):
        for candidate in group:
            if candidate not in combined:
                combined.append(candidate)
    if len(combined) == 1:
        return combined[0], "combined_unique_candidate", combined
    return None, "", combined


def candidate_row_dict(
    *,
    final_formulation_id: str,
    paper_key: str,
    field_name: str,
    candidate: TableRowCandidate,
    match_rule: str,
    match_score: int,
    source_value_text: str,
    source_value_normalized: str,
) -> dict[str, str]:
    return {
        "final_formulation_id": final_formulation_id,
        "paper_key": paper_key,
        "field_name": field_name,
        "source_table_id": candidate.source_table_id,
        "source_table_row_id": candidate.source_table_row_id,
        "source_table_path": repo_rel(candidate.source_table_path),
        "source_value_text": source_value_text,
        "source_value_normalized": source_value_normalized,
        "match_rule": match_rule,
        "match_score": str(match_score),
        "row_text": candidate.joined_text,
    }


def resolved_row_dict(
    *,
    row: dict[str, str],
    field_name: str,
    status: str,
    source_table_id: str = "",
    source_table_row_id: str = "",
    source_table_path: str = "",
    source_value_text: str = "",
    source_value_normalized: str = "",
    binding_rule_used: str = "",
    binding_confidence_class: str = "",
    row_text: str = "",
) -> dict[str, str]:
    return {
        "final_formulation_id": normalize_text(row.get("final_formulation_id")),
        "paper_key": normalize_text(row.get("key")),
        "field_name": field_name,
        "source_table_id": source_table_id,
        "source_table_row_id": source_table_row_id,
        "source_value_text": source_value_text,
        "source_value_normalized": source_value_normalized,
        "binding_rule_used": binding_rule_used,
        "binding_confidence_class": binding_confidence_class,
        "binding_status": status,
        "source_table_path": source_table_path,
        "source_row_text": row_text,
    }


def build_summary_markdown(
    *,
    attempted_fields: int,
    resolved_rows: list[dict[str, str]],
) -> str:
    status_counter = Counter(row["binding_status"] for row in resolved_rows)
    field_counter: dict[str, Counter[str]] = defaultdict(Counter)
    for row in resolved_rows:
        field_counter[row["field_name"]][row["binding_status"]] += 1
    lines = [
        "# Table Row Binding Summary v1",
        "",
        "## Contract",
        "- deterministic only",
        "- no formulation membership change",
        "- no direct value filling",
        "- row-local binding only",
        "",
        "## Counts",
        f"- attempted field bindings: `{attempted_fields}`",
        f"- resolved_row_local: `{status_counter.get('resolved_row_local', 0)}`",
        f"- ambiguous_multiple_rows: `{status_counter.get('ambiguous_multiple_rows', 0)}`",
        f"- no_matching_row: `{status_counter.get('no_matching_row', 0)}`",
        f"- unsupported_table_shape: `{status_counter.get('unsupported_table_shape', 0)}`",
        f"- parse_failed: `{status_counter.get('parse_failed', 0)}`",
        "",
        "## By Field",
    ]
    for field_name in PRIORITY_FIELDS:
        counter = field_counter.get(field_name, Counter())
        lines.append(
            "- "
            + field_name
            + f": resolved={counter.get('resolved_row_local', 0)}, ambiguous={counter.get('ambiguous_multiple_rows', 0)}, "
            + f"no_match={counter.get('no_matching_row', 0)}, unsupported={counter.get('unsupported_table_shape', 0)}, parse_failed={counter.get('parse_failed', 0)}"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic table-row binding surface for frozen Step 1 rows.")
    parser.add_argument("--final-table-tsv", required=True, type=Path)
    parser.add_argument("--decision-trace-tsv", type=Path, default=None)
    parser.add_argument("--relation-records-tsv", type=Path, default=None)
    parser.add_argument("--resolved-relation-fields-tsv", type=Path, default=None)
    parser.add_argument("--scope-manifest-tsv", type=Path, default=None)
    parser.add_argument("--paper-key", action="append", default=[])
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    final_rows = read_tsv_rows(args.final_table_tsv.resolve())
    selected_keys = {normalize_text(key) for key in args.paper_key if normalize_text(key)}
    if selected_keys:
        final_rows = [row for row in final_rows if normalize_text(row.get("key")) in selected_keys]

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    table_cache_by_path: dict[str, TableCache] = {}
    candidate_rows: list[dict[str, str]] = []
    resolved_rows: list[dict[str, str]] = []
    attempted_fields = 0

    for row in final_rows:
        candidate_tables = discover_candidate_tables(row)
        for spec in FIELD_SPECS:
            if not target_field_requires_binding(row, spec):
                continue
            attempted_fields += 1
            if not candidate_tables:
                resolved_rows.append(
                    resolved_row_dict(
                        row=row,
                        field_name=spec.field_name,
                        status="no_matching_row",
                    )
                )
                continue

            best_candidate: TableRowCandidate | None = None
            best_rule = ""
            all_candidates: list[TableRowCandidate] = []
            unsupported_shape = False

            for table_path in candidate_tables:
                cache = table_cache_by_path.get(str(table_path))
                if cache is None:
                    cache = build_table_cache(table_path)
                    table_cache_by_path[str(table_path)] = cache
                if cache.field_columns.get(spec.field_name) is None:
                    unsupported_shape = True
                    continue
                matched_row, rule, considered = select_bound_row(row, table_path, cache)
                all_candidates.extend(considered)
                if matched_row is not None and not best_candidate:
                    best_candidate = matched_row
                    best_rule = rule

            deduped_candidates: list[TableRowCandidate] = []
            seen_row_ids: set[str] = set()
            for candidate in all_candidates:
                if candidate.source_table_row_id in seen_row_ids:
                    continue
                seen_row_ids.add(candidate.source_table_row_id)
                deduped_candidates.append(candidate)

            for candidate in deduped_candidates:
                source_value_text = candidate.field_values.get(spec.field_name, "")
                source_value_normalized = candidate.field_normalized.get(spec.field_name, "")
                match_score = 0
                if source_value_normalized and source_value_normalized == target_field_value_norm(row, spec):
                    match_score += 1
                if normalize_token(row.get("representative_source_raw_formulation_label")) in candidate.label_candidates:
                    match_score += 1
                candidate_rows.append(
                    candidate_row_dict(
                        final_formulation_id=normalize_text(row.get("final_formulation_id")),
                        paper_key=normalize_text(row.get("key")),
                        field_name=spec.field_name,
                        candidate=candidate,
                        match_rule=best_rule or "candidate_pool",
                        match_score=match_score,
                        source_value_text=source_value_text,
                        source_value_normalized=source_value_normalized,
                    )
                )

            if best_candidate is None:
                status = "unsupported_table_shape" if unsupported_shape else "ambiguous_multiple_rows" if len(deduped_candidates) > 1 else "no_matching_row"
                resolved_rows.append(resolved_row_dict(row=row, field_name=spec.field_name, status=status))
                continue

            source_value_text = best_candidate.field_values.get(spec.field_name, "")
            source_value_normalized = best_candidate.field_normalized.get(spec.field_name, "")
            if not source_value_text:
                resolved_rows.append(
                    resolved_row_dict(
                        row=row,
                        field_name=spec.field_name,
                        status="parse_failed",
                        source_table_id=best_candidate.source_table_id,
                        source_table_row_id=best_candidate.source_table_row_id,
                        source_table_path=repo_rel(best_candidate.source_table_path),
                        binding_rule_used=best_rule,
                        binding_confidence_class="field_column_missing_or_blank",
                        row_text=best_candidate.joined_text,
                    )
                )
                continue

            confidence = {
                "supporting_snippet_exact": "high",
                "article_label_exact": "high",
                "snippet_and_label_overlap": "high",
                "combined_unique_candidate": "medium",
                "unique_numeric_signature": "medium",
            }.get(best_rule, "medium")
            resolved_rows.append(
                resolved_row_dict(
                    row=row,
                    field_name=spec.field_name,
                    status="resolved_row_local",
                    source_table_id=best_candidate.source_table_id,
                    source_table_row_id=best_candidate.source_table_row_id,
                    source_table_path=repo_rel(best_candidate.source_table_path),
                    source_value_text=source_value_text,
                    source_value_normalized=source_value_normalized or target_field_value_norm(row, spec),
                    binding_rule_used=best_rule,
                    binding_confidence_class=confidence,
                    row_text=best_candidate.joined_text,
                )
            )

    candidate_fieldnames = [
        "final_formulation_id",
        "paper_key",
        "field_name",
        "source_table_id",
        "source_table_row_id",
        "source_table_path",
        "source_value_text",
        "source_value_normalized",
        "match_rule",
        "match_score",
        "row_text",
    ]
    resolved_fieldnames = [
        "final_formulation_id",
        "paper_key",
        "field_name",
        "source_table_id",
        "source_table_row_id",
        "source_value_text",
        "source_value_normalized",
        "binding_rule_used",
        "binding_confidence_class",
        "binding_status",
        "source_table_path",
        "source_row_text",
    ]
    write_tsv(out_dir / CANDIDATES_NAME, candidate_fieldnames, candidate_rows)
    write_tsv(out_dir / RESOLVED_NAME, resolved_fieldnames, resolved_rows)
    (out_dir / SUMMARY_NAME).write_text(
        build_summary_markdown(attempted_fields=attempted_fields, resolved_rows=resolved_rows),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "attempted_field_bindings": attempted_fields,
                "resolved_row_local_total": sum(1 for row in resolved_rows if row["binding_status"] == "resolved_row_local"),
                "status_counts": dict(Counter(row["binding_status"] for row in resolved_rows)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
