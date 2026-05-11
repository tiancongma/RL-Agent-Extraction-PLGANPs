#!/usr/bin/env python3
"""Diagnostic clean-text / selector visibility audit for characterization metrics.

This script is intentionally diagnostic-only.  It never reads raw PDFs, raw HTML,
protocol notes, or GT-completion excerpts.  It starts from Layer3 residual cells
and asks where each GT-reported measurement value first disappears from the
lawful Stage2 evidence chain:

clean text -> selected evidence blocks / prompt pack -> LLM semantic object ->
Stage2 compatibility projection -> Stage5 final row -> Layer3 compare.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.stage2_sampling_labels.extract_semantic_stage2_objects_v2 import build_prompt_render_bundle

TARGET_FIELDS = {
    "particle_size_nm",
    "pdi",
    "zeta_mV",
    "ee_percent",
    "encapsulation_efficiency_percent",
    "dl_percent",
    "lc_percent",
    "loading_content_percent",
}

FIELD_TO_STAGE2_PREFIX = {
    "particle_size_nm": "size_nm",
    "pdi": "pdi",
    "zeta_mV": "zeta_mV",
    "ee_percent": "encapsulation_efficiency_percent",
    "encapsulation_efficiency_percent": "encapsulation_efficiency_percent",
    "dl_percent": "dl_percent",
    "lc_percent": "loading_content_percent",
    "loading_content_percent": "loading_content_percent",
}

FIELD_CONTEXT_TERMS = {
    "particle_size_nm": ["size", "diameter", "z-average", "z average", "major axis", "hydrodynamic"],
    "pdi": ["pdi", "p.i", "pi ", "polydispersity"],
    "zeta_mV": ["zeta", "ζ", "zp", "z-potential", "z potential"],
    "ee_percent": ["ee", "e.e", "encapsulation", "entrapment"],
    "encapsulation_efficiency_percent": ["ee", "e.e", "encapsulation", "entrapment"],
    "dl_percent": ["d.l", "dl", "drug loading", "loading efficiency"],
    "lc_percent": ["l.c", "lc", "loading content", "drug content", "payload"],
    "loading_content_percent": ["l.c", "lc", "loading content", "drug content", "payload"],
}

OUTPUT_COLUMNS = [
    "paper_key",
    "field_name",
    "gt_formulation_id",
    "matched_system_formulation_id",
    "compare_status",
    "gt_value_raw",
    "clean_text_value_visibility",
    "clean_text_locator",
    "selector_visibility",
    "selector_artifact_ref",
    "prompt_visibility",
    "prompt_artifact_ref",
    "llm_authorization_status",
    "downstream_boundary",
    "first_failure_boundary",
    "notes",
]

SUMMARY_COLUMNS = ["field_name", "compare_status", "first_failure_boundary", "count"]


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: Iterable[Mapping[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def normalize_numeric_tokens(value: str) -> List[str]:
    """Return conservative numeric tokens from a GT value.

    We intentionally do not evaluate arithmetic or use fuzzy semantic matching.
    The audit asks whether the same reported value is visibly present in lawful
    text/evidence surfaces.  Tokens shorter than one digit are ignored; mean±SD
    strings yield multiple tokens and any token can prove visibility.
    """
    if not value:
        return []
    cleaned = value.replace("−", "-").replace("–", "-").replace("—", "-")
    tokens = re.findall(r"[-+]?\d+(?:\.\d+)?", cleaned)
    out: List[str] = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        # Drop pure signs or pathologically short punctuation-only matches.
        if not re.search(r"\d", token):
            continue
        out.append(token)
    # Stable de-duplication.
    return list(dict.fromkeys(out))


def token_pattern(token: str) -> re.Pattern[str]:
    escaped = re.escape(token)
    if "." in token:
        # Allow common OCR/trailing-zero variants only by exact decimal text.
        return re.compile(rf"(?<![\d.]){escaped}(?![\d.])")
    return re.compile(rf"(?<![\d.]){escaped}(?![\d.])")


def normalize_visibility_text(text: str) -> str:
    return text.replace("−", "-").replace("–", "-").replace("—", "-")


def find_token_visibility(text: str, tokens: Sequence[str]) -> Tuple[bool, str]:
    if not text or not tokens:
        return False, ""
    searchable = normalize_visibility_text(text)
    for token in tokens:
        m = token_pattern(token).search(searchable)
        if m:
            start = max(0, m.start() - 90)
            end = min(len(searchable), m.end() + 90)
            snippet = re.sub(r"\s+", " ", searchable[start:end]).strip()
            return True, f"token={token};snippet={snippet}"
    return False, ""


def paragraph_locator_for_token(text: str, source_path: str, tokens: Sequence[str]) -> str:
    if not text or not tokens:
        return ""
    # Paragraph indexing mirrors cleaned text diagnostics approximately; it is
    # only a locator for audit triage, not a benchmark evidence quote.
    paragraphs = re.split(r"\n\s*\n", text)
    char_cursor = 0
    for idx, para in enumerate(paragraphs, start=1):
        found, snippet = find_token_visibility(para, tokens)
        if found:
            return f"{source_path}#paragraph:{idx};{snippet}"
        char_cursor += len(para) + 2
    found, snippet = find_token_visibility(text, tokens)
    if found:
        return f"{source_path}#char;{snippet}"
    return ""


def load_key2txt(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1]
    return mapping


def load_prompt_index(prompt_preview_tsv: Path) -> Dict[str, Dict[str, str]]:
    if not prompt_preview_tsv.exists():
        return {}
    return {row.get("document_key", "") or row.get("paper_key", ""): row for row in read_tsv(prompt_preview_tsv)}


def load_semantic_index(semantic_jsonl: Path) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    if not semantic_jsonl.exists():
        return out
    with semantic_jsonl.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            key = obj.get("paper_key") or obj.get("document_key")
            if key:
                out[key] = obj
    return out


def selected_evidence_text(stage2_run_dir: Path, paper_key: str) -> Tuple[str, str, Dict]:
    p = stage2_run_dir / "semantic_stage2_objects" / "evidence_blocks" / paper_key / "evidence_blocks_v1.json"
    if not p.exists():
        return "", "", {}
    obj = json.loads(p.read_text(encoding="utf-8"))
    blocks = obj.get("evidence_blocks") or []
    text = "\n\n".join(str(b.get("text_content", "")) for b in blocks)
    return text, str(p), obj


def semantic_text_for_authorization(obj: Optional[Mapping]) -> str:
    if not obj:
        return ""
    # Dump only the semantic object.  This is replayed LLM output / semantic
    # contract, not raw external source.  It helps separate "prompt visible but
    # not authorized" from downstream projection loss.
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def index_stage_rows(rows: List[Dict[str, str]], key_field: str, id_fields: Sequence[str]) -> Dict[Tuple[str, str], List[Dict[str, str]]]:
    idx: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = row.get(key_field, "")
        for id_field in id_fields:
            fid = row.get(id_field, "")
            if key and fid:
                idx[(key, fid)].append(row)
    return idx


def row_field_value(row: Mapping[str, str], field_name: str) -> str:
    prefix = FIELD_TO_STAGE2_PREFIX.get(field_name, field_name)
    candidates = [
        f"{prefix}_value",
        f"{prefix}_value_text",
        field_name,
        f"{field_name}_value",
        f"{field_name}_value_text",
    ]
    for col in candidates:
        val = row.get(col, "")
        if val:
            return val
    return ""


def row_has_metric_binding(row: Mapping[str, str], field_name: str, tokens: Sequence[str]) -> bool:
    hay_cols = [
        "table_cell_bindings_json",
        "table_row_variable_assignments_json",
        "change_descriptions",
        "supporting_evidence_refs",
        "evidence_span_text",
    ]
    hay = "\n".join(str(row.get(col, "")) for col in hay_cols)
    prefix = FIELD_TO_STAGE2_PREFIX.get(field_name, field_name)
    if prefix and prefix in hay:
        return True
    found, _ = find_token_visibility(hay, tokens)
    return found


def selector_has_metric_context(evidence_text: str, field_name: str) -> bool:
    if not evidence_text:
        return False
    hay = normalize_visibility_text(evidence_text).lower()
    for term in FIELD_CONTEXT_TERMS.get(field_name, []):
        if term.lower() in hay:
            return True
    return False


def classify_row(
    compare_row: Mapping[str, str],
    repo_root: Path,
    stage2_run_dir: Path,
    key2txt: Mapping[str, str],
    prompt_index: Mapping[str, Mapping[str, str]],
    semantic_index: Mapping[str, Mapping],
    weak_idx: Mapping[Tuple[str, str], List[Dict[str, str]]],
    final_idx: Mapping[Tuple[str, str], List[Dict[str, str]]],
) -> Dict[str, str]:
    paper_key = compare_row.get("paper_key", "")
    field_name = compare_row.get("field_name", "")
    gt_value = compare_row.get("gt_value_raw", "")
    gt_formulation_id = compare_row.get("gt_formulation_id", "")
    system_id = compare_row.get("matched_system_formulation_id", "")
    tokens = normalize_numeric_tokens(gt_value)

    clean_rel = key2txt.get(paper_key, "")
    clean_path = repo_root / clean_rel if clean_rel else Path("")
    clean_text = ""
    if clean_path and clean_path.exists():
        clean_text = clean_path.read_text(encoding="utf-8", errors="replace")
    clean_visible, _ = find_token_visibility(clean_text, tokens)
    clean_locator = paragraph_locator_for_token(clean_text, clean_rel, tokens) if clean_visible else (clean_rel or "")

    ev_text, ev_path, ev_obj = selected_evidence_text(stage2_run_dir, paper_key)
    selector_visible, selector_snippet = find_token_visibility(ev_text, tokens)
    selector_metric_context = selector_has_metric_context(ev_text, field_name)
    prompt_rendered_text = build_prompt_render_bundle(ev_obj).get("evidence_text", "") if ev_obj else ""
    prompt_value_visible, prompt_snippet = find_token_visibility(str(prompt_rendered_text), tokens)
    prompt_metric_context = selector_has_metric_context(str(prompt_rendered_text), field_name)
    prompt_row = prompt_index.get(paper_key, {})
    all_blocks_included = (prompt_row.get("all_selected_blocks_included", "").strip().lower() in {"yes", "true", "1"})
    uses_evidence_pack = (prompt_row.get("uses_evidence_pack_only", "").strip().lower() in {"yes", "true", "1"})
    if prompt_value_visible and all_blocks_included:
        prompt_visibility = "value_visible_in_prompt_pack"
    elif prompt_value_visible and not all_blocks_included:
        prompt_visibility = "value_visible_in_rendered_prompt_but_prompt_inclusion_uncertain"
    elif selector_visible and not prompt_value_visible:
        prompt_visibility = "selector_value_present_but_rendered_prompt_missing_numeric_value"
    elif prompt_metric_context and all_blocks_included:
        prompt_visibility = "metric_context_visible_but_numeric_value_absent_from_summary"
    elif selector_metric_context and all_blocks_included:
        prompt_visibility = "selector_metric_context_present_prompt_numeric_absent"
    else:
        # Prompt previews are only head/tail, so do not overclaim absence from
        # prompt beyond evidence-pack omission.
        prompt_visibility = "not_visible_in_selected_evidence_pack"
    prompt_ref = prompt_row.get("evidence_artifact_path", "") or ev_path

    sem_text = semantic_text_for_authorization(semantic_index.get(paper_key))
    sem_visible, _ = find_token_visibility(sem_text, tokens)
    if sem_visible:
        llm_status = "llm_output_mentions_value_or_token"
    elif prompt_value_visible:
        llm_status = "prompt_visible_but_no_value_in_semantic_output"
    elif selector_visible or selector_metric_context or prompt_metric_context:
        llm_status = "selector_or_prompt_context_visible_but_numeric_value_absent"
    elif clean_visible:
        llm_status = "not_applicable_selector_omitted"
    else:
        llm_status = "not_applicable_clean_text_absent"

    weak_rows = weak_idx.get((paper_key, system_id), [])
    final_rows = final_idx.get((paper_key, system_id), [])
    weak_value = next((row_field_value(r, field_name) for r in weak_rows if row_field_value(r, field_name)), "")
    final_value = next((row_field_value(r, field_name) for r in final_rows if row_field_value(r, field_name)), "")
    weak_binding = any(row_has_metric_binding(r, field_name, tokens) for r in weak_rows)
    final_binding = any(row_has_metric_binding(r, field_name, tokens) for r in final_rows)

    compare_status = compare_row.get("compare_status", "")
    if compare_status == "blocked_alignment":
        downstream = "alignment_blocked_before_metric_projection"
        first = "endpoint_or_alignment_blocked"
    elif not clean_visible:
        downstream = "clean_text_gate"
        first = "source_value_absent_from_clean_text"
    elif not prompt_value_visible:
        if selector_visible or selector_metric_context or prompt_metric_context:
            downstream = "selector_summary_numeric_visibility"
            first = "selector_included_summary_but_not_numeric_table"
        else:
            downstream = "selector_prompt_gate"
            first = "clean_text_has_value_but_selector_omitted"
    elif prompt_value_visible and not sem_visible:
        downstream = "llm_semantic_authorization"
        first = "llm_prompt_has_value_but_llm_did_not_authorize"
    elif sem_visible and not (weak_value or weak_binding):
        downstream = "stage2_compatibility_projection"
        first = "llm_authorized_but_stage2_projection_lost"
    elif (weak_value or weak_binding) and not (final_value or final_binding):
        downstream = "stage5_materialization"
        first = "stage2_projected_but_stage5_materialization_lost"
    elif (final_value or final_binding) and compare_status == "missing_in_system":
        downstream = "layer3_compare_visibility"
        first = "stage5_has_evidence_but_compare_surface_blank"
    else:
        downstream = "endpoint_or_value_policy"
        first = "endpoint_or_alignment_blocked" if compare_status in {"present_but_mismatch", "blocked_alignment"} else "stage5_has_evidence_but_compare_surface_blank"

    notes = []
    if not tokens:
        notes.append("no_numeric_token_from_gt_value")
    if weak_value:
        notes.append(f"stage2_value={weak_value[:80]}")
    if final_value:
        notes.append(f"stage5_value={final_value[:80]}")
    if weak_binding:
        notes.append("stage2_binding_or_assignment_mentions_metric")
    if final_binding:
        notes.append("stage5_binding_or_evidence_mentions_metric")
    if selector_snippet:
        notes.append(selector_snippet[:180])
    if prompt_snippet and prompt_snippet != selector_snippet:
        notes.append("prompt:" + prompt_snippet[:160])
    if (selector_metric_context or prompt_metric_context) and not prompt_value_visible:
        notes.append("selected_or_prompt_evidence_has_metric_context_but_not_gt_numeric_token")

    return {
        "paper_key": paper_key,
        "field_name": field_name,
        "gt_formulation_id": gt_formulation_id,
        "matched_system_formulation_id": system_id,
        "compare_status": compare_status,
        "gt_value_raw": gt_value,
        "clean_text_value_visibility": "value_token_found_in_clean_text" if clean_visible else "value_token_not_found_in_clean_text",
        "clean_text_locator": clean_locator,
        "selector_visibility": "value_token_found_in_selected_evidence" if selector_visible else "value_token_not_found_in_selected_evidence",
        "selector_artifact_ref": ev_path,
        "prompt_visibility": prompt_visibility,
        "prompt_artifact_ref": prompt_ref,
        "llm_authorization_status": llm_status,
        "downstream_boundary": downstream,
        "first_failure_boundary": first,
        "notes": "; ".join(notes),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--compare-cells-tsv", required=True)
    ap.add_argument("--stage2-run-dir", required=True)
    ap.add_argument("--stage5-final-table-tsv", required=True)
    ap.add_argument("--key2txt-tsv", default="data/cleaned/index/key2txt.tsv")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--include-status", action="append", default=["missing_in_system", "present_but_mismatch", "blocked_alignment"])
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    compare_path = (repo / args.compare_cells_tsv).resolve() if not Path(args.compare_cells_tsv).is_absolute() else Path(args.compare_cells_tsv)
    stage2_dir = (repo / args.stage2_run_dir).resolve() if not Path(args.stage2_run_dir).is_absolute() else Path(args.stage2_run_dir)
    final_path = (repo / args.stage5_final_table_tsv).resolve() if not Path(args.stage5_final_table_tsv).is_absolute() else Path(args.stage5_final_table_tsv)
    output_dir = (repo / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)

    compare_rows = read_tsv(compare_path)
    final_rows = read_tsv(final_path)
    weak_rows = read_tsv(stage2_dir / "semantic_to_widerow_adapter" / "weak_labels__v7pilot_r3_fixparse.tsv")
    key2txt = load_key2txt((repo / args.key2txt_tsv).resolve() if not Path(args.key2txt_tsv).is_absolute() else Path(args.key2txt_tsv))
    prompt_index = load_prompt_index(stage2_dir / "analysis" / "stage2_prompt_preview_v1.tsv")
    semantic_index = load_semantic_index(stage2_dir / "semantic_stage2_objects" / "semantic_stage2_v2_objects.jsonl")

    weak_idx = index_stage_rows(weak_rows, "key", ["formulation_id", "local_instance_id"])
    final_idx = index_stage_rows(final_rows, "key", ["final_formulation_id", "representative_source_formulation_id", "formulation_id", "local_instance_id"])

    wanted_status = set(args.include_status or [])
    audited: List[Dict[str, str]] = []
    for row in compare_rows:
        if row.get("field_name") not in TARGET_FIELDS:
            continue
        if row.get("compare_status") not in wanted_status:
            continue
        if not row.get("gt_value_raw"):
            continue
        audited.append(classify_row(row, repo, stage2_dir, key2txt, prompt_index, semantic_index, weak_idx, final_idx))

    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "measurement_cleantext_selector_visibility_audit_v1.tsv"
    write_tsv(detail_path, audited, OUTPUT_COLUMNS)

    counts: Counter[Tuple[str, str, str]] = Counter()
    for row in audited:
        counts[(row["field_name"], row["compare_status"], row["first_failure_boundary"])] += 1
    summary_rows = [
        {
            "field_name": field,
            "compare_status": status,
            "first_failure_boundary": boundary,
            "count": str(count),
        }
        for (field, status, boundary), count in sorted(counts.items())
    ]
    write_tsv(output_dir / "measurement_cleantext_selector_visibility_summary_v1.tsv", summary_rows, SUMMARY_COLUMNS)

    run_context = output_dir / "RUN_CONTEXT.md"
    run_context.write_text(
        "# RUN_CONTEXT\n\n"
        "run_id: 055_characterization_metric_cleantext_selector_visibility_audit_diagnostic\n\n"
        "benchmark_valid: no\n\n"
        "run_type: diagnostic-only clean-text / selector visibility audit\n\n"
        "source_compare_cells_tsv: " + str(compare_path.relative_to(repo) if compare_path.is_relative_to(repo) else compare_path) + "\n\n"
        "source_stage2_run_dir: " + str(stage2_dir.relative_to(repo) if stage2_dir.is_relative_to(repo) else stage2_dir) + "\n\n"
        "source_stage5_final_table_tsv: " + str(final_path.relative_to(repo) if final_path.is_relative_to(repo) else final_path) + "\n\n"
        "outputs:\n"
        f"- {detail_path.relative_to(repo) if detail_path.is_relative_to(repo) else detail_path}\n"
        f"- {(output_dir / 'measurement_cleantext_selector_visibility_summary_v1.tsv').relative_to(repo) if (output_dir / 'measurement_cleantext_selector_visibility_summary_v1.tsv').is_relative_to(repo) else output_dir / 'measurement_cleantext_selector_visibility_summary_v1.tsv'}\n\n"
        "notes: This audit uses cleaned text, selected evidence block artifacts, replayed semantic Stage2 objects, Stage2 compatibility rows, and Stage5 final rows. It does not read raw PDFs/raw HTML/GT as evidence. Numeric token presence is a conservative visibility signal, not GT authority.\n",
        encoding="utf-8",
    )

    print(f"audited_rows={len(audited)}")
    print(f"detail={detail_path}")
    print(f"summary={output_dir / 'measurement_cleantext_selector_visibility_summary_v1.tsv'}")
    print("top_boundaries:")
    for boundary, count in Counter(row["first_failure_boundary"] for row in audited).most_common():
        print(f"  {boundary}\t{count}")


if __name__ == "__main__":
    main()
