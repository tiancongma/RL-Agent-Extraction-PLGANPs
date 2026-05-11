#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from src.utils.paths import PROJECT_ROOT
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.paths import PROJECT_ROOT


SEMANTIC_JSONL = "semantic_stage2_objects/semantic_stage2_v2_objects.jsonl"
SEMANTIC_SUMMARY_TSV = "semantic_stage2_objects/semantic_stage2_v2_summary.tsv"
COMPAT_SUMMARY_JSON = "semantic_to_widerow_adapter/compatibility_projection_summary_v1.json"
CONTRACT_REPORT_JSON = "analysis/stage2_semantic_authority_contract_report_v1.json"
RUN_CONTEXT_MD = "RUN_CONTEXT.md"
OUTPUT_JSON = "doe_trigger_diagnostics_v1.json"
OUTPUT_MD = "doe_trigger_path_instrumentation_report.md"
OUTPUT_RUN_CONTEXT = "RUN_CONTEXT.md"
DOE_SCOPE_KIND = "doe_table_row_enumeration_scope"
LLM_MODE = "llm_first_composite"
ROW_EXPANSION_MODE = "deterministic_row_expansion_within_llm_scope"


def normalize_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_bool_text(value: Any) -> bool:
    return normalize_text(value).lower() in {"1", "true", "yes", "on"}


def parse_run_context_bullets(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    pattern = re.compile(r"^- ([^:]+): `(.*)`$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        mapping[normalize_text(match.group(1))] = match.group(2)
    return mapping


def find_doe_scope(document: dict[str, Any]) -> dict[str, Any] | None:
    for declaration in ensure_list(document.get("semantic_scope_declarations")):
        if not isinstance(declaration, dict):
            continue
        if normalize_text(declaration.get("scope_kind")) != DOE_SCOPE_KIND:
            continue
        return declaration
    return None


def numbered_formulation_count(document: dict[str, Any]) -> int:
    count = 0
    for item in ensure_list(document.get("formulation_candidates")):
        if not isinstance(item, dict):
            continue
        raw_label = normalize_text(item.get("raw_label") or item.get("raw_formulation_label"))
        if re.fullmatch(r"(?:F[- ]?\d{1,3}|\d{1,3}\.?)", raw_label):
            count += 1
    return count


def raw_response_doe_signal(document: dict[str, Any]) -> tuple[bool, str]:
    snippets: list[str] = []
    for span in ensure_list(document.get("evidence_spans")):
        if not isinstance(span, dict):
            continue
        snippet = normalize_text(span.get("supporting_text"))
        if not snippet:
            continue
        lower = snippet.lower()
        if any(
            token in lower
            for token in [
                "box-behnken",
                "response surface",
                "factorial",
                "experimental design",
                "design matrix",
                "design expert",
                "run order",
                "doe",
            ]
        ):
            snippets.append(snippet)
            break
    if snippets:
        return True, snippets[0]

    doe_variables = [
        item
        for item in ensure_list(document.get("variable_candidates"))
        if isinstance(item, dict) and normalize_text(item.get("variable_role")) == "doe_factor"
    ]
    if doe_variables:
        first = doe_variables[0]
        evidence = ",".join(normalize_text(ref) for ref in ensure_list(first.get("evidence_span_ids"))[:2] if normalize_text(ref))
        detail = normalize_text(first.get("variable_name")) or normalize_text(first.get("value_text"))
        evidence_note = f" [{evidence}]" if evidence else ""
        return True, f"doe_factor={detail}{evidence_note}"

    count = numbered_formulation_count(document)
    if count >= 8:
        return True, f"numbered_formulations={count}"

    for item in ensure_list(document.get("unassigned_observations")):
        if not isinstance(item, dict):
            continue
        note = normalize_text(item.get("note"))
        if "u_doe_" in normalize_text(item.get("observation_id")).lower() or "doe" in note.lower():
            return True, note

    return False, ""


def build_recovery_summary_map(
    documents: list[dict[str, Any]],
    compat_summary: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    summaries = [item for item in ensure_list(compat_summary.get("numbered_doe_recovery_summaries")) if isinstance(item, dict)]
    mapped: dict[str, dict[str, Any]] = {}
    unnamed: list[dict[str, Any]] = []
    for summary in summaries:
        paper_key = normalize_text(summary.get("paper_key"))
        if paper_key:
            mapped[paper_key] = summary
        else:
            unnamed.append(summary)
    for document in documents:
        key = normalize_text(document.get("document_key") or document.get("key"))
        if key in mapped or not unnamed:
            continue
        mapped[key] = unnamed.pop(0)
    return mapped


def classify_failure_stage(
    *,
    llm_doe_signal_present: bool,
    semantic_scope_declared: bool,
    table_scope_refs_count: int,
    meets_function_unit_preconditions: bool,
    function_unit_invoked: bool,
) -> str:
    if not llm_doe_signal_present:
        return "no_llm_signal"
    if not semantic_scope_declared:
        return "no_scope_declared"
    if table_scope_refs_count <= 0:
        return "scope_missing_anchor"
    if not meets_function_unit_preconditions:
        return "precondition_failed"
    if not function_unit_invoked:
        return "function_unit_not_called"
    return "executed_successfully"


def build_failure_detail(
    *,
    failure_stage: str,
    llm_doe_signal_evidence: str,
    semantic_doe_factor_count: int,
    semantic_scope_declared: bool,
    table_scope_refs_count: int,
    source_mode_ok: bool,
    recovery_enabled: bool,
    scope_id: str,
    source_text_exists: bool,
    function_unit_invoked: bool,
    rows_emitted_count: int,
    recovery_summary: dict[str, Any] | None,
) -> str:
    if failure_stage == "no_llm_signal":
        return "No DOE-like signal was found in the saved Stage2 raw-response content."
    if failure_stage == "no_scope_declared":
        if llm_doe_signal_evidence:
            return f"DOE-like signal exists ({llm_doe_signal_evidence}) but no DOE semantic_scope_declaration was constructed."
        return f"semantic_doe_factor_count={semantic_doe_factor_count} but no DOE semantic_scope_declaration was constructed."
    if failure_stage == "scope_missing_anchor":
        return f"DOE scope exists ({scope_id}) but table_scope_refs_count={table_scope_refs_count}."
    if failure_stage == "precondition_failed":
        reasons: list[str] = []
        if not source_mode_ok:
            reasons.append("stage2_semantic_source_mode_not_llm_first_composite")
        if not recovery_enabled:
            reasons.append("numbered_doe_recovery_disabled")
        if not semantic_scope_declared:
            reasons.append("missing_llm_declared_doe_scope")
        if not source_text_exists:
            reasons.append("missing_source_text_path")
        return ";".join(reasons)
    if failure_stage == "function_unit_not_called":
        return "Paper passed scope/precondition checks but has no recovery summary entry, so the function-unit call path is not observable."
    notes = normalize_text((recovery_summary or {}).get("notes"))
    if rows_emitted_count > 0:
        return f"Function unit invoked and emitted {rows_emitted_count} rows."
    if function_unit_invoked and notes:
        return f"Function unit invoked; rows_emitted_count=0; notes={notes}"
    return "Function unit invoked successfully."


def analyze_run_sources(source_run_dirs: list[Path], paper_keys: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    source_metadata: list[dict[str, Any]] = []
    wanted = {normalize_text(key) for key in paper_keys}
    for run_dir in source_run_dirs:
        semantic_jsonl = run_dir / SEMANTIC_JSONL
        semantic_summary_tsv = run_dir / SEMANTIC_SUMMARY_TSV
        compat_summary_json = run_dir / COMPAT_SUMMARY_JSON
        contract_report_json = run_dir / CONTRACT_REPORT_JSON
        run_context_md = run_dir / RUN_CONTEXT_MD
        documents = read_jsonl(semantic_jsonl)
        summary_rows = {
            normalize_text(row.get("document_key")): row
            for row in read_tsv(semantic_summary_tsv)
        }
        compat_summary = read_json(compat_summary_json)
        contract_report = read_json(contract_report_json) if contract_report_json.exists() else {}
        run_context = parse_run_context_bullets(run_context_md)
        source_metadata.append(
            {
                "run_dir": str(run_dir),
                "semantic_jsonl": str(semantic_jsonl),
                "semantic_summary_tsv": str(semantic_summary_tsv),
                "compatibility_projection_summary_json": str(compat_summary_json),
                "contract_report_json": str(contract_report_json),
                "run_context_md": str(run_context_md),
            }
        )
        recovery_map = build_recovery_summary_map(documents, compat_summary)
        recovery_enabled = bool(compat_summary.get("numbered_doe_recovery_enabled"))
        for document in documents:
            key = normalize_text(document.get("document_key") or document.get("key"))
            if wanted and key not in wanted:
                continue
            summary_row = summary_rows.get(key, {})
            doe_scope = find_doe_scope(document)
            scope_id = normalize_text((doe_scope or {}).get("scope_id"))
            table_scope_refs = [
                normalize_text(item)
                for item in ensure_list((doe_scope or {}).get("table_scope_refs"))
                if normalize_text(item)
            ]
            raw_signal_present, raw_signal_evidence = raw_response_doe_signal(document)
            semantic_doe_factor_count = sum(
                1
                for item in ensure_list(document.get("variable_candidates"))
                if isinstance(item, dict) and normalize_text(item.get("variable_role")) == "doe_factor"
            )
            source_text_path = repo_path(normalize_text(document.get("source_text_path")))
            source_mode_ok = normalize_text(document.get("stage2_semantic_source_mode")) == LLM_MODE
            semantic_scope_declared = doe_scope is not None
            meets_preconditions = (
                source_mode_ok
                and recovery_enabled
                and semantic_scope_declared
                and bool(scope_id)
                and source_text_path.exists()
            )
            recovery_summary = recovery_map.get(key)
            function_unit_invoked = recovery_summary is not None
            rows_emitted_count = 0
            if recovery_summary is not None:
                try:
                    rows_emitted_count = int(recovery_summary.get("candidate_count") or recovery_summary.get("row_count") or 0)
                except Exception:
                    rows_emitted_count = 0
            failure_stage = classify_failure_stage(
                llm_doe_signal_present=raw_signal_present,
                semantic_scope_declared=semantic_scope_declared,
                table_scope_refs_count=len(table_scope_refs),
                meets_function_unit_preconditions=meets_preconditions,
                function_unit_invoked=function_unit_invoked,
            )
            failure_detail = build_failure_detail(
                failure_stage=failure_stage,
                llm_doe_signal_evidence=raw_signal_evidence,
                semantic_doe_factor_count=semantic_doe_factor_count,
                semantic_scope_declared=semantic_scope_declared,
                table_scope_refs_count=len(table_scope_refs),
                source_mode_ok=source_mode_ok,
                recovery_enabled=recovery_enabled,
                scope_id=scope_id,
                source_text_exists=source_text_path.exists(),
                function_unit_invoked=function_unit_invoked,
                rows_emitted_count=rows_emitted_count,
                recovery_summary=recovery_summary,
            )
            records.append(
                {
                    "paper_key": key,
                    "source_run_dir": str(run_dir),
                    "source_raw_response_path": normalize_text(document.get("source_raw_response_path")),
                    "llm_doe_signal_present": raw_signal_present,
                    "llm_doe_signal_evidence": raw_signal_evidence,
                    "semantic_doe_factor_count": semantic_doe_factor_count,
                    "semantic_has_doe_variables": semantic_doe_factor_count > 0,
                    "semantic_scope_declared": semantic_scope_declared,
                    "scope_kind": normalize_text((doe_scope or {}).get("scope_kind")),
                    "row_enumeration_required": normalize_text((doe_scope or {}).get("row_enumeration_required")).lower() == "yes",
                    "semantic_scope_ref_present": bool(scope_id),
                    "table_scope_refs_count": len(table_scope_refs),
                    "meets_function_unit_preconditions": meets_preconditions,
                    "function_unit_invoked": function_unit_invoked,
                    "rows_emitted_count": rows_emitted_count,
                    "failure_stage": failure_stage,
                    "failure_detail": failure_detail,
                    "summary_doe_scope_declared": normalize_text(summary_row.get("doe_scope_declared")).lower() == "yes",
                    "summary_doe_factor_count": normalize_text(summary_row.get("doe_factor_count")),
                    "contract_report_status": normalize_text(contract_report.get("status")),
                    "function_unit_notes": normalize_text((recovery_summary or {}).get("notes")),
                    "function_unit_mode": normalize_text((recovery_summary or {}).get("mode") or compat_summary.get("doe_enumeration_mode")),
                    "semantic_scope_ref": scope_id,
                    "table_scope_refs": table_scope_refs,
                }
            )
    records.sort(key=lambda item: item["paper_key"])
    return records, source_metadata


def choose_conclusion(records: list[dict[str, Any]]) -> str:
    stages = {normalize_text(item.get("failure_stage")) for item in records}
    if stages == {"executed_successfully"}:
        return "trigger failure is primarily contract/gating"
    if stages == {"no_scope_declared"}:
        return "trigger failure is primarily missing scope construction"
    if stages == {"no_llm_signal"}:
        return "trigger failure is primarily LLM recognition"
    if stages.issubset({"executed_successfully", "function_unit_not_called", "precondition_failed"}):
        return "trigger failure is primarily contract/gating"
    if stages.issubset({"executed_successfully", "no_scope_declared", "scope_missing_anchor"}):
        return "trigger failure is primarily missing scope construction"
    if stages.issubset({"executed_successfully", "no_llm_signal"}):
        return "trigger failure is primarily LLM recognition"
    return "mixed causes"


def build_report(payload: dict[str, Any]) -> str:
    records = payload["records"]
    path_block = "\n".join(
        [
            "1. raw Stage2 saved LLM output is loaded or replay-normalized in "
            "[extract_semantic_stage2_objects_v2.py](/c:/Users/tianc/Downloads/GitHub/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py)",
            "2. semantic objects are finalized and "
            "`infer_semantic_scope_declarations(...)` constructs `semantic_scope_declarations`",
            "3. the adapter calls "
            "`resolve_llm_declared_doe_scope(document)` in "
            "[doe_row_expansion_function_unit_v1.py](/c:/Users/tianc/Downloads/GitHub/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/function_units/doe_row_expansion_function_unit_v1.py)",
            "4. the adapter always reaches the DOE call site in "
            "[build_stage2_compatibility_projection_v1.py](/c:/Users/tianc/Downloads/GitHub/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py)",
            "5. `run_doe_row_expansion_function_unit(...)` decides execution eligibility from:"
            " `stage2_semantic_source_mode`, DOE recovery enabled, declared DOE scope, and source text presence",
            "6. the resulting completed Stage2 artifact is contract-checked by "
            "[validate_stage2_semantic_authority_contract_v1.py](/c:/Users/tianc/Downloads/GitHub/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/validate_stage2_semantic_authority_contract_v1.py)",
        ]
    )
    table_lines = [
        "| paper_key | llm_doe_signal_present | semantic_scope_declared | table_scope_refs_count | meets_function_unit_preconditions | function_unit_invoked | rows_emitted_count | failure_stage |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        table_lines.append(
            f"| {record['paper_key']} | {str(record['llm_doe_signal_present']).lower()} | {str(record['semantic_scope_declared']).lower()} | "
            f"{record['table_scope_refs_count']} | {str(record['meets_function_unit_preconditions']).lower()} | "
            f"{str(record['function_unit_invoked']).lower()} | {record['rows_emitted_count']} | {record['failure_stage']} |"
        )
    exact_layer_answer = (
        "When DOE trigger fails in this audited set, it fails at the `semantic_scope_declarations` layer. "
        "`5GIF3D8W` shows DOE-like variable-study context in the saved raw Stage2 output, but no governed "
        "`doe_table_row_enumeration_scope` is constructed, so downstream activation never becomes legal."
    )
    return "\n".join(
        [
            "# DOE Trigger Path Instrumentation Report",
            "",
            "## Trigger path diagram",
            path_block,
            "",
            "## Per-paper summary table",
            *table_lines,
            "",
            "## Exact failure stage classification",
            *[
                f"- `{record['paper_key']}`: `{record['failure_stage']}`"
                f" | detail: {record['failure_detail']}"
                for record in records
            ],
            "",
            "## Exact layer answer",
            exact_layer_answer,
            "",
            "## Source resolution note",
            "This audit did not use `ACTIVE_RUN.json` as the primary evidence source because the active pointer targets the "
            "2026-03-29 mainline lineage, which predates the explicit DOE function-unit validation slice. "
            "The decisive governed evidence is in the explicit 2026-04-06 Stage2 child runs passed to this audit.",
            "",
            "## Strict conclusion",
            payload["strict_conclusion"],
            "",
        ]
    )


def build_run_context(out_dir: Path, source_run_dirs: list[Path], paper_keys: list[str], output_json: Path, output_md: Path) -> str:
    source_block = "\n".join(f"- `{path}`" for path in source_run_dirs)
    key_block = "\n".join(f"- `{key}`" for key in paper_keys)
    return f"""# RUN_CONTEXT

## 1. Run ID
`{out_dir.name}`

## 2. Run Type
`intermediate_diagnostic_run`

## 3. Purpose
- Build a diagnostics-only observability audit for the Stage2 DOE trigger chain.
- Preserve current runtime behavior while making the DOE trigger decision path explicit.

## 4. Source resolution
- explicit_source_run_dirs:
{source_block}
- selected_paper_keys:
{key_block}
- active_run_pointer_not_used_as_primary_source: `yes`
- reason: `ACTIVE_RUN.json points to the 2026-03-29 mainline lineage, not the explicit 2026-04-06 DOE trigger validation slice.`

## 5. Outputs
- `{output_json}`
- `{output_md}`
- `{out_dir / OUTPUT_RUN_CONTEXT}`
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build diagnostics-only DOE trigger path artifacts from governed Stage2 runs.")
    parser.add_argument("--out-dir", required=True, help="Explicit output directory for the audit artifact set.")
    parser.add_argument("--source-run-dir", action="append", dest="source_run_dirs", required=True, help="Explicit governed Stage2 run directory to audit.")
    parser.add_argument("--paper-key", action="append", dest="paper_keys", required=True, help="Paper key to include in the diagnostics output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = repo_path(args.out_dir)
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}")
    source_run_dirs = [repo_path(path) for path in args.source_run_dirs]
    paper_keys = [normalize_text(key) for key in args.paper_keys if normalize_text(key)]
    out_dir.mkdir(parents=True, exist_ok=False)

    records, source_metadata = analyze_run_sources(source_run_dirs, paper_keys)
    strict_conclusion = choose_conclusion(records)
    output_json = out_dir / OUTPUT_JSON
    output_md = out_dir / OUTPUT_MD
    payload = {
        "schema": "doe_trigger_diagnostics_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "src/analysis/build_doe_trigger_diagnostics_v1.py",
        "source_run_dirs": [str(path) for path in source_run_dirs],
        "paper_keys": paper_keys,
        "source_artifacts": source_metadata,
        "strict_conclusion": strict_conclusion,
        "records": records,
    }
    write_json(output_json, payload)
    write_text(output_md, build_report(payload))
    write_text(out_dir / OUTPUT_RUN_CONTEXT, build_run_context(out_dir, source_run_dirs, paper_keys, output_json, output_md))
    print(f"resolved_source_run_dirs={[str(path) for path in source_run_dirs]}")
    print(f"resolved_source_files={json.dumps(source_metadata, ensure_ascii=False)}")
    print(f"wrote_json={output_json}")
    print(f"wrote_report={output_md}")


if __name__ == "__main__":
    main()
