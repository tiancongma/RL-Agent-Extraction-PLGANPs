#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

try:
    from src.utils.paths import (
        DATA_CLEANED_INDEX_DIR,
        DATA_RESULTS_DIR,
        PROJECT_ROOT,
        dataset_tables_root,
    )
    from src.utils.run_id import resolve_results_write_target
    from src.stage2_sampling_labels.denoise_stage2_source_text_s2_1b_v1 import run_denoise_projection
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.paths import (
        DATA_CLEANED_INDEX_DIR,
        DATA_RESULTS_DIR,
        PROJECT_ROOT,
        dataset_tables_root,
    )
    from src.utils.run_id import resolve_results_write_target
    from src.stage2_sampling_labels.denoise_stage2_source_text_s2_1b_v1 import run_denoise_projection


SEMANTIC_SUBDIR = "semantic_stage2_objects"
COMPAT_SUBDIR = "semantic_to_widerow_adapter"
SEMANTIC_JSONL = "semantic_stage2_v2_objects.jsonl"
SEMANTIC_SUMMARY = "semantic_stage2_v2_summary.tsv"
FINAL_STAGE2_TSV = "weak_labels__v7pilot_r3_fixparse.tsv"
FINAL_STAGE2_JSONL = "weak_labels__v7pilot_r3_fixparse.jsonl"
STAGE2_RUN_METADATA_JSON = "stage2_run_metadata_v1.json"
STAGE2_CONTRACT_REPORT_JSON = "stage2_semantic_authority_contract_report_v1.json"
STAGE2_SEMANTIC_SOURCE_MODE = "llm_first_composite"
REQUEST_SUMMARY_NAME = "request_summary.tsv"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def run_command(command: list[str]) -> None:
    print("running_command=" + " ".join(str(part) for part in command))
    subprocess.run(command, cwd=PROJECT_ROOT, text=True, check=True)


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def normalize_text(value: str) -> str:
    return str(value or "").strip()


def normalize_key(row: dict[str, str]) -> str:
    return normalize_text(row.get("key") or row.get("paper_key") or row.get("zotero_key"))


def infer_text_source_type(text_path: str) -> str:
    lower = normalize_text(text_path).lower()
    if lower.endswith(".html.txt"):
        return "html"
    if lower.endswith(".pdf.txt"):
        return "pdf"
    return ""


def load_key2txt_map(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Authoritative key2txt surface not found: {path}")
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            key = normalize_text(row[0])
            text_path = normalize_text(row[1])
            if key and text_path:
                out[key] = text_path
    return out


def load_key2structure_map(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            key = normalize_text(row.get("key"))
            if key:
                out[key] = {field: normalize_text(value).replace("\\\\", "/") for field, value in row.items()}
    return out


def resolve_project_file(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def to_repo_rel(path_value: Path) -> str:
    return str(path_value.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")


def infer_dataset_id_from_table_dir(table_dir: Path) -> str:
    try:
        return table_dir.resolve().parent.parent.name
    except Exception:
        return ""


def iter_stage2_table_dir_candidates(*, key: str, text_path: Path | None) -> list[Path]:
    candidates: list[Path] = []
    if text_path is not None:
        candidates.append(text_path.parent.parent / "tables" / key)
    cleaned_root = PROJECT_ROOT / "data" / "cleaned"
    if cleaned_root.exists():
        for dataset_root in sorted(path for path in cleaned_root.iterdir() if path.is_dir()):
            candidates.append(dataset_root / "tables" / key)
    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def resolve_stage2_table_binding(row: dict[str, str]) -> tuple[str, Path | None]:
    key = normalize_key(row)
    dataset_id = normalize_text(row.get("dataset_id"))
    text_path_value = normalize_text(row.get("text_path"))
    text_path: Path | None = None
    if text_path_value:
        text_path = resolve_project_file(text_path_value)

    if dataset_id and key:
        explicit_dir = dataset_tables_root(dataset_id) / key
        if explicit_dir.exists():
            return dataset_id, explicit_dir.resolve()

    if not key:
        return dataset_id, None

    for candidate in iter_stage2_table_dir_candidates(key=key, text_path=text_path):
        if candidate.exists():
            inferred_dataset_id = infer_dataset_id_from_table_dir(candidate)
            return inferred_dataset_id or dataset_id, candidate
    return dataset_id, None


def refresh_stage2_text_bindings(selected_rows: list[dict[str, str]], key2txt_path: Path) -> list[dict[str, str]]:
    key2txt_map = load_key2txt_map(key2txt_path)
    refreshed_rows: list[dict[str, str]] = []
    missing_bindings: list[str] = []
    missing_files: list[str] = []

    for row in selected_rows:
        refreshed = dict(row)
        key = normalize_key(row)
        explicit_text_path = normalize_text(row.get("text_path"))
        if explicit_text_path:
            text_path = explicit_text_path.replace("\\", "/")
            resolved_text_path = resolve_project_file(text_path)
            if not resolved_text_path.exists():
                missing_files.append(f"{key or '<missing_key>'}: {resolved_text_path}")
            refreshed["text_path"] = text_path
            refreshed["text_source_type"] = normalize_text(row.get("text_source_type")) or infer_text_source_type(text_path)
            refreshed["text_available"] = "yes" if resolved_text_path.exists() else "missing_file"
            refreshed_rows.append(refreshed)
            continue
        text_path = key2txt_map.get(key, "")
        if not key or not text_path:
            missing_bindings.append(key or "<missing_key>")
            refreshed_rows.append(refreshed)
            continue
        resolved_text_path = resolve_project_file(text_path)
        if not resolved_text_path.exists():
            missing_files.append(f"{key}: {resolved_text_path}")
        refreshed["text_path"] = text_path
        refreshed["text_source_type"] = infer_text_source_type(text_path)
        refreshed["text_available"] = "yes" if resolved_text_path.exists() else "missing_file"
        refreshed_rows.append(refreshed)

    if missing_bindings:
        raise FileNotFoundError(
            "Selected Stage2 scope is missing authoritative key2txt bindings for: "
            + ", ".join(sorted(missing_bindings))
        )
    if missing_files:
        raise FileNotFoundError(
            "Selected Stage2 scope resolved to missing clean-text files: "
            + "; ".join(sorted(missing_files))
        )
    return refreshed_rows


def refresh_stage2_structure_bindings(selected_rows: list[dict[str, str]], key2structure_path: Path) -> list[dict[str, str]]:
    key2structure_map = load_key2structure_map(key2structure_path)
    refreshed_rows: list[dict[str, str]] = []
    for row in selected_rows:
        refreshed = dict(row)
        key = normalize_key(row)
        binding = key2structure_map.get(key, {})
        structure_path = normalize_text(refreshed.get("structure_path")) or normalize_text(binding.get("structure_path"))
        if structure_path:
            refreshed["structure_path"] = structure_path.replace("\\", "/")
            refreshed["structure_available"] = "yes" if resolve_project_file(refreshed["structure_path"]).exists() else "missing_file"
        else:
            refreshed["structure_path"] = ""
            refreshed["structure_available"] = "no"
        if not normalize_text(refreshed.get("table_dir")) and normalize_text(binding.get("tables_dir")):
            refreshed["table_dir"] = normalize_text(binding.get("tables_dir")).replace("\\", "/")
        refreshed_rows.append(refreshed)
    return refreshed_rows


def refresh_stage2_table_bindings(selected_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    refreshed_rows: list[dict[str, str]] = []
    for row in selected_rows:
        refreshed = dict(row)
        explicit_table_dir = normalize_text(refreshed.get("table_dir"))
        if explicit_table_dir:
            explicit_table_dir = explicit_table_dir.replace("\\", "/")
            refreshed["table_dir"] = explicit_table_dir
            refreshed["table_available"] = "yes" if resolve_project_file(explicit_table_dir).exists() else "missing_file"
            refreshed_rows.append(refreshed)
            continue
        resolved_dataset_id, table_dir = resolve_stage2_table_binding(refreshed)
        if resolved_dataset_id:
            refreshed["dataset_id"] = resolved_dataset_id
        if table_dir is not None and table_dir.exists():
            refreshed["table_dir"] = to_repo_rel(table_dir)
            refreshed["table_available"] = "yes"
        else:
            refreshed["table_dir"] = ""
            refreshed["table_available"] = "no"
        refreshed_rows.append(refreshed)
    return refreshed_rows


def apply_s2_1b_denoise_projection_to_manifest_rows(
    selected_rows: list[dict[str, str]],
    *,
    run_dir: Path,
) -> list[dict[str, str]]:
    """Materialize S2-1b projections without overwriting raw clean-text authority.

    ``text_path`` remains the raw/current clean-text audit authority. Downstream
    S2-2 consumes ``source_s2_1b_denoised_text_path`` when present and records
    both surfaces in provenance.
    """
    inputs: list[tuple[str, Path]] = []
    for row in selected_rows:
        key = normalize_key(row)
        text_path_value = normalize_text(row.get("text_path"))
        if not key or not text_path_value:
            continue
        text_path = resolve_project_file(text_path_value)
        if not text_path.exists():
            raise FileNotFoundError(f"Missing source text for S2-1b denoise projection: {key}: {text_path}")
        inputs.append((key, text_path))
    if not inputs:
        return [dict(row) for row in selected_rows]

    summary_path = run_denoise_projection(inputs=inputs, run_dir=run_dir)
    hydrated_rows: list[dict[str, str]] = []
    for row in selected_rows:
        hydrated = dict(row)
        key = normalize_key(row)
        if key:
            denoised_path = run_dir / SEMANTIC_SUBDIR / "s2_1b_denoised_text" / f"{key}.txt"
            audit_path = run_dir / SEMANTIC_SUBDIR / "s2_1b_denoise_audit" / f"{key}_s2_1b_denoise_audit_v1.json"
            if denoised_path.exists():
                hydrated["source_text_projection"] = "s2_1b_denoised"
                hydrated["source_raw_clean_text_path"] = normalize_text(row.get("text_path"))
                hydrated["source_s2_1b_denoised_text_path"] = str(denoised_path)
                hydrated["s2_1b_denoise_audit_path"] = str(audit_path)
                hydrated["s2_1b_denoise_summary_path"] = str(summary_path)
        hydrated_rows.append(hydrated)
    return hydrated_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the governed composite Stage2 path: LLM semantic discovery followed by deterministic post-LLM completion."
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Explicit legacy compatibility run_id. New writes default to MDEC084 v2 bucket/child naming when omitted.",
    )
    parser.add_argument(
        "--run-dir",
        default="",
        help="Explicit results run directory. Supports legacy roots and v2 child execution paths under data/results/.",
    )
    parser.add_argument(
        "--execution-cue",
        default="stage2",
        help="Future-facing child cue used only when auto-allocating a new v2 child execution path.",
    )
    parser.add_argument("--manifest-tsv", required=True)
    parser.add_argument("--paper-key", action="append", dest="paper_keys", default=[])
    parser.add_argument(
        "--source-mode",
        choices=["live_llm", "legacy_llm_replay"],
        default="live_llm",
        help="Run live Stage2 extraction or replay saved raw responses through the maintained Stage2 completion path.",
    )
    parser.add_argument("--legacy-raw-responses-dir", default="")
    parser.add_argument(
        "--fallback-legacy-raw-responses-dir",
        default="",
        help="Optional richer legacy raw-response directory used when replayed live-v2 raw responses collapse to the minimal shrunken contract.",
    )
    parser.add_argument("--llm-backend", choices=["gemini", "nvidia", "ollama"], default="gemini")
    parser.add_argument("--model", default="", help="Live-call model. Required only with --source-mode live_llm unless --stop-before-live-call is set.")
    parser.add_argument("--max-text-chars", type=int, default=18000)
    parser.add_argument("--request-timeout-seconds", type=int, default=180)
    parser.add_argument("--request-retries", type=int, default=1)
    parser.add_argument("--retry-sleep-sec", type=float, default=3.0)
    parser.add_argument(
        "--stop-before-live-call",
        action="store_true",
        help="Materialize maintained pre-LLM Stage2 artifacts only and stop before any S2-4b live or replay raw-response handling.",
    )
    parser.add_argument(
        "--stage1-table-cell-sidecar-root",
        default="",
        help="Optional explicit Stage1 table-cell sidecar root passed to S2-2 table authority reconstruction.",
    )
    return parser.parse_args()


def build_run_context(
    *,
    run_id: str,
    run_dir: Path,
    run_dir_kind: str,
    run_selection_mode: str,
    bucket_dir: Path,
    manifest_tsv: Path,
    selected_keys: list[str],
    source_mode: str,
    llm_backend: str,
    model: str,
    max_text_chars: int,
    legacy_raw_responses_dir: Path | None,
    semantic_dir: Path,
    compat_dir: Path,
) -> str:
    key_block = "\n".join(f"- `{key}`" for key in selected_keys) if selected_keys else "- `all manifest rows`"
    legacy_note = f"- legacy_raw_responses_dir: `{legacy_raw_responses_dir}`" if legacy_raw_responses_dir else "- legacy_raw_responses_dir: ``"
    table_mode = os.getenv("STAGE2_TABLE_MODE", "full")
    first_column_enhancement = os.getenv("STAGE2_TABLE_SUMMARY_FIRST_COLUMN_ENHANCEMENT", "")
    input_packing_mode = os.getenv("STAGE2_INPUT_EVIDENCE_PACKING_MODE", "")
    doe_recovery = os.getenv("STAGE2_ENABLE_NUMBERED_DOE_RECOVERY", "")
    doe_mode = os.getenv("STAGE2_DOE_ENUMERATION_MODE", "")
    doe_min_rows = os.getenv("STAGE2_NUMBERED_DOE_MIN_ROWS", "8")
    return f"""# RUN_CONTEXT

## 1. Run ID
`{run_id}`

## 1a. Run Path
- run_dir: `{run_dir}`
- run_dir_kind: `{run_dir_kind}`
- run_selection_mode: `{run_selection_mode}`
- bucket_dir: `{bucket_dir}`

## 2. Run Type
`intermediate_diagnostic_run`

Benchmark reporting rule:
- This run is `diagnostic-only, not benchmark-valid final output`.
- It builds the composite Stage2 output only.
- Benchmark-valid system reporting still requires downstream Stage3 and Stage5 execution.

## 3. Purpose
- Run the governed composite Stage2 path for the declared manifest scope.
- Keep Stage2 internal structure explicit:
  1. LLM semantic discovery
  2. deterministic post-LLM completion for downstream readiness
- Produce the only authoritative Stage2 output contract consumed by Stage3.

  ## 4. Stage2 composite contract
  - stage2_semantic_source_mode: `{STAGE2_SEMANTIC_SOURCE_MODE}`
  - formal S2-2 boundary:
    - `clean text -> governed evidence package (pre-LLM)`
  - explicit internal candidate-segmentation boundary inside S2-2:
    - `clean text / extracted tables -> candidate segmentation -> evidence-driven selector -> governed evidence package`
  - canonical candidate-segmentation artifact:
    - `{semantic_dir / 'candidate_blocks'} / <paper_key> / candidate_blocks_v1.json`
  - canonical S2-2 artifact:
    - `{semantic_dir / 'evidence_blocks'} / <paper_key> / evidence_blocks_v1.json`
  - maintained candidate-segmentation profile:
    - `section_aware_candidate_segmentation_v1`
    - section-aware prose splitting active
    - table isolation active when table assets exist
    - conservative candidate-level noise filtering active
  - maintained S2-2 selector:
    - deterministic evidence-driven evidence selection
    - conservative noise filtering plus weak importance ordering
    - pre-LLM archetype detection is metadata only and does not drive selection
    - no second LLM at this boundary
  - Stage2 internal intermediate:
    - LLM semantic discovery objects under `{semantic_dir}`
  - S2-2 prompt observability is derived from the canonical evidence artifact:
    - `{run_dir / 'analysis' / 'stage2_prompt_preview_v1.tsv'}`
  - Stage2 authoritative final output:
    - deterministic post-LLM completion under `{compat_dir}`
  - Stage3 must consume only the completed Stage2 artifact, not raw LLM semantic objects alone.
- No formulation candidate may enter the authoritative Stage2 artifact unless it
  is traceable to:
  - `llm_semantic_discovery`
  - or an explicitly declared governed fallback mode
- Deterministic Stage2 completion may expand, normalize, resolve, or bridge
  rows, but it must not silently become the semantic source of the formulation
  universe in this run mode.
- DOE row expansion is allowed only within LLM-declared DOE scope in this run
  mode.
- Governed DOE deterministic expansion is isolated inside:
  - `src/stage2_sampling_labels/function_units/doe_row_expansion_function_unit_v1.py`
- Sequential optimization resolution is allowed only within LLM-declared
  document scope when explicit stagewise selection evidence exists and no DOE
  scope is declared.
- Governed sequential optimization resolution is isolated inside:
  - `src/stage2_sampling_labels/function_units/sequential_optimization_interpreter_v1.py`

## 5. Scope and inputs
- manifest_tsv: `{manifest_tsv}`
- selected_paper_keys:
{key_block}
- source_mode: `{source_mode}`
- source_mode_note: `{('saved_raw_response_replay' if source_mode == 'legacy_llm_replay' else 'live_llm_execution')}`
- llm_backend: `{llm_backend}`
- model: `{model}`
- max_text_chars: `{max_text_chars}`
- stage2_table_mode: `{table_mode}`
- stage2_table_summary_first_column_enhancement: `{first_column_enhancement or '0'}`
- stage2_input_evidence_packing_mode: `{input_packing_mode or 'off'}`
- stage2_enable_numbered_doe_recovery: `{doe_recovery or '0'}`
- stage2_doe_enumeration_mode: `{doe_mode or 'off'}`
- stage2_numbered_doe_min_rows: `{doe_min_rows}`
{legacy_note}

## 6. Exact script execution order
1. `src/stage2_sampling_labels/run_stage2_composite_v1.py`
2. `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`
3. `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`

  ## 7. Outputs
  - formal S2 internal candidate-segmentation artifacts:
    - `{semantic_dir / 'candidate_blocks'} / <paper_key> / candidate_blocks_v1.json`
  - formal S2-2 evidence artifacts:
    - `{semantic_dir / 'evidence_blocks'} / <paper_key> / evidence_blocks_v1.json`
  - intermediate semantic objects:
    - `{semantic_dir / SEMANTIC_JSONL}`
    - `{semantic_dir / SEMANTIC_SUMMARY}`
  - derived S2-2 observability artifacts:
    - `{run_dir / 'analysis' / 'candidate_segmentation_debug_v1.tsv'}`
    - `{run_dir / 'analysis' / 'stage2_prompt_preview_v1.tsv'}`
    - `{run_dir / 'analysis' / 'table_selection_debug_v1.json'}`
  - final Stage2 completion artifacts:
    - `{compat_dir / FINAL_STAGE2_TSV}`
  - `{compat_dir / FINAL_STAGE2_JSONL}`
  - `{compat_dir / 'compatibility_projection_trace_v1.tsv'}`
  - `{compat_dir / 'compatibility_projection_summary_v1.json'}`
- run context:
  - `{run_dir / 'RUN_CONTEXT.md'}`
- machine-readable run metadata:
  - `{run_dir / STAGE2_RUN_METADATA_JSON}`
- contract-validation report:
  - `{run_dir / 'analysis' / STAGE2_CONTRACT_REPORT_JSON}`

## 8. Evaluation guardrail
- Raw LLM semantic objects are an internal Stage2 intermediate only.
- Structural Stage2 evaluation for downstream readiness must target the completed Stage2 artifact after deterministic post-LLM completion.
- Direct comparison of raw LLM semantic objects to formulation-level GT is diagnostic-only failure localization and is not the authoritative Stage2 evaluation object.
"""


def main() -> None:
    args = parse_args()
    # Load repo-local environment variables before any live Stage2 backend probe.
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

    manifest_tsv = repo_path(args.manifest_tsv)
    if not manifest_tsv.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_tsv}")

    manifest_rows = read_tsv(manifest_tsv)
    selected_keys = [str(key).strip() for key in args.paper_keys if str(key).strip()]
    if selected_keys:
        selected_rows = [row for row in manifest_rows if str(row.get("key", "")).strip() in selected_keys]
        found_keys = {str(row.get("key", "")).strip() for row in selected_rows}
        missing = [key for key in selected_keys if key not in found_keys]
        if missing:
            raise ValueError(f"Manifest missing requested paper keys: {missing}")
    else:
        selected_rows = manifest_rows
        selected_keys = [str(row.get("key", "")).strip() for row in manifest_rows if str(row.get("key", "")).strip()]

    target = resolve_results_write_target(
        results_root=DATA_RESULTS_DIR,
        default_child_cue=args.execution_cue,
        explicit_run_dir=repo_path(args.run_dir) if str(args.run_dir).strip() else None,
        explicit_legacy_run_id=args.run_id,
    )
    run_dir = Path(target["run_dir"])
    run_id = target["run_basename"]
    run_dir_kind = target["path_kind"]
    run_selection_mode = target["selection_mode"]
    bucket_dir = Path(target["bucket_dir"])
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    if run_dir_kind == "v2_child_execution":
        bucket_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=False)

    key2txt_tsv = (DATA_CLEANED_INDEX_DIR / "key2txt.tsv").resolve()
    key2structure_tsv = (DATA_CLEANED_INDEX_DIR / "key2structure.tsv").resolve()
    selected_rows = refresh_stage2_text_bindings(selected_rows, key2txt_tsv)
    selected_rows = refresh_stage2_structure_bindings(selected_rows, key2structure_tsv)
    selected_rows = refresh_stage2_table_bindings(selected_rows)
    selected_rows = apply_s2_1b_denoise_projection_to_manifest_rows(selected_rows, run_dir=run_dir)

    selected_manifest_tsv = run_dir / "targeted_manifest.tsv"
    if selected_rows:
        write_tsv(selected_manifest_tsv, list(selected_rows[0].keys()), selected_rows)
    else:
        raise ValueError("No manifest rows selected for Stage2 execution.")

    semantic_dir = run_dir / SEMANTIC_SUBDIR
    compat_dir = run_dir / COMPAT_SUBDIR
    # S2-1b materializes pre-LLM text-projection artifacts under semantic_dir
    # before S2-2 creates candidate/evidence artifacts.
    semantic_dir.mkdir(parents=True, exist_ok=True)
    compat_dir.mkdir(parents=True, exist_ok=False)

    legacy_raw_responses_dir: Path | None = None
    if args.source_mode == "legacy_llm_replay":
        if not str(args.legacy_raw_responses_dir).strip():
            raise ValueError("--legacy-raw-responses-dir is required for legacy_llm_replay mode.")
        legacy_raw_responses_dir = repo_path(args.legacy_raw_responses_dir)
        if not legacy_raw_responses_dir.exists():
            raise FileNotFoundError(f"Legacy raw responses directory not found: {legacy_raw_responses_dir}")

    fallback_legacy_raw_responses_dir: Path | None = None
    if str(args.fallback_legacy_raw_responses_dir).strip():
        fallback_legacy_raw_responses_dir = repo_path(args.fallback_legacy_raw_responses_dir)
        if not fallback_legacy_raw_responses_dir.exists():
            raise FileNotFoundError(
                f"Fallback legacy raw responses directory not found: {fallback_legacy_raw_responses_dir}"
            )

    extractor_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "stage2_sampling_labels" / "extract_semantic_stage2_objects_v2.py"),
        "--manifest-tsv",
        str(selected_manifest_tsv),
        "--out-dir",
        str(semantic_dir),
        "--source-mode",
        args.source_mode,
        "--model",
        args.model,
        "--llm-backend",
        args.llm_backend,
        "--max-text-chars",
        str(args.max_text_chars),
        "--request-timeout-seconds",
        str(max(1, min(int(args.request_timeout_seconds), 180))),
        "--request-retries",
        str(max(0, min(int(args.request_retries), 1))),
        "--retry-sleep-sec",
        str(args.retry_sleep_sec),
    ]
    if args.stop_before_live_call:
        extractor_cmd.append("--stop-before-live-call")
    if str(args.stage1_table_cell_sidecar_root).strip():
        extractor_cmd.extend(["--stage1-table-cell-sidecar-root", str(repo_path(args.stage1_table_cell_sidecar_root))])
    for key in selected_keys:
        extractor_cmd.extend(["--paper-key", key])
    if legacy_raw_responses_dir is not None:
        extractor_cmd.extend(["--legacy-raw-responses-dir", str(legacy_raw_responses_dir)])
    if fallback_legacy_raw_responses_dir is not None:
        extractor_cmd.extend(["--fallback-legacy-raw-responses-dir", str(fallback_legacy_raw_responses_dir)])
    run_context = build_run_context(
        run_id=run_id,
        run_dir_kind=run_dir_kind,
        run_selection_mode=run_selection_mode,
        bucket_dir=bucket_dir,
        run_dir=run_dir,
        manifest_tsv=manifest_tsv,
        selected_keys=selected_keys,
        source_mode=args.source_mode,
        llm_backend=args.llm_backend,
        model=args.model,
        max_text_chars=args.max_text_chars,
        legacy_raw_responses_dir=legacy_raw_responses_dir,
        semantic_dir=semantic_dir,
        compat_dir=compat_dir,
    )
    (run_dir / "RUN_CONTEXT.md").write_text(run_context, encoding="utf-8")
    run_command(extractor_cmd)

    request_summary_path = run_dir / "analysis" / REQUEST_SUMMARY_NAME
    success_count = 0
    failure_count = 0
    if request_summary_path.exists():
        request_rows = read_tsv(request_summary_path)
        success_count = sum(1 for row in request_rows if normalize_text(row.get("status")) == "success")
        failure_count = sum(1 for row in request_rows if normalize_text(row.get("status")) != "success")

    contract_report_path = run_dir / "analysis" / STAGE2_CONTRACT_REPORT_JSON
    compatibility_projection_executed = False
    compatibility_projection_status = "not_run"
    contract_validation_status = "not_run"
    authority_reattachment_status = "not_run"
    authority_reattachment_sidecar_root = run_dir / "semantic_stage2_objects" / "authority_reattachment"
    if not args.stop_before_live_call and success_count > 0:
        authority_reattachment_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "src" / "stage2_sampling_labels" / "build_semantic_authority_reattachment_v1.py"),
            "--semantic-jsonl",
            str(semantic_dir / SEMANTIC_JSONL),
            "--payload-root",
            str(semantic_dir / "normalized_table_payloads"),
            "--out-dir",
            str(run_dir),
        ]
        try:
            run_command(authority_reattachment_cmd)
            authority_reattachment_status = "success"
        except subprocess.CalledProcessError:
            authority_reattachment_status = "failed"

        compat_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "src" / "stage2_sampling_labels" / "build_stage2_compatibility_projection_v1.py"),
            "--input-jsonl",
            str(semantic_dir / SEMANTIC_JSONL),
            "--output-dir",
            str(compat_dir),
            "--authority-sidecar",
            str(run_dir),
        ]
        try:
            run_command(compat_cmd)
            compatibility_projection_executed = True
            compatibility_projection_status = "success"
        except subprocess.CalledProcessError:
            compatibility_projection_status = "failed"

        if compatibility_projection_executed:
            try:
                run_command(
                    [
                        sys.executable,
                        str(PROJECT_ROOT / "src" / "stage2_sampling_labels" / "validate_stage2_semantic_authority_contract_v1.py"),
                        "--semantic-jsonl",
                        str(semantic_dir / SEMANTIC_JSONL),
                        "--stage2-tsv",
                        str(compat_dir / FINAL_STAGE2_TSV),
                        "--report-out",
                        str(contract_report_path),
                    ]
                )
                contract_validation_status = "success"
            except subprocess.CalledProcessError:
                contract_validation_status = "failed"

    run_context = build_run_context(
        run_id=run_id,
        run_dir_kind=run_dir_kind,
        run_selection_mode=run_selection_mode,
        bucket_dir=bucket_dir,
        run_dir=run_dir,
        manifest_tsv=manifest_tsv,
        selected_keys=selected_keys,
        source_mode=args.source_mode,
        llm_backend=args.llm_backend,
        model=args.model,
        max_text_chars=args.max_text_chars,
        legacy_raw_responses_dir=legacy_raw_responses_dir,
        semantic_dir=semantic_dir,
        compat_dir=compat_dir,
    )
    (run_dir / "RUN_CONTEXT.md").write_text(run_context, encoding="utf-8")
    run_metadata = {
        "schema": "stage2_run_metadata_v1",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "stage2_semantic_source_mode": STAGE2_SEMANTIC_SOURCE_MODE,
        "stage2_entrypoint": "src/stage2_sampling_labels/run_stage2_composite_v1.py",
        "stage2_internal_semantic_extractor": "src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py",
        "stage2_internal_source_denoise_projection": "src/stage2_sampling_labels/denoise_stage2_source_text_s2_1b_v1.py",
        "stage2_internal_source_denoise_boundary": "s2_1b_high_confidence_source_denoise_projection",
        "stage2_internal_source_denoise_artifact_pattern": "semantic_stage2_objects/s2_1b_denoised_text/<paper_key>.txt",
        "stage2_internal_source_denoise_audit_pattern": "semantic_stage2_objects/s2_1b_denoise_audit/<paper_key>_s2_1b_denoise_audit_v1.json",
        "stage2_internal_source_denoise_summary": str(run_dir / "analysis" / "s2_1b_denoise_summary_v1.tsv"),
        "stage2_internal_pre_llm_boundary": "s2_2_clean_text_to_governed_evidence_package",
        "stage2_internal_candidate_segmentation_boundary": "clean_text_and_extracted_tables_to_candidate_blocks",
        "stage2_internal_candidate_artifact_pattern": "semantic_stage2_objects/candidate_blocks/<paper_key>/candidate_blocks_v1.json",
        "stage2_internal_candidate_segmentation_profile": "section_aware_candidate_segmentation_v1",
        "stage2_internal_candidate_table_isolation": "active_when_table_assets_exist",
        "stage2_internal_candidate_noise_filtering": "conservative_high_confidence",
        "stage2_internal_pre_llm_evidence_artifact_pattern": "semantic_stage2_objects/evidence_blocks/<paper_key>/evidence_blocks_v1.json",
        "stage2_internal_pre_llm_selection_mode": "evidence_priority_v1",
        "stage2_internal_pre_llm_archetype_policy": "metadata_only_no_selection_overlay",
        "stage2_prompt_preview_relationship": "derived_from_evidence_blocks_v1_json",
        "stage2_internal_completion": "src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py",
        "stage2_internal_semantic_authority_reattachment": "src/stage2_sampling_labels/build_semantic_authority_reattachment_v1.py",
        "stage2_internal_semantic_authority_reattachment_status": authority_reattachment_status,
        "stage2_internal_semantic_authority_reattachment_sidecar_root": str(authority_reattachment_sidecar_root),
        "stage2_internal_doe_function_unit": "src/stage2_sampling_labels/function_units/doe_row_expansion_function_unit_v1.py",
        "stage2_internal_sequential_optimization_function_unit": "src/stage2_sampling_labels/function_units/sequential_optimization_interpreter_v1.py",
        "stage2_contract_validation_report": str(contract_report_path),
        "request_timeout_seconds": max(1, min(int(args.request_timeout_seconds), 180)),
        "request_retries": max(0, min(int(args.request_retries), 1)),
        "request_summary_path": str(request_summary_path),
        "success_count": success_count,
        "failure_count": failure_count,
        "compatibility_projection_executed": compatibility_projection_executed,
        "compatibility_projection_status": compatibility_projection_status,
        "contract_validation_status": contract_validation_status,
        "stop_before_live_call": args.stop_before_live_call,
        "source_mode": args.source_mode,
        "llm_backend": args.llm_backend,
        "model": args.model,
        "legacy_raw_responses_dir": str(legacy_raw_responses_dir) if legacy_raw_responses_dir is not None else "",
        "stage1_table_cell_sidecar_root": str(repo_path(args.stage1_table_cell_sidecar_root)) if str(args.stage1_table_cell_sidecar_root).strip() else "",
    }
    (run_dir / STAGE2_RUN_METADATA_JSON).write_text(
        json.dumps(run_metadata, indent=2),
        encoding="utf-8",
    )
    run_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "src" / "utils" / "update_run_context_with_feature_activation_v1.py"),
            "--run-dir",
            str(run_dir),
        ]
    )

    print(f"run_id={run_id}")
    print(f"run_dir={run_dir}")
    print(f"semantic_dir={semantic_dir}")
    print(f"compat_dir={compat_dir}")
    print(f"success_count={success_count}")
    print(f"failure_count={failure_count}")
    print(f"compatibility_projection_status={compatibility_projection_status}")
    print(f"contract_validation_status={contract_validation_status}")
    print(f"completed_at={datetime.now().isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()
