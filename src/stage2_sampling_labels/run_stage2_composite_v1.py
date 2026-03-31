#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    from src.utils.paths import DATA_RESULTS_DIR, PROJECT_ROOT
    from src.utils.run_id import is_valid_run_id
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.paths import DATA_RESULTS_DIR, PROJECT_ROOT
    from src.utils.run_id import is_valid_run_id


SEMANTIC_SUBDIR = "semantic_stage2_objects"
COMPAT_SUBDIR = "semantic_to_widerow_adapter"
SEMANTIC_JSONL = "semantic_stage2_v2_objects.jsonl"
SEMANTIC_SUMMARY = "semantic_stage2_v2_summary.tsv"
FINAL_STAGE2_TSV = "weak_labels__v7pilot_r3_fixparse.tsv"
FINAL_STAGE2_JSONL = "weak_labels__v7pilot_r3_fixparse.jsonl"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the governed composite Stage2 path: LLM semantic discovery followed by deterministic post-LLM completion."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--manifest-tsv", required=True)
    parser.add_argument("--paper-key", action="append", dest="paper_keys", default=[])
    parser.add_argument("--source-mode", choices=["live_llm", "legacy_llm_replay"], default="live_llm")
    parser.add_argument("--legacy-raw-responses-dir", default="")
    parser.add_argument("--llm-backend", choices=["gemini", "nvidia"], default="gemini")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--max-text-chars", type=int, default=18000)
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--retry-sleep-sec", type=float, default=3.0)
    return parser.parse_args()


def build_run_context(
    *,
    run_id: str,
    run_dir: Path,
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
    return f"""# RUN_CONTEXT

## 1. Run ID
`{run_id}`

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
- Stage2 internal intermediate:
  - LLM semantic discovery objects under `{semantic_dir}`
- Stage2 authoritative final output:
  - deterministic post-LLM completion under `{compat_dir}`
- Stage3 must consume only the completed Stage2 artifact, not raw LLM semantic objects alone.

## 5. Scope and inputs
- manifest_tsv: `{manifest_tsv}`
- selected_paper_keys:
{key_block}
- source_mode: `{source_mode}`
- llm_backend: `{llm_backend}`
- model: `{model}`
- max_text_chars: `{max_text_chars}`
{legacy_note}

## 6. Exact script execution order
1. `src/stage2_sampling_labels/run_stage2_composite_v1.py`
2. `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`
3. `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`

## 7. Outputs
- intermediate semantic objects:
  - `{semantic_dir / SEMANTIC_JSONL}`
  - `{semantic_dir / SEMANTIC_SUMMARY}`
- final Stage2 completion artifacts:
  - `{compat_dir / FINAL_STAGE2_TSV}`
  - `{compat_dir / FINAL_STAGE2_JSONL}`
  - `{compat_dir / 'compatibility_projection_trace_v1.tsv'}`
  - `{compat_dir / 'compatibility_projection_summary_v1.json'}`
- run context:
  - `{run_dir / 'RUN_CONTEXT.md'}`

## 8. Evaluation guardrail
- Raw LLM semantic objects are an internal Stage2 intermediate only.
- Structural Stage2 evaluation for downstream readiness must target the completed Stage2 artifact after deterministic post-LLM completion.
- Direct comparison of raw LLM semantic objects to formulation-level GT is diagnostic-only failure localization and is not the authoritative Stage2 evaluation object.
"""


def main() -> None:
    args = parse_args()
    if not is_valid_run_id(args.run_id):
        raise ValueError(f"Invalid --run-id: {args.run_id}")

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

    run_dir = (DATA_RESULTS_DIR / args.run_id).resolve()
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)

    selected_manifest_tsv = run_dir / "targeted_manifest.tsv"
    if selected_rows:
        write_tsv(selected_manifest_tsv, list(selected_rows[0].keys()), selected_rows)
    else:
        raise ValueError("No manifest rows selected for Stage2 execution.")

    semantic_dir = run_dir / SEMANTIC_SUBDIR
    compat_dir = run_dir / COMPAT_SUBDIR
    semantic_dir.mkdir(parents=True, exist_ok=False)
    compat_dir.mkdir(parents=True, exist_ok=False)

    legacy_raw_responses_dir: Path | None = None
    if args.source_mode == "legacy_llm_replay":
        if not str(args.legacy_raw_responses_dir).strip():
            raise ValueError("--legacy-raw-responses-dir is required for legacy_llm_replay mode.")
        legacy_raw_responses_dir = repo_path(args.legacy_raw_responses_dir)
        if not legacy_raw_responses_dir.exists():
            raise FileNotFoundError(f"Legacy raw responses directory not found: {legacy_raw_responses_dir}")

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
        "--request-retries",
        str(args.request_retries),
        "--retry-sleep-sec",
        str(args.retry_sleep_sec),
    ]
    for key in selected_keys:
        extractor_cmd.extend(["--paper-key", key])
    if legacy_raw_responses_dir is not None:
        extractor_cmd.extend(["--legacy-raw-responses-dir", str(legacy_raw_responses_dir)])
    run_command(extractor_cmd)

    compat_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "stage2_sampling_labels" / "build_stage2_compatibility_projection_v1.py"),
        "--input-jsonl",
        str(semantic_dir / SEMANTIC_JSONL),
        "--output-dir",
        str(compat_dir),
    ]
    run_command(compat_cmd)

    run_context = build_run_context(
        run_id=args.run_id,
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

    print(f"run_dir={run_dir}")
    print(f"semantic_dir={semantic_dir}")
    print(f"compat_dir={compat_dir}")
    print(f"completed_at={datetime.now().isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()
