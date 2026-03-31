#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    from src.utils.active_data_source import resolve_artifact_path, resolve_run_context
    from src.utils.paths import DATA_RESULTS_DIR, PROJECT_ROOT
    from src.utils.run_id import is_valid_run_id
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.active_data_source import resolve_artifact_path, resolve_run_context
    from src.utils.paths import DATA_RESULTS_DIR, PROJECT_ROOT
    from src.utils.run_id import is_valid_run_id


TARGET_PAPERS = ["WIVUCMYG", "UFXX9WXE", "5GIF3D8W"]
DEFAULT_HISTORICAL_RAW_RESPONSES_DIR = (
    "data/results/run_20260313_0950_f4912f3_dev15_current_merged_benchmark_v1/"
    "weak_labels_v7pilot_r3_fixparse/raw_responses"
)
DEFAULT_HISTORICAL_LEGACY_TSV = (
    "data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/"
    "weak_labels_v7pilot_r3_fixparse/weak_labels__v7pilot_r3_fixparse.tsv"
)
DEFAULT_REPLAY_V2_JSONL = (
    "data/results/run_20260331_1015_03e5d25_threepaper_stage2_v2_comparison_v1/"
    "semantic_stage2_v2/semantic_stage2_v2_objects.jsonl"
)


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def run_command(command: list[str]) -> None:
    print("running_command=" + " ".join(str(part) for part in command))
    subprocess.run(command, cwd=PROJECT_ROOT, text=True, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the governed three-paper semantic-intermediate comparison wrapper on top of the composite Stage2 entrypoint."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-mode", choices=["legacy_llm_replay", "live_llm"], default="legacy_llm_replay")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--llm-backend", choices=["gemini", "nvidia"], default="gemini")
    parser.add_argument("--max-text-chars", type=int, default=18000)
    parser.add_argument("--historical-raw-responses-dir", default=DEFAULT_HISTORICAL_RAW_RESPONSES_DIR)
    parser.add_argument("--historical-legacy-tsv", default=DEFAULT_HISTORICAL_LEGACY_TSV)
    parser.add_argument("--replay-v2-jsonl", default=DEFAULT_REPLAY_V2_JSONL)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--active-run-id", default="")
    return parser.parse_args()


def build_run_context(
    *,
    run_id: str,
    run_dir: Path,
    source_run_context: dict[str, object],
    scope_manifest_tsv: Path,
    comparison_dir: Path,
    source_mode: str,
    model: str,
    llm_backend: str,
    max_text_chars: int,
    historical_raw_dir: Path,
    historical_legacy_tsv: Path,
    replay_v2_jsonl: Path,
    current_semantic_jsonl: Path,
    current_compat_tsv: Path,
    gt_skeleton_tsv: Path,
) -> str:
    bullet_keys = "\n".join(
        [
            "- `WIVUCMYG`: DOE factors plus pH-coded row identity stress.",
            "- `UFXX9WXE`: numbered DOE row multiplicity and boundary stability stress.",
            "- `5GIF3D8W`: boundary drift and identity instability stress.",
        ]
    )
    return f"""# RUN_CONTEXT

## 1. Run ID
`{run_id}`

## 2. Run Type
`component_regression_run`

Benchmark reporting rule:
- This run is `diagnostic-only, not benchmark-valid final output`.
- It is a semantic-intermediate comparison slice layered on top of the governed composite Stage2 runner.
- It does not replace the active benchmark mainline.

## 3. Purpose
- Run the governed composite Stage2 path for exactly three targeted papers.
- Compare the raw LLM semantic-discovery intermediate against maintained comparator surfaces.
- Preserve the observed semantic-intermediate findings without treating them as authoritative Stage2 completion judgment.

## 4. Correct interpretation guardrail
- The composite Stage2 contract is:
  1. LLM semantic discovery
  2. deterministic post-LLM completion
- This wrapper compares only the raw semantic-discovery intermediate.
- The authoritative Stage2 evaluation object is the completed Stage2 artifact consumed by Stage3, not the raw LLM intermediate alone.

## 5. Source authority resolution
- source_resolution: `{source_run_context['resolution_source']}`
- source_run_id: `{source_run_context['run_id']}`
- source_run_dir: `{source_run_context['run_dir']}`
- source_scope_manifest_tsv: `{scope_manifest_tsv}`
- current_semantic_jsonl: `{current_semantic_jsonl}`
- current_compat_tsv: `{current_compat_tsv}`
- gt_skeleton_tsv: `{gt_skeleton_tsv}`
- historical_raw_responses_dir: `{historical_raw_dir}`
- historical_legacy_tsv: `{historical_legacy_tsv}`
- replay_v2_jsonl: `{replay_v2_jsonl}`

## 6. Paper selection
{bullet_keys}

## 7. Exact script execution order
1. `src/utils/run_threepaper_stage2_v2_comparison.py`
2. `src/stage2_sampling_labels/run_stage2_composite_v1.py`
3. `src/analysis/build_stage2_v2_threepaper_comparison_pack.py`

## 8. Backend configuration
- source_mode: `{source_mode}`
- llm_backend: `{llm_backend}`
- model: `{model}`
- max_text_chars: `{max_text_chars}`

## 9. Comparison outputs
- `{comparison_dir / 'paper_level_counts.tsv'}`
- `{comparison_dir / 'boundary_review.tsv'}`
- `{comparison_dir / 'component_completeness_review.tsv'}`
- `{comparison_dir / 'expression_richness_review.tsv'}`
- `{comparison_dir / 'variable_detection_review.tsv'}`
- `{comparison_dir / 'ambiguity_handling_review.tsv'}`
- `{comparison_dir / 'structural_comparison_summary.tsv'}`
- `{comparison_dir / 'structural_comparison_report.md'}`

## 10. Non-promotion statement
- This run does not modify `data/results/ACTIVE_RUN.json`.
- This wrapper is comparison-only and must not be treated as the Stage2 definition.
- Future promotion requires evaluation of the completed Stage2 output, not the raw LLM intermediate alone.
"""


def main() -> None:
    args = parse_args()
    if not is_valid_run_id(args.run_id):
        raise ValueError(f"Invalid --run-id: {args.run_id}")

    source_run_context = resolve_run_context(
        explicit_run_dir=repo_path(args.run_dir) if str(args.run_dir).strip() else None,
        explicit_run_id=args.active_run_id,
    )
    scope_manifest_tsv = resolve_artifact_path(
        explicit_path=None,
        run_context=source_run_context,
        pointer_key="scope_manifest_tsv",
        required=True,
    )
    current_semantic_jsonl = resolve_artifact_path(
        explicit_path=None,
        run_context=source_run_context,
        pointer_key="stage2_semantic_objects_jsonl",
        required=True,
    )
    current_compat_tsv = resolve_artifact_path(
        explicit_path=None,
        run_context=source_run_context,
        pointer_key="stage2_compatibility_tsv",
        required=True,
    )
    gt_skeleton_tsv = resolve_artifact_path(
        explicit_path=None,
        run_context=source_run_context,
        pointer_key="gt_skeleton_tsv",
        required=True,
    )
    assert scope_manifest_tsv is not None
    assert current_semantic_jsonl is not None
    assert current_compat_tsv is not None
    assert gt_skeleton_tsv is not None

    historical_raw_dir = repo_path(args.historical_raw_responses_dir)
    historical_legacy_tsv = repo_path(args.historical_legacy_tsv)
    replay_v2_jsonl = repo_path(args.replay_v2_jsonl)

    print(f"resolved_source_run_dir={source_run_context['run_dir']}")
    print(f"resolved_scope_manifest_tsv={scope_manifest_tsv}")
    print(f"resolved_current_semantic_jsonl={current_semantic_jsonl}")
    print(f"resolved_current_compat_tsv={current_compat_tsv}")
    print(f"resolved_gt_skeleton_tsv={gt_skeleton_tsv}")
    print(f"resolved_historical_raw_responses_dir={historical_raw_dir}")
    print(f"resolved_historical_legacy_tsv={historical_legacy_tsv}")
    print(f"resolved_replay_v2_jsonl={replay_v2_jsonl}")

    stage2_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "stage2_sampling_labels" / "run_stage2_composite_v1.py"),
        "--run-id",
        args.run_id,
        "--manifest-tsv",
        str(scope_manifest_tsv),
        "--source-mode",
        args.source_mode,
        "--llm-backend",
        args.llm_backend,
        "--model",
        args.model,
        "--max-text-chars",
        str(args.max_text_chars),
    ]
    for paper_key in TARGET_PAPERS:
        stage2_cmd.extend(["--paper-key", paper_key])
    if args.source_mode == "legacy_llm_replay":
        stage2_cmd.extend(["--legacy-raw-responses-dir", str(historical_raw_dir)])
    run_command(stage2_cmd)

    run_dir = (DATA_RESULTS_DIR / args.run_id).resolve()
    comparison_dir = run_dir / "analysis" / "stage2_v2_threepaper_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    stage2_semantic_jsonl = run_dir / "semantic_stage2_objects" / "semantic_stage2_v2_objects.jsonl"

    write_tsv(
        run_dir / "paper_selection.tsv",
        ["key", "selection_reason", "slice_role"],
        [
            {"key": "WIVUCMYG", "selection_reason": "DOE factors and coded pH row identity stress.", "slice_role": "doe_factors"},
            {"key": "UFXX9WXE", "selection_reason": "Numbered DOE row multiplicity and boundary stability stress.", "slice_role": "numbered_rows"},
            {"key": "5GIF3D8W", "selection_reason": "Boundary drift and identity instability stress.", "slice_role": "boundary_drift"},
        ],
    )

    comparison_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "analysis" / "build_stage2_v2_threepaper_comparison_pack.py"),
        "--llm-v2-jsonl",
        str(stage2_semantic_jsonl),
        "--out-dir",
        str(comparison_dir),
        "--current-semantic-jsonl",
        str(current_semantic_jsonl),
        "--current-compat-tsv",
        str(current_compat_tsv),
        "--historical-legacy-tsv",
        str(historical_legacy_tsv),
        "--replay-v2-jsonl",
        str(replay_v2_jsonl),
        "--gt-skeleton-tsv",
        str(gt_skeleton_tsv),
        "--run-id",
        str(source_run_context["run_id"]),
    ]
    for paper_key in TARGET_PAPERS:
        comparison_cmd.extend(["--paper-key", paper_key])
    run_command(comparison_cmd)

    run_context = build_run_context(
        run_id=args.run_id,
        run_dir=run_dir,
        source_run_context=source_run_context,
        scope_manifest_tsv=scope_manifest_tsv,
        comparison_dir=comparison_dir,
        source_mode=args.source_mode,
        model=args.model,
        llm_backend=args.llm_backend,
        max_text_chars=args.max_text_chars,
        historical_raw_dir=historical_raw_dir,
        historical_legacy_tsv=historical_legacy_tsv,
        replay_v2_jsonl=replay_v2_jsonl,
        current_semantic_jsonl=current_semantic_jsonl,
        current_compat_tsv=current_compat_tsv,
        gt_skeleton_tsv=gt_skeleton_tsv,
    )
    (run_dir / "RUN_CONTEXT.md").write_text(run_context, encoding="utf-8")

    print(f"run_dir={run_dir}")
    print(f"comparison_dir={comparison_dir}")
    print(f"completed_at={datetime.now().isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()
