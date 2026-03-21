#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _bootstrap_import_paths() -> None:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "src" / "utils" / "paths.py").exists():
            sys.path.insert(0, str(p))
            return


_bootstrap_import_paths()

from src.stage5_benchmark.formulation_core_signature_v1 import (  # noqa: E402
    _self_test,
    build_formulation_core_signature_v1,
)
from src.utils import paths  # noqa: E402
from src.utils.run_id import is_valid_run_id, validate_artifact_subdir  # noqa: E402
from src.utils.run_latest import inputs_fingerprint, write_latest  # noqa: E402


def _sanitize_out_subdir(s: str) -> str:
    try:
        return validate_artifact_subdir(s, param_name="--out-subdir")
    except ValueError as exc:
        raise ValueError(
            "ERROR: --out-subdir is required when reusing a run_id and must be a functional artifact path under data/results/<run_id>/ without repeating a nested run_id or timestamp/hash token."
        ) from exc


def find_default_input_tsv() -> Path:
    result_dir = paths.DATA_RESULTS_DIR
    candidates = sorted(
        result_dir.glob("run_*_goren18_*/weak_labels__*.tsv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    fallback = result_dir / "run_20260219_1623_780eb83_goren18_weaklabels_v1" / "weak_labels__gemini.tsv"
    return fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Formulation Core Signature v1 with auditable hash, quality/risk scoring, "
            "and merge gates A/B/C."
        )
    )
    parser.add_argument(
        "--input-tsv",
        default=str(find_default_input_tsv()),
        help="Input extracted TSV (default: latest goren18 weak_labels TSV under data/results).",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Required deterministic run_id from preflight.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Optional output directory. Default: data/results/<run_id>/formulation_core_signature_v1",
    )
    parser.add_argument(
        "--out-subdir",
        default="",
        help="Optional subdirectory under data/results/<run_id>/ for run variants (e.g., iter_001).",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run lightweight self-test and exit.",
    )
    parser.add_argument(
        "--derived-values-tsv",
        default="",
        help=(
            "Optional derived_values.tsv path for DoE factor traces. If empty, auto-detect under "
            "data/results/<run_id>/benchmark_goren_2025/ or .../formulation_core_signature_v1/."
        ),
    )
    return parser.parse_args()


def resolve_derived_values_path(run_id: str, input_tsv: Path, derived_values_arg: str) -> Path | None:
    if derived_values_arg.strip():
        p = Path(derived_values_arg).resolve()
        return p if p.exists() else None
    base = paths.DATA_RESULTS_DIR / run_id
    candidates = [
        base / "benchmark_goren_2025" / "derived_values.tsv",
        base / "formulation_core_signature_v1" / "derived_values.tsv",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    # fallback from input path run folder
    if "data" in input_tsv.parts and "results" in input_tsv.parts:
        try:
            idx = input_tsv.parts.index("results")
            run_folder = Path(*input_tsv.parts[: idx + 2]) / run_id
            c2 = run_folder / "benchmark_goren_2025" / "derived_values.tsv"
            if c2.exists():
                return c2.resolve()
        except Exception:
            pass
    return None


def main() -> None:
    args = parse_args()
    if args.self_test:
        _self_test()
        print("[OK] self-test passed")
        return

    input_tsv = Path(args.input_tsv).resolve()
    if not input_tsv.exists():
        raise FileNotFoundError(f"input TSV not found: {input_tsv}")

    run_id = str(args.run_id or "").strip()
    if not run_id:
        raise ValueError(
            "ERROR: --run-id is required. Generate/reuse a run_id via: python -m src.utils.run_preflight ..."
        )
    if not is_valid_run_id(run_id):
        raise ValueError(f"Resolved run_id is invalid: {run_id}")
    out_subdir = _sanitize_out_subdir(args.out_subdir)

    latest_path = write_latest(
        run_id=run_id,
        meta={
            "created_at": datetime.now(timezone.utc).isoformat(),
            "subset": "goren18",
            "stage": "formulation_core_signature",
            "inputs_fingerprint": inputs_fingerprint([input_tsv]),
            "note": "run_formulation_core_signature_v1",
        },
    )
    base_run_dir = paths.DATA_RESULTS_DIR / run_id / out_subdir
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (base_run_dir / "formulation_core_signature_v1")
    if args.out_dir:
        try:
            out_dir.resolve().relative_to(base_run_dir.resolve())
        except Exception:
            raise ValueError(
                f"ERROR: --out-dir must be under data/results/<run_id>/<out-subdir>/. Got: {out_dir}"
            )
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_tsv, sep="\t", dtype=str).fillna("")
    derived_values_path = resolve_derived_values_path(run_id=run_id, input_tsv=input_tsv, derived_values_arg=args.derived_values_tsv)
    derived_df = pd.DataFrame()
    doe_trace_enabled = False
    if derived_values_path is not None:
        try:
            derived_df = pd.read_csv(derived_values_path, sep="\t", dtype=str).fillna("")
            doe_trace_enabled = not derived_df.empty
        except Exception:
            derived_df = pd.DataFrame()
            doe_trace_enabled = False

    outputs = build_formulation_core_signature_v1(
        df=df,
        run_id=run_id,
        input_tsv=str(input_tsv),
        derived_values_df=derived_df if doe_trace_enabled else None,
        derived_values_path=str(derived_values_path) if derived_values_path is not None else "",
    )

    p_core = out_dir / "formulation_core_v1.tsv"
    p_assign = out_dir / "instance_assignment_v1.tsv"
    p_trace = out_dir / "signature_trace_v1.tsv"
    p_log = out_dir / "build_log.json"

    outputs.core_df.to_csv(p_core, sep="\t", index=False)
    outputs.assignment_df.to_csv(p_assign, sep="\t", index=False)
    outputs.trace_df.to_csv(p_trace, sep="\t", index=False)
    p_log.write_text(json.dumps(outputs.build_log, ensure_ascii=False, indent=2), encoding="utf-8")

    top_risks = outputs.build_log.get("top_risk_reasons", [])
    print(f"run_id={run_id}")
    print(f"latest_pointer={latest_path}")
    print(f"input_tsv={input_tsv}")
    print(f"out_dir={out_dir}")
    print(f"doe_trace_enabled={doe_trace_enabled}")
    print(f"derived_values_path={str(derived_values_path) if derived_values_path is not None else ''}")
    print(f"n_instances={outputs.build_log.get('n_instances', 0)}")
    print(f"n_cores={outputs.build_log.get('n_cores', 0)}")
    print(f"count_rows_with_doe_signature={outputs.build_log.get('count_rows_with_doe_signature', 0)}")
    print(f"count_cores_with_doe_signature={outputs.build_log.get('count_cores_with_doe_signature', 0)}")
    print(f"auto_merged_count={outputs.build_log.get('auto_merged_count', 0)}")
    print(f"unresolved_count={outputs.build_log.get('unresolved_count', 0)}")
    print(f"top_10_risk_reasons={json.dumps(top_risks, ensure_ascii=False)}")
    spot = outputs.assignment_df.copy()
    if "doe_signature_canon" in spot.columns:
        spot = spot.sort_values(
            by=["doe_signature_canon", "doc_key", "formulation_id"],
            ascending=[False, True, True],
        )
    spot = spot.head(3)
    print("[spot_check_top3]")
    if spot.empty:
        print("(empty)")
    else:
        cols = ["doc_key", "formulation_id", "doe_signature_canon", "doe_signature_source", "signature_string"]
        for _, r in spot.iterrows():
            print(
                f"zotero_key={r.get('doc_key','')}\tformulation_id={r.get('formulation_id','')}\t"
                f"doe_signature_canon={r.get('doe_signature_canon','')}\t"
                f"doe_signature_source={r.get('doe_signature_source','')}\t"
                f"signature_string={r.get('signature_string','')}"
            )
    print(f"output_formulation_core={p_core}")
    print(f"output_instance_assignment={p_assign}")
    print(f"output_signature_trace={p_trace}")
    print(f"output_build_log={p_log}")


if __name__ == "__main__":
    main()
