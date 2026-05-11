#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_RUN_ID = "run_20260219_1623_780eb83_goren18_weaklabels_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run core-level alignment eval by reusing run_alignment_eval_v1 logic."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--core-projected-tsv",
        default="",
        help="Default: data/results/<run_id>/benchmark_goren_2025/core_eval_v1/core_projected_to_curated.tsv",
    )
    parser.add_argument(
        "--curated-tsv",
        default="data/benchmark/goren_2025/overlap_goren18_v1/goren18_curated_overlap_subset.tsv",
    )
    parser.add_argument("--modes", default="strict,relaxed,canonicalized")
    parser.add_argument(
        "--out-dir",
        default="",
        help="Default: data/results/<run_id>/benchmark_goren_2025/core_eval_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"data/results/{run_id}/benchmark_goren_2025/core_eval_v1")
    core_projected = Path(args.core_projected_tsv) if args.core_projected_tsv else out_dir / "core_projected_to_curated.tsv"
    curated_tsv = Path(args.curated_tsv)

    missing = [str(p) for p in [core_projected, curated_tsv] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input file(s): {missing}")

    out_dir.mkdir(parents=True, exist_ok=True)
    base_script = Path("src/stage5_benchmark/run_alignment_eval_v1.py")
    if not base_script.exists():
        raise FileNotFoundError(f"Required reusable alignment script not found: {base_script}")

    cmd = [
        sys.executable,
        str(base_script),
        "--run-id",
        run_id,
        "--projected-tsv",
        str(core_projected),
        "--curated-tsv",
        str(curated_tsv),
        "--modes",
        str(args.modes),
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)

    # Re-label outputs to core-specific names while keeping base files intact.
    copy_map = {
        out_dir / "alignment_rows.tsv": out_dir / "core_alignment_rows.tsv",
        out_dir / "metrics_summary.json": out_dir / "core_metrics_summary.json",
        out_dir / "failure_types.tsv": out_dir / "core_failure_types.tsv",
    }
    for src, dst in copy_map.items():
        if src.exists():
            shutil.copyfile(src, dst)

    print(f"output_alignment_rows={out_dir / 'core_alignment_rows.tsv'}")
    print(f"output_metrics={out_dir / 'core_metrics_summary.json'}")
    print(f"output_failure_types={out_dir / 'core_failure_types.tsv'}")


if __name__ == "__main__":
    main()
