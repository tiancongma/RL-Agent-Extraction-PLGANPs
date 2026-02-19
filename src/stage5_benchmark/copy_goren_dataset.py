"""
Copy the external Goren 2025 benchmark CSV into this repository.

Usage examples:
    python src/stage5_benchmark/copy_goren_dataset.py \
        --source-csv "C:\\absolute\\path\\to\\NP_dataset_formulations.csv"

    python src/stage5_benchmark/copy_goren_dataset.py \
        --source-csv "D:\\datasets\\goren_2025\\NP_dataset_formulations.csv" \
        --dest "data/benchmark/goren_2025/NP_dataset_formulations.csv"
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_DEST = Path("data/benchmark/goren_2025/NP_dataset_formulations.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy external Goren benchmark CSV into repository benchmark folder."
    )
    parser.add_argument(
        "--source-csv",
        required=True,
        help="Absolute path to external NP_dataset_formulations.csv",
    )
    parser.add_argument(
        "--dest",
        default=str(DEFAULT_DEST),
        help="Destination path inside repo.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source = Path(args.source_csv)
    dest = Path(args.dest)

    if not source.is_absolute():
        raise ValueError("--source-csv must be an absolute path.")

    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Source CSV not found: {source}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)

    print(f"Copied benchmark CSV to: {dest}")


if __name__ == "__main__":
    main()
