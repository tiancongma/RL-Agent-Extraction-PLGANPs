#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage5_benchmark.build_minimal_final_output_v1 import build_minimal_final_output


FULL_DEV15_INPUT_TSV = PROJECT_ROOT / Path(
    r"data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/weak_labels_v7pilot_r3_fixparse/weak_labels__v7pilot_r3_fixparse.tsv"
)
FULL_DEV15_RELATION_DIR = PROJECT_ROOT / Path(
    r"data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/lineage/children/34_dev15_v2_full_gt_compare/run_20260320_1317_f54824a_dev15_v2_full_gt_compare_no_llm_v1/formulation_relation_v1"
)
BLOCKER_INPUT_TSV = PROJECT_ROOT / Path(
    r"data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/lineage/children/30_dev15_v2_blocker_gate/run_20260320_1111_f54824a_dev15_v2_blocker_gate_no_llm_v1/weak_labels_v7pilot_r3_fixparse/weak_labels__v7pilot_r3_fixparse.tsv"
)
BLOCKER_RELATION_DIR = PROJECT_ROOT / Path(
    r"data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/lineage/children/31_stage5_descendant_filter_validation/run_20260320_1141_f54824a_dev15_v2_blocker_stage5_descendant_filter_validation_v1/formulation_relation_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministic Stage5 descendant-filter regression check for BB3JUVW7 and known negative-control descendant rows."
    )
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def parse_json_list(raw_value: str) -> list[str]:
    text = str(raw_value or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [text]
    if isinstance(data, list):
        return [str(item).strip() for item in data if str(item).strip()]
    return [str(data).strip()] if str(data).strip() else []


def count_paper_rows(rows: list[dict[str, str]], paper_id: str) -> int:
    return sum(1 for row in rows if str(row.get("key", "")).strip() == paper_id)


def assert_condition(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_stage5_case(input_tsv: Path, relation_dir: Path, out_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    build_minimal_final_output(
        input_tsv=input_tsv,
        relation_records_tsv=relation_dir / "formulation_relation_records_v1.tsv",
        resolved_relation_fields_tsv=relation_dir / "resolved_relation_fields_v1.tsv",
        out_dir=out_dir,
    )
    final_rows = read_tsv(out_dir / "final_formulation_table_v1.tsv")
    decision_rows = read_tsv(out_dir / "final_output_decision_trace_v1.tsv")
    return final_rows, decision_rows


def main() -> None:
    args = parse_args()
    temp_root = Path(tempfile.mkdtemp(prefix="stage5_descendant_regression_"))
    try:
        full_out = temp_root / "full_dev15"
        blocker_out = temp_root / "blocker_subset"

        full_final_rows, full_decision_rows = run_stage5_case(
            input_tsv=FULL_DEV15_INPUT_TSV,
            relation_dir=FULL_DEV15_RELATION_DIR,
            out_dir=full_out,
        )
        _blocker_final_rows, blocker_decision_rows = run_stage5_case(
            input_tsv=BLOCKER_INPUT_TSV,
            relation_dir=BLOCKER_RELATION_DIR,
            out_dir=blocker_out,
        )

        bb_rows = [row for row in full_final_rows if row.get("key") == "BB3JUVW7"]
        bb_source_ids = {
            source_id.strip()
            for row in bb_rows
            for source_id in parse_json_list(row.get("source_candidate_ids", ""))
            if source_id.strip()
        }
        expected_bb_ids = {"F2.2", "F2.4", "F2.5", "F2.6", "F2.7"}
        assert_condition(len(bb_rows) == 12, f"BB3JUVW7 final count expected 12, observed {len(bb_rows)}")
        assert_condition(
            expected_bb_ids.issubset(bb_source_ids),
            f"BB3JUVW7 missing restored descendant rows: {sorted(expected_bb_ids - bb_source_ids)}",
        )

        assert_condition(
            count_paper_rows(full_final_rows, "WIVUCMYG") == 26,
            f"WIVUCMYG final count expected 26, observed {count_paper_rows(full_final_rows, 'WIVUCMYG')}",
        )

        blocker_expect_filtered = {
            ("BXCV5XWB", "F1_Blank"),
            ("BXCV5XWB", "F1_FITC"),
            ("QLYKLPKT", "T3_F1_Sucrose_2_percent"),
            ("QLYKLPKT", "T4_F1_OptimalPLGA_ITZ_NS_PK"),
        }
        filtered_pairs = {
            (row.get("zotero_key", "").strip(), row.get("source_formulation_id", "").strip())
            for row in blocker_decision_rows
            if row.get("decision_rule") == "parent_linked_non_synthesis_descendant_variant"
        }
        for pair in blocker_expect_filtered:
            assert_condition(
                pair in filtered_pairs,
                f"Expected descendant row to remain filtered: {pair[0]}::{pair[1]}",
            )

        print(
            json.dumps(
                {
                    "status": "ok",
                    "bb3juvw7_final_count": len(bb_rows),
                    "bb3juvw7_restored_ids": sorted(expected_bb_ids),
                    "wivucmyg_final_count": count_paper_rows(full_final_rows, "WIVUCMYG"),
                    "negative_control_rows_still_filtered": sorted(f"{paper}::{fid}" for paper, fid in blocker_expect_filtered),
                    "temp_root": str(temp_root),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        if not args.keep_temp:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
