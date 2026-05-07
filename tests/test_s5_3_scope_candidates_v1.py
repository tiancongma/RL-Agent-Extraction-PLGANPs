from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.stage5_benchmark import build_s5_3_scope_candidates_v1 as scope_policy


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


OBS_COLUMNS = [
    "paper_key",
    "formulation_id",
    "field_name",
    "scope_trigger",
    "source_locator",
    "upstream_boundary",
    "why_s5_2_failed",
    "source_observability_status",
    "row_identity_status",
    "direct_or_derived",
    "source_context",
]


class S5_3ScopeCandidatesPolicyTests(unittest.TestCase):
    def test_source_observable_unmapped_s2_2_cell_is_eligible_without_gt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final_table = root / "inputs" / "final.tsv"
            observability = root / "inputs" / "observability.tsv"
            out_dir = root / "out"
            write_tsv(final_table, ["paper_key", "final_formulation_id"], [{"paper_key": "PAPER_A", "final_formulation_id": "F1"}])
            write_tsv(
                observability,
                OBS_COLUMNS,
                [
                    {
                        "paper_key": "PAPER_A",
                        "formulation_id": "F1",
                        "field_name": "particle_size_nm",
                        "scope_trigger": "s2_2_unmapped_row_local_cell",
                        "source_locator": "table_cell_grid_v1.tsv#row=7;col=size",
                        "upstream_boundary": "S2-2",
                        "why_s5_2_failed": "header_alias_not_mapped",
                        "source_observability_status": "source_observable",
                        "row_identity_status": "resolved",
                        "direct_or_derived": "direct",
                        "source_context": "formulation_row",
                    }
                ],
            )

            self.assertEqual(
                scope_policy.main(
                    ["--final-table-tsv", str(final_table), "--source-observability-tsv", str(observability), "--out-dir", str(out_dir)]
                ),
                0,
            )

            rows = read_tsv(out_dir / scope_policy.SCOPE_CANDIDATES_TSV_NAME)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["eligible_for_s5_3"], "yes")
            self.assertEqual(rows[0]["exclusion_reason"], "")
            summary = json.loads((out_dir / scope_policy.SUMMARY_JSON_NAME).read_text(encoding="utf-8"))
            self.assertEqual(summary["gt_consumed"], "no")
            self.assertEqual(summary["live_llm_calls"], "no")
            self.assertEqual(summary["benchmark_valid"], "no")
            self.assertEqual(summary["eligible_for_s5_3_rows"], 1)
            context = (out_dir / scope_policy.RUN_CONTEXT_NAME).read_text(encoding="utf-8")
            self.assertIn("Empty final-table schema slots alone are never eligible", context)
            self.assertIn("gt_inputs_consumed: `no`", context)
            self.assertIn("live_llm_calls: `no`", context)
            self.assertIn(str(final_table.resolve()), context)
            self.assertIn(str(observability.resolve()), context)

    def test_blank_final_slot_without_source_observability_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final_table = root / "final.tsv"
            observability = root / "observability.tsv"
            out_dir = root / "out"
            write_tsv(final_table, ["paper_key", "formulation_id", "particle_size_nm"], [{"paper_key": "PAPER_B", "formulation_id": "F2", "particle_size_nm": ""}])
            write_tsv(
                observability,
                OBS_COLUMNS,
                [
                    {
                        "paper_key": "PAPER_B",
                        "formulation_id": "F2",
                        "field_name": "particle_size_nm",
                        "scope_trigger": "",
                        "source_locator": "",
                        "upstream_boundary": "",
                        "why_s5_2_failed": "",
                        "source_observability_status": "",
                        "row_identity_status": "resolved",
                        "direct_or_derived": "direct",
                        "source_context": "",
                    }
                ],
            )

            scope_policy.main(["--final-table-tsv", str(final_table), "--source-observability-tsv", str(observability), "--out-dir", str(out_dir)])

            rows = read_tsv(out_dir / scope_policy.SCOPE_CANDIDATES_TSV_NAME)
            self.assertEqual(rows[0]["eligible_for_s5_3"], "no")
            self.assertEqual(rows[0]["exclusion_reason"], "blank_field_or_no_source_observability_signal")

    def test_excludes_derived_unresolved_identity_unrelated_and_nonfixed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final_table = root / "final.tsv"
            observability = root / "observability.tsv"
            out_dir = root / "out"
            write_tsv(final_table, ["paper_key", "formulation_id"], [{"paper_key": "PAPER_C", "formulation_id": "F3"}])
            base = {
                "paper_key": "PAPER_C",
                "formulation_id": "F3",
                "field_name": "drug_mass_mg",
                "scope_trigger": "typed_direct_candidate_unassigned",
                "source_locator": "evidence.tsv#r1",
                "upstream_boundary": "S5-2",
                "why_s5_2_failed": "not_promoted",
                "source_observability_status": "source_observable",
                "row_identity_status": "resolved",
                "direct_or_derived": "direct",
                "source_context": "formulation_row",
            }
            rows = [
                {**base, "direct_or_derived": "derived"},
                {**base, "field_name": "zeta_potential_mv", "row_identity_status": "blocked_alignment"},
                {**base, "field_name": "pdi", "source_observability_status": "ambiguous"},
                {**base, "field_name": "ee_percent", "source_context": "assay"},
                {**base, "paper_key": "PAPER_NOT_FIXED", "formulation_id": "F9"},
            ]
            write_tsv(observability, OBS_COLUMNS, rows)

            scope_policy.main(["--final-table-tsv", str(final_table), "--source-observability-tsv", str(observability), "--out-dir", str(out_dir)])

            output_rows = read_tsv(out_dir / scope_policy.SCOPE_CANDIDATES_TSV_NAME)
            self.assertEqual([row["eligible_for_s5_3"] for row in output_rows], ["no", "no", "no", "no", "no"])
            self.assertEqual(
                [row["exclusion_reason"] for row in output_rows],
                [
                    "derived_only_candidate_for_direct_field",
                    "row_identity_alignment_unresolved",
                    "ambiguous_source_scope",
                    "value_only_in_unrelated_context",
                    "row_not_in_fixed_stage5_universe",
                ],
            )

    def test_declared_triggers_and_missing_required_policy_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final_table = root / "final.tsv"
            observability = root / "observability.tsv"
            out_dir = root / "out"
            write_tsv(final_table, ["paper_key", "formulation_id"], [{"paper_key": "PAPER_D", "formulation_id": "F4"}])
            rows = [
                {
                    "paper_key": "PAPER_D",
                    "formulation_id": "F4",
                    "field_name": "ee_percent",
                    "scope_trigger": "upstream_reported_s5_2_failed",
                    "source_locator": "decision_trace.tsv#row=1",
                    "upstream_boundary": "S5-2",
                    "why_s5_2_failed": "validator_rejected_header_alias",
                    "source_observability_status": "source_observable",
                    "row_identity_status": "resolved",
                    "direct_or_derived": "direct",
                    "source_context": "formulation_row",
                },
                {
                    "paper_key": "PAPER_D",
                    "formulation_id": "F4",
                    "field_name": "polymer_identity",
                    "scope_trigger": "stage3_shared_value_not_promoted",
                    "source_locator": "resolved_relation_fields_v1.tsv#row=2",
                    "upstream_boundary": "Stage3",
                    "why_s5_2_failed": "shared_value_not_promoted",
                    "source_observability_status": "source_observable",
                    "row_identity_status": "resolved",
                    "direct_or_derived": "direct",
                    "source_context": "formulation_row",
                },
                {
                    "paper_key": "PAPER_D",
                    "formulation_id": "F4",
                    "field_name": "pdi",
                    "scope_trigger": "evidence_block_mentions_value_type",
                    "source_locator": "evidence_blocks_v1.json#block=5",
                    "upstream_boundary": "S2-2b",
                    "why_s5_2_failed": "not_mapped",
                    "source_observability_status": "not_reported",
                    "row_identity_status": "resolved",
                    "direct_or_derived": "direct",
                    "source_context": "formulation_row",
                },
                {
                    "paper_key": "PAPER_D",
                    "formulation_id": "F4",
                    "field_name": "",
                    "scope_trigger": "typed_direct_candidate_unassigned",
                    "source_locator": "evidence.tsv#row=3",
                    "upstream_boundary": "S5-2",
                    "why_s5_2_failed": "not_promoted",
                    "source_observability_status": "source_observable",
                    "row_identity_status": "resolved",
                    "direct_or_derived": "direct",
                    "source_context": "formulation_row",
                },
                {
                    "paper_key": "PAPER_D",
                    "formulation_id": "F4",
                    "field_name": "zeta_potential_mv",
                    "scope_trigger": "typed_direct_candidate_unassigned",
                    "source_locator": "evidence.tsv#row=4",
                    "upstream_boundary": "",
                    "why_s5_2_failed": "not_promoted",
                    "source_observability_status": "source_observable",
                    "row_identity_status": "resolved",
                    "direct_or_derived": "direct",
                    "source_context": "formulation_row",
                },
                {
                    "paper_key": "PAPER_D",
                    "formulation_id": "F4",
                    "field_name": "drug_loading_percent",
                    "scope_trigger": "typed_direct_candidate_unassigned",
                    "source_locator": "evidence.tsv#row=5",
                    "upstream_boundary": "S5-2",
                    "why_s5_2_failed": "",
                    "source_observability_status": "source_observable",
                    "row_identity_status": "resolved",
                    "direct_or_derived": "direct",
                    "source_context": "formulation_row",
                },
            ]
            write_tsv(observability, OBS_COLUMNS, rows)

            scope_policy.main(["--final-table-tsv", str(final_table), "--source-observability-tsv", str(observability), "--out-dir", str(out_dir)])

            output_rows = read_tsv(out_dir / scope_policy.SCOPE_CANDIDATES_TSV_NAME)
            self.assertEqual([row["eligible_for_s5_3"] for row in output_rows], ["yes", "yes", "no", "no", "no", "no"])
            self.assertEqual(
                [row["exclusion_reason"] for row in output_rows],
                ["", "", "not_reported_no_source", "missing_field_name", "missing_upstream_boundary", "missing_s5_2_failure_reason"],
            )

    def test_requires_explicit_input_paths_and_does_not_infer_active_run_or_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final_table = root / "missing_final.tsv"
            observability = root / "observability.tsv"
            write_tsv(observability, OBS_COLUMNS, [])
            with self.assertRaises(FileNotFoundError):
                scope_policy.main(
                    ["--final-table-tsv", str(final_table), "--source-observability-tsv", str(observability), "--out-dir", str(root / "out")]
                )


if __name__ == "__main__":
    unittest.main()
