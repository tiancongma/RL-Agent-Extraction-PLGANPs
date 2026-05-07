import csv
import tempfile
import unittest
from pathlib import Path

from src.stage2_sampling_labels.audit_pre_llm_acceptance_gate_v1 import (
    evaluate_pre_llm_acceptance_row,
    write_pre_llm_acceptance_gate,
)


class TestPreLlmAcceptanceGate(unittest.TestCase):
    def test_passes_when_clean_selector_prompt_and_sidecars_are_explicitly_ready(self):
        row = evaluate_pre_llm_acceptance_row(
            {
                "paper_key": "PASS1",
                "clean_text_status": "pass",
                "structure_sidecar_status": "loaded",
                "table_cell_sidecar_status": "consumed",
                "candidate_has_ee_or_loading_signal": "yes",
                "selected_has_ee_or_loading_signal": "yes",
                "candidate_has_preparation_core": "yes",
                "selected_has_preparation_core": "yes",
                "prompt_size_policy_status": "healthy",
                "s2_3_ready_overall": "pass",
                "selector_authority_filter_violations": "0",
                "selected_table_authority_status": "backed_by_full_authority",
                "tail_noise_with_weak_ee_signal": "no",
            }
        )
        self.assertEqual(row["pre_llm_acceptance_status"], "pass_for_live_llm")
        self.assertEqual(row["first_failure_boundary"], "none")
        self.assertEqual(row["benchmark_valid"], "no")
        self.assertEqual(row["diagnostic_only"], "yes")

    def test_holds_when_selected_evidence_misses_existing_ee_signal(self):
        row = evaluate_pre_llm_acceptance_row(
            {
                "paper_key": "MISS_EE",
                "clean_text_status": "pass",
                "structure_sidecar_status": "loaded",
                "table_cell_sidecar_status": "consumed",
                "candidate_has_ee_or_loading_signal": "yes",
                "selected_has_ee_or_loading_signal": "no",
                "candidate_has_preparation_core": "yes",
                "selected_has_preparation_core": "yes",
                "prompt_size_policy_status": "healthy",
                "s2_3_ready_overall": "pass",
                "selector_authority_filter_violations": "0",
                "selected_table_authority_status": "backed_by_full_authority",
            }
        )
        self.assertEqual(row["pre_llm_acceptance_status"], "hold_for_selector_or_cleantext_review")
        self.assertEqual(row["first_failure_boundary"], "stage2_selector_missing_ee_or_loading_signal")
        self.assertIn("selected_evidence_missing_ee_or_loading_signal", row["gate_reasons"])

    def test_holds_on_silent_missing_sidecar_but_allows_explicit_not_applicable(self):
        missing = evaluate_pre_llm_acceptance_row(
            {
                "paper_key": "SIDE_MISSING",
                "clean_text_status": "pass",
                "structure_sidecar_status": "missing",
                "table_cell_sidecar_status": "not_configured",
                "candidate_has_ee_or_loading_signal": "no",
                "selected_has_ee_or_loading_signal": "no",
                "candidate_has_preparation_core": "no",
                "selected_has_preparation_core": "no",
                "prompt_size_policy_status": "healthy",
                "s2_3_ready_overall": "pass",
                "selector_authority_filter_violations": "0",
                "selected_table_authority_status": "no_selected_tables",
            }
        )
        self.assertEqual(missing["first_failure_boundary"], "stage1_structure_sidecar_missing")
        self.assertIn("structure_sidecar_missing_or_silent", missing["gate_reasons"])

        explicit = evaluate_pre_llm_acceptance_row(
            {
                "paper_key": "SIDE_NA",
                "clean_text_status": "pass",
                "structure_sidecar_status": "explicitly_not_applicable",
                "table_cell_sidecar_status": "explicitly_not_applicable",
                "candidate_has_ee_or_loading_signal": "no",
                "selected_has_ee_or_loading_signal": "no",
                "candidate_has_preparation_core": "no",
                "selected_has_preparation_core": "no",
                "prompt_size_policy_status": "healthy",
                "s2_3_ready_overall": "pass",
                "selector_authority_filter_violations": "0",
                "selected_table_authority_status": "no_selected_tables",
            }
        )
        self.assertEqual(explicit["pre_llm_acceptance_status"], "pass_for_live_llm")

    def test_write_gate_splits_pass_and_hold_manifests(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_tsv = root / "inputs.tsv"
            input_tsv.write_text(
                "paper_key\tclean_text_status\tstructure_sidecar_status\ttable_cell_sidecar_status\t"
                "candidate_has_ee_or_loading_signal\tselected_has_ee_or_loading_signal\t"
                "candidate_has_preparation_core\tselected_has_preparation_core\t"
                "prompt_size_policy_status\ts2_3_ready_overall\tselector_authority_filter_violations\t"
                "selected_table_authority_status\n"
                "PASS\tpass\tloaded\tconsumed\tyes\tyes\tyes\tyes\thealthy\tpass\t0\tbacked_by_full_authority\n"
                "HOLD\tpass\tloaded\tconsumed\tyes\tno\tyes\tyes\thealthy\tpass\t0\tbacked_by_full_authority\n",
                encoding="utf-8",
            )
            out_dir = root / "out"
            write_pre_llm_acceptance_gate(input_tsv=input_tsv, out_dir=out_dir)

            with (out_dir / "pre_llm_acceptance_gate_v1.tsv").open(encoding="utf-8", newline="") as handle:
                report_rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(report_rows), 2)
            self.assertTrue((out_dir / "pre_llm_acceptance_pass_manifest_v1.tsv").exists())
            self.assertTrue((out_dir / "pre_llm_acceptance_hold_manifest_v1.tsv").exists())
            with (out_dir / "pre_llm_acceptance_pass_manifest_v1.tsv").open(encoding="utf-8", newline="") as handle:
                pass_rows = list(csv.DictReader(handle, delimiter="\t"))
            with (out_dir / "pre_llm_acceptance_hold_manifest_v1.tsv").open(encoding="utf-8", newline="") as handle:
                hold_rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual([r["paper_key"] for r in pass_rows], ["PASS"])
            self.assertEqual([r["paper_key"] for r in hold_rows], ["HOLD"])


if __name__ == "__main__":
    unittest.main()
