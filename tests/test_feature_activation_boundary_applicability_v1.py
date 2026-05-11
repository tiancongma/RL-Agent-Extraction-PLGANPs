import json
import tempfile
import unittest
from pathlib import Path

from src.utils.build_feature_activation_report_v1 import (
    compute_activation_gate,
    expected_for_run,
    observe_s2_2_duplicate_table_suppression,
    observe_variant_aware_gt_authority_switch,
)


class FeatureActivationBoundaryApplicabilityTests(unittest.TestCase):
    def test_compare_only_feature_not_expected_for_stage2_boundary(self):
        surfaces = {
            "stage2_active": True,
            "benchmark_compare": False,
            "run_context": True,
        }
        matrix_row = {
            "run_context": "required",
            "benchmark_compare": "required",
        }

        self.assertEqual(
            expected_for_run("variant_aware_gt_authority_switch", matrix_row, surfaces),
            "no",
        )

    def test_duplicate_table_suppression_not_expected_without_duplicate_event(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            evidence_path = run_dir / "semantic_stage2_objects" / "evidence_blocks" / "PAPER1" / "evidence_blocks_v1.json"
            evidence_path.parent.mkdir(parents=True)
            evidence_path.write_text(
                json.dumps(
                    {
                        "paper_key": "PAPER1",
                        "evidence_blocks": [],
                        "duplicate_table_suppression_events": [],
                    }
                ),
                encoding="utf-8",
            )
            surfaces = {
                "stage2_active": True,
                "evidence_blocks_paths": [evidence_path],
            }
            matrix_row = {"stage2_active": "required"}

            self.assertEqual(
                expected_for_run("s2_2_duplicate_table_suppression", matrix_row, surfaces),
                "no",
            )
            observed = observe_s2_2_duplicate_table_suppression(run_dir, surfaces)
            report_rows = [
                {
                    "feature_id": "s2_2_duplicate_table_suppression",
                    "expected_for_run": "no",
                    "activation_status": observed["activation_status"],
                }
            ]
            gate = compute_activation_gate(report_rows)
            self.assertNotEqual(gate["run_activation_gate"], "fail")

    def test_stage5_variant_governance_not_expected_without_final_boundary(self):
        surfaces = {
            "stage2_active": True,
            "stage5_final": False,
            "run_context": True,
        }
        matrix_row = {"run_context": "required", "stage5_final": "required"}

        self.assertEqual(
            expected_for_run("family_variant_retention_governance", matrix_row, surfaces),
            "no",
        )
    def test_variant_aware_gt_authority_accepts_frozen_layer1_counts_tsv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            counts_path = run_dir / "final_table_vs_gt_counts_by_doi.tsv"
            counts_path.write_text(
                "doi\tpaper_key\tgt_count\tpred_count\tdelta_count\tcount_status\tcompare_mode\tbenchmark_valid\tgt_authority_file\tpred_source_file\tmatched_rows\tmissing_rows\tspurious_rows\n"
                "10.1/example\tPAPER1\t1\t1\t0\tmatch\tdiagnostic\tno\tdata/cleaned/gt_authority/v1/dev15_layer1_gt_counts.tsv\tfinal_formulation_table_v1.tsv\t\t\t\n",
                encoding="utf-8",
            )

            observed = observe_variant_aware_gt_authority_switch(run_dir, {})

            self.assertEqual(observed["activation_status"], "active")
            self.assertIn("dev15_layer1_gt_counts.tsv", observed["evidence_detail"])


if __name__ == "__main__":
    unittest.main()
