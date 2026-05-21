from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.stage5_benchmark import build_evidence_binding_risk_assessment_v1 as risk


class EvidenceBindingRiskAssessmentTests(unittest.TestCase):
    def test_pack_status_maps_to_risk_without_resolving_sources(self):
        pack = {
            "paper_key": "P1",
            "final_formulation_id": "P1__F1",
            "field_name": "drug_mass_mg",
            "frozen_value": "5",
            "binding_status": "missing_evidence_anchor",
            "assignment_path": "unresolved",
        }
        row = risk.assess_pack_risk(pack)
        self.assertEqual(row["risk_level"], "high")
        self.assertEqual(row["risk_type"], "missing_evidence_anchor")
        self.assertEqual(row["evidence_status"], "missing_evidence_anchor")

    def test_supported_and_blank_values_are_low_risk(self):
        supported = risk.assess_pack_risk({
            "paper_key": "P1",
            "final_formulation_id": "P1__F1",
            "field_name": "polymer_mass_mg",
            "frozen_value": "50",
            "binding_status": "relation_supported",
            "assignment_path": "stage3_relation_resolved",
        })
        blank = risk.assess_pack_risk({
            "paper_key": "P1",
            "final_formulation_id": "P1__F1",
            "field_name": "ee_percent",
            "frozen_value": "",
            "binding_status": "blank_value",
            "assignment_path": "blank_value",
        })
        self.assertEqual(supported["risk_level"], "low")
        self.assertEqual(blank["risk_level"], "low")

    def test_missing_exact_value_evidence_is_high_risk(self):
        row = risk.assess_pack_risk({
            "paper_key": "P1",
            "final_formulation_id": "P1__F1",
            "field_name": "encapsulation_efficiency_percent_value",
            "frozen_value": "89",
            "binding_status": "missing_exact_value_evidence",
            "assignment_path": "direct_same_table_row",
            "binding_strength": "weak_or_missing",
            "evidence_contains_exact_value": "no",
        })
        self.assertEqual(row["risk_level"], "high")
        self.assertEqual(row["risk_type"], "weak_binding_chain")
        self.assertEqual(row["evidence_status"], "missing_exact_value_evidence")
        self.assertEqual(row["evidence_contains_exact_value"], "no")

    def test_rollups_choose_highest_risk_per_group(self):
        rows = [
            {"paper_key": "P1", "final_formulation_id": "P1__F1", "field_name": "a", "risk_level": "low", "risk_type": "supported"},
            {"paper_key": "P1", "final_formulation_id": "P1__F1", "field_name": "b", "risk_level": "high", "risk_type": "missing_evidence_anchor"},
            {"paper_key": "P1", "final_formulation_id": "P1__F2", "field_name": "a", "risk_level": "medium", "risk_type": "coded_value_supported_decode_pending"},
        ]
        formulation = risk.rollup_risk(rows, group_field="final_formulation_id")
        by_group = {row["group_key"]: row for row in formulation}
        self.assertEqual(by_group["P1__F1"]["risk_level"], "high")
        self.assertEqual(by_group["P1__F1"]["high_count"], "1")
        self.assertEqual(by_group["P1__F2"]["risk_level"], "medium")

    def test_row_review_queue_rolls_up_core_fields_only(self):
        rows = [
            {
                "paper_key": "P1",
                "final_formulation_id": "P1__F1",
                "field_name": "encapsulation_efficiency_percent_value",
                "risk_level": "low",
                "risk_type": "supported",
                "frozen_value_present": "yes",
                "binding_strength": "direct_row",
                "assignment_path": "stage3_relation_resolved",
            },
            {
                "paper_key": "P1",
                "final_formulation_id": "P1__F1",
                "field_name": "polymer_identity_final",
                "risk_level": "high",
                "risk_type": "weak_binding_chain",
                "frozen_value_present": "yes",
                "binding_strength": "weak_or_missing",
                "assignment_path": "unresolved",
            },
        ]
        queue = risk.build_row_review_queue_rows(rows)
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["row_risk_level"], "high")
        self.assertIn("polymer_identity_final", queue[0]["core_fields_high_risk"])
        self.assertEqual(queue[0]["review_scope"], "row")

    def test_run_risk_assessment_writes_row_level_no_gt_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack_path = root / "packs.jsonl"
            packs = [
                {
                    "paper_key": "P1",
                    "final_formulation_id": "P1__F1",
                    "field_name": "encapsulation_efficiency_percent_value",
                    "frozen_value": "89±5",
                    "binding_status": "direct_supported",
                    "assignment_path": "direct_same_table_row",
                    "binding_strength": "direct_row",
                },
                {
                    "paper_key": "P1",
                    "final_formulation_id": "P1__F1",
                    "field_name": "polymer_identity_final",
                    "frozen_value": "PLGA",
                    "binding_status": "missing_evidence_anchor",
                    "assignment_path": "unresolved",
                    "binding_strength": "weak_or_missing",
                },
                {
                    "paper_key": "P1",
                    "final_formulation_id": "P1__F1",
                    "field_name": "retention_reason",
                    "frozen_value": "kept",
                    "binding_status": "missing_evidence_anchor",
                    "assignment_path": "unresolved",
                    "binding_strength": "weak_or_missing",
                },
            ]
            pack_path.write_text("\n".join(json.dumps(p) for p in packs) + "\n", encoding="utf-8")
            out_dir = root / "out"
            result = risk.run_risk_assessment(pack_path=pack_path, out_dir=out_dir)
            self.assertEqual(result["input_pack_count"], 3)
            self.assertEqual(result["field_row_count"], 2)
            self.assertEqual(result["row_review_queue_count"], 1)
            self.assertTrue((out_dir / "evidence_binding_field_risk_v1.tsv").exists())
            self.assertTrue((out_dir / "evidence_binding_row_review_queue_v1.tsv").exists())
            self.assertTrue((out_dir / "evidence_binding_formulation_risk_v1.tsv").exists())
            self.assertTrue((out_dir / "evidence_binding_paper_risk_v1.tsv").exists())
            self.assertTrue((out_dir / "RUN_CONTEXT.md").exists())


if __name__ == "__main__":
    unittest.main()
