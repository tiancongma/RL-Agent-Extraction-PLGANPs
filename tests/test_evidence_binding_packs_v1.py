from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.stage5_benchmark import build_evidence_binding_packs_v1 as packs


class EvidenceBindingPackBuilderTests(unittest.TestCase):
    def test_status_and_assignment_taxonomy_contains_required_values(self):
        self.assertIn("coded_value_supported_decode_pending", packs.BINDING_STATUSES)
        self.assertIn("relation_path_missing", packs.BINDING_STATUSES)
        self.assertIn("missing_exact_value_evidence", packs.BINDING_STATUSES)
        self.assertIn("stage3_relation_resolved", packs.ASSIGNMENT_PATHS)
        self.assertIn("doe_factor_decode", packs.ASSIGNMENT_PATHS)

    def test_relation_supported_requires_relation_source_path(self):
        row = {"final_formulation_id": "P1__F1", "polymer_mass_mg": "50", "table_id": "Table 1"}
        relation_index = {
            ("P1__F1", "polymer_mass_mg"): {
                "resolution_rule": "shared_method_context",
                "source_relation_row_ids": "rel-1",
                "field_value": "50",
                "source_table_id": "Table 1",
                "source_table_row_id": "F1",
            }
        }
        pack = packs.build_field_pack(row, "polymer_mass_mg", relation_index=relation_index)
        self.assertEqual(pack["binding_status"], "relation_supported")
        self.assertEqual(pack["assignment_path"], "stage3_relation_resolved")
        self.assertEqual(pack["relation_path"]["source_relation_row_ids"], "rel-1")
        self.assertEqual(pack["source_label"], "Table 1")
        self.assertIn("F1", pack["source_locator_text"])
        self.assertEqual(pack["binding_strength"], "scoped_table")
        self.assertIn("field=polymer_mass_mg", pack["review_display_text"])

    def test_relation_lookup_uses_base_field_for_value_columns(self):
        row = {"final_formulation_id": "P1__F1", "encapsulation_efficiency_percent_value": "89±5"}
        relation_index = {
            ("P1__F1", "encapsulation_efficiency_percent"): {
                "source_relation_row_ids": "rel-2",
                "field_value": "89±5",
                "source_metric_header": "Encapsulation efficiency (%)",
            }
        }
        pack = packs.build_field_pack(
            row,
            "encapsulation_efficiency_percent_value",
            relation_index=relation_index,
        )
        self.assertEqual(pack["binding_status"], "relation_supported")
        self.assertEqual(pack["metric_match_basis"], "Encapsulation efficiency (%)")
        self.assertEqual(pack["evidence_contains_exact_value"], "yes")

    def test_numeric_direct_support_requires_exact_value_in_evidence_text(self):
        supported = packs.build_field_pack(
            {
                "final_formulation_id": "P1__F1",
                "encapsulation_efficiency_percent_value": "89",
                "encapsulation_efficiency_percent_value_text": "F1 EE (%) 89 ± 5",
            },
            "encapsulation_efficiency_percent_value",
            relation_index={},
        )
        self.assertEqual(supported["binding_status"], "direct_supported")
        self.assertEqual(supported["value_evidence_text"], "F1 EE (%) 89 ± 5")
        self.assertEqual(supported["evidence_contains_exact_value"], "yes")

        missing_exact = packs.build_field_pack(
            {
                "final_formulation_id": "P1__F1",
                "encapsulation_efficiency_percent_value": "89",
                "encapsulation_efficiency_percent_scope": "Table 1",
                "evidence_span_text": "F1 was prepared as a PLGA nanoparticle.",
            },
            "encapsulation_efficiency_percent_value",
            relation_index={},
        )
        self.assertEqual(missing_exact["binding_status"], "missing_exact_value_evidence")
        self.assertEqual(missing_exact["value_evidence_text"], "")
        self.assertEqual(missing_exact["evidence_contains_exact_value"], "no")
        self.assertIn("F1 was prepared", missing_exact["row_identity_evidence_text"])

    def test_relation_value_without_relation_path_is_missing_not_supported(self):
        row = {"final_formulation_id": "P1__F1", "polymer_mass_mg": "50"}
        relation_index = {
            ("P1__F1", "polymer_mass_mg"): {
                "resolution_rule": "shared_method_context",
                "source_relation_row_ids": "",
                "field_value": "50",
            }
        }
        pack = packs.build_field_pack(row, "polymer_mass_mg", relation_index=relation_index)
        self.assertEqual(pack["binding_status"], "relation_path_missing")
        self.assertEqual(pack["assignment_path"], "stage3_relation_resolved")

    def test_blank_and_supporting_refs_are_not_direct_support(self):
        blank = packs.build_field_pack({"final_formulation_id": "P1__F1", "ee_percent": ""}, "ee_percent", relation_index={})
        self.assertEqual(blank["binding_status"], "blank_value")
        row = {"final_formulation_id": "P1__F1", "drug_mass_mg": "5", "supporting_evidence_refs": "[{\"target_field_name\": \"drug mass\"}]"}
        pack = packs.build_field_pack(row, "drug_mass_mg", relation_index={})
        self.assertEqual(pack["binding_status"], "missing_evidence_anchor")
        self.assertEqual(pack["supporting_refs_class"], "broad_anchor")

    def test_write_jsonl_and_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "out"
            rows = [
                {"final_formulation_id": "P1__F1", "polymer_mass_mg": "50", "drug_mass_mg": ""},
            ]
            relation_index = {
                ("P1__F1", "polymer_mass_mg"): {
                    "resolution_rule": "shared_method_context",
                    "source_relation_row_ids": "rel-1",
                    "field_value": "50",
                }
            }
            result = packs.write_pack_outputs(rows=rows, fields=["polymer_mass_mg", "drug_mass_mg"], relation_index=relation_index, out_dir=out, source_manifest={"active_run_id": "test"})
            self.assertTrue((out / "evidence_binding_packs_v1.jsonl").exists())
            self.assertTrue((out / "evidence_binding_field_summary_v1.tsv").exists())
            lines = (out / "evidence_binding_packs_v1.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            first = json.loads(lines[0])
            self.assertIn(first["binding_status"], packs.BINDING_STATUSES)
            self.assertEqual(result["pack_count"], 2)


if __name__ == "__main__":
    unittest.main()
