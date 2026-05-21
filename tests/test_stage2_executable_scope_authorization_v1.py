import json
import tempfile
import unittest
from pathlib import Path

from src.stage2_sampling_labels.extract_semantic_stage2_objects_v2 import (
    convert_legacy_raw_response_to_v2,
)
from src.stage2_sampling_labels.table_row_expansion_v1 import row_values_look_like_aggregate_variant_list
from src.utils.paths import DATA_RESULTS_DIR


class Stage2ExecutableScopeAuthorizationTest(unittest.TestCase):
    def test_executable_scope_authorization_projects_to_shrunken_contract(self):
        with tempfile.TemporaryDirectory(dir=DATA_RESULTS_DIR) as temp_dir:
            root = Path(temp_dir)
            text_path = root / "paper.txt"
            text_path.write_text("placeholder", encoding="utf-8")
            raw_path = root / "TESTKEY__stage2_v2_raw_response.json"
            raw_payload = {
                "paper_key": "TESTKEY",
                "paper_level_formulation_status": "has_formulation_records",
                "semantic_authorization_note": "authorize table expansion only",
                "formulation_scope_authorizations": [
                    {
                        "scope_id": "scope_001",
                        "source_scope_type": "table",
                        "source_scope_ref": "Table 2",
                        "semantic_role": "doe_run_matrix",
                        "formulation_bearing_status": "formulation_bearing",
                        "row_universe_signal": "doe_runs",
                        "downstream_expansion_required": "yes",
                        "expected_expansion_unit": "run_id",
                        "authority_locator": {
                            "primary_table_ref": "Table 2",
                            "factor_column_hints": ["X1", "X2"],
                            "metric_column_hints": ["Y1"],
                        },
                        "linked_scope_refs": ["Table 1"],
                        "decoding_status": "requires_linked_factor_map",
                        "expansion_guardrails": [],
                        "why_authorized": "Rows are DOE runs.",
                        "confidence": "high",
                    }
                ],
                "non_formulation_scope_classifications": [],
                "cross_scope_expansion_links": [
                    {
                        "link_type": "factor_map_to_run_matrix",
                        "source_scope_ref": "Table 1",
                        "target_scope_ref": "Table 2",
                        "binding_basis": "same_table_caption",
                        "downstream_action": "join_for_doe_decoding",
                        "confidence": "medium",
                    }
                ],
                "coverage_check": {
                    "formulation_bearing_scope_count": 1,
                    "linked_supporting_scope_count": 1,
                    "unclear_scope_count": 0,
                    "notes_on_possible_omissions": "",
                },
            }
            raw_path.write_text(json.dumps(raw_payload), encoding="utf-8")
            document = convert_legacy_raw_response_to_v2(
                record={
                    "key": "TESTKEY",
                    "doi": "10.test/example",
                    "title": "Test paper",
                    "text_path": str(text_path),
                },
                raw_response_path=raw_path,
                raw_response_text=raw_path.read_text(encoding="utf-8"),
                authority_metadata={},
            )

        self.assertEqual(document["source_raw_response_schema"], "stage2_executable_scope_authorization_v1")
        self.assertEqual(document["table_scopes"][0]["table_id"], "Table 2")
        self.assertEqual(document["table_scopes"][0]["scope_kind"], "doe_table")
        self.assertTrue(document["table_scopes"][0]["is_formulation_bearing"])
        self.assertTrue(document["table_scopes"][0]["is_doe"])
        self.assertEqual(document["semantic_signals"]["primary_variable_names"], ["X1", "X2"])
        self.assertTrue(document["semantic_signals"]["has_variable_sweep"])
        self.assertTrue(document["semantic_signals"]["has_parent_child_table_relation"])
        self.assertEqual(document["formulation_candidates"][0]["candidate_kind"], "formulation_family")
        self.assertEqual(document["table_formulation_scopes"][0]["table_type"], "doe_table")
        self.assertEqual(document["executable_scope_authorizations"][0]["scope_id"], "scope_001")

    def test_hbnps_single_row_is_not_aggregate_variant_list(self):
        self.assertFalse(
            row_values_look_like_aggregate_variant_list(
                ["HbNPs-7 | 50 | 1 | 466.7 | 0.391 | 16.6 | 23.9 | Yes"]
            )
        )
        self.assertTrue(
            row_values_look_like_aggregate_variant_list(
                ["blank NPs | FITC-NPs | drug-loaded NPs"]
            )
        )


if __name__ == "__main__":
    unittest.main()
