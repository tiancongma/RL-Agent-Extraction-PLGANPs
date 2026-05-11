import json
import tempfile
import unittest
from pathlib import Path

from src.stage2_sampling_labels.build_semantic_authority_reattachment_v1 import (
    build_reattachment_for_document,
    load_payload_records,
    normalize_alias_token,
)


class SemanticAuthorityReattachmentTests(unittest.TestCase):
    def test_normalize_alias_token_removes_extension_and_case_noise(self):
        self.assertEqual(
            normalize_alias_token(" Data/Tables/PAPER__table_01__PDF_table.csv "),
            "paper table 01 pdf table",
        )
        self.assertEqual(normalize_alias_token("Table 1"), "table 1")

    def test_load_payload_records_indexes_table_aliases(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            payload_path = root / "PAPER" / "normalized_table_payloads_v1.json"
            payload_path.parent.mkdir(parents=True)
            payload_path.write_text(json.dumps({
                "paper_key": "PAPER",
                "normalized_table_payloads": [
                    {
                        "table_id": "Table 1",
                        "source_table_reference": "data/tables/PAPER__table_01__pdf_table.csv",
                        "source_table_asset_id": "PAPER__table_01__pdf_table",
                        "source_caption_or_title": "Optimization matrix",
                        "normalized_csv_path": "payloads/PAPER__table_01__pdf_table__normalized.csv",
                        "authority_rank": 1,
                        "authority_score": 9.5,
                    }
                ],
            }))
            records = load_payload_records(root, "PAPER")
            self.assertEqual(len(records), 1)
            self.assertIn("table 1", records[0].aliases)
            self.assertIn("paper table 01 pdf table", records[0].aliases)
            self.assertIn("optimization matrix", records[0].aliases)

    def test_build_reattachment_resolves_exact_table_scope_to_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            payload_path = root / "PAPER" / "normalized_table_payloads_v1.json"
            payload_path.parent.mkdir(parents=True)
            payload_path.write_text(json.dumps({
                "paper_key": "PAPER",
                "normalized_table_payloads": [
                    {
                        "table_id": "Table 1",
                        "source_table_reference": "data/tables/PAPER__table_01__pdf_table.csv",
                        "source_table_asset_id": "PAPER__table_01__pdf_table",
                        "source_caption_or_title": "Formulation factors",
                        "normalized_csv_path": "payloads/PAPER__table_01__pdf_table__normalized.csv",
                        "authority_rank": 1,
                        "authority_score": 10,
                    }
                ],
            }))
            doc = {
                "paper_key": "PAPER",
                "table_scopes": [
                    {
                        "scope_id": "scope-1",
                        "table_id": "Table 1",
                        "scope_kind": "formulation_table",
                        "is_formulation_bearing": True,
                        "source_table_asset_id": "PAPER__table_01__pdf_table",
                    }
                ],
            }
            sidecar, rows = build_reattachment_for_document(doc, root, None)
            self.assertEqual(sidecar["diagnostic_only"], True)
            self.assertEqual(sidecar["benchmark_valid"], False)
            self.assertEqual(sidecar["summary"]["resolved_signal_count"], 1)
            binding = sidecar["reattachments"][0]
            self.assertEqual(binding["resolution_status"], "resolved")
            self.assertEqual(binding["selected_authority_record"]["table_id"], "Table 1")
            self.assertEqual(rows[0]["resolution_status"], "resolved")

    def test_exact_source_locator_beats_broad_low_authority_table_label(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            payload_path = root / "PAPER" / "normalized_table_payloads_v1.json"
            payload_path.parent.mkdir(parents=True)
            payload_path.write_text(json.dumps({
                "paper_key": "PAPER",
                "normalized_table_payloads": [
                    {
                        "table_id": "Table 1",
                        "source_table_reference": "data/tables/PAPER__table_01__pdf_table.csv",
                        "source_table_asset_id": "PAPER__table_01__pdf_table",
                        "authority_rank": 9,
                        "authority_score": -20,
                    },
                    {
                        "table_id": "Table 15",
                        "source_table_reference": "data/tables/PAPER__table_15__pdf_table.csv",
                        "source_table_asset_id": "PAPER__table_15__pdf_table",
                        "authority_rank": 1,
                        "authority_score": 80,
                    },
                ],
            }))
            doc = {
                "paper_key": "PAPER",
                "table_scopes": [
                    {
                        "scope_id": "scope-1",
                        "table_id": "Table 1",
                        "parent_table_hint": "Table 1",
                        "source_table_asset_id": "PAPER__table_15__pdf_table",
                        "source_table_reference": "data/tables/PAPER__table_15__pdf_table.csv",
                        "is_formulation_bearing": True,
                    }
                ],
            }
            sidecar, _ = build_reattachment_for_document(doc, root, None)
            binding = sidecar["reattachments"][0]
            self.assertEqual(binding["resolution_status"], "resolved")
            self.assertEqual(binding["selected_authority_record"]["table_id"], "Table 15")
            self.assertNotIn("raw_cells", binding["selected_authority_record"])
            self.assertNotIn("normalized_rows", binding["selected_authority_record"])

    def test_no_semantic_table_signal_creates_no_authority_target(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            payload_path = root / "PAPER" / "normalized_table_payloads_v1.json"
            payload_path.parent.mkdir(parents=True)
            payload_path.write_text(json.dumps({
                "paper_key": "PAPER",
                "normalized_table_payloads": [
                    {"table_id": "Table 1", "source_table_reference": "a.csv"}
                ],
            }))
            doc = {"paper_key": "PAPER", "table_scopes": [{"table_id": "Table 1", "is_formulation_bearing": False}]}
            sidecar, rows = build_reattachment_for_document(doc, root, None)
            self.assertEqual(sidecar["summary"]["semantic_signal_count"], 0)
            self.assertEqual(sidecar["reattachments"], [])
            self.assertEqual(rows, [])

    def test_build_reattachment_records_ambiguous_and_unresolved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            payload_path = root / "PAPER" / "normalized_table_payloads_v1.json"
            payload_path.parent.mkdir(parents=True)
            payload_path.write_text(json.dumps({
                "paper_key": "PAPER",
                "normalized_table_payloads": [
                    {"table_id": "Table 1", "source_table_reference": "a.csv", "authority_rank": 2},
                    {"table_id": "Table 1", "source_table_reference": "b.csv", "authority_rank": 1},
                ],
            }))
            doc = {
                "paper_key": "PAPER",
                "table_scopes": [
                    {"scope_id": "ambig", "table_id": "Table 1", "is_formulation_bearing": True},
                    {"scope_id": "missing", "table_id": "Table 9", "is_formulation_bearing": True},
                ],
            }
            sidecar, _ = build_reattachment_for_document(doc, root, None)
            statuses = [item["resolution_status"] for item in sidecar["reattachments"]]
            self.assertEqual(statuses, ["ambiguous", "unresolved"])
            self.assertEqual(sidecar["summary"]["ambiguous_signal_count"], 1)
            self.assertEqual(sidecar["summary"]["unresolved_signal_count"], 1)


if __name__ == "__main__":
    unittest.main()
