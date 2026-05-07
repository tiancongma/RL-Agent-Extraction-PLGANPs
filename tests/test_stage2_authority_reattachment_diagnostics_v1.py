import json
import tempfile
import unittest
from pathlib import Path

from src.stage2_sampling_labels.validate_stage2_semantic_authority_contract_v1 import (
    summarize_authority_reattachment_sidecar,
)
from src.stage2_sampling_labels.build_stage2_compatibility_projection_v1 import (
    load_authority_sidecar,
    run_projection,
    LEGACY_TSV_NAME,
    TRACE_TSV_NAME,
)


class Stage2AuthorityReattachmentDiagnosticsTest(unittest.TestCase):
    def _write_t05_sidecar(self, root: Path, *, status: str = "resolved") -> Path:
        sidecar = {
            "contract_version": "s2_5b_semantic_authority_reattachment_v1",
            "diagnostic_only": True,
            "benchmark_valid": False,
            "paper_key": "PAPER1",
            "payload_root": str(root / "payloads"),
            "summary": {
                "semantic_signal_count": 1,
                "resolved_signal_count": 1 if status == "resolved" else 0,
                "ambiguous_signal_count": 1 if status == "ambiguous" else 0,
                "unresolved_signal_count": 1 if status == "unresolved" else 0,
            },
            "reattachments": [
                {
                    "scope_id": "scope_table_1",
                    "signal_family": "table_formulation_scopes",
                    "resolution_status": status,
                    "selected_authority_record": {
                        "table_id": "Table 1",
                        "source_table_asset_id": "paper1_table_1",
                        "source_table_reference": "paper1_table_1.csv",
                        "payload_artifact_path": str(root / "payloads" / "PAPER1" / "normalized_table_payloads_v1.json"),
                        "normalized_csv_path": "data/cleaned/tables/PAPER1/table_1.csv",
                    },
                    "grid_path": str(root / "grid" / "table_cell_grid_v1.tsv"),
                }
            ],
        }
        path = root / "semantic_stage2_objects" / "authority_reattachment" / "PAPER1" / "semantic_authority_reattachment_v1.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sidecar), encoding="utf-8")
        return root

    def test_summarize_t05_directory_sidecars_counts_resolution_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_t05_sidecar(root, status="ambiguous")
            summary = summarize_authority_reattachment_sidecar(root)
            self.assertEqual(summary["semantic_signal_count"], 1)
            self.assertEqual(summary["reattached_target_count"], 0)
            self.assertEqual(summary["ambiguous_target_count"], 1)
            self.assertEqual(summary["unresolved_target_count"], 0)
            self.assertEqual(summary["paper_count"], 1)
            self.assertEqual(summary["status"], "diagnostic_complete")

    def test_load_t05_sidecar_directory_provides_projection_locator_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sidecar_root = self._write_t05_sidecar(root, status="resolved")
            entries = load_authority_sidecar(sidecar_root)
            entry = entries["PAPER1"]
            self.assertEqual(entry["resolution_status"], "resolved")
            self.assertEqual(entry["authority_payload_root"], str(root / "payloads"))
            self.assertEqual(entry["table_scope_locators"][0]["source_table_reference"], "paper1_table_1.csv")
            self.assertEqual(entry["reattachment_diagnostics"][0]["table_cell_grid_ref"], str(root / "grid" / "table_cell_grid_v1.tsv"))

    def test_load_t05_sidecar_directory_keeps_unresolved_diagnostic_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sidecar_root = self._write_t05_sidecar(root, status="unresolved")
            entries = load_authority_sidecar(sidecar_root)
            entry = entries["PAPER1"]
            self.assertEqual(entry["resolution_status"], "unresolved")
            self.assertEqual(entry["table_scope_locators"], [])
            self.assertEqual(entry["reattachment_diagnostics"][0]["resolution_status"], "unresolved")
            self.assertEqual(entry["reattachment_diagnostics"][0]["authority_target_id"], "paper1_table_1.csv")

    def test_projection_trace_exposes_reattachment_diagnostic_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            semantic_path = root / "semantic.jsonl"
            semantic_doc = {
                "document_key": "PAPER1",
                "paper_key": "PAPER1",
                "doi": "10/test",
                "stage2_semantic_source_mode": "llm_first_composite",
                "table_scopes": [{"table_id": "Table 1", "scope_kind": "formulation_table", "confidence": "high"}],
                "semantic_signals": {
                    "has_variable_sweep": False,
                    "has_sequential_optimization": False,
                    "has_parent_child_table_relation": False,
                    "has_downstream_non_synthesis_variants": False,
                    "has_measurement_only_variants": False,
                    "primary_preparation_method_hint": "nanoprecipitation",
                    "primary_variable_names": [],
                    "selected_condition_hints": [],
                },
                "formulation_candidates": [
                    {
                        "candidate_id": "PAPER1_F1",
                        "candidate_kind": "single_formulation",
                        "instance_role": "synthesis_core",
                        "status": "reported",
                        "confidence": "high",
                    }
                ],
                "semantic_scope_declarations": [{"scope_id": "doc_scope", "scope_kind": "document_formulation_scope"}],
                "formulation_identity_candidates": [
                    {
                        "formulation_candidate_id": "PAPER1_F1",
                        "raw_formulation_label": "F1",
                        "semantic_scope_ref": "doc_scope",
                        "semantic_scope_authority": "llm_declared_scope",
                    }
                ],
            }
            semantic_path.write_text(json.dumps(semantic_doc) + "\n", encoding="utf-8")
            sidecar_root = self._write_t05_sidecar(root, status="resolved")
            out_dir = root / "out"
            summary = run_projection(
                input_path=semantic_path,
                output_dir=out_dir,
                contract_path=out_dir / "contract.tsv",
                authority_sidecar_path=sidecar_root,
            )
            self.assertTrue((out_dir / LEGACY_TSV_NAME).exists())
            self.assertEqual(summary["authority_reattachment_diagnostics"]["reattached_target_count"], 1)
            trace_text = (out_dir / TRACE_TSV_NAME).read_text(encoding="utf-8")
            self.assertIn("authority_target_id", trace_text.splitlines()[0])
            self.assertIn("paper1_table_1.csv", trace_text)
            self.assertIn(str(root / "grid" / "table_cell_grid_v1.tsv"), trace_text)
            self.assertIn("resolved", trace_text)

    def test_projection_trace_exposes_missing_sidecar_status_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            semantic_path = root / "semantic.jsonl"
            semantic_doc = {
                "document_key": "PAPER1",
                "paper_key": "PAPER1",
                "doi": "10/test",
                "stage2_semantic_source_mode": "llm_first_composite",
                "table_scopes": [{"table_id": "Table 1", "scope_kind": "formulation_table", "confidence": "high"}],
                "semantic_signals": {
                    "has_variable_sweep": False,
                    "has_sequential_optimization": False,
                    "has_parent_child_table_relation": False,
                    "has_downstream_non_synthesis_variants": False,
                    "has_measurement_only_variants": False,
                    "primary_preparation_method_hint": "nanoprecipitation",
                    "primary_variable_names": [],
                    "selected_condition_hints": [],
                },
                "formulation_candidates": [
                    {
                        "candidate_id": "PAPER1_F1",
                        "candidate_kind": "single_formulation",
                        "instance_role": "synthesis_core",
                        "status": "reported",
                        "confidence": "high",
                    }
                ],
                "semantic_scope_declarations": [{"scope_id": "doc_scope", "scope_kind": "document_formulation_scope"}],
                "formulation_identity_candidates": [
                    {
                        "formulation_candidate_id": "PAPER1_F1",
                        "raw_formulation_label": "F1",
                        "semantic_scope_ref": "doc_scope",
                        "semantic_scope_authority": "llm_declared_scope",
                    }
                ],
            }
            semantic_path.write_text(json.dumps(semantic_doc) + "\n", encoding="utf-8")
            out_dir = root / "out_missing"
            summary = run_projection(
                input_path=semantic_path,
                output_dir=out_dir,
                contract_path=out_dir / "contract.tsv",
                authority_sidecar_path=root / "does_not_exist",
            )
            self.assertEqual(summary["authority_reattachment_diagnostics"]["status"], "missing")
            trace_text = (out_dir / TRACE_TSV_NAME).read_text(encoding="utf-8")
            self.assertIn("authority_reattachment_diagnostic", trace_text)
            self.assertIn("missing", trace_text)


if __name__ == "__main__":
    unittest.main()
