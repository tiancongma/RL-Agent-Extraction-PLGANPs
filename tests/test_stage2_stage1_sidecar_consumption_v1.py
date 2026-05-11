import json
import tempfile
import unittest
from pathlib import Path

from src.stage2_sampling_labels import extract_semantic_stage2_objects_v2 as extractor


class TestStage2Stage1SidecarConsumptionV1(unittest.TestCase):
    def _write_sidecar(self, root: Path, paper_key: str) -> None:
        sidecar_dir = root / paper_key
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            ("t001", 1, 1, "Formulation", "yes", ["Formulation"]),
            ("t001", 1, 2, "Size (nm)", "yes", ["Size (nm)"]),
            ("t001", 2, 1, "F1", "no", ["Formulation"]),
            ("t001", 2, 2, "123", "no", ["Size (nm)"]),
            ("t001", 3, 1, "F2", "no", ["Formulation"]),
            ("t001", 3, 2, "456", "no", ["Size (nm)"]),
            ("t002", 1, 1, "Unselected", "yes", ["Unselected"]),
            ("t002", 2, 1, "MUST_NOT_DEFINE_CANDIDATE", "no", ["Unselected"]),
        ]
        with (sidecar_dir / "stage1_table_cells_v1.jsonl").open("w", encoding="utf-8") as handle:
            for table_id, row_index, col_index, value, is_header, header_path in rows:
                rec = {
                    "paper_key": paper_key,
                    "source_type": "HTML",
                    "source_path": "/tmp/PAPER.html",
                    "parser": "current",
                    "parser_variant": "diagnostic_bakeoff_v1",
                    "table_id": table_id,
                    "table_source_kind": "html_dom_table",
                    "page": "",
                    "bbox_json": "{}",
                    "caption": "Table 1" if table_id == "t001" else "Article metadata",
                    "caption_binding_rule": "synthetic_test_caption",
                    "continuation_group_id": "",
                    "continuation_binding_rule": "",
                    "noise_class": "confirmed_noise" if table_id == "t002" else "keep",
                    "noise_reason": "article_metadata_table" if table_id == "t002" else "",
                    "row_index": str(row_index),
                    "col_index": str(col_index),
                    "rowspan": "1",
                    "colspan": "1",
                    "raw_cell_text": value,
                    "normalized_cell_text": value,
                    "is_header_cell": is_header,
                    "header_scope": "col" if is_header == "yes" else "",
                    "header_path_json": json.dumps(header_path),
                    "row_label_text": "",
                    "column_label_text": header_path[-1],
                    "source_block_id": table_id,
                    "source_hash": "sha1:synthetic",
                    "warnings_json": "[]",
                }
                handle.write(json.dumps(rec) + "\n")

    def test_stage1_sidecar_matrix_preserves_coordinates_without_semantic_candidate_creation(self):
        with tempfile.TemporaryDirectory(dir=extractor.PROJECT_ROOT) as td:
            root = Path(td) / "tables_cell_sidecar"
            self._write_sidecar(root, "PAPER1")

            rows = extractor.load_stage1_table_cell_sidecar_rows(root, "PAPER1")
            groups = extractor.group_stage1_sidecar_cells_by_table(rows)
            matrix, metadata = extractor.stage1_sidecar_cells_to_matrix(groups["t001"])

            self.assertEqual(matrix, [["Formulation", "Size (nm)"], ["F1", "123"], ["F2", "456"]])
            self.assertEqual(metadata["stage1_cell_sidecar_table_id"], "t001")
            self.assertEqual(metadata["stage1_cell_sidecar_cell_count"], 6)
            self.assertNotIn("MUST_NOT_DEFINE_CANDIDATE", " | ".join(" | ".join(row) for row in matrix))

    def test_stage1_sidecar_can_be_loaded_from_manifest_file_path_without_parser_specific_interface(self):
        with tempfile.TemporaryDirectory(dir=extractor.PROJECT_ROOT) as td:
            root = Path(td) / "tables_cell_sidecar"
            self._write_sidecar(root, "PAPER1")
            sidecar_file = root / "PAPER1" / "stage1_table_cells_v1.jsonl"

            effective_root = extractor.resolve_stage1_table_cell_sidecar_root_for_record(
                {"key": "PAPER1", "stage1_table_cell_sidecar_path": str(sidecar_file)},
                configured_root=None,
            )
            rows = extractor.load_stage1_table_cell_sidecar_rows(effective_root, "PAPER1")
            groups = extractor.group_stage1_sidecar_cells_by_table(rows)
            matrix, metadata = extractor.stage1_sidecar_cells_to_matrix(groups["t001"])

            self.assertEqual(matrix[1], ["F1", "123"])
            self.assertEqual(metadata["stage1_cell_sidecar_status"], "consumed")
            self.assertEqual(metadata["stage1_cell_sidecar_caption_binding_rule"], "synthetic_test_caption")

    def test_stage1_structure_sidecar_metadata_attaches_without_defining_candidate_universe(self):
        with tempfile.TemporaryDirectory(dir=extractor.PROJECT_ROOT) as td:
            tmp = Path(td)
            text = "PLGA nanoparticles were prepared by solvent evaporation with PVA stabilizer.\n\nResults are summarized elsewhere."
            text_path = tmp / "PAPER1.txt"
            text_path.write_text(text, encoding="utf-8")
            sidecar_path = tmp / "PAPER1.structure.json"
            sidecar_path.write_text(
                json.dumps(
                    {
                        "key": "PAPER1",
                        "blocks": [
                            {
                                "block_id": "b1",
                                "type": "paragraph",
                                "block_text": "PLGA nanoparticles were prepared by solvent evaporation with PVA stabilizer.",
                                "section_id": "sec_methods_1",
                                "section_label": "Methods",
                                "section_kind": "methods",
                                "noise_class": "keep",
                                "noise_reason": "",
                            },
                            {
                                "block_id": "b2",
                                "type": "paragraph",
                                "block_text": "MUST_NOT_DEFINE_CANDIDATE",
                                "section_id": "sec_refs_1",
                                "section_label": "References",
                                "section_kind": "references",
                                "noise_class": "terminal_noise",
                                "noise_reason": "references_section",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            artifact, bundle = extractor.build_candidate_segmentation_artifact(
                record={"key": "PAPER1"},
                manifest_path=tmp / "manifest.tsv",
                text_path=text_path,
                table_dir=None,
                producer_script="test",
                stage1_structure_sidecar_path=sidecar_path,
            )

            candidate_texts = [candidate["text_content"] for candidate in artifact["candidate_blocks"]]
            self.assertFalse(any("MUST_NOT_DEFINE_CANDIDATE" in text for text in candidate_texts))
            matched = next(candidate for candidate in artifact["candidate_blocks"] if "solvent evaporation" in candidate["text_content"])
            self.assertEqual(matched["stage1_section_id"], "sec_methods_1")
            self.assertEqual(matched["stage1_section_label"], "Methods")
            self.assertEqual(matched["stage1_section_kind"], "methods")
            self.assertEqual(matched["stage1_noise_class"], "keep")
            self.assertEqual(matched["stage1_structure_match_rule"], "exact_normalized_block_text")
            selector_match = next(candidate for candidate in bundle["selector_candidates"] if candidate["candidate_id"] == matched["candidate_id"])
            self.assertEqual(selector_match["stage1_section_kind"], "methods")
            self.assertTrue(artifact["feature_activation_snapshot"]["stage1_structure_sidecar_metadata"])
            self.assertEqual(artifact["stage1_structure_sidecar_status"], "loaded")

    def test_stage1_structure_sidecar_noise_metadata_does_not_hard_drop_candidate_or_overwrite_stage2_section(self):
        with tempfile.TemporaryDirectory(dir=extractor.PROJECT_ROOT) as td:
            tmp = Path(td)
            text_path = tmp / "PAPER1.txt"
            text_path.write_text("Preparation of PLGA nanoparticles used acetone and aqueous PVA under stirring.", encoding="utf-8")
            sidecar_path = tmp / "PAPER1.structure.json"
            sidecar_path.write_text(
                json.dumps(
                    {
                        "key": "PAPER1",
                        "blocks": [
                            {
                                "block_id": "b1",
                                "type": "paragraph",
                                "block_text": "Preparation of PLGA nanoparticles used acetone and aqueous PVA under stirring.",
                                "section_id": "sec_intro_1",
                                "section_label": "Introduction",
                                "section_kind": "introduction",
                                "noise_class": "soft_noise",
                                "noise_reason": "introductory_context",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            artifact, _bundle = extractor.build_candidate_segmentation_artifact(
                record={"key": "PAPER1"},
                manifest_path=tmp / "manifest.tsv",
                text_path=text_path,
                table_dir=None,
                producer_script="test",
                stage1_structure_sidecar_path=sidecar_path,
            )

            matched = next(candidate for candidate in artifact["candidate_blocks"] if "PLGA nanoparticles" in candidate["text_content"])
            self.assertEqual(matched["stage1_section_kind"], "introduction")
            self.assertEqual(matched["stage1_noise_class"], "soft_noise")
            self.assertNotEqual(matched["section_kind"], matched["stage1_section_kind"])
            self.assertIn("stage1_noise:soft_noise", matched["quality_flags"])
            self.assertIn(matched, artifact["candidate_blocks"])

    def test_missing_stage1_structure_sidecar_preserves_legacy_candidate_shape(self):
        with tempfile.TemporaryDirectory(dir=extractor.PROJECT_ROOT) as td:
            tmp = Path(td)
            text_path = tmp / "PAPER1.txt"
            text_path.write_text("PLGA nanoparticles were prepared by solvent evaporation with PVA stabilizer.", encoding="utf-8")

            artifact, _bundle = extractor.build_candidate_segmentation_artifact(
                record={"key": "PAPER1"},
                manifest_path=tmp / "manifest.tsv",
                text_path=text_path,
                table_dir=None,
                producer_script="test",
            )

            matched = next(candidate for candidate in artifact["candidate_blocks"] if "solvent evaporation" in candidate["text_content"])
            self.assertNotIn("stage1_section_kind", matched)
            self.assertFalse(artifact["feature_activation_snapshot"]["stage1_structure_sidecar_metadata"])
            self.assertEqual(artifact["stage1_structure_sidecar_status"], "missing")

    def test_s2_1b_denoised_projection_is_consumed_while_raw_text_remains_audit_authority(self):
        with tempfile.TemporaryDirectory(dir=extractor.PROJECT_ROOT) as td:
            tmp = Path(td)
            raw_text_path = tmp / "PAPER1_raw.txt"
            denoised_text_path = tmp / "PAPER1_s2_1b.txt"
            raw_text_path.write_text(
                "Journals & Books\nPLGA nanoparticles were prepared by solvent evaporation with PVA stabilizer.",
                encoding="utf-8",
            )
            denoised_text_path.write_text(
                "PLGA nanoparticles were prepared by solvent evaporation with PVA stabilizer.",
                encoding="utf-8",
            )
            record = {
                "key": "PAPER1",
                "source_s2_1b_denoised_text_path": str(denoised_text_path),
                "s2_1b_denoise_audit_path": str(tmp / "PAPER1_s2_1b_audit.json"),
                "s2_1b_denoise_summary_path": str(tmp / "s2_1b_summary.tsv"),
            }

            candidate_artifact, bundle = extractor.build_candidate_segmentation_artifact(
                record=record,
                manifest_path=tmp / "manifest.tsv",
                text_path=raw_text_path,
                table_dir=None,
                producer_script="test",
            )
            evidence_artifact, _debug = extractor.build_evidence_blocks_artifact(
                record={**record, "doi": "", "title": "Synthetic"},
                manifest_path=tmp / "manifest.tsv",
                text_path=raw_text_path,
                table_dir=None,
                max_chars=2000,
                producer_script="test",
                candidate_artifact_path=tmp / "candidate_blocks_v1.json",
                segmentation_bundle=bundle,
            )

            self.assertEqual(candidate_artifact["source_clean_text_path"], extractor.to_repo_rel(raw_text_path))
            self.assertEqual(candidate_artifact["source_text_projection"], "s2_1b_denoised")
            self.assertEqual(candidate_artifact["source_s2_1b_denoised_text_path"], extractor.to_repo_rel(denoised_text_path))
            self.assertEqual(candidate_artifact["s2_1b_denoise_summary_path"], extractor.to_repo_rel(tmp / "s2_1b_summary.tsv"))
            self.assertTrue(candidate_artifact["feature_activation_snapshot"]["s2_1b_source_denoise_projection"])
            candidate_text = "\n".join(candidate["text_content"] for candidate in candidate_artifact["candidate_blocks"])
            self.assertNotIn("Journals & Books", candidate_text)
            self.assertIn("solvent evaporation", candidate_text)
            self.assertEqual(evidence_artifact["source_clean_text_path"], extractor.to_repo_rel(raw_text_path))
            self.assertEqual(evidence_artifact["source_text_projection"], "s2_1b_denoised")
            self.assertEqual(evidence_artifact["s2_1b_denoise_summary_path"], extractor.to_repo_rel(tmp / "s2_1b_summary.tsv"))
            self.assertEqual(evidence_artifact["input_contract"]["source_text_projection"], "s2_1b_denoised")
            self.assertTrue(evidence_artifact["feature_activation_snapshot"]["s2_1b_source_denoise_projection"])

    def test_build_normalized_payload_uses_sidecar_only_for_already_selected_candidate(self):
        with tempfile.TemporaryDirectory(dir=extractor.PROJECT_ROOT) as td:
            tmp = Path(td)
            sidecar_root = tmp / "tables_cell_sidecar"
            self._write_sidecar(sidecar_root, "PAPER1")
            out_dir = tmp / "semantic_stage2_objects"
            evidence_path = tmp / "evidence_blocks" / "PAPER1" / "evidence_blocks_v1.json"
            evidence_artifact = {
                "evidence_blocks": [
                    {
                        "candidate_id": "PAPER1__candidate_table__01",
                        "is_table_derived": True,
                    }
                ]
            }
            segmentation_bundle = {
                "selector_candidates": [
                    {
                        "candidate_id": "PAPER1__candidate_table__01",
                        "candidate_kind": "table",
                        "table_role_hint": "formulation",
                        "origin_locator": "synthetic_table_01.csv",
                        "item": {
                            "rows": [["FallbackHeader"], ["fallback_value"]],
                            "meta": {"caption_or_title": "Table 1"},
                            "representation_status": "raw_summary",
                            "selector_readiness_label": "ready",
                        },
                    },
                    {
                        "candidate_id": "PAPER1__candidate_table__02",
                        "candidate_kind": "table",
                        "table_role_hint": "formulation",
                        "origin_locator": "synthetic_table_02.csv",
                        "item": {
                            "rows": [["UnselectedFallback"], ["unselected"]],
                            "meta": {"caption_or_title": "Table 2"},
                            "representation_status": "raw_summary",
                            "selector_readiness_label": "ready",
                        },
                    },
                ]
            }

            artifact, _validation_rows = extractor.build_normalized_table_payload_artifact(
                record={"key": "PAPER1"},
                out_dir=out_dir,
                producer_script="test",
                evidence_artifact_path=evidence_path,
                evidence_artifact=evidence_artifact,
                segmentation_bundle=segmentation_bundle,
                stage1_table_cell_sidecar_root=sidecar_root,
            )

            payloads = artifact["normalized_table_payloads"]
            self.assertEqual(len(payloads), 1)
            payload = payloads[0]
            flattened_raw_cells = " | ".join(" | ".join(row) for row in payload["raw_cells"])
            self.assertIn("Formulation", flattened_raw_cells)
            self.assertIn("456", flattened_raw_cells)
            self.assertNotIn("FallbackHeader", flattened_raw_cells)
            self.assertNotIn("MUST_NOT_DEFINE_CANDIDATE", flattened_raw_cells)
            self.assertEqual(payload["stage1_cell_sidecar_status"], "consumed")
            self.assertEqual(payload["stage1_cell_sidecar_match_rule"], "selected_candidate_ordinal_to_sidecar_table_ordinal")
            self.assertEqual(payload["stage1_cell_sidecar_table_id"], "t001")
            self.assertEqual(payload["stage1_cell_sidecar_noise_class"], "keep")
            self.assertEqual(payload["stage1_cell_sidecar_caption_binding_rule"], "synthetic_test_caption")
            self.assertEqual(payload["source_caption_binding_status"], "unbound")
            self.assertEqual(payload["source_caption_or_title"], "")

    def test_normalized_payload_uses_exact_sidecar_caption_over_polluted_candidate_caption(self):
        with tempfile.TemporaryDirectory(dir=extractor.PROJECT_ROOT) as td:
            tmp = Path(td)
            sidecar_root = tmp / "tables_cell_sidecar"
            self._write_sidecar(sidecar_root, "PAPER1")
            out_dir = tmp / "semantic_stage2_objects"
            evidence_path = tmp / "evidence_blocks" / "PAPER1" / "evidence_blocks_v1.json"
            evidence_artifact = {
                "evidence_blocks": [
                    {"candidate_id": "PAPER1__candidate_table__01", "is_table_derived": True}
                ]
            }
            segmentation_bundle = {
                "selector_candidates": [
                    {
                        "candidate_id": "PAPER1__candidate_table__01",
                        "candidate_kind": "table",
                        "table_role_hint": "formulation",
                        "origin_locator": "synthetic_table_01.csv",
                        "item": {
                            "rows": [["FallbackHeader"], ["fallback_value"]],
                            "meta": {
                                "table_id": "t001",
                                "caption_or_title": "Table 2 Wrong adjacent caption with publisher prose",
                                "caption_binding_source": "csv_body_untrusted",
                                "caption_binding_status": "untrusted",
                            },
                            "representation_status": "raw_summary",
                            "selector_readiness_label": "ready",
                        },
                    }
                ]
            }

            artifact, _validation_rows = extractor.build_normalized_table_payload_artifact(
                record={"key": "PAPER1"},
                out_dir=out_dir,
                producer_script="test",
                evidence_artifact_path=evidence_path,
                evidence_artifact=evidence_artifact,
                segmentation_bundle=segmentation_bundle,
                stage1_table_cell_sidecar_root=sidecar_root,
            )

            payload = artifact["normalized_table_payloads"][0]
            self.assertEqual(payload["stage1_cell_sidecar_match_rule"], "stable_table_id_to_sidecar_table_id")
            self.assertEqual(payload["source_caption_or_title"], "Table 1")
            self.assertEqual(payload["source_caption_binding_source"], "stage1_cell_sidecar")
            self.assertEqual(payload["source_caption_binding_status"], "trusted_exact_table_id")
            self.assertNotIn("Wrong adjacent", payload["source_caption_or_title"])

    def test_confirmed_noise_stage1_sidecar_table_is_not_promoted_to_selected_payload(self):
        with tempfile.TemporaryDirectory(dir=extractor.PROJECT_ROOT) as td:
            tmp = Path(td)
            sidecar_root = tmp / "tables_cell_sidecar"
            self._write_sidecar(sidecar_root, "PAPER1")
            out_dir = tmp / "semantic_stage2_objects"
            evidence_path = tmp / "evidence_blocks" / "PAPER1" / "evidence_blocks_v1.json"
            evidence_artifact = {
                "evidence_blocks": [
                    {
                        "candidate_id": "PAPER1__candidate_table__02",
                        "is_table_derived": True,
                    }
                ]
            }
            segmentation_bundle = {
                "selector_candidates": [
                    {
                        "candidate_id": "PAPER1__candidate_table__02",
                        "candidate_kind": "table",
                        "table_role_hint": "formulation",
                        "origin_locator": "synthetic_table_02.csv",
                        "item": {
                            "rows": [["FallbackHeader"], ["fallback_value"]],
                            "meta": {"caption_or_title": "Table 2"},
                            "representation_status": "raw_summary",
                            "selector_readiness_label": "ready",
                        },
                    },
                ]
            }

            artifact, _validation_rows = extractor.build_normalized_table_payload_artifact(
                record={"key": "PAPER1"},
                out_dir=out_dir,
                producer_script="test",
                evidence_artifact_path=evidence_path,
                evidence_artifact=evidence_artifact,
                segmentation_bundle=segmentation_bundle,
                stage1_table_cell_sidecar_root=sidecar_root,
            )

            payloads = artifact["normalized_table_payloads"]
            self.assertEqual(len(payloads), 1)
            payload = payloads[0]
            flattened_raw_cells = " | ".join(" | ".join(row) for row in payload["raw_cells"])
            self.assertIn("FallbackHeader", flattened_raw_cells)
            self.assertNotIn("MUST_NOT_DEFINE_CANDIDATE", flattened_raw_cells)
            self.assertEqual(payload["stage1_cell_sidecar_status"], "not_available")
    def test_corrupted_csv_does_not_downgrade_clean_text_compact_table_payload(self):
        with tempfile.TemporaryDirectory(dir=extractor.PROJECT_ROOT) as td:
            tmp = Path(td)
            out_dir = tmp / "semantic_stage2_objects"
            evidence_path = tmp / "evidence_blocks" / "PAPER1" / "evidence_blocks_v1.json"
            text_path = tmp / "PAPER1.pdf.txt"
            text_path.write_text(
                "Methods before table.\n"
                "Table 1. Composition of nanoparticle formulations. "
                "Formulation PLGA_mg Drug_mg Size_nm "
                "F1 50 5 123 F2 75 7.5 145 F3 100 10 166.\n"
                "Following text after the compact table.",
                encoding="utf-8",
            )
            evidence_artifact = {
                "evidence_blocks": [
                    {"candidate_id": "PAPER1__candidate_table__01", "is_table_derived": True}
                ]
            }
            segmentation_bundle = {
                "selector_candidates": [
                    {
                        "candidate_id": "PAPER1__candidate_table__01",
                        "candidate_kind": "table",
                        "table_role_hint": "formulation",
                        "origin_locator": "bad_blank_table_1.csv",
                        "item": {
                            "rows": [[""]],
                            "meta": {"table_id": "Table 1", "caption_or_title": "Table 1"},
                            "representation_status": "unrepaired_corrupted",
                            "selector_readiness_label": "weak",
                        },
                    }
                ]
            }

            artifact, _validation_rows = extractor.build_normalized_table_payload_artifact(
                record={"key": "PAPER1", "text_path": str(text_path)},
                out_dir=out_dir,
                producer_script="test",
                evidence_artifact_path=evidence_path,
                evidence_artifact=evidence_artifact,
                segmentation_bundle=segmentation_bundle,
            )

            payload = artifact["normalized_table_payloads"][0]
            self.assertIn("recover_clean_text_compact_inline_table", payload["normalization_actions"])
            self.assertEqual(payload["source_table_reference"], f"{extractor.to_repo_rel(text_path)}#Table 1")
            self.assertEqual(payload["normalized_matrix"][0], ["Formulation", "PLGA_mg", "Drug_mg", "Size_nm"])
            self.assertEqual(payload["normalized_matrix"][1], ["F1", "50", "5", "123"])
            self.assertEqual(payload["normalized_matrix"][3], ["F3", "100", "10", "166"])
            self.assertEqual(payload["data_row_count"], 3)
            self.assertEqual(payload["representation_status"], "clean_text_compact_table_recovered")

    def test_normalized_payload_preserves_full_size_table_asset_reference(self):
        with tempfile.TemporaryDirectory(dir=extractor.PROJECT_ROOT) as td:
            tmp = Path(td)
            out_dir = tmp / "semantic_stage2_objects"
            evidence_path = tmp / "evidence_blocks" / "PAPER1" / "evidence_blocks_v1.json"
            source_csv = tmp / "publisher_assets" / "full_size_table_2.csv"
            source_csv.parent.mkdir(parents=True)
            source_csv.write_text(
                "Formulation,Drug loading (mg KGN/mg nanoparticles),Particle diameter (nm)\n"
                "NP-HA,0.467 ± 0.192,166.6 ± 34.48\n",
                encoding="utf-8",
            )
            evidence_artifact = {
                "evidence_blocks": [
                    {"candidate_id": "PAPER1__candidate_table__01", "is_table_derived": True}
                ]
            }
            segmentation_bundle = {
                "selector_candidates": [
                    {
                        "candidate_id": "PAPER1__candidate_table__01",
                        "candidate_kind": "table",
                        "table_role_hint": "formulation",
                        "origin_locator": str(source_csv),
                        "item": {
                            "rows": [["Formulation", "Drug loading"], ["NP-HA", "0.467 ± 0.192"]],
                            "meta": {
                                "caption_or_title": "KGN-loaded nanoparticle properties",
                                "source_table_reference": "https://example.org/article/tables/2",
                                "source_table_asset_local_path": str(source_csv),
                                "table_source_kind": "html_full_size_table_asset",
                            },
                            "representation_status": "raw_summary",
                            "selector_readiness_label": "ready",
                        },
                    }
                ]
            }

            artifact, _validation_rows = extractor.build_normalized_table_payload_artifact(
                record={"key": "PAPER1"},
                out_dir=out_dir,
                producer_script="test",
                evidence_artifact_path=evidence_path,
                evidence_artifact=evidence_artifact,
                segmentation_bundle=segmentation_bundle,
            )

            payload = artifact["normalized_table_payloads"][0]
            self.assertEqual(payload["source_table_reference"], "https://example.org/article/tables/2")
            self.assertEqual(payload["source_table_asset_local_path"], str(source_csv))
            self.assertEqual(payload["source_table_asset_origin"], "html_full_size_table_asset")


if __name__ == "__main__":
    unittest.main()
