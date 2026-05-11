import json
import tempfile
import unittest
from pathlib import Path

from src.stage2_sampling_labels.extract_semantic_stage2_objects_v2 import (
    build_raw_metric_value_evidence_lines,
    build_table_summary_lines,
    classify_execution_table_type,
    classify_table_source_role,
    collect_summary_table_candidates,
    compact_value_evidence_lines_from_payload,
    extract_informative_header_parts,
    payload_inclusion_class,
    payload_usage_role,
    prompt_semantic_summary_line,
    render_prompt_block,
    render_selected_table_candidate,
    repair_table_representation,
    render_value_evidence_payload_blocks,
    summary_header_context,
    summary_header_row_count,
)


FORBIDDEN_PRE_LLM_SEMANTIC_LABELS = [
    "table_role_hint",
    "semantic_summary: formulation",
    "semantic_summary: optimization",
    "semantic_summary: row-local measurement",
]


class Stage2TableSummaryNumericVisibilityTests(unittest.TestCase):
    def test_table_summary_is_structural_and_has_no_pre_llm_semantic_labels(self):
        rows = [
            ["Formulation", "Polymer", "Surfactant", "Organic solvent"],
            ["F1", "PLGA", "PVA", "acetone"],
            ["F2", "PLGA", "PVA", "ethyl acetate"],
        ]
        item = {
            "path": Path("data/cleaned/goren_2025/tables/TEST/TEST__table_00__pdf_table.csv"),
            "rows": rows,
            "meta": {"caption_or_title": "Source Table 1", "n_rows": len(rows), "n_cols": len(rows[0])},
        }
        rendered = "\n".join(build_table_summary_lines(item, enhancement_enabled=False))
        self.assertIn("summary_contract: structural_prompt_view_only", rendered)
        self.assertIn("column_schema", rendered)
        self.assertIn("sample_row_1", rendered)
        self.assertIn("sample_row_2", rendered)
        for forbidden in FORBIDDEN_PRE_LLM_SEMANTIC_LABELS:
            self.assertNotIn(forbidden, rendered)
        self.assertNotIn("full_numeric_rows", rendered)

    def test_prompt_semantic_summary_line_is_neutral_for_all_pre_llm_tables(self):
        for role in ["design matrix", "characterization", "formulation", "optimization", "results", ""]:
            line = prompt_semantic_summary_line(role)
            self.assertIn("structural_prompt_view_only", line)
            self.assertIn("LLM must decide", line)
            self.assertNotIn(role if role else "__never__", line.replace("LLM", ""))

    def test_characterization_table_summary_stays_semantic_only_without_row_local_metric_values(self):
        rows = [
            ["Formulation", "Size (nm)", "PDI", "Zeta potential (mV)", "EE (%)", "DL (%)", "LC (%)", "Yield (%)", "Release (%)"],
            ["NP1", "101.1", "0.11", "-12.1", "45.1", "5.1", "6.1", "80.1", "10.1"],
            ["NP2", "102.2", "0.12", "-12.2", "45.2", "5.2", "6.2", "80.2", "10.2"],
            ["NP3", "103.3", "0.13", "-12.3", "45.3", "5.3", "6.3", "80.3", "10.3"],
            ["NP4", "104.4", "0.14", "-12.4", "45.4", "5.4", "6.4", "80.4", "10.4"],
            ["NP5", "105.5", "0.15", "-12.5", "45.5", "5.5", "6.5", "80.5", "10.5"],
        ]
        item = {
            "path": Path("data/cleaned/goren_2025/tables/TEST/TEST__table_01__pdf_table.csv"),
            "rows": rows,
            "meta": {"caption_or_title": "Characterization-only table.", "n_rows": len(rows), "n_cols": len(rows[0])},
            "representation_status": "raw_summary",
            "repair_primary_source": "stage1_selected_table_asset",
            "selector_readiness_label": "ready",
        }
        rendered = "\n".join(build_table_summary_lines(item, enhancement_enabled=False))
        self.assertIn("summary_contract: structural_prompt_view_only", rendered)
        self.assertIn("sample_row_1", rendered)
        self.assertIn("sample_row_2", rendered)
        self.assertNotIn("sample_row_3", rendered)
        self.assertNotIn("metric_value_rows", rendered)
        self.assertNotIn("physical_row_", rendered)
        self.assertNotIn("value_evidence_only", rendered)
        self.assertNotIn("NP4", rendered)
        for forbidden in FORBIDDEN_PRE_LLM_SEMANTIC_LABELS:
            self.assertNotIn(forbidden, rendered)

    def test_non_metric_table_summary_does_not_emit_metric_value_rows(self):
        rows = [
            ["Reference", "Title"],
            ["1", "Prior work"],
            ["2", "Another prior work"],
        ]
        item = {
            "path": Path("data/cleaned/goren_2025/tables/TEST/TEST__table_02__pdf_table.csv"),
            "rows": rows,
            "meta": {"caption_or_title": "References", "n_rows": len(rows), "n_cols": len(rows[0])},
        }
        rendered = "\n".join(build_table_summary_lines(item, enhancement_enabled=False))
        self.assertNotIn("metric_value_rows", rendered)
        self.assertNotIn("value_evidence_only", rendered)

    def test_table_summary_sample_rows_skip_publisher_chrome_without_losing_source_rows(self):
        rows = [
            ["0", "1"],
            ["Dovepress", "Gatifloxacin NPs for CNS tuberculosis"],
            [
                "for formulations NPG1, NPG2, and NPG3, respectively",
                "2. Abbott NJ, Patabendige AA, Dolman DE, Yusof SR, Begley DJ.",
            ],
            ["International Journal of Nanomedicine 2017:12", "submit your manuscript | www.dovepress.com"],
            ["PLGA NPs prepared with two surfactants", "showed improved passage across the BBB"],
        ]
        item = {
            "path": Path("data/cleaned/goren_2025/tables/5ZXYABSU/5ZXYABSU__table_16__pdf_table.csv"),
            "rows": rows,
            "meta": {"caption_or_title": "", "n_rows": len(rows), "n_cols": len(rows[0])},
        }
        rendered = "\n".join(build_table_summary_lines(item, enhancement_enabled=False))
        self.assertIn("sample_row_1: for formulations NPG1, NPG2, and NPG3, respectively", rendered)
        self.assertIn("sample_row_2: PLGA NPs prepared with two surfactants", rendered)
        self.assertNotIn("sample_row_1: Dovepress", rendered)
        self.assertNotIn("submit your manuscript", rendered)
        self.assertNotIn("www.dovepress.com", rendered)

    def test_multirow_summary_preserves_full_header_schema_and_two_complete_rows(self):
        rows = [
            ["", "Composition", "Composition", "Process", "Process", "Results"],
            ["Formulation", "PLGA (mg)", "Drug (mg)", "PVA (%)", "Sonication (s)", "Size (nm)"],
            ["F1", "10", "1", "1.0", "30", "101"],
            ["F2", "20", "2", "1.5", "45", "112"],
            ["F3", "30", "3", "2.0", "60", "123"],
        ]
        item = {
            "path": Path("data/cleaned/goren_2025/tables/TEST/TEST__table_03__pdf_table.csv"),
            "rows": rows,
            "meta": {"caption_or_title": "Source Table 3", "n_rows": len(rows), "n_cols": len(rows[0])},
        }
        rendered = "\n".join(build_table_summary_lines(item, enhancement_enabled=False))
        self.assertIn("column_schema", rendered)
        self.assertIn("Composition PLGA (mg)", rendered)
        self.assertIn("Process Sonication (s)", rendered)
        self.assertIn("Results Size (nm)", rendered)
        self.assertIn("sample_row_1: F1 | 10 | 1 | 1.0 | 30 | 101", rendered)
        self.assertIn("sample_row_2: F2 | 20 | 2 | 1.5 | 45 | 112", rendered)
        self.assertNotIn("sample_row_3", rendered)
        for forbidden in FORBIDDEN_PRE_LLM_SEMANTIC_LABELS:
            self.assertNotIn(forbidden, rendered)

    def test_source_native_table_caption_binding_uses_marker_content_signature(self):
        table_dir = Path("data/cleaned/goren_2025/tables/5GIF3D8W")
        path = table_dir / "5GIF3D8W__table_04__pdf_table.csv"
        rows = [
            ["0", "1", "2", "3", "4"],
            ["", "Formulation characters for the optimized nanoparticle formulations", "", "", ""],
            ["", "PLGA 50/50 (Mean ± SD)", "", "PLGA 75/25 (Mean ± SD)", ""],
            ["", "Empty", "Drug loaded", "Empty", "Drug loaded"],
            ["0", "Diameter (nm)", "87.2 ± 0.25", "91.8 ± 2.74", "96.9 ± 1.06"],
            ["1", "PIa", "0.14 ± 0.01", "0.13 ± 0.01", "0.12 ± 0.01"],
        ]
        repaired = repair_table_representation(
            path=path,
            rows=rows,
            meta={"n_rows": len(rows), "n_cols": len(rows[0])},
            quality_flags=[],
            filtered_noise_rows=0,
            table_dir=table_dir,
        )
        self.assertEqual(repaired["meta"].get("source_native_table_id"), "Table 1")
        self.assertIn(
            "Formulation characters for the optimized nanoparticle formulations",
            repaired["meta"].get("caption_or_title", ""),
        )
        self.assertEqual(repaired["meta"].get("caption_binding_status"), "trusted_content_match")
        rendered = "\n".join(build_table_summary_lines(repaired, enhancement_enabled=False))
        self.assertIn("- table_id: Table 1", rendered)
        self.assertIn("- title_or_caption: TABLE 1 Formulation characters", rendered)
        self.assertIn("sample_row_1", rendered)
        self.assertNotIn("sample_row_3", rendered)
        for forbidden in FORBIDDEN_PRE_LLM_SEMANTIC_LABELS:
            self.assertNotIn(forbidden, rendered)

    def test_numeric_index_prelude_multilevel_header_expands_column_schema(self):
        rows = [
            ["0", "1", "2", "3", "4"],
            ["", "Formulation characters for the optimized nanoparticle formulations", "", "", ""],
            ["", "PLGA 50/50 (Mean ± SD)", "", "PLGA 75/25 (Mean ± SD)", ""],
            ["", "Empty", "Drug loaded", "Empty", "Drug loaded"],
            ["0", "PIa", "PIb", "PIIa", "PIIb"],
            ["1", "88.1 ± 2.1", "91.2 ± 1.9", "102.4 ± 3.1", "107.5 ± 2.8"],
        ]
        item = {
            "path": Path("data/cleaned/goren_2025/tables/TEST/TEST__table_04__pdf_table.csv"),
            "rows": rows,
            "meta": {
                "caption_or_title": "Formulation table with drug/polymer/loading variables.",
                "n_rows": len(rows),
                "n_cols": len(rows[0]),
            },
        }
        rendered = "\n".join(build_table_summary_lines(item, enhancement_enabled=False))
        self.assertIn(
            "Formulation characters for the optimized nanoparticle formulations PLGA 50/50 (Mean ± SD) Empty",
            rendered,
        )
        self.assertIn(
            "Formulation characters for the optimized nanoparticle formulations PLGA 75/25 (Mean ± SD) Drug loaded",
            rendered,
        )
        self.assertIn("sample_row_1: 0 | PIa | PIb | PIIa | PIIb", rendered)
        self.assertIn("sample_row_2: 1 | 88.1 ± 2.1 | 91.2 ± 1.9 | 102.4 ± 3.1 | 107.5 ± 2.8", rendered)
        self.assertIn("title_or_caption: (not available)", rendered)
        self.assertNotIn("key_columns", rendered)
        for forbidden in FORBIDDEN_PRE_LLM_SEMANTIC_LABELS:
            self.assertNotIn(forbidden, rendered)

    def test_render_prompt_block_uses_neutral_text_block_label_not_pre_llm_paragraph_roles(self):
        rendered = render_prompt_block(
            {
                "block_id": "PAPER__method__001",
                "block_type": "text",
                "evidence_kind": "method",
                "text_content": "The particles were prepared by nanoprecipitation.",
            },
            summary_enhanced=False,
        )["rendered_text"]
        self.assertTrue(rendered.startswith("[TEXT_BLOCK]\n"))
        self.assertNotIn("[METHOD]", rendered)
        self.assertNotIn("[MATERIALS]", rendered)
        self.assertNotIn("[SUPPORTING]", rendered)

    def test_selected_table_candidate_does_not_rebind_polluted_csv_caption_as_title(self):
        rendered = render_selected_table_candidate(
            {
                "source_type": "table_summary",
                "origin_locator": "data/cleaned/goren_2025/tables/QLYKLPKT/QLYKLPKT__table_09__pdf_table.csv",
                "text_content": "- title_or_caption: Table 2 Physicochemical properties of Plga-ITZ-Ns with The morphology of lyophilized PLGA-ITZ-NS was - key_columns: stale",
            }
        )
        title_line = next(line for line in rendered.splitlines() if line.startswith("- title_or_caption:"))
        self.assertEqual("- title_or_caption: (not available)", title_line)
        self.assertNotIn("The morphology of lyophilized PLGA-ITZ-NS was", title_line)
        self.assertNotIn("key_columns", rendered)

    def test_selected_table_candidate_uses_trusted_cleantext_caption_when_unambiguous(self):
        rendered = render_selected_table_candidate(
            {
                "source_type": "table_summary",
                "origin_locator": "data/cleaned/goren_2025/tables/UFXX9WXE/UFXX9WXE__table_10__pdf_table.csv",
                "text_content": "- title_or_caption: stale",
            }
        )
        title_line = next(line for line in rendered.splitlines() if line.startswith("- title_or_caption:"))
        self.assertEqual(
            "- title_or_caption: Table 1: Independent and dependent variables levels in Box-Behnken design.",
            title_line,
        )
        self.assertNotIn("drug entrapment and drug loading", title_line)
        self.assertNotIn("key_columns", rendered)

    def test_selected_table_candidate_rejects_prose_mentions_and_body_spillover_as_titles(self):
        cases = [
            "data/cleaned/goren_2025/tables/5ZXYABSU/5ZXYABSU__table_08__pdf_table.csv",
            "data/cleaned/goren_2025/tables/5GIF3D8W/5GIF3D8W__table_03__pdf_table.csv",
            "data/cleaned/goren_2025/tables/PA3SPZ28/PA3SPZ28__table_07__pdf_table.csv",
            "data/cleaned/goren_2025/tables/RHMJWZX8/RHMJWZX8__table_10__pdf_table.csv",
        ]
        for origin_locator in cases:
            with self.subTest(origin_locator=origin_locator):
                rendered = render_selected_table_candidate(
                    {
                        "source_type": "table_summary",
                        "origin_locator": origin_locator,
                        "text_content": "- title_or_caption: stale - key_columns: stale",
                    }
                )
                title_line = next(line for line in rendered.splitlines() if line.startswith("- title_or_caption:"))
                self.assertEqual("- title_or_caption: (not available)", title_line)
                self.assertNotIn("key_columns", rendered)

    def test_collect_summary_table_candidates_keeps_csv_body_caption_untrusted_when_ambiguous(self):
        candidates = collect_summary_table_candidates(
            Path("data/cleaned/goren_2025/tables/QLYKLPKT")
        )
        candidate = next(item for item in candidates if item["path"].name == "QLYKLPKT__table_09__pdf_table.csv")

        caption = candidate["meta"].get("caption_or_title", "")
        self.assertEqual(caption, "")
        self.assertIn("csv_body_caption_ambiguous", candidate["repair_warnings"])
        self.assertIn("csv_body_caption_untrusted", candidate["repair_actions"])
        self.assertIn("The morphology of lyophilized PLGA-ITZ-NS was", candidate["meta"].get("untrusted_recovered_caption_or_title", ""))
        self.assertNotIn("Formulation table with drug/polymer/loading variables", json.dumps(candidate["meta"]))

    def test_collect_summary_table_candidates_binds_unambiguous_cleantext_caption(self):
        candidates = collect_summary_table_candidates(
            Path("data/cleaned/goren_2025/tables/UFXX9WXE")
        )
        candidate = next(item for item in candidates if item["path"].name == "UFXX9WXE__table_10__pdf_table.csv")

        self.assertEqual(
            candidate["meta"].get("caption_or_title"),
            "Table 1: Independent and dependent variables levels in Box-Behnken design.",
        )
        self.assertEqual(candidate["meta"].get("caption_binding_source"), "source_clean_text_table_caption")
        self.assertEqual(candidate["meta"].get("caption_binding_status"), "trusted_unambiguous")
        self.assertNotIn("drug entrapment and drug loading", candidate["meta"].get("caption_or_title", ""))

    def test_value_evidence_payload_renders_compact_metric_matrix(self):
        payload = {
            "table_id": "Table 3",
            "source_table_reference": "data/cleaned/tables/TEST__table_03.csv",
            "source_caption_or_title": "Characterization of nanoparticles.",
            "payload_usage_role": "row_local_value_evidence",
            "value_evidence_only": True,
            "table_role_hint": "characterization",
            "normalized_matrix": [
                ["Batch", "Size (nm)", "P.I.", "ZP (mV)", "E.E. (%)"],
                ["F1", "188.1", "0.12", "-22.4", "91.2"],
                ["F2", "201.5", "0.18", "-18.6", "89.7"],
            ],
        }
        rendered = "\n".join(compact_value_evidence_lines_from_payload(payload))
        self.assertIn("VALUE_EVIDENCE_TABLE", rendered)
        self.assertIn("row_local_value_evidence", rendered)
        self.assertIn("value_evidence_only", rendered)
        self.assertIn("F2", rendered)
        self.assertIn("201.5", rendered)
        self.assertIn("89.7", rendered)

    def test_universe_authority_payload_does_not_render_as_value_only_matrix(self):
        payload = {
            "payload_usage_role": "formulation_universe_authority",
            "value_evidence_only": False,
            "table_role_hint": "formulation",
            "normalized_matrix": [
                ["Formulation", "Drug", "Polymer"],
                ["F1", "Drug A", "PLGA"],
            ],
        }
        self.assertEqual(compact_value_evidence_lines_from_payload(payload), [])

    def test_flattened_result_payload_renders_raw_metric_value_rows(self):
        rows = [
            ["text"],
            ["NPg1 176.6±11.6 -18.6±0.4 34.1±0.1"],
            ["NPG2 176.5±2.9 -20.1±1.0 28.2±0.8"],
        ]
        rendered = "\n".join(
            build_raw_metric_value_evidence_lines(
                rows,
                table_role_hint="results",
                table_source_role="noise_or_nonformulation_table",
            )
        )
        self.assertIn("raw_metric_value_rows", rendered)
        self.assertIn("NPg1", rendered)
        self.assertIn("176.6", rendered)
        self.assertIn("34.1", rendered)

    def test_release_payload_does_not_render_raw_metric_value_rows(self):
        rows = [
            ["time", "release"],
            ["1 h", "12.5%"],
            ["2 h", "24.0%"],
        ]
        self.assertEqual(
            build_raw_metric_value_evidence_lines(
                rows,
                table_role_hint="results",
                table_source_role="release_profile_table",
            ),
            [],
        )

    def test_factorial_metric_matrix_rows_are_retained_beyond_first_twelve_candidates(self):
        rows = [["fragment"]]
        for idx in range(1, 18):
            rows.append([f"introductory prose {idx} with 11.1 ± 0.1 but no row-local matrix"])
        rows.extend(
            [
                [" | 1 | (cid:4)1 | (cid:4)1 | (cid:4)1 | 36.5 ± 2.21 | 126.6 ± 4.16"],
                [" | 10 | 0 | (cid:4)1 | (cid:4)1 | 61.6 ± 2.56 | 144.1 ± 2.13"],
                [" | 15 | 0 | 0 | 1 | 63.1 ± 2.17 | 165.2 ± 7.26"],
            ]
        )
        rendered = "\n".join(
            build_raw_metric_value_evidence_lines(
                rows,
                table_role_hint="results",
                table_source_role="reference_spillover_table",
            )
        )
        self.assertIn("36.5", rendered)
        self.assertIn("144.1", rendered)
        self.assertIn("165.2", rendered)

    def test_g_uncertainty_metric_rows_are_exposed_for_flattened_tables(self):
        rows = [
            ["fragment"],
            ["Diameter (nm) 274G3 278G15 280G19"],
            ["PI 0.455G0.130 0.412G0.051 0.376G0.104"],
        ]
        rendered = "\n".join(
            build_raw_metric_value_evidence_lines(
                rows,
                table_role_hint="results",
                table_source_role="reference_spillover_table",
            )
        )
        self.assertIn("274G3", rendered)
        self.assertIn("0.412G0.051", rendered)

    def test_paired_g_uncertainty_numeric_rows_are_exposed_for_compact_result_tables(self):
        rows = [
            ["fragment"],
            ["400 0.8 342G18 85G5 | 1200 2.4 918G9 77G1"],
            ["600 1.2 529G57 88G9 | 1400 2.8 1162G80 83G6"],
        ]
        rendered = "\n".join(
            build_raw_metric_value_evidence_lines(
                rows,
                table_role_hint="results",
                table_source_role="reference_spillover_table",
            )
        )
        self.assertIn("77G1", rendered)
        self.assertIn("83G6", rendered)

    def test_formulation_authority_payload_can_echo_metric_rows_as_value_evidence(self):
        payload = {
            "payload_usage_role": "formulation_universe_authority",
            "value_evidence_only": False,
            "table_source_role": "formulation_composition_table",
            "source_table_reference": "table_04.csv",
            "table_id": "Table 4",
            "normalized_matrix": [
                ["", "Empty", "Drug loaded", "Empty", "Drug loaded"],
                ["Diameter (nm)", "87.2 ± 0.25", "91.8 ± 2.74", "96.9 ± 1.06", "103.7 ± 2.98"],
                ["PIa", "0.14 ± 0.01", "0.13 ± 0.01", "0.12 ± 0.01", "0.14 ± 0.01"],
                ["ZPb (mV)", "−18.3 ± 0.52", "−21.23 ± 1.04", "−17.2 ± 0.51", "−28.06 ± 0.39"],
            ],
        }
        rendered = "\n".join(compact_value_evidence_lines_from_payload(payload))
        self.assertIn("0.13 ± 0.01", rendered)
        self.assertIn("−21.23 ± 1.04", rendered)
        self.assertIn("value_evidence_only", rendered)

    def test_value_evidence_payload_rendering_prioritizes_metric_payloads_beyond_first_eight(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paper_dir = root / "PAPER1"
            paper_dir.mkdir(parents=True)
            payloads = []
            for idx in range(9):
                payloads.append(
                    {
                        "payload_usage_role": "row_local_value_evidence",
                        "value_evidence_only": True,
                        "table_role_hint": "results",
                        "table_source_role": "reference_spillover_table",
                        "source_table_reference": f"noise_{idx}.csv",
                        "normalized_matrix": [["fragment"], [f"NPx{idx} 101.{idx}±0.1 -10.{idx}±0.2 20.{idx}±0.3"]],
                    }
                )
            payloads.append(
                {
                    "payload_usage_role": "row_local_value_evidence",
                    "value_evidence_only": True,
                    "table_role_hint": "results",
                    "table_source_role": "reference_spillover_table",
                    "source_table_reference": "late_metric_matrix.csv",
                    "normalized_matrix": [["fragment"], [" | 10 | 0 | (cid:4)1 | (cid:4)1 | 61.6 ± 2.56 | 144.1 ± 2.13"]],
                }
            )
            (paper_dir / "normalized_table_payloads_v1.json").write_text(
                json.dumps({"normalized_table_payloads": payloads}),
                encoding="utf-8",
            )
            rendered = "\n".join(
                render_value_evidence_payload_blocks(
                    {"paper_key": "PAPER1", "authority_payload_root": str(root)}
                )
            )
        self.assertIn("late_metric_matrix.csv", rendered)
        self.assertIn("144.1", rendered)
    def test_collect_summary_table_candidates_reads_manifest_selected_full_size_csv_asset(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
            root = Path(tmpdir)
            table_dir = root / "tables" / "PAPER1"
            asset_dir = root / "publisher_assets"
            table_dir.mkdir(parents=True)
            asset_dir.mkdir(parents=True)
            asset_csv = asset_dir / "full_size_table_2.csv"
            asset_csv.write_text(
                "Formulation,Drug loading (mg KGN/mg nanoparticles),Particle diameter (nm)\n"
                "NP-HA,0.467 ± 0.192,166.6 ± 34.48\n",
                encoding="utf-8",
            )
            (table_dir / "tables_manifest.json").write_text(
                json.dumps(
                    {
                        "paper_key": "PAPER1",
                        "tables": [],
                        "selected_table_assets": [
                            {
                                "local_path": str(asset_csv),
                                "href_raw": "https://example.org/article/tables/2",
                                "table_id": "Table 2",
                                "caption_or_title": "KGN-loaded nanoparticle properties",
                                "table_source_kind": "html_full_size_table_asset",
                                "asset_kind": "csv",
                                "selected": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            candidates = collect_summary_table_candidates(table_dir)

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate["path"], asset_csv)
        flattened = " | ".join(" | ".join(row) for row in candidate["rows"])
        self.assertIn("0.467", flattened)
        self.assertIn("166.6", flattened)
        self.assertEqual(candidate["repair_primary_source"], "stage1_selected_table_asset")
        self.assertEqual(candidate["meta"]["table_source_kind"], "html_full_size_table_asset")
        self.assertEqual(candidate["meta"]["source_table_reference"], "https://example.org/article/tables/2")
    def _payload_contract_for_rows(self, rows, *, caption=""):
        header_parts = extract_informative_header_parts(rows)
        meta = {"caption_or_title": caption} if caption else {}
        signal_text = " ".join(header_parts + [caption]).strip()
        source_role = classify_table_source_role(header_parts, {**meta, "_signal_text": signal_text})
        header_span = summary_header_row_count(rows)
        normalization_metadata = {
            "numbered_row_count": sum(
                1 for row in rows[header_span:] if row and str(row[0]).strip().rstrip(".").isdigit()
            )
        }
        table_type = classify_execution_table_type(
            rows,
            meta=meta,
            table_role_hint="",
            normalization_metadata=normalization_metadata,
        )
        payload = {
            "raw_cells": rows,
            "row_identity_signals": {"first_column_labels": [row[0] for row in rows[header_span:] if row]},
            "header_structure": {"header_rows": summary_header_context(rows)[0]},
            "source_caption_or_title": caption,
            "table_source_role": source_role,
            "table_type": table_type,
            "data_row_count": max(0, len(rows) - header_span),
            "representation_status": "raw_summary",
        }
        inclusion = payload_inclusion_class(payload)
        usage = payload_usage_role({**payload, "table_inclusion_class": inclusion})
        return source_role, table_type, inclusion, usage

    def test_ufxx9wxe_numbered_doe_matrix_is_payload_universe_authority_not_prompt_full_table(self):
        rows = [
            ["0", "1", "2", "3", "4", "5", "6", "7"],
            ["", "", "", "Table 2: Effect of independent process variables on dependent variable.", "", "", "", ""],
            ["", "", "", "", "", "", "% Drug", ""],
            ["", "PLGA", "Poloxamer", "w/o phase", "Drug conc.", "z-Average d.nm", "", "PDI"],
            ["Formulation", "", "", "", "", "", "entrapment", ""],
            ["", "mg/mL", "mg/mL", "volume ratio", "mg/mL", "(±SD)", "", "(±SD)"],
            ["1.", "35", "2", "6", "1", "211 ± 0.11", "70 ± 1.3", "0.183 ± 0.002"],
            ["2.", "35", "2", "6", "5", "220 ± 0.8", "88.48 ± 0.8", "0.150 ± 0.003"],
            ["3.", "10", "8.50", "10", "3", "176 ± 0.5", "83 ± 0.5", "0.048 ± 0.001"],
        ]
        source_role, table_type, inclusion, usage = self._payload_contract_for_rows(rows)
        self.assertIn(source_role, {"formulation_composition_table", "preparation_parameter_table"})
        self.assertIn(table_type, {"DOE_table", "mixed_table", "formulation_table"})
        self.assertEqual(inclusion, "must_include")
        self.assertEqual(usage, "formulation_universe_authority")
        rendered = "\n".join(build_table_summary_lines({"path": Path("data/cleaned/goren_2025/tables/UFXX9WXE/UFXX9WXE__table_13__pdf_table.csv"), "rows": rows, "meta": {}}, enhancement_enabled=False))
        self.assertIn("column_schema", rendered)
        self.assertIn("PLGA", rendered)
        self.assertIn("Poloxamer", rendered)
        self.assertNotIn("table_role_hint", rendered)
        self.assertNotIn("semantic_summary", rendered)

    def test_prose_spillover_numeric_labels_do_not_become_universe_authority(self):
        rows = [
            ["0", "1"],
            ["4", "BioMed Research International"],
            ["NPs pellet was redispersed in 2 mL methanolic PBS buffer", "than the control cells indicate a reduction in the rate of cell"],
            ["solution (pH 6.4, 30% v/v methanol). Methanolic PBS was", "proliferation. Conversely, a higher absorbance rate indicates"],
            ["used, as lorazepam being poorly water soluble requires", "another prose continuation"],
        ]
        _source_role, _table_type, inclusion, usage = self._payload_contract_for_rows(rows)
        self.assertNotEqual(inclusion, "must_include")
        self.assertEqual(usage, "row_local_value_evidence")

    def test_bb3juvw7_numeric_row_composition_tables_are_internal_payload_authority(self):
        composition_rows = [
            ["Composition Artemether (mg)", "Composition PLGA (mg)", "Composition PVA (mg)", "Composition Acetone (mL)", "Composition Aqueous phase (mL)", "Particle size (nm)"],
            ["5", "75", "75", "5", "15", "190.2 ± 18.0"],
            ["5", "75", "150", "5", "15", "214.3 ± 6.2"],
            ["10", "75", "300", "10", "30", "196.8 ± 1.1"],
        ]
        process_rows = [
            ["Process conditions Film thickness (µm)", "Process conditions PLGA type (lactide: glycolide)", "Process conditions Extent of stretching", "Process conditions Liquefaction method", "Process conditions Incubation period (min)", "Major axis (nm)"],
            ["100", "75:25", "4x", "Acetone", "15", "234.1 ± 61.7"],
            ["150", "75:25", "4x", "Acetone", "15", "295.1 ± 64.9"],
            ["100", "50:50", "4x", "Acetone", "15", "211.3 ± 44.1"],
        ]
        for rows in [composition_rows, process_rows]:
            with self.subTest(rows=rows[0][0]):
                _source_role, table_type, inclusion, usage = self._payload_contract_for_rows(rows)
                self.assertIn(table_type, {"DOE_table", "mixed_table", "formulation_table", "parameter_sweep_table"})
                self.assertEqual(inclusion, "must_include")
                self.assertEqual(usage, "formulation_universe_authority")

    def test_s2_4a_summary_contract_blocks_raw_table_text_when_summary_unavailable(self):
        rendered = render_prompt_block(
            {
                "block_id": "raw_table_block",
                "block_type": "table",
                "evidence_kind": "supporting",
                "origin_locator": "data/cleaned/goren_2025/tables/TEST/MISSING__table_00__pdf_table.csv#row:1",
                "text_content": "Formulation | Size (nm)\nF1 | 123.4\nF2 | 456.7",
            },
            summary_enhanced=False,
        )

        self.assertEqual(rendered["rendered_text"], "")
        self.assertNotIn("123.4", rendered["rendered_text"])
        self.assertEqual(rendered["summary_applied"], "no")
        self.assertEqual(
            rendered["reason_for_full_table"],
            "summary_table_unavailable_blocked_by_summary_only_contract",
        )

    def test_s2_4a_blocks_stale_table_summary_text_when_source_summary_unavailable(self):
        rendered = render_prompt_block(
            {
                "block_id": "stale_summary_block",
                "block_type": "table",
                "source_type": "table_summary",
                "origin_locator": "data/cleaned/goren_2025/tables/TEST/MISSING__table_99__pdf_table.csv",
                "text_content": (
                    "[TABLE_SUMMARY missing] - key_columns: Formulation | Size "
                    "- table_role_hint: formulation "
                    "- semantic_summary: formulation composition/identity cues"
                ),
            },
            summary_enhanced=False,
        )

        self.assertEqual(rendered["rendered_text"], "")
        self.assertEqual(rendered["summary_applied"], "no")
        self.assertEqual(
            rendered["reason_for_full_table"],
            "summary_table_unavailable_blocked_by_summary_only_contract",
        )
        self.assertNotIn("table_role_hint", rendered["rendered_text"])
        self.assertNotIn("semantic_summary", rendered["rendered_text"])
        self.assertNotIn("key_columns", rendered["rendered_text"])


if __name__ == "__main__":
    unittest.main()
