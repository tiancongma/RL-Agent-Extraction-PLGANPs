import json
import tempfile
import unittest
from pathlib import Path

from src.stage2_sampling_labels.extract_semantic_stage2_objects_v2 import (
    build_raw_metric_value_evidence_lines,
    build_table_summary_lines,
    compact_value_evidence_lines_from_payload,
    prompt_semantic_summary_line,
    render_value_evidence_payload_blocks,
)


class Stage2TableSummaryNumericVisibilityTests(unittest.TestCase):
    def test_formulation_summary_exposes_semantic_cues_without_full_numeric_authority(self):
        rows = [
            ["Formulation", "Polymer", "Surfactant", "Organic solvent"],
            ["F1", "PLGA", "PVA", "acetone"],
            ["F2", "PLGA", "PVA", "ethyl acetate"],
        ]
        item = {
            "path": Path("data/cleaned/goren_2025/tables/TEST/TEST__table_00__pdf_table.csv"),
            "rows": rows,
            "meta": {"caption_or_title": "Formulation composition of PLGA nanoparticles", "n_rows": len(rows), "n_cols": len(rows[0])},
        }
        rendered = "\n".join(build_table_summary_lines(item, enhancement_enabled=False))
        self.assertIn("semantic_summary", rendered)
        self.assertIn("nanoparticle preparation", rendered)
        self.assertIn("row/value authority remains in the S2-2 payload/grid", rendered)
        self.assertNotIn("full_numeric_rows", rendered)

    def test_design_summary_points_to_s2_2_payload_grid_for_value_authority(self):
        line = prompt_semantic_summary_line("design matrix")
        self.assertIn("formulation design/process variables", line)
        self.assertIn("S2-2 payload/grid", line)

    def test_characterization_summary_not_promoted_to_formulation_universe_authority(self):
        line = prompt_semantic_summary_line("characterization")
        self.assertIn("measurement/context evidence only", line)
        self.assertIn("not formulation-universe", line)

    def test_characterization_table_summary_exposes_row_local_metric_values_beyond_three_samples(self):
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
        self.assertIn("metric_value_rows", rendered)
        self.assertIn("NP4", rendered)
        self.assertIn("104.4", rendered)
        self.assertIn("45.4", rendered)
        self.assertIn("6.4", rendered)
        self.assertIn("80.4", rendered)
        self.assertIn("value_evidence_only", rendered)

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


if __name__ == "__main__":
    unittest.main()
