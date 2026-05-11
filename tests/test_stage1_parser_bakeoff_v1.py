import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.stage1_cleaning import run_stage1_parser_bakeoff_v1 as bakeoff
from src.stage1_cleaning.pdf2clean import (
    annotate_blocks_with_sections,
    build_sidecar_payload,
    extract_marker_markdown_blocks,
    extract_marker_markdown_tables,
    finalize_blocks,
)


class TestStage1ParserBakeoffV1(unittest.TestCase):
    def test_current_html_parser_writes_diagnostic_artifacts_without_active_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            html = root / "PAPER001.html"
            html.write_text(
                "<html><body><h1>Title</h1><p>Preparation text.</p>"
                "<table><caption>Table 1</caption>"
                "<tr><th>Formulation</th><th>Size (nm)</th></tr>"
                "<tr><td>F1</td><td>123</td></tr>"
                "</table></body></html>",
                encoding="utf-8",
            )
            manifest = root / "manifest.tsv"
            manifest.write_text(
                "key\ttitle\tpdf\thtml\ttable_dir\n"
                f"PAPER001\tSynthetic\t\t{html}\t\n",
                encoding="utf-8",
            )
            scope = root / "scope.tsv"
            scope.write_text("key\nPAPER001\n", encoding="utf-8")
            out_dir = root / "bakeoff"

            result = bakeoff.run_bakeoff(
                manifest_path=manifest,
                scope_keys_path=scope,
                out_dir=out_dir,
                parser_selection="current",
            )

            self.assertEqual(result["status"], "completed")
            self.assertTrue((out_dir / "parser_bakeoff_summary_v1.tsv").exists())
            self.assertTrue((out_dir / "parser_bakeoff_blocks_v1.jsonl").exists())
            self.assertTrue((out_dir / "parser_bakeoff_tables_v1.jsonl").exists())
            self.assertTrue((out_dir / "parser_bakeoff_cells_v1.jsonl").exists())
            self.assertTrue((out_dir / "parser_bakeoff_warnings_v1.tsv").exists())
            self.assertTrue((out_dir / "RUN_CONTEXT.md").exists())
            self.assertTrue((out_dir / "tables_cell_sidecar" / "PAPER001" / "stage1_table_cells_v1.jsonl").exists())
            self.assertTrue((out_dir / "tables_cell_sidecar" / "stage1_table_cells_manifest_v1.tsv").exists())
            self.assertFalse((root / "text").exists(), "bakeoff must not write active Stage1 text outputs")

            with (out_dir / "parser_bakeoff_summary_v1.tsv").open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f, delimiter="\t"))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["paper_key"], "PAPER001")
            self.assertEqual(rows[0]["parser"], "current")
            self.assertEqual(rows[0]["status"], "ok")
            self.assertEqual(rows[0]["benchmark_valid"], "no")
            expected_sidecar_path = str((out_dir / "tables_cell_sidecar" / "PAPER001" / "stage1_table_cells_v1.jsonl").resolve())
            self.assertEqual(rows[0]["stage1_table_cell_sidecar_path"], expected_sidecar_path)
            self.assertEqual(rows[0]["stage1_table_cell_sidecar_available"], "yes")

            with (out_dir / "tables_cell_sidecar" / "stage1_table_cells_manifest_v1.tsv").open(encoding="utf-8") as f:
                sidecar_manifest_rows = list(csv.DictReader(f, delimiter="\t"))
            self.assertEqual(len(sidecar_manifest_rows), 1)
            self.assertEqual(sidecar_manifest_rows[0]["paper_key"], "PAPER001")
            self.assertEqual(sidecar_manifest_rows[0]["stage1_table_cell_sidecar_path"], expected_sidecar_path)
            self.assertEqual(sidecar_manifest_rows[0]["stage1_table_cell_sidecar_available"], "yes")
            self.assertEqual(sidecar_manifest_rows[0]["source_type"], "HTML")

            cells = [json.loads(line) for line in (out_dir / "parser_bakeoff_cells_v1.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertGreaterEqual(len(cells), 4)
            self.assertEqual(cells[0]["parser"], "current")
            self.assertEqual(cells[0]["paper_key"], "PAPER001")
            self.assertIn("raw_cell_text", cells[0])
            self.assertNotIn("semantic_label", cells[0])

    def test_stage1_table_cell_sidecar_manifest_uses_source_agnostic_path_columns_for_pdf_cells(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = bakeoff.write_stage1_cell_sidecars(
                root,
                [
                    {
                        "paper_key": "PDF001",
                        "source_type": "PDF",
                        "source_path": "data/raw/PDF001.pdf",
                        "parser": "marker",
                        "parser_variant": "diagnostic_bakeoff_v1",
                        "table_id": "t001",
                        "table_source_kind": "marker_markdown_table",
                        "row_index": "1",
                        "col_index": "1",
                        "raw_cell_text": "EE (%)",
                        "normalized_cell_text": "EE (%)",
                        "is_header_cell": "yes",
                    }
                ],
            )

            expected_sidecar_path = str((root / "tables_cell_sidecar" / "PDF001" / "stage1_table_cells_v1.jsonl").resolve())
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["paper_key"], "PDF001")
            self.assertEqual(rows[0]["stage1_table_cell_sidecar_path"], expected_sidecar_path)
            self.assertEqual(rows[0]["stage1_table_cell_sidecar_available"], "yes")
            self.assertEqual(rows[0]["source_type"], "PDF")
            self.assertEqual(rows[0]["parsers"], "marker")
            manifest_path = root / "tables_cell_sidecar" / "stage1_table_cells_manifest_v1.tsv"
            with manifest_path.open(encoding="utf-8") as f:
                manifest_rows = list(csv.DictReader(f, delimiter="\t"))
            self.assertEqual(manifest_rows[0]["stage1_table_cell_sidecar_path"], expected_sidecar_path)

    def test_marker_markdown_pdf_tables_emit_structure_cells(self):
        marker_text = """
Intro paragraph.

Table 1 Encapsulation parameters of XAN and 3-MeOXAN in PLGA nanospheres

| Theoretical concentration (mg/mL) | XAN nanospheres | 3-MeOXAN nanospheres |
| --- | --- | --- |
| 50 | 13.0 ± 1.1 | 19.0 ± 0.6 |
| 60 | 20.0 ± 2.4 | 24.9 ± 4.6 |

After table.
"""

        tables, cells = extract_marker_markdown_tables(
            marker_text,
            paper_key="PDF001",
            source_path=Path("/tmp/source.pdf"),
            parser="marker",
            parser_variant="diagnostic_bakeoff_v1",
        )

        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0]["format"], "marker_markdown_table")
        self.assertIn("Encapsulation parameters", tables[0]["caption"])
        self.assertEqual(tables[0]["caption_binding_rule"], "nearest_preceding_table_caption_line")
        self.assertEqual(len(cells), 9)
        header_cells = [c for c in cells if c["is_header_cell"] == "yes"]
        value_cells = [c for c in cells if c["is_header_cell"] == "no"]
        self.assertEqual(len(header_cells), 3)
        self.assertEqual(len(value_cells), 6)
        self.assertEqual(value_cells[0]["raw_cell_text"], "50")
        self.assertEqual(value_cells[1]["column_label_text"], "XAN nanospheres")
        self.assertEqual(value_cells[1]["caption_binding_rule"], "nearest_preceding_table_caption_line")
        self.assertEqual(json.loads(value_cells[1]["header_path_json"]), ["XAN nanospheres"])
        self.assertEqual(value_cells[1]["row_label_text"], "50")
        self.assertNotIn("semantic_label", value_cells[1])

    def test_marker_bakeoff_binds_table_captions_from_structural_block_adjacency(self):
        blocks = [
            {"block_id": "b0001", "order": 1, "type": "paragraph", "text": "Methods prose"},
            {"block_id": "b0002", "order": 2, "type": "paragraph", "text": "<span id=\"page-5-0\"></span>Table 3 Encapsulation parameters of XAN in PLGA nanocapsules"},
            {
                "block_id": "b0003",
                "order": 3,
                "type": "table",
                "text": "| Formulation | EE (%) |\n| --- | --- |\n| F1 | 77 |",
            },
        ]
        tables = [
            {
                "paper_key": "PDF001",
                "parser": "marker",
                "table_id": "t001",
                "block_text": "| Formulation | EE (%) |\n| --- | --- |\n| F1 | 77 |",
                "caption": "",
                "caption_binding_rule": "",
            }
        ]
        cells = [
            {"paper_key": "PDF001", "parser": "marker", "table_id": "t001", "caption": "", "caption_binding_rule": ""}
        ]

        bound_tables, bound_cells = bakeoff.bind_marker_table_captions_from_blocks(blocks, tables, cells)

        self.assertEqual(bound_tables[0]["caption"], "Table 3 Encapsulation parameters of XAN in PLGA nanocapsules")
        self.assertEqual(bound_tables[0]["caption_binding_rule"], "preceding_structural_block_caption")
        self.assertEqual(bound_tables[0]["caption_source_block_id"], "b0002")
        self.assertEqual(bound_cells[0]["caption"], bound_tables[0]["caption"])
        self.assertEqual(bound_cells[0]["caption_binding_rule"], "preceding_structural_block_caption")
        self.assertEqual(bound_cells[0]["caption_source_block_id"], "b0002")

    def test_marker_markdown_caption_binding_handles_span_prefixed_caption_lines(self):
        marker_text = """
<span id="page-5-0"></span>Table 3 Encapsulation parameters of XAN in PLGA nanocapsules

| Formulation | EE (%) |
| --- | --- |
| F1 | 77 |
"""
        tables, cells = extract_marker_markdown_tables(
            marker_text,
            paper_key="PDF001",
            source_path=Path("/tmp/source.pdf"),
            parser="marker",
            parser_variant="diagnostic_bakeoff_v1",
        )
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0]["caption"], "Table 3 Encapsulation parameters of XAN in PLGA nanocapsules")
        self.assertEqual(tables[0]["caption_binding_rule"], "nearest_preceding_table_caption_line")
        self.assertEqual(cells[0]["caption"], tables[0]["caption"])

    def test_marker_markdown_adjacent_compatible_tables_emit_continuation_metadata(self):
        marker_text = """
Table 2 Formulation composition and characterization

| Formulation | PLGA (mg) | Size (nm) |
| --- | --- | --- |
| F1 | 10 | 120 |

| Formulation | PLGA (mg) | Size (nm) |
| --- | --- | --- |
| F2 | 20 | 140 |
"""
        tables, cells = extract_marker_markdown_tables(
            marker_text,
            paper_key="PDF001",
            source_path=Path("/tmp/source.pdf"),
            parser="marker",
            parser_variant="diagnostic_bakeoff_v1",
        )

        self.assertEqual(len(tables), 2)
        self.assertNotEqual(tables[0]["table_id"], tables[1]["table_id"])
        self.assertEqual(tables[0]["continuation_group_id"], tables[1]["continuation_group_id"])
        self.assertEqual(tables[1]["continuation_binding_rule"], "adjacent_compatible_marker_markdown_table")
        second_fragment_cells = [cell for cell in cells if cell["table_id"] == tables[1]["table_id"]]
        self.assertTrue(second_fragment_cells)
        self.assertEqual(second_fragment_cells[0]["continuation_group_id"], tables[1]["continuation_group_id"])

    def test_marker_markdown_cells_normalize_scientific_symbols_with_provenance(self):
        marker_text = """
Table 3 Characterization

| Formulation | Size | ζ-potential |
| --- | --- | --- |
| F1 | 12 ± 3 μm | −25 mV |
"""
        _tables, cells = extract_marker_markdown_tables(
            marker_text,
            paper_key="PDF001",
            source_path=Path("/tmp/source.pdf"),
            parser="marker",
            parser_variant="diagnostic_bakeoff_v1",
        )

        size_cell = next(cell for cell in cells if cell["raw_cell_text"] == "12 ± 3 μm")
        zeta_header = next(cell for cell in cells if cell["raw_cell_text"] == "ζ-potential")
        zeta_value = next(cell for cell in cells if cell["raw_cell_text"] == "−25 mV")
        self.assertEqual(size_cell["normalized_cell_text"], "12 +/- 3 um")
        self.assertEqual(zeta_header["normalized_cell_text"], "zeta-potential")
        self.assertEqual(zeta_value["normalized_cell_text"], "-25 mV")
        self.assertIn("plus_minus_to_ascii", json.loads(size_cell["warnings_json"]))
        self.assertIn("micro_to_u", json.loads(size_cell["warnings_json"]))
        self.assertIn("unicode_minus_to_ascii", json.loads(zeta_value["warnings_json"]))
        self.assertIn("zeta_to_text", json.loads(zeta_header["warnings_json"]))

    def test_marker_markdown_article_chrome_table_tagged_noise_without_dropping_scientific_table(self):
        marker_text = """
| Article type | Research article |
| --- | --- |
| Submitted | 01 Jan 2024 |

Table 4 Formulation characterization

| Formulation | Size (nm) |
| --- | --- |
| F1 | 120 |
"""
        tables, cells = extract_marker_markdown_tables(
            marker_text,
            paper_key="PDF001",
            source_path=Path("/tmp/source.pdf"),
            parser="marker",
            parser_variant="diagnostic_bakeoff_v1",
        )

        self.assertEqual(len(tables), 2)
        self.assertEqual(tables[0]["noise_class"], "confirmed_noise")
        self.assertIn("article_metadata_table", tables[0]["noise_reason"])
        self.assertEqual(tables[1]["noise_class"], "keep")
        first_table_cells = [cell for cell in cells if cell["table_id"] == tables[0]["table_id"]]
        self.assertTrue(first_table_cells)
        self.assertEqual(first_table_cells[0]["noise_class"], "confirmed_noise")

    def test_marker_markdown_blocks_preserve_headings_for_section_model(self):
        marker_text = """
# Article Title

## 1 Introduction

Background text should remain visible.

## 2 Materials and methods

PLGA nanoparticles were prepared by nanoprecipitation.

## References

[1] A reference that should be tagged but not deleted.
"""
        blocks = finalize_blocks(extract_marker_markdown_blocks(marker_text))
        annotated, sections = annotate_blocks_with_sections(blocks)

        heading_texts = [block["text"] for block in annotated if block["type"] == "heading"]
        self.assertIn("2 Materials and methods", heading_texts)
        self.assertIn("References", heading_texts)
        method_block = next(block for block in annotated if "nanoprecipitation" in block["text"])
        self.assertEqual(method_block["section_label"], "2 Materials and methods")
        self.assertEqual(method_block["section_kind"], "methods")
        reference_block = next(block for block in annotated if "A reference" in block["text"])
        self.assertEqual(reference_block["section_kind"], "references")
        self.assertIn(reference_block["noise_class"], {"suppressible_noise", "terminal_noise"})
        self.assertIn("section_path_json", reference_block)
        self.assertTrue(sections)
        self.assertTrue(any(section["section_label"] == "References" for section in sections))

    def test_stage1_sidecar_payload_emits_sections_additively(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            txt_path = root / "PDF001.pdf.txt"
            text = "2 Materials and methods\n\nPLGA nanoparticles were prepared."
            txt_path.write_text(text, encoding="utf-8")
            blocks = finalize_blocks([
                {"type": "heading", "text": "2 Materials and methods", "heading_level": 2},
                {"type": "paragraph", "text": "PLGA nanoparticles were prepared."},
            ])
            payload = build_sidecar_payload(
                key="PDF001",
                source_type="PDF",
                txt_path=txt_path,
                text=text,
                parse_payload={
                    "blocks": blocks,
                    "tables": [],
                    "metadata": {"parser": "marker", "parse_quality": "high", "warnings": []},
                    "reading_order_source": "marker_native",
                },
            )

            self.assertEqual(payload["metadata"]["section_model_version"], "stage1_section_model_v1")
            self.assertEqual(len(payload["sections"]), 1)
            self.assertEqual(payload["sections"][0]["section_kind"], "methods")
            self.assertEqual(payload["blocks"][1]["section_label"], "2 Materials and methods")
            self.assertEqual(payload["blocks"][1]["noise_class"], "keep")
            self.assertIn("PLGA nanoparticles were prepared", text)

    def test_marker_missing_is_reported_as_diagnostic_row_not_crash(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pdf = root / "PAPER002.pdf"
            pdf.write_bytes(b"%PDF-1.4\n% synthetic placeholder\n")
            manifest = root / "manifest.tsv"
            manifest.write_text(
                "key\ttitle\tpdf\thtml\ttable_dir\n"
                f"PAPER002\tSynthetic PDF\t{pdf}\t\t\n",
                encoding="utf-8",
            )
            scope = root / "scope.tsv"
            scope.write_text("PAPER002\n", encoding="utf-8")
            out_dir = root / "bakeoff"

            with patch.object(bakeoff, "extract_marker_pdf_blocks", side_effect=RuntimeError("marker_not_available:ImportError:no module named marker")):
                result = bakeoff.run_bakeoff(
                    manifest_path=manifest,
                    scope_keys_path=scope,
                    out_dir=out_dir,
                    parser_selection="marker",
                )

            self.assertEqual(result["status"], "completed")
            with (out_dir / "parser_bakeoff_summary_v1.tsv").open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f, delimiter="\t"))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["parser"], "marker")
            self.assertEqual(rows[0]["status"], "marker_unavailable")
            self.assertIn("marker_not_available", rows[0]["notes"])
            self.assertEqual(rows[0]["benchmark_valid"], "no")

            warning_text = (out_dir / "parser_bakeoff_warnings_v1.tsv").read_text(encoding="utf-8")
            self.assertIn("marker_unavailable", warning_text)


if __name__ == "__main__":
    unittest.main()
