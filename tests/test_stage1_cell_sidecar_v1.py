import json
import tempfile
import unittest
from pathlib import Path

from src.stage1_cleaning import run_stage1_parser_bakeoff_v1 as bakeoff


class TestStage1CellSidecarV1(unittest.TestCase):
    def test_write_stage1_cell_sidecars_groups_by_paper_and_writes_required_schema(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "diagnostic"
            rows = [
                {
                    "paper_key": "PAPER002",
                    "source_type": "HTML",
                    "source_path": "/tmp/PAPER002.html",
                    "parser": "current",
                    "parser_variant": "diagnostic_bakeoff_v1",
                    "table_id": "t002",
                    "table_source_kind": "html_dom_table",
                    "page": "",
                    "bbox_json": "{}",
                    "caption": "Table 2",
                    "row_index": 2,
                    "col_index": 1,
                    "rowspan": 1,
                    "colspan": 1,
                    "raw_cell_text": "F2",
                    "normalized_cell_text": "F2",
                    "is_header_cell": "no",
                    "header_scope": "",
                    "header_path_json": "[]",
                    "row_label_text": "F2",
                    "column_label_text": "Formulation",
                    "source_block_id": "t002",
                    "source_hash": "sha1:bbb",
                    "warnings_json": "[]",
                    "semantic_label": "must_not_leak",
                },
                {
                    "paper_key": "PAPER001",
                    "source_type": "HTML",
                    "source_path": "/tmp/PAPER001.html",
                    "parser": "current",
                    "parser_variant": "diagnostic_bakeoff_v1",
                    "table_id": "t001",
                    "table_source_kind": "html_dom_table",
                    "page": "",
                    "bbox_json": "{}",
                    "caption": "Table 1",
                    "row_index": 1,
                    "col_index": 1,
                    "rowspan": 1,
                    "colspan": 1,
                    "raw_cell_text": "Formulation",
                    "normalized_cell_text": "Formulation",
                    "is_header_cell": "yes",
                    "header_scope": "col",
                    "header_path_json": "[\"Formulation\"]",
                    "row_label_text": "",
                    "column_label_text": "Formulation",
                    "source_block_id": "t001",
                    "source_hash": "sha1:aaa",
                    "warnings_json": "[]",
                    "variable_role": "must_not_leak",
                },
            ]

            manifest_rows = bakeoff.write_stage1_cell_sidecars(out_dir, rows)

            self.assertEqual(len(manifest_rows), 2)
            paper1_path = out_dir / "tables_cell_sidecar" / "PAPER001" / "stage1_table_cells_v1.jsonl"
            paper2_path = out_dir / "tables_cell_sidecar" / "PAPER002" / "stage1_table_cells_v1.jsonl"
            manifest_path = out_dir / "tables_cell_sidecar" / "stage1_table_cells_manifest_v1.tsv"
            self.assertTrue(paper1_path.exists())
            self.assertTrue(paper2_path.exists())
            self.assertTrue(manifest_path.exists())

            first = json.loads(paper1_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(list(first.keys()), bakeoff.STAGE1_TABLE_CELL_COLUMNS)
            self.assertNotIn("semantic_label", first)
            self.assertNotIn("variable_role", first)
            self.assertEqual(json.loads(first["header_path_json"]), ["Formulation"])
            self.assertEqual(json.loads(first["warnings_json"]), [])

    def test_current_html_bakeoff_writes_run_scoped_cell_sidecar_without_active_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            html = root / "PAPER001.html"
            html.write_text(
                "<html><body><table><caption>Table 1</caption>"
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
            self.assertTrue((out_dir / "parser_bakeoff_cells_v1.jsonl").exists())
            sidecar = out_dir / "tables_cell_sidecar" / "PAPER001" / "stage1_table_cells_v1.jsonl"
            self.assertTrue(sidecar.exists())
            self.assertFalse((root / "text").exists(), "diagnostic sidecar must not write active Stage1 text outputs")
            self.assertFalse((root / "key2txt.tsv").exists(), "diagnostic sidecar must not write active key2txt mappings")
            rows = [json.loads(line) for line in sidecar.read_text(encoding="utf-8").splitlines()]
            self.assertGreaterEqual(len(rows), 4)
            self.assertEqual(list(rows[0].keys()), bakeoff.STAGE1_TABLE_CELL_COLUMNS)
            self.assertEqual(rows[0]["paper_key"], "PAPER001")
            self.assertNotIn("semantic_label", rows[0])


if __name__ == "__main__":
    unittest.main()
