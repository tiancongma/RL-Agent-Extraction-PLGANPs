import json
import tempfile
import unittest
from pathlib import Path

from src.stage1_cleaning.pdf2clean import extract_html_native_table_cells


class TestStage1HtmlNativeTableExtractionV1(unittest.TestCase):
    def test_extracts_dom_table_cells_with_span_sections_caption_and_blank_cells(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            html_path = root / "DOC001.html"
            html_path.write_text(
                """
                <html><body>
                  <p>Before table.</p>
                  <table id="formulation-table">
                    <caption>Table 1. Formulation metrics</caption>
                    <thead>
                      <tr><th rowspan="2" scope="col">Formulation</th><th colspan="2" scope="colgroup">Measurements</th></tr>
                      <tr><th scope="col">Size (nm)</th><th scope="col">P.I.</th></tr>
                    </thead>
                    <tbody>
                      <tr><th scope="row">F1</th><td>123</td><td></td></tr>
                      <tr><th scope="row">F2</th><td colspan="2">not measured</td></tr>
                    </tbody>
                    <tfoot>
                      <tr><td colspan="3">Values are mean ± SD.</td></tr>
                    </tfoot>
                  </table>
                </body></html>
                """,
                encoding="utf-8",
            )

            cells = extract_html_native_table_cells(html_path, doc_key="DOC001")

        self.assertEqual(len(cells), 10)
        first = cells[0]
        self.assertEqual(first["paper_key"], "DOC001")
        self.assertEqual(first["source_type"], "HTML")
        self.assertEqual(first["parser"], "beautifulsoup_dom_table")
        self.assertEqual(first["parser_variant"], "html_native_table_cells_v1")
        self.assertEqual(first["table_id"], "t001")
        self.assertEqual(first["table_source_kind"], "html_dom_table")
        self.assertEqual(first["caption"], "Table 1. Formulation metrics")
        self.assertEqual(first["row_index"], 1)
        self.assertEqual(first["col_index"], 1)
        self.assertEqual(first["rowspan"], 2)
        self.assertEqual(first["colspan"], 1)
        self.assertEqual(first["raw_cell_text"], "Formulation")
        self.assertEqual(first["is_header_cell"], "yes")
        self.assertEqual(first["header_scope"], "col")
        self.assertEqual(json.loads(first["warnings_json"]), [])
        self.assertNotIn("semantic_label", first)
        self.assertNotIn("variable_role", first)

        blank = [cell for cell in cells if cell["row_index"] == 3 and cell["col_index"] == 3][0]
        self.assertEqual(blank["raw_cell_text"], "")
        self.assertEqual(blank["normalized_cell_text"], "")
        self.assertEqual(blank["row_label_text"], "F1")
        self.assertEqual(blank["column_label_text"], "P.I.")
        self.assertEqual(json.loads(blank["header_path_json"]), ["Measurements", "P.I."])

        tfoot = cells[-1]
        self.assertEqual(tfoot["row_index"], 5)
        self.assertEqual(tfoot["colspan"], 3)
        self.assertIn("Values are mean", tfoot["raw_cell_text"])


if __name__ == "__main__":
    unittest.main()
