import csv
import tempfile
import unittest
from pathlib import Path

from src.stage1_cleaning.build_stage1_unified_current_marker_v1 import (
    infer_source_type,
    load_key2structure,
    load_key2txt,
)


class TestStage1UnifiedHtmlPriorityV1(unittest.TestCase):
    def test_key2txt_prefers_html_binding_over_pdf_duplicate(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "key2txt.tsv"
            path.write_text(
                "DOC1\tdata/cleaned/content/text/DOC1.html.txt\n"
                "DOC1\tdata/cleaned/content/text/DOC1.pdf.txt\n"
                "DOC2\tdata/cleaned/content/text/DOC2.pdf.txt\n",
                encoding="utf-8",
            )

            key2txt = load_key2txt(path)

        self.assertEqual(key2txt["DOC1"], "data/cleaned/content/text/DOC1.html.txt")
        self.assertEqual(key2txt["DOC2"], "data/cleaned/content/text/DOC2.pdf.txt")

    def test_key2structure_prefers_html_structure_over_pdf_duplicate(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "key2structure.tsv"
            rows = [
                {
                    "key": "DOC1",
                    "txt_path": "data/cleaned/content/text/DOC1.pdf.txt",
                    "structure_path": "data/cleaned/content/structure/DOC1.pdf.json",
                    "tables_dir": "",
                    "parser": "pymupdf_fallback",
                },
                {
                    "key": "DOC1",
                    "txt_path": "data/cleaned/content/text/DOC1.html.txt",
                    "structure_path": "data/cleaned/content/structure/DOC1.html.json",
                    "tables_dir": "",
                    "parser": "trafilatura",
                },
            ]
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["key", "txt_path", "structure_path", "tables_dir", "parser"],
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerows(rows)

            key2structure = load_key2structure(path)

        self.assertEqual(
            key2structure["DOC1"]["structure_path"],
            "data/cleaned/content/structure/DOC1.html.json",
        )
        self.assertEqual(key2structure["DOC1"]["parser"], "trafilatura")

    def test_infer_source_type_prefers_selected_html_path_over_manifest_pdf_default(self):
        source_type = infer_source_type(
            {
                "text_source_type": "pdf",
                "text_path": "data/cleaned/content/text/DOC1.html.txt",
                "pdf": "source.pdf",
                "html": "source.html",
            },
            {"source_type": "HTML"},
        )

        self.assertEqual(source_type, "HTML")


if __name__ == "__main__":
    unittest.main()
