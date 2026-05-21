import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.stage1_cleaning import build_stage1_unified_current_marker_v1 as unified


class Stage1UnifiedMarkerTablePromotionTests(unittest.TestCase):
    def test_promotes_marker_table_cells_to_table_dir_and_sidecar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            current_text = root / "PAPER1.pdf.txt"
            marker_text = root / "marker_clean_text_v1.md"
            marker_structure = root / "marker_structure_v1.json"
            out_root = root / "out"
            current_text.write_text("current text", encoding="utf-8")
            marker_text.write_text("marker text", encoding="utf-8")
            marker_structure.write_text(
                json.dumps(
                    {
                        "blocks": [
                            {"block_id": "caption1", "block_type": "Caption", "text": "Table 1. Formulations."},
                            {
                                "block_id": "table1",
                                "block_type": "Table",
                                "page_index": 0,
                                "bbox": [0, 0, 100, 60],
                                "text": "F1 100 80",
                            },
                            {"block_id": "c11", "block_type": "TableCell", "page_index": 0, "bbox": [0, 0, 40, 20], "text": "Formulation"},
                            {"block_id": "c12", "block_type": "TableCell", "page_index": 0, "bbox": [45, 0, 80, 20], "text": "Size"},
                            {"block_id": "c21", "block_type": "TableCell", "page_index": 0, "bbox": [0, 25, 40, 45], "text": "F1"},
                            {"block_id": "c22", "block_type": "TableCell", "page_index": 0, "bbox": [45, 25, 80, 45], "text": "100"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            rec = unified.process_one(
                "PAPER1",
                {"key": "PAPER1", "pdf": "paper.pdf"},
                {"PAPER1": str(current_text)},
                {},
                {
                    "PAPER1": {
                        "marker_text_path": str(marker_text),
                        "marker_structure_path": str(marker_structure),
                    }
                },
                {},
                out_root,
            )

            self.assertEqual(rec["table_available"], "yes")
            self.assertEqual(rec["stage1_table_cell_sidecar_available"], "yes")
            table_dir = Path(rec["table_dir"])
            self.assertTrue(table_dir.exists())
            csv_path = table_dir / "PAPER1__marker_table_01.csv"
            self.assertTrue(csv_path.exists())
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows, [["Formulation", "Size"], ["F1", "100"]])

            structure = json.loads((out_root / "PAPER1" / "unified_structure_v1.json").read_text(encoding="utf-8"))
            self.assertEqual(len(structure["tables"]), 1)
            self.assertEqual(structure["table_available"], "yes")

    def test_preserves_manifest_table_dir_when_marker_tables_exist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            current_text = root / "PAPER0.pdf.txt"
            marker_text = root / "marker_clean_text_v1.md"
            marker_structure = root / "marker_structure_v1.json"
            curated_table_dir = root / "curated_tables" / "PAPER0"
            out_root = root / "out"
            curated_table_dir.mkdir(parents=True)
            (curated_table_dir / "PAPER0__table_01__pdf_table.csv").write_text("Run,Value\nF1,1\n", encoding="utf-8")
            current_text.write_text("current text", encoding="utf-8")
            marker_text.write_text("marker text", encoding="utf-8")
            marker_structure.write_text(
                json.dumps(
                    {
                        "blocks": [
                            {"block_id": "caption1", "block_type": "Caption", "text": "Table 1. Formulations."},
                            {"block_id": "table1", "block_type": "Table", "page_index": 0, "bbox": [0, 0, 100, 60], "text": "F1 100"},
                            {"block_id": "c11", "block_type": "TableCell", "page_index": 0, "bbox": [0, 0, 40, 20], "text": "Formulation"},
                            {"block_id": "c12", "block_type": "TableCell", "page_index": 0, "bbox": [45, 0, 80, 20], "text": "Size"},
                            {"block_id": "c21", "block_type": "TableCell", "page_index": 0, "bbox": [0, 25, 40, 45], "text": "F1"},
                            {"block_id": "c22", "block_type": "TableCell", "page_index": 0, "bbox": [45, 25, 80, 45], "text": "100"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            rec = unified.process_one(
                "PAPER0",
                {
                    "key": "PAPER0",
                    "pdf": "paper.pdf",
                    "table_dir": str(curated_table_dir),
                    "table_available": "yes",
                },
                {"PAPER0": str(current_text)},
                {},
                {
                    "PAPER0": {
                        "marker_text_path": str(marker_text),
                        "marker_structure_path": str(marker_structure),
                    }
                },
                {},
                out_root,
            )

            self.assertEqual(Path(rec["table_dir"]).resolve(), curated_table_dir.resolve())
            self.assertEqual(rec["table_available"], "yes")
            self.assertEqual(rec["stage1_table_cell_sidecar_available"], "no")
            self.assertTrue((out_root / "PAPER0" / "tables" / "PAPER0__marker_table_01.csv").exists())

    def test_projects_existing_sidecar_to_manifest_table_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sidecar = root / "stage1_table_cells_v1.jsonl"
            sidecar.write_text(
                "\n".join(
                    [
                        json.dumps({"paper_key": "PAPER2", "table_id": "t1", "row_index": "1", "col_index": "1", "raw_cell_text": "Run"}),
                        json.dumps({"paper_key": "PAPER2", "table_id": "t1", "row_index": "1", "col_index": "2", "raw_cell_text": "EE"}),
                        json.dumps({"paper_key": "PAPER2", "table_id": "t1", "row_index": "2", "col_index": "1", "raw_cell_text": "F1"}),
                        json.dumps({"paper_key": "PAPER2", "table_id": "t1", "row_index": "2", "col_index": "2", "raw_cell_text": "80"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            tables, table_dir, available = unified.extract_table_assets_from_stage1_sidecar(
                paper_key="PAPER2",
                sidecar_path=str(sidecar),
                paper_out=root / "out" / "PAPER2",
            )

            self.assertEqual(available, "yes")
            self.assertEqual(len(tables), 1)
            with (Path(table_dir) / "PAPER2__sidecar_table_01.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows, [["Run", "EE"], ["F1", "80"]])

    def test_preserves_manifest_table_dir_when_sidecar_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            html_text = root / "PAPER_SIDE.html.txt"
            sidecar = root / "stage1_table_cells_v1.jsonl"
            curated_table_dir = root / "curated_tables" / "PAPER_SIDE"
            out_root = root / "out"
            curated_table_dir.mkdir(parents=True)
            (curated_table_dir / "PAPER_SIDE__table_01__html_table.csv").write_text("Run,EE\nF1,80\n", encoding="utf-8")
            html_text.write_text("html text", encoding="utf-8")
            sidecar.write_text(
                "\n".join(
                    [
                        json.dumps({"paper_key": "PAPER_SIDE", "table_id": "t1", "row_index": "1", "col_index": "1", "raw_cell_text": "Run"}),
                        json.dumps({"paper_key": "PAPER_SIDE", "table_id": "t1", "row_index": "2", "col_index": "1", "raw_cell_text": "F1"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            rec = unified.process_one(
                "PAPER_SIDE",
                {
                    "key": "PAPER_SIDE",
                    "html": "paper.html",
                    "table_dir": str(curated_table_dir),
                    "table_available": "yes",
                },
                {"PAPER_SIDE": str(html_text)},
                {},
                {},
                {
                    "PAPER_SIDE": {
                        "stage1_table_cell_sidecar_path": str(sidecar),
                        "stage1_table_cell_sidecar_available": "yes",
                    }
                },
                out_root,
            )

            self.assertEqual(Path(rec["table_dir"]).resolve(), curated_table_dir.resolve())
            self.assertEqual(rec["table_available"], "yes")
            self.assertEqual(rec["stage1_table_cell_sidecar_path"], str(sidecar))
            self.assertEqual(rec["stage1_table_cell_sidecar_available"], "yes")

    def test_inherits_explicit_upstream_table_authority_only_when_manifest_lacks_it(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            current_text = root / "PAPER_AUTH.pdf.txt"
            inherited_table_dir = root / "run18" / "PAPER_AUTH" / "tables"
            inherited_sidecar = root / "run18" / "PAPER_AUTH" / "stage1_table_cells_v1.jsonl"
            out_root = root / "out"
            inherited_table_dir.mkdir(parents=True)
            inherited_sidecar.parent.mkdir(parents=True, exist_ok=True)
            current_text.write_text("current text", encoding="utf-8")
            (inherited_table_dir / "PAPER_AUTH__sidecar_table_01.csv").write_text("Run,EE\nF1,80\n", encoding="utf-8")
            inherited_sidecar.write_text(
                json.dumps(
                    {
                        "paper_key": "PAPER_AUTH",
                        "table_id": "t1",
                        "row_index": "1",
                        "col_index": "1",
                        "raw_cell_text": "Run",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            authority_row = {
                "paper_key": "PAPER_AUTH",
                "table_dir": str(inherited_table_dir),
                "table_available": "yes",
                "stage1_table_cell_sidecar_path": str(inherited_sidecar),
                "stage1_table_cell_sidecar_available": "yes",
                "_table_authority_manifest_source": "data/results/campaign/18/stage1_manifest.tsv",
            }
            merged = unified.merge_table_authority_row(
                {
                    "key": "PAPER_AUTH",
                    "pdf": "paper.pdf",
                    "table_dir": "",
                    "stage1_table_cell_sidecar_path": "",
                },
                authority_row,
            )

            rec = unified.process_one(
                "PAPER_AUTH",
                merged,
                {"PAPER_AUTH": str(current_text)},
                {},
                {},
                {},
                out_root,
            )

            self.assertEqual(Path(rec["table_dir"]).resolve(), inherited_table_dir.resolve())
            self.assertEqual(rec["table_available"], "yes")
            self.assertEqual(Path(rec["stage1_table_cell_sidecar_path"]).resolve(), inherited_sidecar.resolve())
            self.assertEqual(rec["stage1_table_cell_sidecar_available"], "yes")
            self.assertEqual(rec["table_authority_inherited"], "yes")
            self.assertEqual(
                rec["table_authority_inheritance_source"],
                "data/results/campaign/18/stage1_manifest.tsv",
            )

    def test_html_current_can_append_richer_same_key_pdf_clean_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            html_text = root / "PAPER3.html.txt"
            pdf_text = root / "PAPER3.pdf.txt"
            out_root = root / "out"
            html_text.write_text("2.4. Nanoparticles preparation\n\n2.5. Characterization\n" * 120, encoding="utf-8")
            pdf_text.write_text(
                (
                    "2.4. Nanoparticles preparation\n"
                    "Batches of PLGA nanoparticles were prepared by a water-in-oil-in-water "
                    "multiple emulsion using 100 mg PLGA, 5 mL dichloromethane, and PVA.\n"
                )
                * 180,
                encoding="utf-8",
            )

            rec = unified.process_one(
                "PAPER3",
                {"key": "PAPER3", "html": "paper.html"},
                {"PAPER3": str(html_text)},
                {},
                {},
                {},
                out_root,
            )

            unified_text = (out_root / "PAPER3" / "unified_clean_text_v1.md").read_text(encoding="utf-8")
            self.assertEqual(rec["source_type"], "HTML")
            self.assertIn("current_pdf_clean_text_supplement", unified_text)
            self.assertIn("Batches of PLGA nanoparticles were prepared", unified_text)

    def test_html_unified_current_can_append_canonical_pdf_clean_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fake_repo = root / "repo"
            canonical_text_dir = fake_repo / "data" / "cleaned" / "content" / "text"
            canonical_text_dir.mkdir(parents=True)
            nested_html = root / "prior_unified_clean_text_v1.md"
            out_root = root / "out"
            nested_html.write_text(
                "<!-- current_clean_text -->\nTitle\nKeywords\nPLGA microspheres\n",
                encoding="utf-8",
            )
            (canonical_text_dir / "PAPER4.pdf.txt").write_text(
                (
                    "Preparation of PLGA microspheres. The particles were prepared "
                    "by double emulsion using 100 mg PLGA, dichloromethane, and PVA.\n"
                )
                * 160,
                encoding="utf-8",
            )

            original_repo = unified.paths.PROJECT_ROOT
            unified.paths.PROJECT_ROOT = fake_repo
            try:
                rec = unified.process_one(
                    "PAPER4",
                    {"key": "PAPER4", "html": "paper.html", "text_source_type": "html"},
                    {"PAPER4": str(nested_html)},
                    {},
                    {},
                    {},
                    out_root,
                )
            finally:
                unified.paths.PROJECT_ROOT = original_repo

            unified_text = (out_root / "PAPER4" / "unified_clean_text_v1.md").read_text(encoding="utf-8")
            self.assertEqual(rec["source_type"], "HTML")
            self.assertIn("current_pdf_clean_text_supplement", unified_text)
            self.assertIn("Preparation of PLGA microspheres", unified_text)


if __name__ == "__main__":
    unittest.main()
