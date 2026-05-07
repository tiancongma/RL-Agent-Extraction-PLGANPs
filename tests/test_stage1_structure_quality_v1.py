import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.stage1_cleaning import score_stage1_structure_quality_v1 as scorer


class TestStage1StructureQualityV1(unittest.TestCase):
    def test_scores_multirrow_header_blank_colspan_caption_and_numeric_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bakeoff = root / "bakeoff"
            out_dir = root / "quality"
            bakeoff.mkdir()
            (bakeoff / "parser_bakeoff_summary_v1.tsv").write_text(
                "paper_key\tparser\tparser_variant\tstatus\tsource_type\tsource_path\ttext_chars\tblock_count\ttable_count\tcell_count\twarning_count\tbenchmark_valid\tnotes\n"
                "PAPER001\tcurrent\tdiagnostic_bakeoff_v1\tok\tHTML\tsource.html\t62\t2\t1\t7\t0\tno\t\n",
                encoding="utf-8",
            )
            blocks = [
                {"paper_key": "PAPER001", "parser": "current", "type": "paragraph", "text": "Alpha 123.45 beta"},
                {"paper_key": "PAPER001", "parser": "current", "type": "table", "text": "Table text"},
            ]
            (bakeoff / "parser_bakeoff_blocks_v1.jsonl").write_text(
                "\n".join(json.dumps(r) for r in blocks) + "\n",
                encoding="utf-8",
            )
            cells = [
                {"paper_key": "PAPER001", "parser": "current", "table_id": "t001", "caption": "Table 1", "row_index": 1, "col_index": 1, "rowspan": 2, "colspan": 1, "raw_cell_text": "Formulation", "normalized_cell_text": "Formulation", "is_header_cell": "yes", "header_path_json": "[\"Formulation\"]"},
                {"paper_key": "PAPER001", "parser": "current", "table_id": "t001", "caption": "Table 1", "row_index": 1, "col_index": 2, "rowspan": 1, "colspan": 2, "raw_cell_text": "Metrics", "normalized_cell_text": "Metrics", "is_header_cell": "yes", "header_path_json": "[\"Metrics\"]"},
                {"paper_key": "PAPER001", "parser": "current", "table_id": "t001", "caption": "Table 1", "row_index": 2, "col_index": 2, "rowspan": 1, "colspan": 1, "raw_cell_text": "Size", "normalized_cell_text": "Size", "is_header_cell": "yes", "header_path_json": "[\"Metrics\", \"Size\"]"},
                {"paper_key": "PAPER001", "parser": "current", "table_id": "t001", "caption": "Table 1", "row_index": 2, "col_index": 3, "rowspan": 1, "colspan": 1, "raw_cell_text": "PDI", "normalized_cell_text": "PDI", "is_header_cell": "yes", "header_path_json": "[\"Metrics\", \"PDI\"]"},
                {"paper_key": "PAPER001", "parser": "current", "table_id": "t001", "caption": "Table 1", "row_index": 3, "col_index": 1, "rowspan": 1, "colspan": 1, "raw_cell_text": "F1", "normalized_cell_text": "F1", "is_header_cell": "no", "header_path_json": "[]"},
                {"paper_key": "PAPER001", "parser": "current", "table_id": "t001", "caption": "Table 1", "row_index": 3, "col_index": 2, "rowspan": 1, "colspan": 1, "raw_cell_text": "", "normalized_cell_text": "", "is_header_cell": "no", "header_path_json": "[]"},
                {"paper_key": "PAPER001", "parser": "current", "table_id": "t001", "caption": "Table 1", "row_index": 3, "col_index": 3, "rowspan": 1, "colspan": 1, "raw_cell_text": "0.12", "normalized_cell_text": "0.12", "is_header_cell": "no", "header_path_json": "[]"},
            ]
            (bakeoff / "parser_bakeoff_cells_v1.jsonl").write_text(
                "\n".join(json.dumps(r) for r in cells) + "\n",
                encoding="utf-8",
            )
            (bakeoff / "parser_bakeoff_warnings_v1.tsv").write_text(
                "paper_key\tparser\tstatus\twarning_code\twarning_detail\n",
                encoding="utf-8",
            )
            anchors = root / "anchors.tsv"
            anchors.write_text(
                "paper_key\tanchor_id\tfragment\n"
                "PAPER001\texact\tAlpha 123.45 beta\n"
                "PAPER001\tnumeric_only\tmissing words 123.45\n"
                "PAPER001\tabsent\tNot present 999\n",
                encoding="utf-8",
            )

            result = scorer.run_structure_quality_scoring(
                bakeoff_dir=bakeoff,
                out_dir=out_dir,
                anchor_fragments_tsv=anchors,
            )

            self.assertEqual(result["status"], "completed")
            with (out_dir / "stage1_structure_quality_by_paper_v1.tsv").open(encoding="utf-8") as f:
                paper_rows = list(csv.DictReader(f, delimiter="\t"))
            self.assertEqual(len(paper_rows), 1)
            row = paper_rows[0]
            self.assertEqual(row["paper_key"], "PAPER001")
            self.assertEqual(row["table_count"], "1")
            self.assertEqual(row["cell_count"], "7")
            self.assertEqual(row["blank_cell_count"], "1")
            self.assertEqual(row["caption_bound_table_count"], "1")
            self.assertEqual(row["anchor_exact_hits"], "1")
            self.assertEqual(row["anchor_numeric_fallback_hits"], "1")
            self.assertEqual(row["anchor_missing_fragments"], "1")
            self.assertEqual(row["benchmark_valid"], "no")

            with (out_dir / "stage1_structure_quality_by_table_v1.tsv").open(encoding="utf-8") as f:
                table_rows = list(csv.DictReader(f, delimiter="\t"))
            self.assertEqual(table_rows[0]["max_columns"], "3")
            self.assertEqual(table_rows[0]["multirow_header_detected"], "yes")
            self.assertEqual(table_rows[0]["colspan_cell_count"], "1")
            self.assertEqual(table_rows[0]["rowspan_cell_count"], "1")


if __name__ == "__main__":
    unittest.main()
