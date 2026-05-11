import csv
import tempfile
import unittest
from pathlib import Path

from src.stage2_sampling_labels.run_stage2_composite_v1 import (
    refresh_stage2_structure_bindings,
    refresh_stage2_table_bindings,
    refresh_stage2_text_bindings,
)


class Stage2CompositeManifestBindingRefreshTests(unittest.TestCase):
    def test_refresh_preserves_explicit_unified_manifest_bindings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            explicit_text = root / "stage1_unified_current_marker_v1" / "PAPER1" / "unified_clean_text_v1.md"
            key2txt_text = root / "legacy_current" / "PAPER1.pdf.txt"
            explicit_structure = root / "stage1_unified_current_marker_v1" / "PAPER1" / "unified_structure_v1.json"
            key2structure_structure = root / "legacy_current" / "PAPER1.structure.json"
            explicit_table_dir = root / "stage1_unified_current_marker_v1" / "PAPER1" / "tables"
            inferred_table_dir = root / "legacy_current" / "tables" / "PAPER1"
            explicit_sidecar = root / "stage1_unified_current_marker_v1" / "PAPER1" / "stage1_table_cells_v1.jsonl"
            for path in [explicit_text, key2txt_text, explicit_structure, key2structure_structure, explicit_sidecar]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}" if path.suffix == ".json" else "x", encoding="utf-8")
            explicit_table_dir.mkdir(parents=True, exist_ok=True)
            inferred_table_dir.mkdir(parents=True, exist_ok=True)

            key2txt_path = root / "key2txt.tsv"
            key2txt_path.write_text(f"PAPER1\t{key2txt_text}\n", encoding="utf-8")
            key2structure_path = root / "key2structure.tsv"
            with key2structure_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["key", "structure_path", "tables_dir"], delimiter="\t")
                writer.writeheader()
                writer.writerow({"key": "PAPER1", "structure_path": str(key2structure_structure), "tables_dir": str(inferred_table_dir)})

            rows = [
                {
                    "key": "PAPER1",
                    "dataset_id": "",
                    "text_path": str(explicit_text),
                    "text_source_type": "stage1_unified_current_marker",
                    "text_available": "yes",
                    "structure_path": str(explicit_structure),
                    "table_dir": str(explicit_table_dir),
                    "table_available": "yes",
                    "stage1_table_cell_sidecar_path": str(explicit_sidecar),
                    "stage1_table_cell_sidecar_available": "yes",
                }
            ]

            refreshed = refresh_stage2_text_bindings(rows, key2txt_path)
            refreshed = refresh_stage2_structure_bindings(refreshed, key2structure_path)
            refreshed = refresh_stage2_table_bindings(refreshed)
            row = refreshed[0]

            self.assertEqual(Path(row["text_path"]).resolve(), explicit_text.resolve())
            self.assertEqual(row["text_source_type"], "stage1_unified_current_marker")
            self.assertEqual(row["text_available"], "yes")
            self.assertEqual(Path(row["structure_path"]).resolve(), explicit_structure.resolve())
            self.assertEqual(Path(row["table_dir"]).resolve(), explicit_table_dir.resolve())
            self.assertEqual(row["table_available"], "yes")
            self.assertEqual(row["stage1_table_cell_sidecar_path"], str(explicit_sidecar))
            self.assertEqual(row["stage1_table_cell_sidecar_available"], "yes")

    def test_refresh_marks_missing_explicit_table_dir_without_falling_back(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            text_path = root / "PAPER1.txt"
            text_path.write_text("text", encoding="utf-8")
            missing_table_dir = root / "missing_unified_tables"
            fallback_table_dir = root / "fallback" / "tables" / "PAPER1"
            fallback_table_dir.mkdir(parents=True)
            rows = [{"key": "PAPER1", "text_path": str(text_path), "table_dir": str(missing_table_dir)}]

            refreshed = refresh_stage2_table_bindings(rows)

            self.assertEqual(Path(refreshed[0]["table_dir"]).resolve(), missing_table_dir.resolve())
            self.assertEqual(refreshed[0]["table_available"], "missing_file")


if __name__ == "__main__":
    unittest.main()
