import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.stage1_cleaning import hydrate_manifest_v1 as hydrate


class TestHydrateManifestV1(unittest.TestCase):
    def _write_tsv(self, path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_hydrates_structure_and_table_cell_sidecar_bindings(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "manifest_current.tsv"
            key2txt = root / "key2txt.tsv"
            key2structure = root / "key2structure.tsv"
            cell_manifest = root / "stage1_table_cells_manifest_v1.tsv"
            out = root / "hydrated.tsv"
            metadata = root / "hydration_metadata.json"

            self._write_tsv(
                manifest,
                ["key", "title"],
                [
                    {"key": "AAA111", "title": "HTML paper"},
                    {"key": "BBB222", "title": "PDF paper"},
                    {"key": "CCC333", "title": "missing sidecars"},
                ],
            )
            self._write_tsv(
                key2txt,
                ["key", "txt_path"],
                [
                    {"key": "AAA111", "txt_path": "data/cleaned/content/text/AAA111.html.txt"},
                    {"key": "BBB222", "txt_path": "data/cleaned/content/text/BBB222.pdf.txt"},
                ],
            )
            self._write_tsv(
                key2structure,
                ["key", "txt_path", "structure_path", "tables_dir", "parser"],
                [
                    {
                        "key": "AAA111",
                        "txt_path": "data/cleaned/content/text/AAA111.html.txt",
                        "structure_path": "data/cleaned/content/structure/AAA111.html.json",
                        "tables_dir": "data/cleaned/goren_2025/tables/AAA111",
                        "parser": "beautifulsoup_fallback",
                    },
                    {
                        "key": "BBB222",
                        "txt_path": "data/cleaned/content/text/BBB222.pdf.txt",
                        "structure_path": "data/cleaned/content/structure/BBB222.pdf.json",
                        "tables_dir": "data/cleaned/goren_2025/tables/BBB222",
                        "parser": "marker",
                    },
                ],
            )
            self._write_tsv(
                cell_manifest,
                [
                    "paper_key",
                    "stage1_table_cell_sidecar_path",
                    "stage1_table_cell_sidecar_available",
                    "source_type",
                    "parsers",
                ],
                [
                    {
                        "paper_key": "AAA111",
                        "stage1_table_cell_sidecar_path": "data/cleaned/content/tables_cell_sidecar/AAA111/stage1_table_cells_v1.jsonl",
                        "stage1_table_cell_sidecar_available": "yes",
                        "source_type": "html",
                        "parsers": "current",
                    },
                    {
                        "paper_key": "BBB222",
                        "stage1_table_cell_sidecar_path": "data/cleaned/content/tables_cell_sidecar/BBB222/stage1_table_cells_v1.jsonl",
                        "stage1_table_cell_sidecar_available": "yes",
                        "source_type": "pdf",
                        "parsers": "marker",
                    },
                ],
            )

            hydrate.hydrate_manifest(
                manifest_tsv=manifest,
                out_tsv=out,
                key2txt_tsv=key2txt,
                key2structure_tsv=key2structure,
                table_cell_sidecar_manifest_tsv=cell_manifest,
                metadata_json=metadata,
            )

            with out.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                self.assertIn("structure_path", reader.fieldnames or [])
                self.assertIn("structure_available", reader.fieldnames or [])
                self.assertIn("stage1_table_cell_sidecar_path", reader.fieldnames or [])
                self.assertIn("stage1_table_cell_sidecar_available", reader.fieldnames or [])
                rows = {row["key"]: row for row in reader}

            self.assertEqual(rows["AAA111"]["structure_path"], "data/cleaned/content/structure/AAA111.html.json")
            self.assertEqual(rows["AAA111"]["structure_available"], "yes")
            self.assertEqual(
                rows["AAA111"]["stage1_table_cell_sidecar_path"],
                "data/cleaned/content/tables_cell_sidecar/AAA111/stage1_table_cells_v1.jsonl",
            )
            self.assertEqual(rows["AAA111"]["stage1_table_cell_sidecar_available"], "yes")
            self.assertEqual(rows["BBB222"]["structure_available"], "yes")
            self.assertEqual(rows["BBB222"]["stage1_table_cell_sidecar_available"], "yes")
            self.assertEqual(rows["CCC333"]["structure_available"], "no")
            self.assertEqual(rows["CCC333"]["stage1_table_cell_sidecar_available"], "no")

            metadata_obj = json.loads(metadata.read_text(encoding="utf-8"))
            self.assertEqual(metadata_obj["structure_available_yes_count"], 2)
            self.assertEqual(metadata_obj["stage1_table_cell_sidecar_available_yes_count"], 2)
            self.assertEqual(metadata_obj["mainline_integration_status"], "maintained_code_only_not_hydrated")

    def test_preserves_existing_dataset_table_and_split_fields_without_overlay_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "manifest_current.tsv"
            key2txt = root / "key2txt.tsv"
            out = root / "hydrated.tsv"

            self._write_tsv(
                manifest,
                [
                    "key",
                    "dataset_id",
                    "table_dir",
                    "table_available",
                    "split_tag",
                    "benchmark_tag",
                ],
                [
                    {
                        "key": "KEEP001",
                        "dataset_id": "goren_2025",
                        "table_dir": "data\\cleaned\\goren_2025\\tables\\KEEP001",
                        "table_available": "yes",
                        "split_tag": "dev_manifest_remaining12_2026-03-10",
                        "benchmark_tag": "DEV15",
                    }
                ],
            )
            self._write_tsv(
                key2txt,
                ["key", "txt_path"],
                [{"key": "KEEP001", "txt_path": "data/cleaned/content/text/KEEP001.html.txt"}],
            )

            hydrate.hydrate_manifest(
                manifest_tsv=manifest,
                out_tsv=out,
                key2txt_tsv=key2txt,
                key2structure_tsv=None,
                table_cell_sidecar_manifest_tsv=None,
            )

            with out.open("r", encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(row["dataset_id"], "goren_2025")
            self.assertEqual(row["table_dir"], "data\\cleaned\\goren_2025\\tables\\KEEP001")
            self.assertEqual(row["table_available"], "yes")
            self.assertEqual(row["split_tag"], "dev_manifest_remaining12_2026-03-10")
            self.assertEqual(row["benchmark_tag"], "DEV15")


if __name__ == "__main__":
    unittest.main()
