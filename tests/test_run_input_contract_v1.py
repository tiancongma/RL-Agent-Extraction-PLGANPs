from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.stage1_cleaning.build_run_input_contract_v1 import build_run_input_contract


class RunInputContractV1Tests(unittest.TestCase):
    def _write_tsv(self, path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_scope_contract_records_selection_without_competing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest_current.tsv"
            out_dir = root / "diagnostic_run"
            fieldnames = [
                "paper_key",
                "title",
                "dataset_id",
                "split_tag",
                "benchmark_tag",
                "text_path",
                "text_source_type",
                "table_dir",
                "table_asset_refs",
                "structure_path",
                "stage1_table_cell_sidecar_path",
                "source_collection",
                "raw_source_path",
            ]
            self._write_tsv(
                manifest,
                fieldnames,
                [
                    {
                        "paper_key": "AAA111",
                        "title": "First source collection paper",
                        "dataset_id": "dev15",
                        "split_tag": "dev",
                        "benchmark_tag": "layer1",
                        "text_path": "data/cleaned/content/text/AAA111.pdf.txt",
                        "text_source_type": "pdf",
                        "table_dir": "data/cleaned/dev15/tables/AAA111",
                        "table_asset_refs": "data/cleaned/dev15/tables/AAA111/tables_manifest.json",
                        "structure_path": "data/cleaned/content/structure/AAA111.structure.json",
                        "stage1_table_cell_sidecar_path": "data/cleaned/content/tables_cell_sidecar/AAA111/stage1_table_cells_v1.jsonl",
                        "source_collection": "zotero_library_a",
                        "raw_source_path": "data/raw/zotero/source_a.jsonl",
                    },
                    {
                        "paper_key": "BBB222",
                        "title": "Second source collection paper",
                        "dataset_id": "dev15",
                        "split_tag": "dev",
                        "benchmark_tag": "layer1",
                        "text_path": "data/cleaned/content/text/BBB222.html.txt",
                        "text_source_type": "html",
                        "table_dir": "data/cleaned/dev15/tables/BBB222",
                        "table_asset_refs": "data/cleaned/dev15/tables/BBB222/tables_manifest.json",
                        "structure_path": "data/cleaned/content/structure/BBB222.structure.json",
                        "stage1_table_cell_sidecar_path": "data/cleaned/content/tables_cell_sidecar/BBB222/stage1_table_cells_v1.jsonl",
                        "source_collection": "zotero_library_b",
                        "raw_source_path": "data/raw/zotero/source_b.jsonl",
                    },
                    {
                        "paper_key": "CCC333",
                        "title": "Held out paper",
                        "dataset_id": "testset",
                        "split_tag": "test",
                        "benchmark_tag": "",
                        "text_path": "data/cleaned/content/text/CCC333.pdf.txt",
                        "text_source_type": "pdf",
                        "table_dir": "",
                        "table_asset_refs": "",
                        "source_collection": "zotero_library_b",
                        "raw_source_path": "data/raw/zotero/source_b.jsonl",
                    },
                ],
            )

            contract = build_run_input_contract(
                canonical_manifest=manifest,
                out_dir=out_dir,
                run_id="unit_diagnostic_run",
                selection_rule_id="dev15_scope_from_canonical_manifest",
                dataset_ids=["dev15"],
                split_tags=["dev"],
                selection_note="unit test diagnostic-only scope",
            )

            contract_path = Path(contract["artifacts"]["run_input_contract_json"])
            scope_path = Path(contract["artifacts"]["run_input_scope_tsv"])
            self.assertTrue(contract_path.exists())
            self.assertTrue(scope_path.exists())
            self.assertFalse((out_dir / "manifest_current.tsv").exists())
            self.assertFalse((out_dir / "manifest.tsv").exists())

            persisted = json.loads(contract_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["schema"], "run_input_contract_v1")
            self.assertTrue(persisted["diagnostic_only"])
            self.assertFalse(persisted["benchmark_valid"])
            self.assertEqual(persisted["canonical_manifest"]["path"], str(manifest))
            self.assertEqual(persisted["canonical_manifest"]["authority_role"], "single_canonical_manifest")
            self.assertFalse(persisted["governance"]["subset_manifest_created"])
            self.assertEqual(persisted["selection"]["selected_paper_keys"], ["AAA111", "BBB222"])
            self.assertIn("selected_source_paths", persisted["selection"])
            self.assertTrue(persisted["selection"]["selected_source_paths"][0]["table_asset_refs"].endswith("tables_manifest.json"))

            with scope_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                self.assertEqual(reader.fieldnames, persisted["scope_schema"]["fields"])
                self.assertIn("table_asset_refs", reader.fieldnames or [])
                self.assertIn("structure_path", reader.fieldnames or [])
                self.assertIn("stage1_table_cell_sidecar_path", reader.fieldnames or [])
                scope_rows = list(reader)
            self.assertEqual([row["paper_key"] for row in scope_rows], ["AAA111", "BBB222"])
            self.assertEqual({row["source_collection"] for row in scope_rows}, {"zotero_library_a", "zotero_library_b"})
            self.assertTrue(all(row["canonical_manifest_path"] == str(manifest) for row in scope_rows))
            self.assertEqual(scope_rows[0]["text_path"], "data/cleaned/content/text/AAA111.pdf.txt")
            self.assertEqual(scope_rows[0]["table_asset_refs"], "data/cleaned/dev15/tables/AAA111/tables_manifest.json")
            self.assertEqual(scope_rows[0]["structure_path"], "data/cleaned/content/structure/AAA111.structure.json")
            self.assertEqual(
                scope_rows[0]["stage1_table_cell_sidecar_path"],
                "data/cleaned/content/tables_cell_sidecar/AAA111/stage1_table_cells_v1.jsonl",
            )
            self.assertEqual(scope_rows[1]["table_asset_root"], "data/cleaned/dev15/tables/BBB222")
            self.assertEqual(scope_rows[1]["text_source_type"], "html")
            self.assertEqual(scope_rows[1]["structure_path"], "data/cleaned/content/structure/BBB222.structure.json")

    def test_requires_explicit_selector(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest_current.tsv"
            self._write_tsv(manifest, ["paper_key"], [{"paper_key": "AAA111"}])
            with self.assertRaises(ValueError):
                build_run_input_contract(
                    canonical_manifest=manifest,
                    out_dir=root / "out",
                    run_id="unit",
                    selection_rule_id="explicit_rule",
                )


if __name__ == "__main__":
    unittest.main()
