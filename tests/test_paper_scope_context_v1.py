from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.stage2_sampling_labels.build_paper_scope_context_v1 import build_paper_scope_contexts


class PaperScopeContextV1Tests(unittest.TestCase):
    def _write_tsv(self, path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_writes_per_paper_scope_context_without_evidence_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text_path = root / "cleaned" / "text" / "AAA111.pdf.txt"
            pdf_path = root / "raw" / "AAA111.pdf"
            table_root = root / "cleaned" / "tables" / "AAA111"
            table_manifest = table_root / "tables_manifest.json"
            text_path.parent.mkdir(parents=True)
            text_path.write_text("clean text", encoding="utf-8")
            pdf_path.parent.mkdir(parents=True)
            pdf_path.write_bytes(b"%PDF")
            table_root.mkdir(parents=True)
            table_manifest.write_text("{}\n", encoding="utf-8")

            manifest = root / "manifest_current.tsv"
            self._write_tsv(
                manifest,
                [
                    "paper_key",
                    "title",
                    "doi",
                    "text_path",
                    "text_source_type",
                    "pdf",
                    "html",
                    "table_dir",
                    "table_asset_refs",
                    "dataset_id",
                    "split_tag",
                    "benchmark_tag",
                    "source_collection",
                    "raw_source_path",
                ],
                [
                    {
                        "paper_key": "AAA111",
                        "title": "A test PLGA paper",
                        "doi": "10.example/a",
                        "text_path": str(text_path),
                        "text_source_type": "pdf",
                        "pdf": str(pdf_path),
                        "html": "",
                        "table_dir": str(table_root),
                        "table_asset_refs": str(table_manifest),
                        "dataset_id": "dev15",
                        "split_tag": "dev",
                        "benchmark_tag": "DEV15",
                        "source_collection": "zotero_a",
                        "raw_source_path": "data/raw/zotero/a.jsonl",
                    }
                ],
            )
            scope = root / "run_input_scope_v1.tsv"
            self._write_tsv(
                scope,
                ["key", "title", "doi", "text_path", "selection_reason"],
                [
                    {
                        "key": "AAA111",
                        "title": "A test PLGA paper",
                        "doi": "10.example/a",
                        "text_path": str(text_path),
                        "selection_reason": "unit_scope",
                    }
                ],
            )

            summary = build_paper_scope_contexts(
                canonical_manifest=manifest,
                scope_tsv=scope,
                out_dir=root / "run",
                run_id="unit_s2_1",
            )

            self.assertTrue(summary["diagnostic_only"])
            self.assertFalse(summary["benchmark_valid"])
            self.assertEqual(summary["paper_count"], 1)
            context_path = root / "run" / "semantic_stage2_objects" / "scope_context" / "AAA111" / "paper_scope_context_v1.json"
            self.assertTrue(context_path.exists())
            context = json.loads(context_path.read_text(encoding="utf-8"))
            self.assertEqual(context["schema"], "paper_scope_context_v1")
            self.assertEqual(context["paper_key"], "AAA111")
            self.assertEqual(context["canonical_manifest"]["authority_role"], "single_canonical_manifest")
            self.assertEqual(context["run_scope"]["selection_reason"], "unit_scope")
            self.assertTrue(context["clean_text"]["exists"])
            self.assertEqual(context["clean_text"]["source_type"], "pdf")
            self.assertTrue(context["source_assets"]["pdf_exists"])
            self.assertTrue(context["table_assets"]["table_asset_root_exists"])
            self.assertEqual(context["table_assets"]["availability_status"], "available")
            self.assertFalse(context["governance"]["does_rank_or_select_evidence"])
            self.assertFalse(context["governance"]["live_llm_call"])
            self.assertTrue(Path(summary["audit_tsv"]).exists())
            self.assertTrue(Path(summary["run_context_md"]).exists())

    def test_records_explicit_table_absence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text_path = root / "cleaned" / "text" / "BBB222.html.txt"
            text_path.parent.mkdir(parents=True)
            text_path.write_text("clean text", encoding="utf-8")
            manifest = root / "manifest_current.tsv"
            self._write_tsv(
                manifest,
                ["paper_key", "text_path", "text_source_type", "table_dir"],
                [{"paper_key": "BBB222", "text_path": str(text_path), "text_source_type": "html", "table_dir": ""}],
            )
            scope = root / "scope.tsv"
            self._write_tsv(scope, ["paper_key", "text_path", "selection_reason"], [{"paper_key": "BBB222", "text_path": str(text_path), "selection_reason": "unit_scope"}])

            build_paper_scope_contexts(canonical_manifest=manifest, scope_tsv=scope, out_dir=root / "run", run_id="unit_s2_1")
            context_path = root / "run" / "semantic_stage2_objects" / "scope_context" / "BBB222" / "paper_scope_context_v1.json"
            context = json.loads(context_path.read_text(encoding="utf-8"))
            self.assertEqual(context["table_assets"]["availability_status"], "explicit_absent")
            self.assertFalse(context["table_assets"]["table_asset_root_exists"])


if __name__ == "__main__":
    unittest.main()
