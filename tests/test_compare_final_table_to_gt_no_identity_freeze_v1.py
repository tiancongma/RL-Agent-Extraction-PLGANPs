from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.stage5_benchmark import compare_final_table_to_gt_v1 as compare_gt


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class CompareFinalTableNoIdentityFreezeTests(unittest.TestCase):
    def test_main_runs_without_identity_freeze_summary_and_omits_freeze_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            out_dir = root / "out"
            final_table = run_dir / "final_formulation_table_v1.tsv"
            gt_counts = root / "gt_counts.tsv"
            scope_manifest = run_dir / "dev15_scope.tsv"

            write_tsv(
                final_table,
                ["key", "formulation_id"],
                [{"key": "PAPER_A", "formulation_id": "F1"}],
            )
            write_tsv(
                gt_counts,
                ["paper_key", "gt_count"],
                [{"paper_key": "PAPER_A", "gt_count": "1"}],
            )
            write_tsv(
                scope_manifest,
                ["key", "doi", "title"],
                [{"key": "PAPER_A", "doi": "10.example/a", "title": "Paper A"}],
            )

            self.assertFalse(
                (run_dir / "audit" / "identity_freeze_guardrail_v1" / "identity_freeze_summary_v1.tsv").exists()
            )

            argv = [
                "compare_final_table_to_gt_v1.py",
                "--run-dir",
                str(run_dir),
                "--final-table-tsv",
                str(final_table),
                "--gt-counts-tsv",
                str(gt_counts),
                "--scope-manifest-tsv",
                str(scope_manifest),
                "--out-dir",
                str(out_dir),
            ]
            def _resolve_artifact_path(*, explicit_path=None, **_kwargs):
                if explicit_path is not None:
                    return explicit_path.resolve()
                raise AssertionError("test supplies explicit artifact paths")

            with (
                patch("sys.argv", argv),
                patch.object(compare_gt, "resolve_artifact_path", side_effect=_resolve_artifact_path),
                patch.object(compare_gt.subprocess, "run") as mock_run,
            ):
                compare_gt.main()

            mock_run.assert_called_once()
            counts_rows = read_tsv(out_dir / compare_gt.COUNTS_NAME)
            self.assertEqual(len(counts_rows), 1)
            self.assertEqual(counts_rows[0]["paper_key"], "PAPER_A")
            self.assertEqual(counts_rows[0]["comparison_status"], "match")

            counts_header = (out_dir / compare_gt.COUNTS_NAME).read_text(encoding="utf-8").splitlines()[0].split("\t")
            audit_header = (out_dir / compare_gt.AUDIT_COUNTS_NAME).read_text(encoding="utf-8").splitlines()[0].split("\t")
            summary = (out_dir / compare_gt.SUMMARY_NAME).read_text(encoding="utf-8")
            run_context = (out_dir / "RUN_CONTEXT.md").read_text(encoding="utf-8")
            metadata = (out_dir / f"{compare_gt.COUNTS_NAME}.metadata.json").read_text(encoding="utf-8")

            forbidden = {
                "identity_freeze_failed",
                "identity_freeze_mode",
                "identity_freeze_summary_tsv",
                "freeze_failed=",
                "--identity-freeze-mode",
            }
            self.assertTrue(forbidden.isdisjoint(counts_header))
            self.assertTrue(forbidden.isdisjoint(audit_header))
            for text in (summary, run_context, metadata):
                for token in forbidden:
                    self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
