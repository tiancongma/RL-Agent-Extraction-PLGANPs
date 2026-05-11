import json
import tempfile
import unittest
from pathlib import Path
from typing import Optional

from src.stage5_benchmark import resolve_evidence_binding_authority_v1 as authority


class EvidenceBindingAuthorityResolutionTests(unittest.TestCase):
    def make_pointer(self, root: Path, terminal_files: dict[str, str], extra: Optional[dict[str, str]] = None) -> Path:
        run_dir = root / "data" / "results" / "run_test"
        run_dir.mkdir(parents=True)
        for value in set(terminal_files.values()) | set((extra or {}).values()):
            if value:
                target = root / value
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("stub\n", encoding="utf-8")
        payload = {
            "active_run_id": "run_test",
            "active_run_dir": str(run_dir),
            "authoritative_terminal_files": terminal_files,
            "lineage_policy": "test",
            "updated_at": "2026-05-05T00:00:00Z",
            "note": "test pointer",
        }
        if extra:
            payload.update(extra)
        pointer = root / "ACTIVE_RUN.json"
        pointer.write_text(json.dumps(payload), encoding="utf-8")
        return pointer

    def test_resolves_canonical_manifest_without_alias_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pointer = self.make_pointer(
                root,
                {
                    "stage5_final_table_tsv": "artifacts/final.tsv",
                    "stage5_decision_trace_tsv": "artifacts/decision.tsv",
                },
                {"active_final_table_tsv": "artifacts/final.tsv"},
            )
            manifest = authority.resolve_authority_manifest(
                pointer_path=pointer,
                semantic_specs=[authority.SEMANTIC_ARTIFACT_SPECS["frozen_final_table"]],
                authority_overrides={},
                require_exists=True,
            )
            row = manifest["artifacts"][0]
            self.assertEqual(row["semantic_name"], "frozen_final_table")
            self.assertEqual(row["selected_key"], "stage5_final_table_tsv")
            self.assertEqual(row["status"], "resolved")
            self.assertEqual(row["alias_conflict"], "no")

    def test_conflicting_aliases_fail_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pointer = self.make_pointer(
                root,
                {
                    "stage5_final_table_tsv": "artifacts/final_a.tsv",
                    "stage5_final_table": "artifacts/final_b.tsv",
                },
            )
            with self.assertRaisesRegex(authority.AuthorityResolutionError, "authority_alias_conflict"):
                authority.resolve_authority_manifest(
                    pointer_path=pointer,
                    semantic_specs=[authority.SEMANTIC_ARTIFACT_SPECS["frozen_final_table"]],
                    authority_overrides={},
                    require_exists=True,
                )

    def test_explicit_authority_field_override_selects_path_and_records_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pointer = self.make_pointer(
                root,
                {
                    "stage5_final_table_tsv": "artifacts/final_a.tsv",
                    "stage5_final_table": "artifacts/final_b.tsv",
                },
            )
            manifest = authority.resolve_authority_manifest(
                pointer_path=pointer,
                semantic_specs=[authority.SEMANTIC_ARTIFACT_SPECS["frozen_final_table"]],
                authority_overrides={"frozen_final_table": "stage5_final_table"},
                require_exists=True,
            )
            row = manifest["artifacts"][0]
            self.assertEqual(row["selected_key"], "stage5_final_table")
            self.assertEqual(row["alias_conflict"], "yes")
            self.assertEqual(row["status"], "resolved_with_explicit_override")


if __name__ == "__main__":
    unittest.main()
