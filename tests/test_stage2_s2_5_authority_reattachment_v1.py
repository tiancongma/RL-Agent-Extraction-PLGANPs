import tempfile
import unittest
from pathlib import Path

from src.stage2_sampling_labels.run_stage2_s2_5_semantic_parsing_v1 import (
    resolve_authority_reattachment_entry,
)


class Stage2S25AuthorityReattachmentTest(unittest.TestCase):
    def test_explicit_authority_override_resolves_combined_raw_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_responses_dir = tmp_path / "combined" / "raw_responses"
            raw_responses_dir.mkdir(parents=True)
            authority_payload_root = tmp_path / "s2_2" / "semantic_stage2_objects" / "normalized_table_payloads"
            payload_dir = authority_payload_root / "PAPER1"
            payload_dir.mkdir(parents=True)
            (payload_dir / "normalized_table_payloads_v1.json").write_text(
                '{"normalized_table_payloads":[{"table_id":"Table 1","source_table_asset_id":"PAPER1__sidecar_table_01"}]}',
                encoding="utf-8",
            )

            entry = resolve_authority_reattachment_entry(
                raw_responses_dir=raw_responses_dir,
                paper_key="PAPER1",
                prompt_cache={},
                explicit_authority_run_dir=str(tmp_path / "s2_2"),
                explicit_authority_payload_root=str(authority_payload_root),
            )

        self.assertEqual("resolved", entry["resolution_status"])
        self.assertEqual("explicit_cli_authority_override", entry["resolution_source"])
        self.assertEqual("", entry["failure_reason"])
        self.assertEqual(1, entry["locator_count"])
        self.assertEqual("Table 1", entry["table_scope_locators"][0]["table_id"])


if __name__ == "__main__":
    unittest.main()
