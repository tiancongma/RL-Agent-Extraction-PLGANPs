from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.stage5_benchmark import build_field_gt_review_workbook_v1 as workbook


class WorkbookEvidenceBindingModeTests(unittest.TestCase):
    def test_default_mode_requires_pack_and_risk_paths(self):
        args = workbook.build_arg_parser().parse_args(["--out-subdir", "x"])
        with self.assertRaisesRegex(ValueError, "evidence_binding_packs"):
            workbook.validate_evidence_binding_workbook_mode(args)

    def test_legacy_mode_allows_missing_pack_and_risk(self):
        args = workbook.build_arg_parser().parse_args(["--out-subdir", "x", "--legacy-evidence-mode"])
        mode = workbook.validate_evidence_binding_workbook_mode(args)
        self.assertEqual(mode["evidence_mode"], "legacy")
        self.assertEqual(mode["evidence_binding_pack_path"], "")

    def test_pack_mode_metadata_records_pack_risk_and_final_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = root / "packs.jsonl"
            risk = root / "risk.tsv"
            final = root / "final.tsv"
            for path in (pack, risk, final):
                path.write_text("ok\n", encoding="utf-8")
            args = workbook.build_arg_parser().parse_args([
                "--out-subdir", "x",
                "--evidence-binding-packs-jsonl", str(pack),
                "--evidence-binding-risk-tsv", str(risk),
            ])
            mode = workbook.validate_evidence_binding_workbook_mode(args, final_table_tsv=final)
            self.assertEqual(mode["evidence_mode"], "evidence_binding_pack")
            self.assertEqual(mode["evidence_binding_pack_path"], str(pack.resolve()))
            self.assertEqual(mode["evidence_binding_risk_path"], str(risk.resolve()))
            self.assertEqual(mode["authority_resolved_final_table_path"], str(final.resolve()))


if __name__ == "__main__":
    unittest.main()
