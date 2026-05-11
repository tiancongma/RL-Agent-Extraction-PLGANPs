import json
import tempfile
import unittest
from pathlib import Path

from src.stage2_sampling_labels.audit_preparation_evidence_sufficiency_v1 import audit_one_paper


class PreparationEvidenceSufficiencyAuditTests(unittest.TestCase):
    def _write_pair(self, tmp: Path, *, clean_text: str, candidates, evidence):
        clean_path = tmp / "cleaned.txt"
        clean_path.write_text(clean_text, encoding="utf-8")
        candidate_path = tmp / "candidate_blocks_v1.json"
        evidence_path = tmp / "evidence_blocks_v1.json"
        candidate_path.write_text(
            json.dumps(
                {
                    "paper_key": "TESTPAPER",
                    "source_clean_text_path": str(clean_path),
                    "candidate_blocks": candidates,
                }
            ),
            encoding="utf-8",
        )
        evidence_path.write_text(
            json.dumps(
                {
                    "paper_key": "TESTPAPER",
                    "source_clean_text_path": str(clean_path),
                    "evidence_blocks": evidence,
                }
            ),
            encoding="utf-8",
        )
        return candidate_path, evidence_path

    def test_candidate_core_missing_is_candidate_segmentation_boundary(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            clean_text = (
                "2.2 Methods\nNanoparticles were prepared by dissolving PLGA 50 mg "
                "and drug 5 mg in acetone; the organic phase was added dropwise "
                "to 4 mL aqueous PVA under stirring."
            )
            c, e = self._write_pair(tmp, clean_text=clean_text, candidates=[], evidence=[])
            row = audit_one_paper(candidate_path=c, evidence_path=e, repo_root=tmp)
            self.assertEqual(row["first_failure_boundary"], "candidate_segmentation_missing_preparation_core")

    def test_clean_text_missing_core_is_source_quality_boundary(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            c, e = self._write_pair(
                tmp,
                clean_text="Nanoparticles were characterized for size and zeta potential only.",
                candidates=[],
                evidence=[],
            )
            row = audit_one_paper(candidate_path=c, evidence_path=e, repo_root=tmp)
            self.assertEqual(row["first_failure_boundary"], "cleaned_text_missing_preparation_core")

    def test_table_noise_overselected_boundary(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            core = (
                "Nanoparticles were prepared by dissolving PLGA 50 mg and drug 5 mg "
                "in acetone, followed by addition to 4 mL aqueous phase."
            )
            block = {"candidate_id": "m1", "block_type": "method", "text_content": core}
            table = {
                "block_id": "t1",
                "block_type": "table",
                "text_content": "Tissue distribution AUC Cmax in rats and organ distribution values.",
            }
            c, e = self._write_pair(tmp, clean_text=core, candidates=[block], evidence=[block, table])
            row = audit_one_paper(candidate_path=c, evidence_path=e, repo_root=tmp)
            self.assertEqual(row["first_failure_boundary"], "table_selector_noise_overselected")

    def test_toc_and_table_captions_do_not_satisfy_cleaned_text_preparation_body(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            navigation_only = (
                "Outline\nAbstract\n1. Introduction\n2. Experimental\n2.1 Materials\n"
                "2.2 Methods\n2.2.1 Production of PLGA nanospheres\n"
                "Table 1 . Initial 2 3 full factorial design, providing upper and lower level values; "
                "cFB, concentration of flurbiprofen (mg/mL); cP188, concentration of poloxamer 188 (15 mg/mL).\n"
                "3.4 Long-term physical stability\n"
                "Fig. 7 . Profiles of nanospheres produced with PLGA as a function of time in aqueous phase.\n"
                "References\n[2] Nanocapsule formation by interfacial polymer deposition following solvent displacement."
            )
            c, e = self._write_pair(tmp, clean_text=navigation_only, candidates=[], evidence=[])
            row = audit_one_paper(candidate_path=c, evidence_path=e, repo_root=tmp)
            self.assertEqual(row["cleaned_text_has_preparation_core"], "no")
            self.assertEqual(row["source_quality_status"], "cleaned_text_missing_method_body")
            self.assertEqual(row["first_failure_boundary"], "cleaned_text_missing_method_body")


if __name__ == "__main__":
    unittest.main()
