from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.stage5_benchmark import validate_evidence_binding_contract_v1 as validator


class EvidenceBindingContractValidatorTests(unittest.TestCase):
    def test_reads_existing_layer3_golden_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed = root / "seed.tsv"
            seed.write_text(
                "paper_key\tformulation_id\tfield_name\textracted_value\tevidence_text\tevidence_anchor_text\tevidence_status_detail\trelation_resolution_rule\treview_warning\tevidence_source_type\n"
                "P1\tF1\tdrug_mass_mg\t5\tdirect text\tanchor\t\t\t\ttext\n",
                encoding="utf-8",
            )
            golden = root / "golden.tsv"
            golden.write_text(
                "case_id\tpaper_key\tformulation_id\tfield_name\texpected_extracted_value_state\texpected_evidence_anchor_state\trequire_not_supported\n"
                "direct_case\tP1\tF1\tdrug_mass_mg\tnonempty\tnonempty\tfalse\n",
                encoding="utf-8",
            )
            result = validator.run_validation(seed_tsv=seed, layer3_golden_cases_tsv=golden)
            self.assertEqual(result["failed"], 0)
            self.assertEqual(result["layer3_passed"], 1)

    def test_validates_binding_pack_status_and_assignment_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed = root / "seed.tsv"
            seed.write_text("paper_key\tformulation_id\tfield_name\n", encoding="utf-8")
            old_golden = root / "old.tsv"
            old_golden.write_text("case_id\tpaper_key\tformulation_id\tfield_name\n", encoding="utf-8")
            pack = root / "packs.jsonl"
            pack.write_text(json.dumps({
                "paper_key": "P1",
                "final_formulation_id": "F1",
                "field_name": "drug_mass_mg",
                "binding_status": "value_only_match",
                "assignment_path": "unresolved",
            }) + "\n", encoding="utf-8")
            binding_golden = root / "binding.tsv"
            binding_golden.write_text(
                "case_id\tpaper_key\tfinal_formulation_id\tfield_name\texpected_binding_status\texpected_assignment_path\n"
                "value_only_stays_unsupported\tP1\tF1\tdrug_mass_mg\tvalue_only_match\tunresolved\n",
                encoding="utf-8",
            )
            result = validator.run_validation(
                seed_tsv=seed,
                layer3_golden_cases_tsv=old_golden,
                evidence_binding_packs_jsonl=pack,
                evidence_binding_golden_cases_tsv=binding_golden,
            )
            self.assertEqual(result["failed"], 0)
            self.assertEqual(result["binding_passed"], 1)


if __name__ == "__main__":
    unittest.main()
