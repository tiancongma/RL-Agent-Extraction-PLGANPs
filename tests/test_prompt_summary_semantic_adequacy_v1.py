import unittest

from src.stage2_sampling_labels.audit_prompt_summary_semantic_adequacy_v1 import (
    assess_prompt_summary_semantic_adequacy,
)


class TestPromptSummarySemanticAdequacy(unittest.TestCase):
    def test_semantic_adequacy_requires_identity_and_formulation_cues_not_all_numeric_rows(self):
        block = {
            "block_type": "table",
            "text_content": "Table 1. Formulations F1-F8 of PLGA nanoparticles prepared with PVA surfactant and drug loading values summarized.",
            "summary_is_lossy": True,
        }
        result = assess_prompt_summary_semantic_adequacy(block)
        self.assertEqual(result["semantic_adequacy"], "adequate")
        self.assertEqual(result["requires_full_numeric_rows"], "no")

    def test_numeric_only_summary_is_semantically_inadequate(self):
        block = {"block_type": "table", "text_content": "1 2 3 4 5 6 7 8 9 10"}
        result = assess_prompt_summary_semantic_adequacy(block)
        self.assertEqual(result["semantic_adequacy"], "inadequate")
        self.assertIn("missing_identity_or_process_signal", result["first_failure_reason"])

    def test_method_block_with_preparation_surface_is_adequate(self):
        block = {"block_type": "method", "text_content": "PLGA nanoparticles were prepared by solvent evaporation using PVA as stabilizer."}
        result = assess_prompt_summary_semantic_adequacy(block)
        self.assertEqual(result["semantic_adequacy"], "adequate")
        self.assertEqual(result["summary_view_is_lossy"], "no")

    def test_prompt_adequacy_is_not_benchmark_or_value_authority(self):
        block = {"block_type": "table", "text_content": "Formulation composition for PLGA particles with surfactant levels."}
        result = assess_prompt_summary_semantic_adequacy(block)
        self.assertEqual(result["benchmark_valid"], "no")
        self.assertEqual(result["value_authority"], "no")

    def test_semantic_summary_design_surface_is_adequate_without_numeric_rows(self):
        block = {
            "block_type": "table",
            "text_content": "- table_role_hint: design matrix - semantic_summary: formulation design/process variables for nanoparticle preparation; row/value authority remains in the S2-2 payload/grid, not this prompt summary.",
        }
        result = assess_prompt_summary_semantic_adequacy(block)
        self.assertEqual(result["semantic_adequacy"], "adequate")
        self.assertEqual(result["requires_full_numeric_rows"], "no")
        self.assertEqual(result["value_authority"], "no")

    def test_short_polymer_abbreviations_do_not_match_inside_unrelated_words(self):
        block = {"block_type": "table", "text_content": "Placebo plant plate table without carrier identity."}
        result = assess_prompt_summary_semantic_adequacy(block)
        self.assertEqual(result["has_identity_signal"], "no")


if __name__ == "__main__":
    unittest.main()
