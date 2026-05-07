import unittest

from src.stage2_sampling_labels.audit_stage1_stage2_visibility_boundary_classifier_v1 import (
    classify_visibility_boundary,
)


class TestStage1Stage2VisibilityBoundaryClassifier(unittest.TestCase):
    def test_cleantext_absent_is_stage1_visibility_first_failure(self):
        row = classify_visibility_boundary(
            cleantext_visibility="absent",
            table_authority_visibility="absent",
            selector_selected=False,
            prompt_adequate=False,
        )
        self.assertEqual(row["first_failure_layer"], "stage1_clean_text_visibility")
        self.assertEqual(row["materialization_allowed"], "no")

    def test_table_absent_after_cleantext_visible_is_stage1_table_authority_failure(self):
        row = classify_visibility_boundary(
            cleantext_visibility="partial",
            table_authority_visibility="absent",
            selector_selected=False,
            prompt_adequate=False,
        )
        self.assertEqual(row["first_failure_layer"], "stage1_table_authority_visibility")

    def test_selector_not_selected_is_not_authority_failure_when_registry_retained(self):
        row = classify_visibility_boundary(
            cleantext_visibility="partial",
            table_authority_visibility="partial",
            selector_selected=False,
            selector_registry_retained=True,
            prompt_adequate=False,
        )
        self.assertEqual(row["first_failure_layer"], "stage2_prompt_summary_semantic_adequacy")
        self.assertEqual(row["selector_ranker_not_filter_status"], "pass")

    def test_selector_drops_unretained_non_noise_candidate_is_boundary_violation(self):
        row = classify_visibility_boundary(
            cleantext_visibility="partial",
            table_authority_visibility="partial",
            selector_selected=False,
            selector_registry_retained=False,
            prompt_adequate=True,
        )
        self.assertEqual(row["first_failure_layer"], "stage2_selector_boundary")
        self.assertEqual(row["selector_ranker_not_filter_status"], "fail")

    def test_selector_violation_is_not_masked_by_other_selected_candidate(self):
        row = classify_visibility_boundary(
            cleantext_visibility="partial",
            table_authority_visibility="partial",
            selector_selected=True,
            selector_registry_retained=False,
            prompt_adequate=True,
        )
        self.assertEqual(row["first_failure_layer"], "stage2_selector_boundary")
        self.assertEqual(row["selector_ranker_not_filter_status"], "fail")

    def test_allowed_hard_drop_rows_do_not_force_selector_boundary_failure(self):
        row = classify_visibility_boundary(
            cleantext_visibility="partial",
            table_authority_visibility="partial",
            selector_selected=False,
            selector_registry_retained=True,
            prompt_adequate=True,
        )
        self.assertNotEqual(row["first_failure_layer"], "stage2_selector_boundary")
        self.assertEqual(row["selector_ranker_not_filter_status"], "pass")

    def test_all_visible_and_adequate_is_visibility_pass_not_benchmark(self):
        row = classify_visibility_boundary(
            cleantext_visibility="partial",
            table_authority_visibility="partial",
            selector_selected=True,
            prompt_adequate=True,
        )
        self.assertEqual(row["first_failure_layer"], "none")
        self.assertEqual(row["visibility_boundary_status"], "pass")
        self.assertEqual(row["benchmark_valid"], "no")


if __name__ == "__main__":
    unittest.main()
