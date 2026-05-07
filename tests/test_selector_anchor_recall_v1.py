import unittest

from src.stage2_sampling_labels.audit_selector_anchor_recall_v1 import (
    audit_selector_registry,
    build_selector_recall_rows,
)


class TestSelectorAnchorRecall(unittest.TestCase):
    def test_candidate_registry_retains_unselected_relevant_blocks(self):
        candidate_artifact = {
            "paper_key": "TESTKEY1",
            "candidate_blocks": [
                {"candidate_id": "c1", "candidate_type": "prose", "text_content": "PLGA nanoparticles were prepared by solvent evaporation.", "section_kind": "preparation"},
                {"candidate_id": "c2", "candidate_type": "prose", "text_content": "Additional formulation variants used different surfactant levels.", "section_kind": "preparation"},
            ],
        }
        evidence_artifact = {"evidence_blocks": [{"candidate_id": "c1", "block_type": "method", "text_content": "PLGA nanoparticles were prepared"}]}
        rows = audit_selector_registry(candidate_artifact, evidence_artifact=evidence_artifact)
        row_by_id = {row["candidate_id"]: row for row in rows}
        self.assertEqual(row_by_id["c2"]["selected_for_prompt"], "no")
        self.assertEqual(row_by_id["c2"]["registry_retained"], "yes")
        self.assertEqual(row_by_id["c2"]["selector_is_authority_filter_violation"], "no")

    def test_selected_evidence_rank_does_not_delete_unselected_blocks(self):
        rows = build_selector_recall_rows(
            paper_key="TESTKEY2",
            candidates=[
                {"candidate_id": "a", "candidate_type": "prose", "priority_score": 10},
                {"candidate_id": "b", "candidate_type": "prose", "priority_score": 1},
            ],
            selected_candidate_ids={"a"},
            suppression_reasons={},
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["selector_rank"], "1")
        self.assertEqual(rows[1]["selected_for_prompt"], "no")
        self.assertEqual(rows[1]["registry_retained"], "yes")

    def test_non_noise_tables_are_preserved_for_authority_even_low_score(self):
        rows = build_selector_recall_rows(
            paper_key="TESTKEY3",
            candidates=[{"candidate_id": "t1", "candidate_type": "table", "table_inclusion_class": "optional_context", "priority_score": -5}],
            selected_candidate_ids=set(),
            suppression_reasons={},
        )
        self.assertEqual(rows[0]["preserved_for_authority"], "yes")
        self.assertEqual(rows[0]["selector_is_authority_filter_violation"], "no")

    def test_confirmed_pure_noise_may_be_excluded_with_explicit_reason(self):
        rows = build_selector_recall_rows(
            paper_key="TESTKEY4",
            candidates=[{"candidate_id": "t_noise", "candidate_type": "table", "table_inclusion_class": "hard_drop"}],
            selected_candidate_ids=set(),
            suppression_reasons={"t_noise": "hard_drop_table_noise"},
        )
        self.assertEqual(rows[0]["preserved_for_authority"], "no")
        self.assertEqual(rows[0]["exclusion_reason"], "hard_drop_table_noise")
        self.assertEqual(rows[0]["selector_is_authority_filter_violation"], "no")

    def test_unretained_non_noise_candidate_is_selector_authority_violation(self):
        rows = build_selector_recall_rows(
            paper_key="TESTKEY5",
            candidates=[{"candidate_id": "c_lost", "candidate_type": "prose", "registry_retained": "no"}],
            selected_candidate_ids=set(),
            suppression_reasons={"c_lost": "method_budget_reached"},
        )
        self.assertEqual(rows[0]["selector_is_authority_filter_violation"], "yes")

    def test_string_hard_drop_reason_and_confirmed_noise_reason_are_hard_drop(self):
        rows = build_selector_recall_rows(
            paper_key="TESTKEY6",
            candidates=[
                {"candidate_id": "n1", "candidate_type": "table", "hard_drop_reason": "confirmed_noise"},
                {"candidate_id": "n2", "candidate_type": "table"},
            ],
            selected_candidate_ids=set(),
            suppression_reasons={"n2": "confirmed_noise"},
        )
        by_id = {row["candidate_id"]: row for row in rows}
        self.assertEqual(by_id["n1"]["preserved_for_authority"], "no")
        self.assertEqual(by_id["n2"]["preserved_for_authority"], "no")
        self.assertEqual(by_id["n1"]["selector_is_authority_filter_violation"], "no")


if __name__ == "__main__":
    unittest.main()
