from __future__ import annotations

import json
from pathlib import Path
import unittest

from src.stage5_benchmark.formulation_review_app_v1 import (
    DECISIONS_NAME,
    ReviewSources,
    append_decision,
    build_review_cards,
    load_source_index,
    read_decisions,
    read_tsv,
    write_metadata,
)


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "formulation_review_app"


class FormulationReviewAppTest(unittest.TestCase):
    def test_build_review_cards_groups_formulation_and_attaches_fields(self) -> None:
        formulation_rows = read_tsv(FIXTURE_ROOT / "audit_ready.tsv")
        seed_rows = read_tsv(FIXTURE_ROOT / "seed_rows.tsv")
        source_index = load_source_index(FIXTURE_ROOT / "source_index.tsv")

        cards = build_review_cards(formulation_rows, seed_rows, source_index=source_index)

        self.assertEqual([card["formulation_id"] for card in cards], ["F1", "F2"])
        self.assertEqual(cards[0]["paper_key"], "PAPER_A")
        self.assertEqual(cards[0]["source_row_count"], 1)
        self.assertEqual(cards[0]["source_controls"]["doi"], "10.1000/test.a")
        self.assertIs(cards[0]["source_controls"]["pdf_available"], True)
        self.assertIs(cards[0]["source_controls"]["html_available"], True)
        self.assertEqual(
            [field["field_name"] for field in cards[0]["fields"]],
            ["encapsulation_efficiency", "particle_size"],
        )
        self.assertEqual(cards[1]["fields"], [])

    def test_load_source_index_resolves_relative_allowlist_paths(self) -> None:
        source_index = load_source_index(FIXTURE_ROOT / "source_index.tsv")

        self.assertIn("PAPER_A", source_index)
        self.assertEqual(source_index["PAPER_A"]["doi"], "10.1000/test.a")
        self.assertTrue(source_index["PAPER_A"]["pdf_path"].is_absolute())
        self.assertTrue(source_index["PAPER_A"]["html_path"].exists())

    def test_load_source_index_supports_upload_copied_files_json(self) -> None:
        source_index = load_source_index(FIXTURE_ROOT / "upload_source_index.tsv")

        self.assertIn("PAPER_A", source_index)
        self.assertEqual(source_index["PAPER_A"]["doi"], "10.1000/test.a")
        self.assertTrue(source_index["PAPER_A"]["pdf_path"].exists())
        self.assertTrue(source_index["PAPER_A"]["html_path"].exists())

    def test_build_review_cards_supports_gpt35_master_table_columns(self) -> None:
        cards = build_review_cards(
            [
                {
                    "paper_key": "PAPER_A",
                    "title": "GPT35 paper",
                    "doi": "10.1000/test.a",
                    "model_formulation_id": "GPT_F1",
                    "raw_formulation_label": "NP-1",
                    "row_role": "formulation",
                    "row_identity_description": "PLGA NP-1; homogenization speed 23,000 rpm",
                    "polymer_name_value": "PLGA",
                    "particle_size_nm_value": "145",
                    "evidence_metrics": "EE was 72%.",
                    "evidence_composition": "PLGA was used.",
                }
            ],
            [],
            source_index=load_source_index(FIXTURE_ROOT / "upload_source_index.tsv"),
        )

        self.assertEqual(cards[0]["formulation_id"], "GPT_F1")
        self.assertEqual(cards[0]["article_formulation_label"], "NP-1")
        self.assertEqual(cards[0]["formulation_label_params"], "PLGA NP-1; homogenization speed 23,000 rpm")
        self.assertEqual(cards[0]["evidence_text"], "EE was 72%.")
        self.assertTrue(cards[0]["source_controls"]["pdf_available"])
        self.assertEqual(
            [field["field_name"] for field in cards[0]["fields"]],
            ["homogenization_speed_rpm", "polymer_name", "particle_size_nm"],
        )
        self.assertEqual(cards[0]["fields"][0]["extracted_value"], "23000")
        self.assertEqual(cards[0]["fields"][0]["extracted_unit"], "rpm")
        self.assertEqual(cards[0]["fields"][1]["evidence_text"], "PLGA was used.")
        self.assertEqual(cards[0]["fields"][2]["evidence_text"], "EE was 72%.")

    def test_build_review_cards_displays_studied_variables_json(self) -> None:
        cards = build_review_cards(
            [
                {
                    "paper_key": "PAPER_A",
                    "formulation_id": "F1",
                    "studied_variables_json": json.dumps(
                        [
                            {
                                "variable_name": "homogenization_speed",
                                "variable_family": "homogenization_speed_rpm",
                                "value": "19000",
                                "unit": "rpm",
                                "scope": "formulation_row",
                                "source": "stage5_studied_variables",
                                "evidence_text": "Homogenization Speed (rpm) = 19,000",
                            }
                        ]
                    ),
                }
            ],
            [],
        )

        self.assertEqual(cards[0]["fields"][0]["field_name"], "homogenization_speed_rpm")
        self.assertEqual(cards[0]["fields"][0]["extracted_value"], "19000")
        self.assertEqual(cards[0]["fields"][0]["extracted_unit"], "rpm")

    def test_append_decision_writes_append_only_jsonl(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            formulation_tsv = (FIXTURE_ROOT / "audit_ready.tsv").resolve()
            seed_tsv = (FIXTURE_ROOT / "seed_rows.tsv").resolve()
            sources = ReviewSources(formulation_tsv=formulation_tsv, seed_rows_tsv=seed_tsv)
            session_id = "test_session"

            record = append_decision(
                tmp_path,
                session_id,
                sources,
                {
                    "paper_key": "PAPER_A",
                    "formulation_id": "F1",
                    "decision_layer": "boundary",
                    "decision": "accept_as_gt_formulation",
                    "reviewer_note": "looks correct",
                },
            )

            self.assertEqual(record["review_session_id"], session_id)
            self.assertEqual(record["decision"], "accept_as_gt_formulation")
            decision_path = tmp_path / DECISIONS_NAME
            self.assertTrue(decision_path.exists())
            rows = read_decisions(tmp_path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["input_formulation_tsv"], str(formulation_tsv))

    def test_append_decision_rejects_missing_identity(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            sources = ReviewSources(formulation_tsv=(FIXTURE_ROOT / "audit_ready.tsv").resolve())

            with self.assertRaisesRegex(ValueError, "paper_key and formulation_id"):
                append_decision(
                    Path(tmp_dir),
                    "test_session",
                    sources,
                    {
                        "paper_key": "",
                        "formulation_id": "",
                        "decision_layer": "boundary",
                        "decision": "accept_as_gt_formulation",
                    },
                )

    def test_write_metadata_records_review_only_boundary(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            sources = ReviewSources(
                formulation_tsv=(FIXTURE_ROOT / "audit_ready.tsv").resolve(),
                seed_rows_tsv=(FIXTURE_ROOT / "seed_rows.tsv").resolve(),
                source_index_tsv=(FIXTURE_ROOT / "source_index.tsv").resolve(),
            )

            metadata_path = write_metadata(tmp_path, "test_session", sources, card_count=2)

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertIs(metadata["benchmark_valid"], False)
            self.assertEqual(metadata["stage_role"], "supporting_nondefault_human_review_surface")
            self.assertEqual(metadata["card_count"], 2)
            self.assertEqual(metadata["input_source_index_tsv"], str((FIXTURE_ROOT / "source_index.tsv").resolve()))


if __name__ == "__main__":
    unittest.main()
