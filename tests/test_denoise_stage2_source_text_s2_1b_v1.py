import csv
import json
import tempfile
from pathlib import Path
import unittest

from src.stage2_sampling_labels.denoise_stage2_source_text_s2_1b_v1 import (
    denoise_text,
    run_denoise_projection,
)
from src.stage2_sampling_labels.run_stage2_composite_v1 import apply_s2_1b_denoise_projection_to_manifest_rows


class TestDenoiseStage2SourceTextS21bV1(unittest.TestCase):
    def test_removes_only_high_confidence_noise_rule_classes(self):
        raw_text = "\n".join([
            "Journals & Books",
            "Downloaded from https://example.org by University on 01 January 2024",
            "Journal of Controlled Release 123 (2020) 45-50",
            "Smith et al. Page 3 of 12",
            "Copyright © 2020 Elsevier Ltd. All rights reserved.",
            "References",
            "[1] Doe J. Unrelated article. Journal Name. 2018;1:2-3.",
            "Recommended articles",
            "Related article metadata: Cited by 17",
            "Materials: PLGA nanoparticles were prepared by nanoprecipitation.",
            "Table 1 Formulation composition of F1 and F2",
            "F1 PLGA 50 mg PVA 1%",
        ])

        result = denoise_text(raw_text, paper_key="PAPER1", input_text_path="input.txt")

        self.assertIn("Materials: PLGA nanoparticles were prepared by nanoprecipitation.", result.denoised_text)
        self.assertIn("Table 1 Formulation composition of F1 and F2", result.denoised_text)
        self.assertIn("F1 PLGA 50 mg PVA 1%", result.denoised_text)
        self.assertNotIn("Journals & Books", result.denoised_text)
        self.assertNotIn("Downloaded from", result.denoised_text)
        self.assertNotIn("Recommended articles", result.denoised_text)
        removed_classes = {event.rule_class for event in result.removed_events}
        expected_classes = {
            "publisher_chrome",
            "download_marker",
            "page_header_footer",
            "author_page_running_line",
            "copyright_or_license_boilerplate",
            "reference_tail",
            "isolated_reference_line",
            "article_recommendation_or_related_articles",
        }
        self.assertTrue(expected_classes.issubset(removed_classes), expected_classes - removed_classes)

    def test_keeps_reference_heading_when_followed_by_formulation_carrythrough(self):
        raw_text = "\n".join([
            "References to Table 2 were used to define formulation F1.",
            "Preparation: nanoparticles were prepared according to Table 2.",
            "Table 2 Formulation design matrix",
        ])

        result = denoise_text(raw_text, paper_key="PAPER2", input_text_path="input.txt")

        self.assertEqual(result.denoised_text, raw_text)
        self.assertEqual(result.removed_events, [])

    def test_writes_denoised_text_audit_json_and_summary_tsv_for_explicit_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "source.txt"
            run_dir = tmp_path / "run"
            input_path.write_text("Journals & Books\nMethods: PLGA NPs were prepared.\n", encoding="utf-8")

            summary_path = run_denoise_projection(
                inputs=[("PAPER3", input_path)],
                run_dir=run_dir,
            )

            output_text = run_dir / "semantic_stage2_objects" / "s2_1b_denoised_text" / "PAPER3.txt"
            audit_json = run_dir / "semantic_stage2_objects" / "s2_1b_denoise_audit" / "PAPER3_s2_1b_denoise_audit_v1.json"
            self.assertEqual(summary_path, run_dir / "analysis" / "s2_1b_denoise_summary_v1.tsv")
            self.assertEqual(output_text.read_text(encoding="utf-8"), "Methods: PLGA NPs were prepared.")
            audit = json.loads(audit_json.read_text(encoding="utf-8"))
            self.assertEqual(audit["paper_key"], "PAPER3")
            self.assertEqual(audit["input_text_path"], str(input_path))
            self.assertEqual(audit["output_text_path"], str(output_text))
            self.assertEqual(audit["removed_events"][0]["rule_class"], "publisher_chrome")

            with summary_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["paper_key"], "PAPER3")
            self.assertEqual(rows[0]["rule_class"], "publisher_chrome")
            self.assertEqual(rows[0]["preservation_exception"], "")

    def test_composite_manifest_rows_are_hydrated_with_s2_1b_projection_without_overwriting_raw_text_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "source.txt"
            run_dir = tmp_path / "run"
            input_path.write_text("Journals & Books\nMethods: PLGA NPs were prepared.", encoding="utf-8")
            rows = [{"key": "PAPER4", "text_path": str(input_path), "title": "Synthetic"}]

            hydrated = apply_s2_1b_denoise_projection_to_manifest_rows(rows, run_dir=run_dir)

            self.assertEqual(hydrated[0]["text_path"], str(input_path))
            denoised_path = Path(hydrated[0]["source_s2_1b_denoised_text_path"])
            audit_path = Path(hydrated[0]["s2_1b_denoise_audit_path"])
            self.assertTrue(denoised_path.exists())
            self.assertTrue(audit_path.exists())
            self.assertEqual(denoised_path.read_text(encoding="utf-8"), "Methods: PLGA NPs were prepared.")
            self.assertEqual(hydrated[0]["source_text_projection"], "s2_1b_denoised")


if __name__ == "__main__":
    unittest.main()
