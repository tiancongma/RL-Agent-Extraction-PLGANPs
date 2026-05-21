from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.stage5_benchmark import build_minimal_final_output_v1 as s5_final
from src.stage5_benchmark import build_s5_3_llm_direct_value_candidates_v1 as s5_3
from src.stage5_benchmark import build_s5_5_derived_values_v1 as s5_5
from src.stage5_benchmark import validate_s5_value_candidates_v1 as s5_4


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class S5ValueLayerContractV1Tests(unittest.TestCase):
    def test_s5_3_default_writes_diagnostic_outputs_with_exact_source_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final_table = root / "inputs" / "exact_final.tsv"
            decision_trace = root / "inputs" / "exact_decision.tsv"
            scope_manifest = root / "inputs" / "exact_scope.tsv"
            source_inventory = root / "inputs" / "exact_source_inventory.tsv"
            out_dir = root / "out" / "diagnostic_s5_3"

            text_path = root / "sources" / "paper_A.txt"
            table_path = root / "sources" / "paper_A_tables.tsv"
            pdf_path = root / "sources" / "paper_A.pdf"
            text_path.parent.mkdir(parents=True)
            text_path.write_text("source text", encoding="utf-8")
            table_path.write_text("table text", encoding="utf-8")
            pdf_path.write_text("%PDF placeholder", encoding="utf-8")

            write_tsv(
                final_table,
                ["paper_key", "final_formulation_id", "field"],
                [{"paper_key": "PAPER_A", "final_formulation_id": "F1", "field": "kept"}],
            )
            write_tsv(decision_trace, ["paper_key", "decision"], [{"paper_key": "PAPER_A", "decision": "keep"}])
            write_tsv(scope_manifest, ["paper_key", "scope"], [{"paper_key": "PAPER_A", "scope": "explicit"}])
            write_tsv(
                source_inventory,
                ["paper_key", "source_text_path", "source_table_path", "source_pdf_path"],
                [
                    {
                        "paper_key": "PAPER_A",
                        "source_text_path": str(text_path),
                        "source_table_path": str(table_path),
                        "source_pdf_path": str(pdf_path),
                    }
                ],
            )

            exit_code = s5_3.main(
                [
                    "--final-table-tsv",
                    str(final_table),
                    "--decision-trace-tsv",
                    str(decision_trace),
                    "--scope-manifest-tsv",
                    str(scope_manifest),
                    "--source-inventory-tsv",
                    str(source_inventory),
                    "--out-dir",
                    str(out_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            candidate_path = out_dir / s5_3.CANDIDATE_TSV_NAME
            audit_path = out_dir / s5_3.AUDIT_TSV_NAME
            prompt_path = out_dir / s5_3.PROMPT_PLACEHOLDER_TSV_NAME
            manifest_path = out_dir / s5_3.INPUT_MANIFEST_TSV_NAME
            context_path = out_dir / s5_3.RUN_CONTEXT_NAME
            raw_responses_dir = out_dir / s5_3.RAW_RESPONSES_DIR_NAME
            for path in [candidate_path, audit_path, prompt_path, manifest_path, context_path]:
                self.assertTrue(path.exists(), f"missing output: {path}")
            self.assertTrue(raw_responses_dir.exists(), f"missing raw responses dir: {raw_responses_dir}")

            # Candidate and audit files are intentionally header-only in the no-backend diagnostic mode.
            self.assertEqual(read_tsv(candidate_path), [])
            self.assertEqual(read_tsv(audit_path), [])

            manifest_rows = read_tsv(manifest_path)
            manifest_by_role = {row["input_role"]: row for row in manifest_rows}
            self.assertEqual(manifest_by_role["final_table_tsv"]["path"], str(final_table.resolve()))
            self.assertEqual(manifest_by_role["decision_trace_tsv"]["path"], str(decision_trace.resolve()))
            self.assertEqual(manifest_by_role["scope_manifest_tsv"]["path"], str(scope_manifest.resolve()))
            self.assertEqual(manifest_by_role["source_inventory_tsv"]["path"], str(source_inventory.resolve()))
            self.assertIn(str(text_path), manifest_by_role["source_inventory_tsv"]["metadata_json"])
            self.assertIn(str(table_path), manifest_by_role["source_inventory_tsv"]["metadata_json"])
            self.assertIn(str(pdf_path), manifest_by_role["source_inventory_tsv"]["metadata_json"])

            prompt_rows = read_tsv(prompt_path)
            self.assertEqual(len(prompt_rows), 1)
            self.assertEqual(prompt_rows[0]["paper_key"], "PAPER_A")
            self.assertEqual(prompt_rows[0]["final_formulation_id"], "F1")
            self.assertEqual(prompt_rows[0]["source_text_path"], str(text_path))
            self.assertEqual(prompt_rows[0]["source_table_path"], str(table_path))
            self.assertEqual(prompt_rows[0]["source_pdf_path"], str(pdf_path))
            self.assertEqual(prompt_rows[0]["prompt_status"], "placeholder_not_sent_no_llm_backend")

            context = context_path.read_text(encoding="utf-8")
            self.assertIn("entrypoint", context)
            self.assertIn(s5_3.ENTRYPOINT, context)
            self.assertIn("boundary_class", context)
            self.assertIn("diagnostic/supporting_boundary", context)
            self.assertIn("benchmark_valid_status", context)
            self.assertIn("benchmark_valid: `no`", context)
            self.assertIn("not benchmark-valid", context)
            self.assertIn(str(final_table.resolve()), context)
            self.assertIn(str(decision_trace.resolve()), context)
            self.assertIn(str(scope_manifest.resolve()), context)
            self.assertIn(str(source_inventory.resolve()), context)
            self.assertIn(str(candidate_path), context)
            self.assertIn(str(audit_path), context)
            self.assertIn(str(prompt_path), context)
            self.assertIn(str(text_path), context)
            self.assertIn(str(table_path), context)
            self.assertIn(str(pdf_path), context)
            self.assertIn("does not use latest-directory lookup", context)

    def test_s5_3_requires_explicit_inputs_and_does_not_infer_latest_or_glob_first_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final_table = root / "run_old" / "final.tsv"
            decision_trace = root / "run_old" / "decision.tsv"
            scope_manifest = root / "run_old" / "scope.tsv"
            out_dir = root / "out"
            tempting_latest = root / "latest" / "source_inventory.tsv"

            write_tsv(final_table, ["paper_key", "final_formulation_id"], [{"paper_key": "PAPER_B", "final_formulation_id": "F2"}])
            write_tsv(decision_trace, ["paper_key", "decision"], [{"paper_key": "PAPER_B", "decision": "keep"}])
            write_tsv(scope_manifest, ["paper_key", "scope"], [{"paper_key": "PAPER_B", "scope": "explicit"}])
            write_tsv(
                tempting_latest,
                ["paper_key", "source_text_path", "source_table_path", "source_pdf_path"],
                [
                    {
                        "paper_key": "PAPER_B",
                        "source_text_path": "/must/not/be/inferred.txt",
                        "source_table_path": "/must/not/be/inferred.tsv",
                        "source_pdf_path": "/must/not/be/inferred.pdf",
                    }
                ],
            )

            exit_code = s5_3.main(
                [
                    "--final-table-tsv",
                    str(final_table),
                    "--decision-trace-tsv",
                    str(decision_trace),
                    "--scope-manifest-tsv",
                    str(scope_manifest),
                    "--out-dir",
                    str(out_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            prompt_rows = read_tsv(out_dir / s5_3.PROMPT_PLACEHOLDER_TSV_NAME)
            self.assertEqual(len(prompt_rows), 1)
            self.assertEqual(prompt_rows[0]["paper_key"], "PAPER_B")
            self.assertEqual(prompt_rows[0]["source_text_path"], "")
            self.assertEqual(prompt_rows[0]["source_table_path"], "")
            self.assertEqual(prompt_rows[0]["source_pdf_path"], "")

            context = (out_dir / s5_3.RUN_CONTEXT_NAME).read_text(encoding="utf-8")
            self.assertNotIn("/must/not/be/inferred", context)
            self.assertIn("All source artifacts are resolved exclusively from explicit CLI arguments", context)

    def test_s5_3_fails_when_required_explicit_path_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final_table = root / "missing_final.tsv"
            decision_trace = root / "decision.tsv"
            scope_manifest = root / "scope.tsv"
            write_tsv(decision_trace, ["paper_key"], [])
            write_tsv(scope_manifest, ["paper_key"], [])

            with self.assertRaises(FileNotFoundError):
                s5_3.main(
                    [
                        "--final-table-tsv",
                        str(final_table),
                        "--decision-trace-tsv",
                        str(decision_trace),
                        "--scope-manifest-tsv",
                        str(scope_manifest),
                        "--out-dir",
                        str(root / "out"),
                    ]
                )

    def test_s5_4_accepts_direct_candidate_with_quote_and_clear_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_tsv = root / "inputs" / s5_3.CANDIDATE_TSV_NAME
            out_dir = root / "out" / "s5_4"
            write_tsv(
                candidate_tsv,
                s5_3.CANDIDATE_COLUMNS,
                [
                    {
                        "paper_key": "PAPER_A",
                        "formulation_id": "F1",
                        "field_name": "drug_loading",
                        "value_text": "12.5",
                        "unit_text": "%",
                        "direct_or_derived": "direct",
                        "evidence_scope": "formulation_row",
                        "source_quote": "Drug loading (%) 12.5",
                    }
                ],
            )

            exit_code = s5_4.main(["--candidate-tsv", str(candidate_tsv), "--out-dir", str(out_dir)])

            self.assertEqual(exit_code, 0)
            for output_name in [
                s5_4.DECISIONS_TSV_NAME,
                s5_4.ACCEPTED_TSV_NAME,
                s5_4.REJECTED_TSV_NAME,
                s5_4.REVIEW_TSV_NAME,
                s5_4.SUMMARY_JSON_NAME,
                s5_4.RUN_CONTEXT_NAME,
            ]:
                self.assertTrue((out_dir / output_name).exists(), f"missing S5-4 output: {output_name}")
            accepted_rows = read_tsv(out_dir / s5_4.ACCEPTED_TSV_NAME)
            rejected_rows = read_tsv(out_dir / s5_4.REJECTED_TSV_NAME)
            review_rows = read_tsv(out_dir / s5_4.REVIEW_TSV_NAME)
            decisions = read_tsv(out_dir / s5_4.DECISIONS_TSV_NAME)
            self.assertEqual(len(accepted_rows), 1)
            self.assertEqual(accepted_rows[0]["s5_4_decision"], "accepted")
            self.assertEqual(accepted_rows[0]["s5_4_reason"], "direct_candidate_has_required_source_quote_and_scope")
            self.assertEqual(rejected_rows, [])
            self.assertEqual(review_rows, [])
            self.assertEqual(decisions[0]["paper_key"], "PAPER_A")

            summary = json.loads((out_dir / s5_4.SUMMARY_JSON_NAME).read_text(encoding="utf-8"))
            self.assertEqual(summary["benchmark_valid"], "no")
            self.assertEqual(summary["candidate_rows"], 1)
            self.assertEqual(summary["accepted_direct_rows"], 1)
            context = (out_dir / s5_4.RUN_CONTEXT_NAME).read_text(encoding="utf-8")
            self.assertIn(s5_4.ENTRYPOINT, context)
            self.assertIn(str(candidate_tsv.resolve()), context)
            self.assertIn(str(out_dir / s5_4.DECISIONS_TSV_NAME), context)
            self.assertIn("Stage5 internal validation boundary", context)
            self.assertIn("benchmark_valid: `no`", context)
            self.assertIn("Does not change formulation membership", context)
            self.assertIn("Does not consult GT values", context)
            self.assertIn("system final-table values", context)

    def test_s5_4_rejects_derived_from_direct_layer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_tsv = root / "inputs" / s5_3.CANDIDATE_TSV_NAME
            out_dir = root / "out"
            write_tsv(
                candidate_tsv,
                s5_3.CANDIDATE_COLUMNS,
                [
                    {
                        "paper_key": "PAPER_B",
                        "formulation_id": "F2",
                        "field_name": "polymer_identity",
                        "value_text": "PLGA",
                        "direct_or_derived": "derived",
                        "evidence_scope": "formulation_row",
                        "source_quote": "PLGA nanoparticles",
                    }
                ],
            )

            self.assertEqual(s5_4.main(["--candidate-tsv", str(candidate_tsv), "--out-dir", str(out_dir)]), 0)

            rejected_rows = read_tsv(out_dir / s5_4.REJECTED_TSV_NAME)
            accepted_rows = read_tsv(out_dir / s5_4.ACCEPTED_TSV_NAME)
            self.assertEqual(accepted_rows, [])
            self.assertEqual(len(rejected_rows), 1)
            self.assertEqual(rejected_rows[0]["s5_4_decision"], "rejected")
            self.assertEqual(rejected_rows[0]["s5_4_reason"], "derived_value_not_allowed_in_direct_layer")
            summary = json.loads((out_dir / s5_4.SUMMARY_JSON_NAME).read_text(encoding="utf-8"))
            self.assertEqual(summary["derived_rejections"], 1)

    def test_s5_4_rejects_missing_source_quote_for_direct_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_tsv = root / "inputs" / s5_3.CANDIDATE_TSV_NAME
            out_dir = root / "out"
            write_tsv(
                candidate_tsv,
                s5_3.CANDIDATE_COLUMNS,
                [
                    {
                        "paper_key": "PAPER_C",
                        "formulation_id": "F3",
                        "field_name": "particle_size_nm",
                        "value_text": "180",
                        "unit_text": "nm",
                        "direct_or_derived": "direct",
                        "evidence_scope": "formulation_row",
                        "source_quote": "",
                    }
                ],
            )

            self.assertEqual(s5_4.main(["--candidate-tsv", str(candidate_tsv), "--out-dir", str(out_dir)]), 0)

            rejected_rows = read_tsv(out_dir / s5_4.REJECTED_TSV_NAME)
            self.assertEqual(len(rejected_rows), 1)
            self.assertEqual(rejected_rows[0]["s5_4_reason"], "missing_source_quote_for_direct_candidate")
            self.assertEqual(read_tsv(out_dir / s5_4.ACCEPTED_TSV_NAME), [])
            summary = json.loads((out_dir / s5_4.SUMMARY_JSON_NAME).read_text(encoding="utf-8"))
            self.assertEqual(summary["missing_quote_rejections"], 1)

    def test_s5_4_marks_ambiguous_scope_as_review_needed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_tsv = root / "inputs" / s5_3.CANDIDATE_TSV_NAME
            out_dir = root / "out"
            write_tsv(
                candidate_tsv,
                s5_3.CANDIDATE_COLUMNS,
                [
                    {
                        "paper_key": "PAPER_D",
                        "formulation_id": "F4",
                        "field_name": "pdi",
                        "value_text": "0.12",
                        "direct_or_derived": "direct",
                        "evidence_scope": "ambiguous",
                        "source_quote": "PDI 0.12",
                    }
                ],
            )

            self.assertEqual(s5_4.main(["--candidate-tsv", str(candidate_tsv), "--out-dir", str(out_dir)]), 0)

            review_rows = read_tsv(out_dir / s5_4.REVIEW_TSV_NAME)
            accepted_rows = read_tsv(out_dir / s5_4.ACCEPTED_TSV_NAME)
            rejected_rows = read_tsv(out_dir / s5_4.REJECTED_TSV_NAME)
            self.assertEqual(accepted_rows, [])
            self.assertEqual(rejected_rows, [])
            self.assertEqual(len(review_rows), 1)
            self.assertEqual(review_rows[0]["s5_4_decision"], "review_needed")
            self.assertEqual(review_rows[0]["s5_4_review_needed"], "yes")
            self.assertEqual(review_rows[0]["needs_review"], "yes")
            self.assertEqual(review_rows[0]["s5_4_reason"], "ambiguous_scope_requires_manual_review")
            summary = json.loads((out_dir / s5_4.SUMMARY_JSON_NAME).read_text(encoding="utf-8"))
            self.assertEqual(summary["review_needed_rows"], 1)
            self.assertEqual(summary["ambiguous_scope_review_rows"], 1)

    def test_s5_5_computes_percent_wv_volume_mass_sidecar_and_never_direct_compare(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted_tsv = root / "inputs" / s5_4.ACCEPTED_TSV_NAME
            out_dir = root / "out" / "s5_5"
            write_tsv(
                accepted_tsv,
                [
                    "paper_key",
                    "formulation_id",
                    "field_name",
                    "value_text",
                    "unit_text",
                    "source_quote",
                    "evidence_scope",
                    "decision",
                ],
                [
                    {
                        "paper_key": "PAPER_MASS",
                        "formulation_id": "F1",
                        "field_name": "concentration_percent_wv",
                        "value_text": "0.5",
                        "unit_text": "%w/v",
                        "source_quote": "0.5% w/v drug solution",
                        "evidence_scope": "formulation_row",
                        "decision": "accepted",
                    },
                    {
                        "paper_key": "PAPER_MASS",
                        "formulation_id": "F1",
                        "field_name": "volume_ml",
                        "value_text": "10",
                        "unit_text": "mL",
                        "source_quote": "final volume 10 mL",
                        "evidence_scope": "formulation_row",
                        "decision": "accepted",
                    },
                ],
            )

            exit_code = s5_5.main(["--accepted-direct-values-tsv", str(accepted_tsv), "--out-dir", str(out_dir)])

            self.assertEqual(exit_code, 0)
            for output_name in [
                s5_5.DERIVED_TSV_NAME,
                s5_5.UNIT_NORMALIZATION_TSV_NAME,
                s5_5.PROVENANCE_TSV_NAME,
                s5_5.REVIEW_TSV_NAME,
                s5_5.SUMMARY_JSON_NAME,
                s5_5.RUN_CONTEXT_NAME,
            ]:
                self.assertTrue((out_dir / output_name).exists(), f"missing S5-5 output: {output_name}")

            derived_rows = read_tsv(out_dir / s5_5.DERIVED_TSV_NAME)
            review_rows = read_tsv(out_dir / s5_5.REVIEW_TSV_NAME)
            provenance_rows = read_tsv(out_dir / s5_5.PROVENANCE_TSV_NAME)
            self.assertEqual(len(derived_rows), 1)
            derived = derived_rows[0]
            self.assertEqual(derived["paper_key"], "PAPER_MASS")
            self.assertEqual(derived["formulation_id"], "F1")
            self.assertEqual(derived["target_field_name"], "derived_mass_mg")
            self.assertEqual(derived["derived_value"], "50")
            self.assertEqual(derived["derived_unit"], "mg")
            self.assertEqual(derived["formula_id"], "percent_wv_x_ml_to_mg_v1")
            self.assertIn("mg = percent_value * 10 * volume_mL", derived["formula_expression"])
            self.assertEqual(derived["input_field_names"], "concentration_percent_wv;volume_ml")
            self.assertIn("0.5", derived["input_values"])
            self.assertIn("10", derived["input_values"])
            self.assertIn("0.5% w/v drug solution", derived["input_source_provenance"])
            self.assertIn("final volume 10 mL", derived["input_source_provenance"])
            self.assertEqual(derived["eligible_for_direct_compare"], "no")
            self.assertEqual(derived["eligible_for_derived_compare"], "yes")
            self.assertEqual(derived["needs_review"], "no")
            self.assertEqual(review_rows, [])
            self.assertEqual(len(provenance_rows), 1)
            self.assertEqual(provenance_rows[0]["source_layer"], "s5_4_accepted_direct_values_v1")

            summary = json.loads((out_dir / s5_5.SUMMARY_JSON_NAME).read_text(encoding="utf-8"))
            self.assertEqual(summary["benchmark_valid"], "no")
            self.assertEqual(summary["derived_rows"], 1)
            self.assertEqual(summary["unit_normalization_rows"], 1)
            self.assertEqual(summary["review_rows"], 0)
            self.assertEqual(summary["eligible_for_direct_compare"], "no")
            context = (out_dir / s5_5.RUN_CONTEXT_NAME).read_text(encoding="utf-8")
            self.assertIn(s5_5.ENTRYPOINT, context)
            self.assertIn(str(accepted_tsv.resolve()), context)
            self.assertIn(str(out_dir / s5_5.DERIVED_TSV_NAME), context)
            self.assertIn("Derived sidecar boundary", context)
            self.assertIn("benchmark_valid: `no`", context)
            self.assertIn("eligible_for_direct_compare: `no`", context)
            self.assertIn("Does not change direct compare outputs", context)
            self.assertIn("Does not change final formulation table", context)

    def test_s5_5_computes_mg_per_ml_volume_mass_derivation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted_tsv = root / "inputs" / s5_4.ACCEPTED_TSV_NAME
            out_dir = root / "out" / "s5_5"
            write_tsv(
                accepted_tsv,
                [
                    "paper_key",
                    "formulation_id",
                    "field_name",
                    "value_text",
                    "unit_text",
                    "source_quote",
                    "evidence_scope",
                    "decision",
                ],
                [
                    {
                        "paper_key": "PAPER_CONC",
                        "formulation_id": "F1",
                        "field_name": "drug_concentration",
                        "value_text": "2.5",
                        "unit_text": "mg/mL",
                        "source_quote": "drug concentration was 2.5 mg/mL",
                        "evidence_scope": "formulation_row",
                        "decision": "accepted",
                    },
                    {
                        "paper_key": "PAPER_CONC",
                        "formulation_id": "F1",
                        "field_name": "volume_ml",
                        "value_text": "4",
                        "unit_text": "mL",
                        "source_quote": "volume was 4 mL",
                        "evidence_scope": "formulation_row",
                        "decision": "accepted",
                    },
                ],
            )

            self.assertEqual(s5_5.main(["--accepted-direct-values-tsv", str(accepted_tsv), "--out-dir", str(out_dir)]), 0)

            derived_rows = read_tsv(out_dir / s5_5.DERIVED_TSV_NAME)
            self.assertEqual(len(derived_rows), 1)
            derived = derived_rows[0]
            self.assertEqual(derived["formula_id"], "mg_per_ml_x_ml_to_mg_v1")
            self.assertEqual(derived["derived_value"], "10")
            self.assertEqual(derived["derived_unit"], "mg")
            self.assertEqual(derived["normalized_value"], "10")
            self.assertEqual(derived["normalized_unit"], "mg")
            self.assertEqual(derived["value_kind"], "derived")
            self.assertEqual(derived["eligible_for_direct_compare"], "no")
            self.assertEqual(derived["eligible_for_modeling"], "yes")

    def test_s5_5_writes_unit_normalization_sidecar_without_final_table_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted_tsv = root / "inputs" / s5_4.ACCEPTED_TSV_NAME
            out_dir = root / "out" / "s5_5"
            write_tsv(
                accepted_tsv,
                [
                    "paper_key",
                    "formulation_id",
                    "field_name",
                    "value_text",
                    "unit_text",
                    "source_quote",
                    "evidence_scope",
                    "decision",
                ],
                [
                    {
                        "paper_key": "PAPER_UNITS",
                        "formulation_id": "F1",
                        "field_name": "drug_mass",
                        "value_text": "0.002",
                        "unit_text": "g",
                        "source_quote": "drug mass 0.002 g",
                        "evidence_scope": "formulation_row",
                        "decision": "accepted",
                    },
                    {
                        "paper_key": "PAPER_UNITS",
                        "formulation_id": "F1",
                        "field_name": "volume",
                        "value_text": "250",
                        "unit_text": "uL",
                        "source_quote": "aqueous volume 250 uL",
                        "evidence_scope": "formulation_row",
                        "decision": "accepted",
                    },
                ],
            )

            self.assertEqual(s5_5.main(["--accepted-direct-values-tsv", str(accepted_tsv), "--out-dir", str(out_dir)]), 0)

            unit_rows = read_tsv(out_dir / s5_5.UNIT_NORMALIZATION_TSV_NAME)
            self.assertEqual(len(unit_rows), 2)
            by_field = {row["source_field_name"]: row for row in unit_rows}
            self.assertEqual(by_field["drug_mass"]["normalized_value"], "2")
            self.assertEqual(by_field["drug_mass"]["normalized_unit"], "mg")
            self.assertEqual(by_field["drug_mass"]["value_kind"], "direct_normalized")
            self.assertEqual(by_field["drug_mass"]["eligible_for_direct_compare"], "no")
            self.assertEqual(by_field["drug_mass"]["eligible_for_modeling"], "yes")
            self.assertEqual(by_field["volume"]["normalized_value"], "0.25")
            self.assertEqual(by_field["volume"]["normalized_unit"], "mL")
            self.assertEqual(by_field["volume"]["formula_id"], "volume_ul_to_ml_v1")

    def test_s5_5_derives_volume_from_mass_and_concentration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted_tsv = root / "inputs" / s5_4.ACCEPTED_TSV_NAME
            out_dir = root / "out" / "s5_5"
            write_tsv(
                accepted_tsv,
                [
                    "paper_key",
                    "formulation_id",
                    "field_name",
                    "value_text",
                    "unit_text",
                    "source_quote",
                    "evidence_scope",
                    "decision",
                ],
                [
                    {
                        "paper_key": "PAPER_VOLUME",
                        "formulation_id": "F1",
                        "field_name": "drug_mass",
                        "value_text": "10",
                        "unit_text": "mg",
                        "source_quote": "drug mass 10 mg",
                        "evidence_scope": "formulation_row",
                        "decision": "accepted",
                    },
                    {
                        "paper_key": "PAPER_VOLUME",
                        "formulation_id": "F1",
                        "field_name": "drug_concentration",
                        "value_text": "2",
                        "unit_text": "mg/mL",
                        "source_quote": "drug concentration 2 mg/mL",
                        "evidence_scope": "formulation_row",
                        "decision": "accepted",
                    },
                ],
            )

            self.assertEqual(s5_5.main(["--accepted-direct-values-tsv", str(accepted_tsv), "--out-dir", str(out_dir)]), 0)

            derived_rows = read_tsv(out_dir / s5_5.DERIVED_TSV_NAME)
            volume_rows = [row for row in derived_rows if row["formula_id"] == "mass_mg_div_concentration_mg_per_ml_to_ml_v1"]
            self.assertEqual(len(volume_rows), 1)
            self.assertEqual(volume_rows[0]["target_field_name"], "drug_derived_volume_mL")
            self.assertEqual(volume_rows[0]["derived_value"], "5")
            self.assertEqual(volume_rows[0]["derived_unit"], "mL")
            self.assertEqual(volume_rows[0]["eligible_for_direct_compare"], "no")

    def test_s5_5_derives_binary_ratio_missing_mass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted_tsv = root / "inputs" / s5_4.ACCEPTED_TSV_NAME
            out_dir = root / "out" / "s5_5"
            write_tsv(
                accepted_tsv,
                [
                    "paper_key",
                    "formulation_id",
                    "field_name",
                    "value_text",
                    "unit_text",
                    "source_quote",
                    "evidence_scope",
                    "decision",
                ],
                [
                    {
                        "paper_key": "PAPER_RATIO",
                        "formulation_id": "F1",
                        "field_name": "drug_polymer_ratio",
                        "value_text": "1:10",
                        "unit_text": "",
                        "source_quote": "drug:polymer ratio was 1:10",
                        "evidence_scope": "formulation_row",
                        "decision": "accepted",
                    },
                    {
                        "paper_key": "PAPER_RATIO",
                        "formulation_id": "F1",
                        "field_name": "drug_mass",
                        "value_text": "5",
                        "unit_text": "mg",
                        "source_quote": "drug mass 5 mg",
                        "evidence_scope": "formulation_row",
                        "decision": "accepted",
                    },
                ],
            )

            self.assertEqual(s5_5.main(["--accepted-direct-values-tsv", str(accepted_tsv), "--out-dir", str(out_dir)]), 0)

            derived_rows = read_tsv(out_dir / s5_5.DERIVED_TSV_NAME)
            ratio_rows = [row for row in derived_rows if row["formula_id"] == "role_ratio_known_mass_to_missing_mass_v1"]
            self.assertEqual(len(ratio_rows), 1)
            self.assertEqual(ratio_rows[0]["target_field_name"], "polymer_derived_mass_mg")
            self.assertEqual(ratio_rows[0]["derived_value"], "50")
            self.assertIn("drug_polymer_ratio", ratio_rows[0]["input_field_names"])
            self.assertEqual(ratio_rows[0]["eligible_for_modeling"], "yes")

    def test_s5_5_preserves_explicit_ratio_role_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted_tsv = root / "inputs" / s5_4.ACCEPTED_TSV_NAME
            out_dir = root / "out" / "s5_5"
            write_tsv(
                accepted_tsv,
                [
                    "paper_key",
                    "formulation_id",
                    "field_name",
                    "value_text",
                    "unit_text",
                    "source_quote",
                    "evidence_scope",
                    "decision",
                ],
                [
                    {
                        "paper_key": "PAPER_RATIO_ORDER",
                        "formulation_id": "F1",
                        "field_name": "polymer_to_drug_ratio",
                        "value_text": "10:1",
                        "unit_text": "",
                        "source_quote": "polymer:drug ratio was 10:1",
                        "evidence_scope": "formulation_row",
                        "decision": "accepted",
                    },
                    {
                        "paper_key": "PAPER_RATIO_ORDER",
                        "formulation_id": "F1",
                        "field_name": "polymer_mass",
                        "value_text": "50",
                        "unit_text": "mg",
                        "source_quote": "polymer mass 50 mg",
                        "evidence_scope": "formulation_row",
                        "decision": "accepted",
                    },
                ],
            )

            self.assertEqual(s5_5.main(["--accepted-direct-values-tsv", str(accepted_tsv), "--out-dir", str(out_dir)]), 0)

            derived_rows = read_tsv(out_dir / s5_5.DERIVED_TSV_NAME)
            ratio_rows = [row for row in derived_rows if row["formula_id"] == "role_ratio_known_mass_to_missing_mass_v1"]
            self.assertEqual(len(ratio_rows), 1)
            self.assertEqual(ratio_rows[0]["target_field_name"], "drug_derived_mass_mg")
            self.assertEqual(ratio_rows[0]["derived_value"], "5")

    def test_s5_5_derives_ternary_ratio_missing_masses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted_tsv = root / "inputs" / s5_4.ACCEPTED_TSV_NAME
            out_dir = root / "out" / "s5_5"
            write_tsv(
                accepted_tsv,
                [
                    "paper_key",
                    "formulation_id",
                    "field_name",
                    "value_text",
                    "unit_text",
                    "source_quote",
                    "evidence_scope",
                    "decision",
                ],
                [
                    {
                        "paper_key": "PAPER_TERNARY",
                        "formulation_id": "F1",
                        "field_name": "drug_polymer_surfactant_ratio",
                        "value_text": "1:10:2",
                        "unit_text": "",
                        "source_quote": "drug:polymer:surfactant ratio was 1:10:2",
                        "evidence_scope": "formulation_row",
                        "decision": "accepted",
                    },
                    {
                        "paper_key": "PAPER_TERNARY",
                        "formulation_id": "F1",
                        "field_name": "drug_mass",
                        "value_text": "5",
                        "unit_text": "mg",
                        "source_quote": "drug mass 5 mg",
                        "evidence_scope": "formulation_row",
                        "decision": "accepted",
                    },
                ],
            )

            self.assertEqual(s5_5.main(["--accepted-direct-values-tsv", str(accepted_tsv), "--out-dir", str(out_dir)]), 0)

            derived_rows = read_tsv(out_dir / s5_5.DERIVED_TSV_NAME)
            by_target = {row["target_field_name"]: row for row in derived_rows if row["formula_id"] == "role_ratio_known_mass_to_missing_mass_v1"}
            self.assertEqual(by_target["polymer_derived_mass_mg"]["derived_value"], "50")
            self.assertEqual(by_target["surfactant_derived_mass_mg"]["derived_value"], "10")
            self.assertEqual(by_target["polymer_derived_mass_mg"]["eligible_for_direct_compare"], "no")
            self.assertEqual(by_target["surfactant_derived_mass_mg"]["eligible_for_modeling"], "yes")

    def test_s5_5_ambiguous_ratio_role_order_routes_to_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted_tsv = root / "inputs" / s5_4.ACCEPTED_TSV_NAME
            out_dir = root / "out" / "s5_5"
            write_tsv(
                accepted_tsv,
                [
                    "paper_key",
                    "formulation_id",
                    "field_name",
                    "value_text",
                    "unit_text",
                    "source_quote",
                    "evidence_scope",
                    "decision",
                ],
                [
                    {
                        "paper_key": "PAPER_REVIEW",
                        "formulation_id": "F1",
                        "field_name": "mass_ratio",
                        "value_text": "1:10",
                        "unit_text": "",
                        "source_quote": "mass ratio was 1:10",
                        "evidence_scope": "formulation_row",
                        "decision": "accepted",
                    },
                    {
                        "paper_key": "PAPER_REVIEW",
                        "formulation_id": "F1",
                        "field_name": "drug_mass",
                        "value_text": "5",
                        "unit_text": "mg",
                        "source_quote": "drug mass 5 mg",
                        "evidence_scope": "formulation_row",
                        "decision": "accepted",
                    },
                ],
            )

            self.assertEqual(s5_5.main(["--accepted-direct-values-tsv", str(accepted_tsv), "--out-dir", str(out_dir)]), 0)

            review_rows = read_tsv(out_dir / s5_5.REVIEW_TSV_NAME)
            ratio_reviews = [row for row in review_rows if row["review_reason"] == "ambiguous_ratio_role_order"]
            self.assertEqual(len(ratio_reviews), 1)
            self.assertEqual(ratio_reviews[0]["formula_id"], "role_ratio_mass_solver_v1")
            self.assertEqual(ratio_reviews[0]["eligible_for_direct_compare"], "no")

    def test_s5_5_insufficient_inputs_route_to_review_without_guessing_derived_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted_tsv = root / "inputs" / s5_4.ACCEPTED_TSV_NAME
            out_dir = root / "out" / "s5_5"
            write_tsv(
                accepted_tsv,
                [
                    "paper_key",
                    "formulation_id",
                    "field_name",
                    "value_text",
                    "unit_text",
                    "source_quote",
                    "evidence_scope",
                    "decision",
                ],
                [
                    {
                        "paper_key": "PAPER_INSUFFICIENT",
                        "formulation_id": "F2",
                        "field_name": "concentration_percent_wv",
                        "value_text": "0.5",
                        "unit_text": "%w/v",
                        "source_quote": "0.5% w/v drug solution",
                        "evidence_scope": "formulation_row",
                        "decision": "accepted",
                    }
                ],
            )

            self.assertEqual(s5_5.main(["--accepted-direct-values-tsv", str(accepted_tsv), "--out-dir", str(out_dir)]), 0)

            derived_rows = read_tsv(out_dir / s5_5.DERIVED_TSV_NAME)
            review_rows = read_tsv(out_dir / s5_5.REVIEW_TSV_NAME)
            self.assertEqual(derived_rows, [])
            self.assertEqual(len(review_rows), 1)
            review = review_rows[0]
            self.assertEqual(review["paper_key"], "PAPER_INSUFFICIENT")
            self.assertEqual(review["formulation_id"], "F2")
            self.assertEqual(review["formula_id"], "percent_wv_x_ml_to_mg_v1")
            self.assertEqual(review["review_reason"], "insufficient_inputs_for_formula")
            self.assertEqual(review["missing_input_field_names"], "volume_ml")
            self.assertEqual(review["available_input_field_names"], "percent_wv")
            self.assertEqual(review["eligible_for_direct_compare"], "no")
            self.assertEqual(review["eligible_for_derived_compare"], "no")
            self.assertEqual(review["needs_review"], "yes")
            summary = json.loads((out_dir / s5_5.SUMMARY_JSON_NAME).read_text(encoding="utf-8"))
            self.assertEqual(summary["derived_rows"], 0)
            self.assertEqual(summary["review_rows"], 1)

    def test_stage5_final_output_value_layer_sidecar_manifest_only_integration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            direct_tsv = root / "inputs" / "accepted_direct.tsv"
            derived_tsv = root / "inputs" / "derived.tsv"
            out_dir = root / "out" / "stage5_final"
            write_tsv(
                direct_tsv,
                ["paper_key", "formulation_id", "field_name", "value_text", "direct_or_derived"],
                [
                    {
                        "paper_key": "PAPER_A",
                        "formulation_id": "F1",
                        "field_name": "drug_loading",
                        "value_text": "12.5",
                        "direct_or_derived": "direct",
                    }
                ],
            )
            write_tsv(
                derived_tsv,
                ["paper_key", "formulation_id", "target_field_name", "derived_value", "eligible_for_direct_compare"],
                [
                    {
                        "paper_key": "PAPER_A",
                        "formulation_id": "F1",
                        "target_field_name": "derived_mass_mg",
                        "derived_value": "50",
                        "eligible_for_direct_compare": "no",
                    },
                    {
                        "paper_key": "PAPER_A",
                        "formulation_id": "F2",
                        "target_field_name": "derived_mass_mg",
                        "derived_value": "25",
                        "eligible_for_direct_compare": "no",
                    },
                ],
            )

            manifest = s5_final.write_stage5_value_layer_sidecar_manifest(
                out_dir=out_dir,
                s5_4_accepted_direct_values_tsv=direct_tsv,
                s5_5_derived_values_tsv=derived_tsv,
                final_row_count_before=3,
                final_row_count_after=3,
            )

            self.assertIsNotNone(manifest)
            manifest_tsv = out_dir / s5_final.VALUE_LAYER_SIDECAR_MANIFEST_TSV_NAME
            manifest_json = out_dir / s5_final.VALUE_LAYER_SIDECAR_MANIFEST_JSON_NAME
            self.assertTrue(manifest_tsv.exists())
            self.assertTrue(manifest_json.exists())
            self.assertTrue((out_dir / s5_final.VALUE_LAYER_DIRECT_COPY_NAME).exists())
            self.assertTrue((out_dir / s5_final.VALUE_LAYER_DERIVED_COPY_NAME).exists())

            rows = read_tsv(manifest_tsv)
            self.assertEqual(len(rows), 2)
            by_role = {row["sidecar_role"]: row for row in rows}
            self.assertEqual(by_role["accepted_direct_values"]["separation"], "direct")
            self.assertEqual(by_role["accepted_direct_values"]["row_count"], "1")
            self.assertEqual(by_role["accepted_direct_values"]["benchmark_valid"], "no")
            self.assertEqual(by_role["accepted_direct_values"]["row_membership_changed"], "no")
            self.assertEqual(by_role["accepted_direct_values"]["input_path"], str(direct_tsv.resolve()))
            self.assertEqual(by_role["derived_values"]["separation"], "derived")
            self.assertEqual(by_role["derived_values"]["row_count"], "2")
            self.assertIn("sidecar_manifest_only", by_role["derived_values"]["integration_mode"])

            payload = json.loads(manifest_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["benchmark_valid"], "no")
            self.assertEqual(payload["direct_derived_separation"], "preserved")
            self.assertEqual(payload["direct_values_merge_status"], "not_merged")
            self.assertEqual(payload["derived_values_merge_status"], "not_merged")
            self.assertEqual(payload["final_row_count_before"], 3)
            self.assertEqual(payload["final_row_count_after"], 3)
            self.assertEqual(payload["row_membership_changed"], "no")

            with self.assertRaises(FileNotFoundError):
                s5_final.write_stage5_value_layer_sidecar_manifest(
                    out_dir=out_dir,
                    s5_4_accepted_direct_values_tsv=root / "missing" / "accepted_direct.tsv",
                    final_row_count_before=3,
                    final_row_count_after=3,
                )


if __name__ == "__main__":
    unittest.main()
