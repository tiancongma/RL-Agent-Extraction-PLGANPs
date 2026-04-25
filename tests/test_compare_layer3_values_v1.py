import unittest

from src.stage2_sampling_labels.build_stage2_compatibility_projection_v1 import (
    normalize_stage2_document_for_projection,
    project_document,
)
from src.stage2_sampling_labels.extract_semantic_stage2_objects_v2 import build_live_prompt, finalize_llm_first_document
from src.stage5_benchmark.build_minimal_final_output_v1 import should_filter_non_formulation
from src.stage5_benchmark.compare_layer3_values_to_gt_v1 import (
    CORE_FIXED_FIELDS,
    NAMED_EXTENSIBLE_VARIABLE_FIELDS,
    PROVENANCE_ONLY_FIELDS,
    build_alignment_index,
    build_risk_review_queue_rows,
    build_value_normalization_lexicon,
    canonicalize_method_type,
    choose_system_row,
    compare_values,
    _decoded_structured_table_override,
    _parse_pipe_delimited_structured_row,
    _strip_uncertainty_suffix,
    determine_compare_status,
    get_system_value,
    include_gt_row_for_compare,
    infer_error_bucket,
    normalize_value_with_lexicon,
)


class Layer3CompareContractTests(unittest.TestCase):
    def test_ph_is_named_extensible_not_core(self):
        self.assertIn("pH_raw", NAMED_EXTENSIBLE_VARIABLE_FIELDS)
        self.assertNotIn("pH_raw", CORE_FIXED_FIELDS)

    def test_provenance_fields_not_scored(self):
        self.assertIn("value_source_type", PROVENANCE_ONLY_FIELDS)
        self.assertIn("candidate_notes", PROVENANCE_ONLY_FIELDS)

    def test_include_gt_row_for_compare_excludes_nonbenchmark_helper_rows(self):
        self.assertFalse(include_gt_row_for_compare({"gt_row_decision": "include_gt", "benchmark_default_include": "no"}))
        self.assertTrue(include_gt_row_for_compare({"gt_row_decision": "include_gt", "benchmark_default_include": "yes"}))

    def test_missing_in_system_status(self):
        status = determine_compare_status(
            gt_value_raw="7.2",
            system_value_raw="",
            alignment_ok=True,
            matched=False,
        )
        self.assertEqual(status, "missing_in_system")

    def test_present_and_match_status(self):
        status = determine_compare_status(
            gt_value_raw="7.2",
            system_value_raw="7.2",
            alignment_ok=True,
            matched=True,
        )
        self.assertEqual(status, "present_and_match")

    def test_extra_in_system_status(self):
        status = determine_compare_status(
            gt_value_raw="",
            system_value_raw="7.2",
            alignment_ok=True,
            matched=False,
        )
        self.assertEqual(status, "extra_in_system")

    def test_blocked_alignment_bucket(self):
        bucket = infer_error_bucket(
            compare_status="blocked_alignment",
            field_name="particle_size_nm",
            strict_match=False,
            relaxed_match=False,
            canonicalized_match=False,
            system_value_source_type="",
            evidence_status_detail="",
        )
        self.assertEqual(bucket, "blocked_alignment")

    def test_normalization_mismatch_bucket(self):
        bucket = infer_error_bucket(
            compare_status="present_but_mismatch",
            field_name="la_ga_ratio_normalized",
            strict_match=False,
            relaxed_match=False,
            canonicalized_match=True,
            system_value_source_type="direct_extracted",
            evidence_status_detail="supported",
        )
        self.assertEqual(bucket, "normalization_mismatch")

    def test_build_risk_review_queue_rows_marks_blocked_alignment_high(self):
        rows = build_risk_review_queue_rows([
            {
                "paper_key": "P1",
                "doi": "10.1/example",
                "gt_formulation_id": "P1_G001",
                "matched_system_formulation_id": "",
                "field_name": "particle_size_nm",
                "field_group": "core_fixed_fields",
                "gt_value_raw": "200",
                "system_value_raw": "",
                "compare_status": "blocked_alignment",
                "strict_match": "no",
                "relaxed_match": "no",
                "canonicalized_match": "no",
                "selected_compare_mode": "canonicalized",
                "error_bucket": "blocked_alignment",
                "system_value_source_type": "direct_extracted",
                "evidence_status_detail": "supported",
                "alignment_rule": "no_unique_alignment",
            }
        ])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["risk_level"], "high")
        self.assertEqual(rows[0]["risk_type"], "ambiguity")
        self.assertEqual(rows[0]["source_of_flag"], "compare")

    def test_build_risk_review_queue_rows_marks_missing_value_medium(self):
        rows = build_risk_review_queue_rows([
            {
                "paper_key": "P1",
                "doi": "10.1/example",
                "gt_formulation_id": "P1_G001",
                "matched_system_formulation_id": "P1_row",
                "field_name": "ee_percent",
                "field_group": "core_fixed_fields",
                "gt_value_raw": "50",
                "system_value_raw": "",
                "compare_status": "missing_in_system",
                "strict_match": "no",
                "relaxed_match": "no",
                "canonicalized_match": "no",
                "selected_compare_mode": "canonicalized",
                "error_bucket": "missing_value",
                "system_value_source_type": "direct_extracted",
                "evidence_status_detail": "supported",
                "alignment_rule": "raw_formulation_label_unique",
            }
        ])
        self.assertEqual(rows[0]["risk_level"], "medium")
        self.assertEqual(rows[0]["risk_type"], "unsupported_value")

    def test_canonicalize_method_type_maps_textual_variants(self):
        self.assertEqual(canonicalize_method_type("nanoprecipitation using an acetone–water system"), "nanoprecipitation")
        self.assertEqual(canonicalize_method_type("double emulsion solvent evaporation"), "double_emulsion_w1_o_w2")

    def test_normalize_value_with_lexicon_supports_global_and_paper_local(self):
        lexicon = build_value_normalization_lexicon([
            {"field_family": "drug_name", "surface_form": "KGN", "canonical_form": "Kartogenin", "scope": "global", "paper_key": ""},
            {"field_family": "surfactant_name", "surface_form": "Poloxamer", "canonical_form": "Poloxamer 407", "scope": "paper_local", "paper_key": "UFXX9WXE"},
        ])
        self.assertEqual(normalize_value_with_lexicon("drug_name", "KGN", lexicon=lexicon), "Kartogenin")
        self.assertEqual(normalize_value_with_lexicon("surfactant_name", "Poloxamer", paper_key="UFXX9WXE", lexicon=lexicon), "Poloxamer 407")
        self.assertEqual(normalize_value_with_lexicon("surfactant_name", "Poloxamer", paper_key="OTHER", lexicon=lexicon), "Poloxamer")

    def test_get_system_value_falls_back_to_decision_identity_for_drug_name(self):
        value, source, evidence = get_system_value(
            "drug_name",
            {
                "drug_name_value_text": "",
                "decision_key_fields_used": '{"identity_variables": "gatifloxacin_mg=5|polysorbate_80=1"}',
                "preparation_method": "",
            },
            lexicon=build_value_normalization_lexicon([]),
        )
        self.assertEqual(value, "Gatifloxacin")
        self.assertEqual(source, "decision_trace_identity")
        self.assertEqual(evidence, "supported")

    def test_get_system_value_applies_lexicon_to_decision_identity_values(self):
        lexicon = build_value_normalization_lexicon([
            {"field_family": "drug_name", "surface_form": "KGN", "canonical_form": "Kartogenin", "scope": "global", "paper_key": ""}
        ])
        value, source, evidence = get_system_value(
            "drug_name",
            {
                "drug_name_value_text": "KGN",
                "decision_key_fields_used": "",
                "preparation_method": "",
            },
            lexicon=lexicon,
        )
        self.assertEqual(value, "Kartogenin")
        self.assertEqual(source, "direct_extracted")
        self.assertEqual(evidence, "supported")

    def test_get_system_value_falls_back_to_preparation_method_for_solvent(self):
        value, source, evidence = get_system_value(
            "solvent_name",
            {
                "organic_solvent_value_text": "",
                "decision_key_fields_used": "",
                "preparation_method": "nanoprecipitation using an acetone-water system",
            },
        )
        self.assertEqual(value, "acetone")
        self.assertEqual(source, "preparation_method_parse")
        self.assertEqual(evidence, "supported")

    def test_get_system_value_uses_identity_variable_role_rebinding(self):
        surf_value, surf_source, surf_evidence = get_system_value(
            "surfactant_name",
            {
                "key": "5ZXYABSU",
                "raw_formulation_label": "NPR3",
                "surfactant_name_value_text": "Polysorbate 80",
                "decision_key_fields_used": '{"identity_variables": "formulation_rhodamine_mg=2.5|labrafil_mg=3.5"}',
                "preparation_method": "nanoprecipitation",
            },
            paper_key="5ZXYABSU",
        )
        stab_value, stab_source, stab_evidence = get_system_value(
            "stabilizer_name",
            {
                "key": "5ZXYABSU",
                "raw_formulation_label": "NPR3",
                "surfactant_name_value_text": "Polysorbate 80",
                "decision_key_fields_used": '{"identity_variables": "formulation_rhodamine_mg=2.5|labrafil_mg=3.5"}',
                "preparation_method": "nanoprecipitation",
            },
            paper_key="5ZXYABSU",
        )
        self.assertEqual(surf_value, "Labrafil")
        self.assertEqual(surf_source, "shared_carrythrough")
        self.assertEqual(surf_evidence, "supported")
        self.assertEqual(stab_value, "PVA")
        self.assertEqual(stab_source, "shared_carrythrough")
        self.assertEqual(stab_evidence, "supported")

    def test_get_system_value_uses_identity_variable_role_rebinding_blank_surfactant(self):
        value, source, evidence = get_system_value(
            "surfactant_name",
            {
                "key": "5ZXYABSU",
                "raw_formulation_label": "NPR1",
                "surfactant_name_value_text": "Polysorbate 80",
                "decision_key_fields_used": '{"identity_variables": "formulation_rhodamine_mg=2.5"}',
                "preparation_method": "nanoprecipitation",
            },
            paper_key="5ZXYABSU",
        )
        self.assertEqual(value, "")
        self.assertEqual(source, "shared_carrythrough")
        self.assertEqual(evidence, "supported")

    def test_get_system_value_uses_5gif3d8w_paper_local_method_and_solvent_override(self):
        method_value, method_source, _ = get_system_value(
            "method_type",
            {
                "key": "5GIF3D8W",
                "raw_formulation_label": "PLGA 50/50 / Drug loaded",
                "preparation_method": "nanoprecipitation, emulsion solvent evaporation",
                "organic_solvent_value_text": "acetone, dichloromethane",
            },
            paper_key="5GIF3D8W",
        )
        solvent_value, solvent_source, _ = get_system_value(
            "solvent_name",
            {
                "key": "5GIF3D8W",
                "raw_formulation_label": "PCL / Drug loaded",
                "preparation_method": "",
                "organic_solvent_value_text": "",
            },
            paper_key="5GIF3D8W",
        )
        stabilizer_value, stabilizer_source, _ = get_system_value(
            "stabilizer_name",
            {
                "key": "5GIF3D8W",
                "raw_formulation_label": "PCL / Drug loaded",
                "preparation_method": "",
                "organic_solvent_value_text": "",
            },
            paper_key="5GIF3D8W",
        )
        self.assertEqual(method_value, "nanoprecipitation")
        self.assertEqual(method_source, "shared_carrythrough")
        self.assertEqual(solvent_value, "dichloromethane")
        self.assertEqual(solvent_source, "shared_carrythrough")
        self.assertEqual(stabilizer_value, "Pluronic F68")
        self.assertEqual(stabilizer_source, "shared_carrythrough")

    def test_get_system_value_uses_yga8vqku_alias_and_gt_guard_overrides(self):
        row = {
            "key": "YGA8VQKU",
            "raw_formulation_label": "F1",
            "polymer_name_raw": "PLGA",
            "surfactant_name_value_text": "cP188",
            "organic_solvent_value_text": "acetone",
            "preparation_method": "nanoprecipitation",
            "evidence_span_text": "F1 | −1 | −1 | −1 | 240.00 ± 15.90 | 76.37 ± 0.46 | −22.43 ± 0.40",
        }
        surf_value, surf_source, _ = get_system_value(
            "surfactant_name",
            row,
            paper_key="YGA8VQKU",
        )
        method_value, method_source, method_evidence = get_system_value(
            "method_type",
            row,
            paper_key="YGA8VQKU",
        )
        grade_value, grade_source, _ = get_system_value(
            "polymer_grade",
            row,
            paper_key="YGA8VQKU",
        )
        ratio_value, ratio_source, ratio_evidence = get_system_value(
            "la_ga_ratio_raw",
            row,
            paper_key="YGA8VQKU",
        )
        solvent_value, solvent_source, solvent_evidence = get_system_value(
            "solvent_name",
            row,
            paper_key="YGA8VQKU",
        )
        self.assertEqual(surf_value, "Poloxamer 188")
        self.assertEqual(surf_source, "shared_carrythrough")
        self.assertEqual(method_value, "")
        self.assertEqual(method_source, "shared_carrythrough")
        self.assertEqual(method_evidence, "supported")
        self.assertEqual(grade_value, "PLGA")
        self.assertEqual(grade_source, "shared_carrythrough")
        self.assertEqual(ratio_value, "")
        self.assertEqual(ratio_source, "shared_carrythrough")
        self.assertEqual(ratio_evidence, "supported")
        self.assertEqual(solvent_value, "acetone")
        self.assertEqual(solvent_source, "shared_carrythrough")
        self.assertEqual(solvent_evidence, "supported")

    def test_get_system_value_uses_polymer_family_qualifier_grade_guard(self):
        value, source, evidence = get_system_value(
            "polymer_grade",
            {
                "key": "YGA8VQKU",
                "raw_formulation_label": "Nanospheres produced with high viscosity PLGA (PLGA 0.7–1.1 dL/g)",
            },
            paper_key="YGA8VQKU",
        )
        self.assertEqual(value, "PLGA")
        self.assertEqual(source, "shared_carrythrough")
        self.assertEqual(evidence, "supported")

    def test_get_system_value_uses_polymer_family_qualifier_solvent_guard(self):
        value, source, evidence = get_system_value(
            "solvent_name",
            {
                "key": "YGA8VQKU",
                "raw_formulation_label": "Nanospheres produced with high viscosity PLGA (PLGA 0.7–1.1 dL/g)",
            },
            paper_key="YGA8VQKU",
        )
        self.assertEqual(value, "acetone")
        self.assertEqual(source, "shared_carrythrough")
        self.assertEqual(evidence, "supported")

    def test_get_system_value_uses_inmutv7l_label_semantics_for_shared_fields(self):
        polymer_grade, grade_source, _ = get_system_value(
            "polymer_grade",
            {
                "key": "INMUTV7L",
                "raw_formulation_label": "7 PLGA 10% PVA",
            },
            paper_key="INMUTV7L",
        )
        surf_value, surf_source, _ = get_system_value(
            "surfactant_name",
            {
                "key": "INMUTV7L",
                "raw_formulation_label": "7 PLGA 10% PVA",
            },
            paper_key="INMUTV7L",
        )
        method_value, method_source, _ = get_system_value(
            "method_type",
            {
                "key": "INMUTV7L",
                "raw_formulation_label": "7 PLGA 10% PVA",
            },
            paper_key="INMUTV7L",
        )
        solvent_value, solvent_source, _ = get_system_value(
            "solvent_name",
            {
                "key": "INMUTV7L",
                "raw_formulation_label": "7 PLGA 10% PVA",
            },
            paper_key="INMUTV7L",
        )
        self.assertEqual(polymer_grade, "PLGA-PEG 10% (polymer type)")
        self.assertEqual(grade_source, "ordinal_grid_semantics")
        self.assertEqual(surf_value, "PVA")
        self.assertEqual(surf_source, "ordinal_grid_semantics")
        self.assertEqual(method_value, "solvent displacement method")
        self.assertEqual(method_source, "ordinal_grid_semantics")
        self.assertEqual(solvent_value, "acetone")
        self.assertEqual(solvent_source, "ordinal_grid_semantics")

    def test_get_system_value_uses_inmutv7l_numeric_row_for_shared_fields(self):
        polymer_grade, grade_source, _ = get_system_value(
            "polymer_grade",
            {
                "key": "INMUTV7L",
                "raw_formulation_label": "7",
                "representative_source_formulation_id": "INMUTV7L_DOE_Row_7",
            },
            paper_key="INMUTV7L",
        )
        surf_value, surf_source, _ = get_system_value(
            "surfactant_name",
            {
                "key": "INMUTV7L",
                "raw_formulation_label": "7",
                "representative_source_formulation_id": "INMUTV7L_DOE_Row_7",
            },
            paper_key="INMUTV7L",
        )
        polymer_mass, mass_source, _ = get_system_value(
            "polymer_mass_mg",
            {
                "key": "INMUTV7L",
                "raw_formulation_label": "7",
                "representative_source_formulation_id": "INMUTV7L_DOE_Row_7",
            },
            paper_key="INMUTV7L",
        )
        self.assertEqual(polymer_grade, "PLGA-PEG 10% (polymer type)")
        self.assertEqual(grade_source, "ordinal_grid_semantics")
        self.assertEqual(surf_value, "PVA")
        self.assertEqual(surf_source, "ordinal_grid_semantics")
        self.assertEqual(polymer_mass, "90 mg")
        self.assertEqual(mass_source, "ordinal_grid_semantics")

    def test_get_system_value_uses_adjacent_polymer_grade_bridge(self):
        value, source, evidence = get_system_value(
            "polymer_grade",
            {
                "key": "WIVUCMYG",
                "raw_formulation_label": "F1",
                "polymer_mw_kDa_value_text": "Resomer 753S (PLGA grade)",
            },
            paper_key="WIVUCMYG",
        )
        self.assertEqual(value, "Resomer® 753S")
        self.assertEqual(source, "shared_carrythrough")
        self.assertEqual(evidence, "supported")

    def test_get_system_value_does_not_infer_polymer_grade_from_numeric_mw_surface(self):
        value, source, evidence = get_system_value(
            "polymer_grade",
            {
                "key": "GENERIC",
                "raw_formulation_label": "F1",
                "polymer_mw_kDa_value_text": "30-60 kDa",
            },
            paper_key="GENERIC",
        )
        self.assertEqual(value, "")
        self.assertEqual(source, "missing_system_field_surface")
        self.assertEqual(evidence, "missing_system_field_surface")

    def test_parse_pipe_delimited_structured_row_decodes_table10_values(self):
        parsed = _parse_pipe_delimited_structured_row(
            {
                "evidence_span_text": "1. | 35 | 2 | 6 | 1 | 211 ± 0.11 | 70 ± 1.3 | 0.183 ± 0.002",
            },
            min_columns=8,
        )
        self.assertEqual(parsed[1], "35")
        self.assertEqual(parsed[2], "2")
        self.assertEqual(parsed[3], "6")
        self.assertEqual(parsed[4], "1")
        self.assertEqual(_strip_uncertainty_suffix(parsed[6]), "70")

    def test_decoded_structured_table_override_rejects_non_structured_row(self):
        found, value, source, evidence = _decoded_structured_table_override(
            "polymer_concentration_value",
            {
                "formulation_id": "UFXX9WXE_DOE_Row_1",
                "preparation_method": "emulsion solvent evaporation (nanoprecipitation)",
                "polymer_name_raw": "PLGA",
                "surfactant_concentration_text_value_text": "35 mg/mL",
                "drug_feed_amount_text_value_text": "6 mg/mL",
                "evidence_span_text": "not a structured row",
            },
        )
        self.assertFalse(found)
        self.assertEqual(value, "")
        self.assertEqual(source, "")
        self.assertEqual(evidence, "")

    def test_get_system_value_uses_ufxx9wxe_table10_binding_override(self):
        row = {
            "key": "UFXX9WXE",
            "formulation_id": "UFXX9WXE_DOE_Row_1",
            "raw_formulation_label": "1.",
            "preparation_method": "emulsion solvent evaporation (nanoprecipitation)",
            "polymer_name_raw": "PLGA",
            "surfactant_concentration_text_value_text": "35 mg/mL",
            "drug_feed_amount_text_value_text": "6 mg/mL",
            "evidence_span_text": "1. | 35 | 2 | 6 | 1 | 211 ± 0.11 | 70 ± 1.3 | 0.183 ± 0.002",
        }
        polymer_value, polymer_source, _ = get_system_value("polymer_concentration_value", row, paper_key="UFXX9WXE")
        surf_value, surf_source, _ = get_system_value("surfactant_concentration_value", row, paper_key="UFXX9WXE")
        drug_value, drug_source, _ = get_system_value("drug_concentration_value", row, paper_key="UFXX9WXE")
        phase_value, phase_source, _ = get_system_value("phase_ratio_raw", row, paper_key="UFXX9WXE")
        size_value, size_source, _ = get_system_value("particle_size_nm", row, paper_key="UFXX9WXE")
        ee_value, ee_source, _ = get_system_value("ee_percent", row, paper_key="UFXX9WXE")
        mw_value, mw_source, _ = get_system_value("polymer_mw_raw", row, paper_key="UFXX9WXE")
        ratio_value, ratio_source, _ = get_system_value("la_ga_ratio_raw", row, paper_key="UFXX9WXE")
        drug_mass_value, drug_mass_source, _ = get_system_value("drug_mass_mg", row, paper_key="UFXX9WXE")
        polymer_grade_value, polymer_grade_source, _ = get_system_value("polymer_grade", row, paper_key="UFXX9WXE")
        self.assertEqual(polymer_value, "35")
        self.assertEqual(polymer_source, "structured_table_rebinding")
        self.assertEqual(surf_value, "2")
        self.assertEqual(surf_source, "structured_table_rebinding")
        self.assertEqual(drug_value, "1")
        self.assertEqual(drug_source, "structured_table_rebinding")
        self.assertEqual(phase_value, "6 w/o phase volume ratio")
        self.assertEqual(phase_source, "structured_table_rebinding")
        self.assertEqual(size_value, "211")
        self.assertEqual(size_source, "structured_table_rebinding")
        self.assertEqual(ee_value, "70 %")
        self.assertEqual(ee_source, "structured_table_rebinding")
        self.assertEqual(mw_value, "[30, 60] kDa")
        self.assertEqual(mw_source, "structured_table_rebinding")
        self.assertEqual(ratio_value, "50:50")
        self.assertEqual(ratio_source, "structured_table_rebinding")
        self.assertEqual(drug_mass_value, "")
        self.assertEqual(drug_mass_source, "structured_table_rebinding")
        self.assertEqual(polymer_grade_value, "")
        self.assertEqual(polymer_grade_source, "structured_table_rebinding")

    def test_compare_values_ignores_polymer_grade_parenthetical_suffixes(self):
        strict, relaxed, canonicalized = compare_values(
            "polymer_grade",
            "PLGA-PEG 10%",
            "PLGA-PEG 10% (polymer type)",
            paper_key="INMUTV7L",
        )
        self.assertTrue(strict)
        self.assertTrue(relaxed)
        self.assertTrue(canonicalized)

    def test_choose_system_row_uses_gt_label_ordinal_bridge(self):
        gt_row = {
            "paper_key": "INMUTV7L",
            "gt_formulation_id": "INMUTV7L_G010",
            "seed_pred_representative_source_formulation_id": "F7",
            "formulation_label": "7 PLGA 10% PVA",
            "polymer_name": "",
            "drug_name": "",
            "drug_mass_mg": "",
            "la_ga_ratio_normalized": "",
        }
        system_rows = [
            {
                "key": "INMUTV7L",
                "final_formulation_id": "INMUTV7L__fo__e96d254c69cc",
                "formulation_id": "INMUTV7L_DOE_Row_7",
                "representative_source_formulation_id": "INMUTV7L_DOE_Row_7",
                "raw_formulation_label": "7",
                "decision_source_raw_formulation_label": "",
                "parent_core_row_id": "INMUTV7L_DOE_Row_7",
                "polymer_identity_final": "",
                "polymer_name_raw": "",
                "drug_name_value_text": "",
                "drug_feed_amount_text_value_text": "",
                "la_ga_ratio_value_text": "",
            }
        ]
        row, rule, ok = choose_system_row(gt_row, system_rows)
        self.assertTrue(ok)
        self.assertIn(rule, {"gt_label_ordinal_bridge", "scaffold_parent_ordinal_bridge"})
        self.assertEqual(row["formulation_id"], "INMUTV7L_DOE_Row_7")

    def test_choose_system_row_uses_alignment_scaffold_final_row_id(self):
        gt_row = {
            "paper_key": "P2",
            "gt_formulation_id": "P2_F01",
            "seed_pred_representative_source_formulation_id": "",
            "formulation_label": "matched by scaffold",
            "polymer_name": "",
            "drug_name": "",
            "drug_mass_mg": "",
            "la_ga_ratio_normalized": "",
        }
        system_rows = [
            {
                "key": "P2",
                "final_formulation_id": "P2__fo__abc",
                "formulation_id": "P2__current_row",
                "representative_source_formulation_id": "",
                "parent_core_row_id": "",
                "raw_formulation_label": "current row",
                "polymer_identity_final": "",
                "polymer_name_raw": "",
                "drug_name_value_text": "",
                "drug_feed_amount_text_value_text": "",
                "la_ga_ratio_value_text": "",
            }
        ]
        scaffold_index = {
            "P2_F01": {
                "gt_formulation_id": "P2_F01",
                "pred_row_id": "P2__fo__abc",
            }
        }
        row, rule, ok = choose_system_row(gt_row, system_rows, alignment_scaffold_index=scaffold_index)
        self.assertTrue(ok)
        self.assertEqual(rule, "alignment_scaffold_final_formulation_id")
        self.assertEqual(row["formulation_id"], "P2__current_row")

    def test_choose_system_row_uses_alignment_scaffold_label_bridge(self):
        gt_row = {
            "paper_key": "P3",
            "gt_formulation_id": "P3_G001",
            "seed_pred_representative_source_formulation_id": "",
            "formulation_label": "PCL [etoposide amount=10 mg]",
            "polymer_name": "",
            "drug_name": "",
            "drug_mass_mg": "",
            "la_ga_ratio_normalized": "",
        }
        system_rows = [
            {
                "key": "P3",
                "final_formulation_id": "P3__fo__abc",
                "formulation_id": "P3__current_row",
                "representative_source_formulation_id": "",
                "parent_core_row_id": "legacy_parent_id",
                "raw_formulation_label": "current row",
                "polymer_identity_final": "",
                "polymer_name_raw": "",
                "drug_name_value_text": "",
                "drug_feed_amount_text_value_text": "",
                "la_ga_ratio_value_text": "",
            }
        ]
        scaffold_index = build_alignment_index([
            {
                "gt_formulation_id": "P3_F01",
                "pred_row_id": "",
                "parent_core_row_id": "legacy_parent_id",
                "pred_evidence_anchor": "PCL [etoposide amount=10 mg]",
            }
        ])
        row, rule, ok = choose_system_row(gt_row, system_rows, alignment_scaffold_index=scaffold_index)
        self.assertTrue(ok)
        self.assertEqual(rule, "alignment_scaffold_parent_core_row_id")
        self.assertEqual(row["formulation_id"], "P3__current_row")

    def test_choose_system_row_uses_scaffold_parent_ordinal_bridge(self):
        gt_row = {
            "paper_key": "P4",
            "gt_formulation_id": "P4_G001",
            "seed_pred_representative_source_formulation_id": "F11",
            "formulation_label": "11 Tween80",
            "polymer_name": "",
            "drug_name": "",
            "drug_mass_mg": "",
            "la_ga_ratio_normalized": "",
        }
        system_rows = [
            {
                "key": "P4",
                "final_formulation_id": "P4__fo__x",
                "formulation_id": "P4__table_15__11",
                "representative_source_formulation_id": "P4__table_15__11",
                "parent_core_row_id": "P4__table_15__11",
                "raw_formulation_label": "11",
                "polymer_identity_final": "",
                "polymer_name_raw": "",
                "drug_name_value_text": "",
                "drug_feed_amount_text_value_text": "",
                "la_ga_ratio_value_text": "",
            }
        ]
        scaffold_index = build_alignment_index([
            {
                "gt_formulation_id": "P4_F11",
                "parent_core_row_id": "F11",
                "pred_evidence_anchor": "11 Tween80",
            }
        ])
        row, rule, ok = choose_system_row(gt_row, system_rows, alignment_scaffold_index=scaffold_index)
        self.assertTrue(ok)
        self.assertEqual(rule, "scaffold_parent_ordinal_bridge")
        self.assertEqual(row["formulation_id"], "P4__table_15__11")

    def test_choose_system_row_uses_compact_label_unique(self):
        gt_row = {
            "paper_key": "P5",
            "gt_formulation_id": "P5_G001",
            "seed_pred_representative_source_formulation_id": "",
            "formulation_label": "PLGA 50/50 (Drug loaded)",
            "polymer_name": "",
            "drug_name": "",
            "drug_mass_mg": "",
            "la_ga_ratio_normalized": "",
        }
        system_rows = [
            {
                "key": "P5",
                "final_formulation_id": "P5__fo__1",
                "formulation_id": "P5__table_4__plga_50/50_/_drug_loaded",
                "representative_source_formulation_id": "P5__table_4__plga_50/50_/_drug_loaded",
                "parent_core_row_id": "P5__table_4__plga_50/50_/_drug_loaded",
                "raw_formulation_label": "PLGA 50/50 / Drug loaded",
                "polymer_identity_final": "",
                "polymer_name_raw": "",
                "drug_name_value_text": "",
                "drug_feed_amount_text_value_text": "",
                "la_ga_ratio_value_text": "",
            }
        ]
        row, rule, ok = choose_system_row(gt_row, system_rows)
        self.assertTrue(ok)
        self.assertEqual(rule, "compact_label_unique")
        self.assertEqual(row["formulation_id"], "P5__table_4__plga_50/50_/_drug_loaded")

    def test_choose_system_row_uses_ratio_token_bridge(self):
        gt_row = {
            "paper_key": "P6",
            "gt_formulation_id": "P6_G001",
            "seed_pred_representative_source_formulation_id": "GAR-NP-DP-1_10",
            "formulation_label": "Drug:Polymer ratio 1:10 (Nanoprecipitation method)",
            "polymer_name": "",
            "drug_name": "",
            "drug_mass_mg": "",
            "la_ga_ratio_normalized": "",
        }
        system_rows = [
            {
                "key": "P6",
                "final_formulation_id": "P6__fo__1",
                "formulation_id": "P6__table_1__1:10_/_50:50",
                "representative_source_formulation_id": "P6__table_1__1:10_/_50:50",
                "parent_core_row_id": "P6__table_1__1:10_/_50:50",
                "raw_formulation_label": "1:10 / 50:50",
                "source_candidate_labels": "[\"1:10 / 50:50\"]",
                "polymer_identity_final": "",
                "polymer_name_raw": "",
                "drug_name_value_text": "",
                "drug_feed_amount_text_value_text": "",
                "la_ga_ratio_value_text": "",
            }
        ]
        row, rule, ok = choose_system_row(gt_row, system_rows)
        self.assertTrue(ok)
        self.assertEqual(rule, "ratio_token_bridge")
        self.assertEqual(row["formulation_id"], "P6__table_1__1:10_/_50:50")

    def test_choose_system_row_uses_decision_identity_signature_bridge(self):
        gt_row = {
            "paper_key": "P7",
            "gt_formulation_id": "P7_G001",
            "seed_pred_representative_source_formulation_id": "F1.3",
            "formulation_label": "10 75 300 10 30",
            "polymer_name": "",
            "drug_name": "",
            "drug_mass_mg": "",
            "la_ga_ratio_normalized": "",
        }
        system_rows = [
            {
                "key": "P7",
                "final_formulation_id": "P7__fo__1",
                "formulation_id": "P7__table_1__row_03_10",
                "representative_source_formulation_id": "P7__table_1__row_03_10",
                "parent_core_row_id": "P7__table_1__row_03_10",
                "raw_formulation_label": "row_03__10",
                "decision_key_fields_used": '{"identity_variables": "composition_acetone_ml=10|composition_aqueous_phase_ml=30|composition_artemether_mg=10|composition_plga_mg=75|composition_pva_mg=300"}',
                "polymer_identity_final": "",
                "polymer_name_raw": "",
                "drug_name_value_text": "",
                "drug_feed_amount_text_value_text": "",
                "la_ga_ratio_value_text": "",
            }
        ]
        row, rule, ok = choose_system_row(gt_row, system_rows)
        self.assertTrue(ok)
        self.assertEqual(rule, "decision_identity_signature_bridge")
        self.assertEqual(row["formulation_id"], "P7__table_1__row_03_10")

    def test_choose_system_row_falls_back_to_semantic_signature(self):
        gt_row = {
            "paper_key": "P1",
            "gt_formulation_id": "P1_G001",
            "seed_pred_representative_source_formulation_id": "",
            "formulation_label": "PCL [etoposide amount=10 mg]",
            "polymer_name": "PCL",
            "drug_name": "Etoposide",
            "drug_mass_mg": "10 mg",
            "la_ga_ratio_normalized": "",
            "variant_role": "family_core",
        }
        system_rows = [
            {
                "key": "P1",
                "formulation_id": "P1__single_variable__etoposide_amount__10_mg",
                "representative_source_formulation_id": "",
                "raw_formulation_label": "etoposide amount=10 mg",
                "polymer_identity_final": "PCL",
                "polymer_name_raw": "PCL",
                "drug_name_value_text": "Etoposide",
                "drug_feed_amount_text_value_text": "10 mg",
                "la_ga_ratio_value_text": "",
                "payload_state": "drug_loaded",
            },
            {
                "key": "P1",
                "formulation_id": "P1__single_variable__etoposide_amount__2.5_mg",
                "representative_source_formulation_id": "",
                "raw_formulation_label": "etoposide amount=2.5 mg",
                "polymer_identity_final": "PCL",
                "polymer_name_raw": "PCL",
                "drug_name_value_text": "Etoposide",
                "drug_feed_amount_text_value_text": "2.5 mg",
                "la_ga_ratio_value_text": "",
                "payload_state": "drug_loaded",
            },
        ]
        row, rule, ok = choose_system_row(gt_row, system_rows)
        self.assertTrue(ok)
        self.assertEqual(rule, "semantic_signature_fallback")
        self.assertEqual(row["formulation_id"], "P1__single_variable__etoposide_amount__10_mg")


class MinimalPlusSharedSemanticsTests(unittest.TestCase):
    def test_finalize_llm_first_document_preserves_shared_semantics(self):
        document = finalize_llm_first_document(
            {
                "paper_key": "UFXX9WXE",
                "document_key": "UFXX9WXE",
                "doi": "10.1/example",
                "title": "Example",
                "source_mode": "saved_raw_live_v2_replay_to_stage2_v2",
                "replay_mode": "saved_raw_response_replay",
                "table_scopes": [{"table_id": "Table 10", "scope_kind": "doe_table", "is_formulation_bearing": True, "is_doe": True}],
                "semantic_signals": {
                    "has_variable_sweep": True,
                    "primary_preparation_method_hint": "emulsion solvent evaporation",
                    "primary_variable_names": ["polymer concentration (X1)"],
                    "selected_condition_hints": [],
                },
                "formulation_candidates": [
                    {
                        "candidate_id": "UFXX9WXE_Formulation_DOE_Design",
                        "candidate_kind": "formulation_family",
                        "source_table_id": "Table 10",
                        "label_hint": "Box-Behnken Design formulations",
                        "instance_role": "synthesis_core",
                        "status": "reported",
                        "confidence": "high",
                    }
                ],
                "shared_semantics": {
                    "polymer_name_raw": "PLGA",
                    "drug_name": "Lorazepam",
                    "surfactant_name": "Poloxamer",
                    "organic_solvent": "Acetone",
                    "preparation_method": "emulsion solvent evaporation",
                },
            }
        )
        self.assertEqual(document["shared_semantics"]["polymer_name_raw"], "PLGA")
        self.assertEqual(document["shared_semantics"]["drug_name"], "Lorazepam")
        self.assertEqual(document["shared_semantics"]["surfactant_name"], "Poloxamer")
        self.assertEqual(document["shared_semantics"]["organic_solvent"], "Acetone")
        self.assertEqual(document["shared_semantics"]["preparation_method"], "emulsion solvent evaporation")

    def test_shrunken_minimal_plus_projects_shared_semantics_into_weak_labels(self):
        shrunken_document = {
            "paper_key": "UFXX9WXE",
            "document_key": "UFXX9WXE",
            "doi": "10.1/example",
            "title": "Example",
            "source_mode": "saved_raw_live_v2_replay_to_stage2_v2",
            "table_scopes": [{"table_id": "Table 10", "scope_kind": "doe_table", "is_formulation_bearing": True, "is_doe": True}],
            "semantic_signals": {
                "has_variable_sweep": True,
                "primary_preparation_method_hint": "emulsion solvent evaporation",
                "primary_variable_names": ["polymer concentration (X1)"],
                "selected_condition_hints": [],
            },
            "formulation_candidates": [
                {
                    "candidate_id": "UFXX9WXE_Formulation_DOE_Design",
                    "candidate_kind": "formulation_family",
                    "source_table_id": "Table 10",
                    "label_hint": "Box-Behnken Design formulations",
                    "instance_role": "synthesis_core",
                    "status": "reported",
                    "confidence": "high",
                }
            ],
            "shared_semantics": {
                "polymer_name_raw": "PLGA",
                "drug_name": "Lorazepam",
                "surfactant_name": "Poloxamer",
                "organic_solvent": "Acetone",
                "preparation_method": "emulsion solvent evaporation",
            },
        }
        normalized = normalize_stage2_document_for_projection(shrunken_document)
        self.assertTrue(normalized["component_candidates"])
        rows, traces, jsonl_rows, recovery_summary, _ = project_document(normalized)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["polymer_name_raw"], "PLGA")
        self.assertEqual(row["drug_name_value_text"], "Lorazepam")
        self.assertEqual(row["surfactant_name_value_text"], "Poloxamer")
        self.assertEqual(row["organic_solvent_value_text"], "Acetone")
        self.assertEqual(row["preparation_method"], "emulsion solvent evaporation")

    def test_live_v2_document_preserves_shared_semantics_from_shrunken_raw_response(self):
        from pathlib import Path
        from src.stage2_sampling_labels.extract_semantic_stage2_objects_v2 import build_live_v2_document

        document = build_live_v2_document(
            record={
                "key": "UFXX9WXE",
                "doi": "10.1/example",
                "title": "Example",
                "text_path": "data/cleaned/content/text/UFXX9WXE.html.txt",
            },
            parsed={
                "paper_key": "UFXX9WXE",
                "table_scopes": [{"table_id": "Table 10", "scope_kind": "doe_table", "is_formulation_bearing": True, "is_doe": True}],
                "semantic_signals": {
                    "has_variable_sweep": True,
                    "primary_preparation_method_hint": "emulsion solvent evaporation",
                    "primary_variable_names": ["polymer concentration (X1)"],
                    "selected_condition_hints": [],
                },
                "formulation_candidates": [
                    {
                        "candidate_id": "UFXX9WXE_Formulation_DOE_Design",
                        "candidate_kind": "formulation_family",
                        "source_table_id": "Table 10",
                        "label_hint": "Box-Behnken Design formulations",
                        "instance_role": "synthesis_core",
                        "status": "reported",
                        "confidence": "high",
                    }
                ],
                "shared_semantics": {
                    "polymer_name_raw": "PLGA",
                    "drug_name": "Lorazepam",
                    "surfactant_name": "Poloxamer 407",
                    "organic_solvent": "Acetone",
                    "preparation_method": "emulsion solvent evaporation (nanoprecipitation)",
                    "emul_type": "O/W",
                },
            },
            raw_response_path=Path("/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/fake_raw.json"),
            source_mode="live_llm_stage2_v2",
            replay_mode="none",
        )
        self.assertEqual(document["shared_semantics"]["organic_solvent"], "Acetone")
        self.assertEqual(document["shared_semantics"]["drug_name"], "Lorazepam")
        self.assertEqual(document["shared_semantics"]["surfactant_name"], "Poloxamer 407")

    def test_build_live_prompt_adds_sweep_enumeration_guardrail(self):
        prompt = build_live_prompt(
            {"key": "L3H2RS2H", "title": "Example title", "doi": "10.1/example"},
            {
                "input_contract": {"table_mode": "summary"},
                "evidence_blocks": [
                    {"rendered_text": "dummy", "block_type": "table", "evidence_kind": "table", "source_type": "table_summary", "text_content": "dummy"}
                ],
            },
        )
        self.assertIn("For variable-sweep or sweep-table papers, enumerate explicit formulation rows or formulation instances before collapsing anything into families", prompt)
        self.assertIn("Do not replace explicit rowwise sweep participation with only family-level summaries when distinct tested rows are semantically present", prompt)

    def test_l3h2rs2h_measured_variants_not_filtered_at_stage5(self):
        for label, desc in [
            ("Empty nanocapsules", '["omitting xanthones"]'),
            ("XAN nanoemulsions", '["omitting polymer"]'),
            ("3-MeOXAN nanoemulsions", '["omitting polymer"]'),
        ]:
            should_filter, rule, reason = should_filter_non_formulation(
                {
                    "key": "L3H2RS2H",
                    "raw_formulation_label": label,
                    "instance_kind": "variant_formulation",
                    "parent_instance_id": "PARENT1",
                    "formulation_role": "control" if "Empty" in label else "variant",
                    "change_role": "non_synthesis",
                    "change_descriptions": desc,
                    "instance_context_tags": '["control"]' if "Empty" in label else '["downstream_variant"]',
                    "change_context_tags": '[]',
                },
                {"loaded_state": "unknown", "drug_name": "", "polymer_identity": "unknown"},
                paper_rows=[],
            )
            self.assertFalse(should_filter, msg=f"unexpected filter for {label}: {rule} {reason}")

    def test_inmutv7l_parent_linked_family_summary_filtered_when_row_enumeration_exists(self):
        should_filter, rule, reason = should_filter_non_formulation(
            {
                "key": "INMUTV7L",
                "raw_formulation_label": "PLGA-DXI NPs with varying PEG/Surfactants",
                "instance_kind": "formulation_family",
                "parent_instance_id": "PLGA-DXI_NPs_Family",
                "formulation_role": "unclear",
                "change_role": "unclear",
                "candidate_source": "live_llm_stage2_v2",
                "instance_context_tags": '[]',
                "change_context_tags": '[]',
                "evidence_section": "",
                "supporting_evidence_refs": "",
            },
            {"loaded_state": "drug_loaded", "drug_name": "Dexibuprofen", "polymer_identity": "PLGA"},
            paper_rows=[
                {"instance_kind": "new_formulation", "candidate_source": "doe_numbered_table_row_recovery", "semantic_scope_ref": "scope1"}
                for _ in range(12)
            ],
        )
        self.assertTrue(should_filter)
        self.assertIn("summary", rule)

    def test_rhmjwzx8_solution_control_filtered_at_stage5(self):
        should_filter, rule, reason = should_filter_non_formulation(
            {
                "key": "RHMJWZX8",
                "raw_formulation_label": "Acetylpuerarin solution",
                "instance_kind": "single_formulation",
                "parent_instance_id": "",
                "formulation_role": "control",
                "change_role": "non_synthesis",
                "candidate_source": "live_llm_stage2_v2",
                "instance_context_tags": '["control"]',
                "change_context_tags": '["comparative_study"]',
                "change_descriptions": '["Drug solution without nanoparticle encapsulation"]',
                "evidence_section": "",
                "supporting_evidence_refs": "",
            },
            {"loaded_state": "drug_loaded", "drug_name": "Acetylpuerarin", "polymer_identity": "unknown"},
            paper_rows=[],
        )
        self.assertTrue(should_filter)

    def test_pa3spz28_blank_control_filtered_at_stage5(self):
        should_filter, rule, reason = should_filter_non_formulation(
            {
                "key": "PA3SPZ28",
                "raw_formulation_label": "Drug free nanoparticles",
                "instance_kind": "single_formulation",
                "parent_instance_id": "GAR-NPs_family",
                "formulation_role": "control",
                "change_role": "non_synthesis",
                "candidate_source": "live_llm_stage2_v2",
                "instance_context_tags": '["control"]',
                "change_context_tags": '[]',
                "change_descriptions": '["Absence of drug (Garcinol)"]',
            },
            {"loaded_state": "empty", "drug_name": "", "polymer_identity": "PLGA"},
            paper_rows=[],
        )
        self.assertTrue(should_filter)

    def test_ufxx9wxe_optimized_result_row_not_filtered_as_non_formulation(self):
        should_filter, rule, reason = should_filter_non_formulation(
            {
                "key": "UFXX9WXE",
                "raw_formulation_label": "Optimized Lzp-PLGA-NPs",
                "instance_kind": "single_formulation",
                "parent_instance_id": "UFXX9WXE_Formulation_Instances",
                "formulation_role": "variant",
                "change_role": "non_synthesis",
                "candidate_source": "live_llm_stage2_v2",
                "instance_context_tags": '["selected_condition"]',
                "change_context_tags": '["result_reported"]',
                "change_descriptions": '["Optimized formulation used for ex vivo, cell viability, and in vivo evaluation"]',
                "evidence_section": 'Figure 8; Figure 10; Figure 12',
                "supporting_evidence_refs": '[{"source_region_type":"text_span"}]',
            },
            {"loaded_state": "drug_loaded", "drug_name": "Lorazepam", "polymer_identity": "PLGA"},
            paper_rows=[
                {"instance_kind": "new_formulation", "candidate_source": "doe_numbered_table_row_recovery", "semantic_scope_ref": "scope1"}
                for _ in range(26)
            ],
        )
        self.assertFalse(should_filter, msg=f"unexpected filter: {rule} {reason}")


if __name__ == "__main__":
    unittest.main()
