import unittest
from unittest.mock import patch

from src.stage2_sampling_labels.build_stage2_compatibility_projection_v1 import (
    normalize_stage2_document_for_projection,
    project_document,
)
from src.stage2_sampling_labels.extract_semantic_stage2_objects_v2 import (
    OLLAMA_SHRUNKEN_LIVE_SCHEMA,
    build_live_prompt,
    call_live_backend,
    finalize_llm_first_document,
    ollama_live_system_prompt,
    should_use_compact_live_prompt,
)
from src.stage5_benchmark.build_minimal_final_output_v1 import (
    apply_global_polymer_material_carrythrough,
    apply_global_preparation_material_carrythrough,
    should_filter_non_formulation,
)
from src.stage5_benchmark.compare_layer3_values_to_gt_v1 import (
    CORE_FIXED_FIELDS,
    NAMED_EXTENSIBLE_VARIABLE_FIELDS,
    PROVENANCE_ONLY_FIELDS,
    build_alignment_index,
    build_cells,
    build_reporting_cells,
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

    def test_build_reporting_cells_merges_role_tolerant_name_fields(self):
        cells = [
            {
                "paper_key": "P1",
                "doi": "10.1/example",
                "gt_formulation_id": "P1_G001",
                "matched_system_formulation_id": "P1_row",
                "field_name": "surfactant_name",
                "field_group": "core_fixed_fields",
                "gt_value_raw": "",
                "system_value_raw": "PVA",
                "compare_status": "extra_in_system",
                "strict_match": "no",
                "relaxed_match": "no",
                "canonicalized_match": "no",
                "selected_compare_mode": "canonicalized",
                "error_bucket": "field_mapping_mismatch",
                "system_value_source_type": "direct_extracted",
                "evidence_status_detail": "supported",
                "alignment_rule": "direct",
            },
            {
                "paper_key": "P1",
                "doi": "10.1/example",
                "gt_formulation_id": "P1_G001",
                "matched_system_formulation_id": "P1_row",
                "field_name": "stabilizer_name",
                "field_group": "core_fixed_fields",
                "gt_value_raw": "PVA",
                "system_value_raw": "",
                "compare_status": "missing_in_system",
                "strict_match": "no",
                "relaxed_match": "no",
                "canonicalized_match": "no",
                "selected_compare_mode": "canonicalized",
                "error_bucket": "missing_value",
                "system_value_source_type": "missing_system_field_surface",
                "evidence_status_detail": "missing_system_field_surface",
                "alignment_rule": "direct",
            },
        ]
        reporting = build_reporting_cells(cells)
        self.assertEqual(len(reporting), 1)
        merged = reporting[0]
        self.assertEqual(merged["field_name"], "emulsifier_stabilizer_name")
        self.assertEqual(merged["gt_value_raw"], "PVA")
        self.assertEqual(merged["system_value_raw"], "PVA")
        self.assertEqual(merged["compare_status"], "present_and_match")
        self.assertEqual(merged["canonicalized_match"], "yes")
        self.assertEqual(merged["system_value_source_type"], "role_tolerant_union_overlay")

    def test_build_reporting_cells_renames_emulsifier_stabilizer_concentration_fields(self):
        cells = [
            {
                "paper_key": "P1",
                "doi": "10.1/example",
                "gt_formulation_id": "P1_G001",
                "matched_system_formulation_id": "P1_row",
                "field_name": "surfactant_concentration_value",
                "field_group": "core_fixed_fields",
                "gt_value_raw": "1 %",
                "system_value_raw": "1 %",
                "compare_status": "present_and_match",
                "strict_match": "yes",
                "relaxed_match": "yes",
                "canonicalized_match": "yes",
                "selected_compare_mode": "canonicalized",
                "error_bucket": "",
                "system_value_source_type": "direct_extracted",
                "evidence_status_detail": "supported",
                "alignment_rule": "direct",
            },
            {
                "paper_key": "P1",
                "doi": "10.1/example",
                "gt_formulation_id": "P1_G001",
                "matched_system_formulation_id": "P1_row",
                "field_name": "surfactant_concentration_unit",
                "field_group": "core_fixed_fields",
                "gt_value_raw": "% w/v",
                "system_value_raw": "% w/v",
                "compare_status": "present_and_match",
                "strict_match": "yes",
                "relaxed_match": "yes",
                "canonicalized_match": "yes",
                "selected_compare_mode": "canonicalized",
                "error_bucket": "",
                "system_value_source_type": "direct_extracted",
                "evidence_status_detail": "supported",
                "alignment_rule": "direct",
            },
        ]
        reporting = build_reporting_cells(cells)
        self.assertEqual(
            sorted(row["field_name"] for row in reporting),
            [
                "emulsifier_stabilizer_concentration_unit",
                "emulsifier_stabilizer_concentration_value",
            ],
        )

    def test_compare_values_matches_emulsifier_stabilizer_name_order_insensitively(self):
        strict, relaxed, canonicalized = compare_values(
            "emulsifier_stabilizer_name",
            "PVA | Poloxamer 188",
            "Poloxamer 188 | PVA",
        )
        self.assertFalse(strict)
        self.assertTrue(relaxed)
        self.assertTrue(canonicalized)

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

    def test_get_system_value_recovers_lc_percent_from_labeled_evidence_span(self):
        value, source, evidence = get_system_value(
            "lc_percent",
            {
                "loading_content_percent_value_text": "",
                "evidence_span_text": "Size (nm)=88.05 ± 2.7 | PDI=0.170 ± 0.05 | Drug loading (DL, %)=9.25 ± 2.8 | Encapsulation efficiency (EE, %)=88.32 ± 3.3",
                "supporting_evidence_refs": "",
            },
        )
        self.assertEqual(value, "9.25 %")
        self.assertEqual(source, "evidence_span_metric_rebinding")
        self.assertEqual(evidence, "supported")

    def test_get_system_value_recovers_lc_percent_from_header_aligned_structured_row(self):
        value, source, evidence = get_system_value(
            "lc_percent",
            {
                "loading_content_percent_value_text": "",
                "evidence_span_text": "MB loaded-PLGAb | 220 ± 4 | 0.19 ± 0.02 | 43.21 ± 2.69 | 0.52 ± 0.19 | 3.12 ± 1.12",
                "supporting_evidence_refs": '[{"target_field_name":"Formulation|Sizes (nm)|P.I.|Yield (%)|D.C. (%)|E.E. (%)"}]',
            },
        )
        self.assertEqual(value, "0.52 %")
        self.assertEqual(source, "evidence_span_metric_rebinding")
        self.assertEqual(evidence, "supported")

    def test_get_system_value_recovers_lc_percent_from_compact_measurement_tail(self):
        value, source, evidence = get_system_value(
            "lc_percent",
            {
                "loading_content_percent_value_text": "",
                "evidence_span_text": "5 | 75 | 75 | 5 | 15 | 190.2 ± 18.0 | 0.06 ± 0.01 | −8.0 ± 0.58 | 78.5 ± 1.8 | 4.9 ± 0.1",
                "supporting_evidence_refs": '[{"target_field_name":"PLGA (mg)|Drug (mg)|PVA (mg)|TPGS (mg)|Other"}]',
            },
        )
        self.assertEqual(value, "4.9 %")
        self.assertEqual(source, "evidence_span_metric_rebinding")
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

    def test_get_system_value_recovers_raw_coded_ph_from_doe_row_last_factor_level(self):
        value, source, evidence = get_system_value(
            "pH_raw",
            {
                "key": "GENERIC_DOE",
                "raw_formulation_label": "F20",
                "formulation_id": "GENERIC_DOE_Row_F20",
                "evidence_span_text": "F20 | 0 | −2.0 | 0 | 0 | 343.80 ± 2.74 | 0.085 ± 0.03 | −8.50 ± 0.61 | 80.34 ± 0.32",
            },
            paper_key="GENERIC_DOE",
        )
        self.assertEqual(value, "0")
        self.assertEqual(source, "coded_doe_factor_rebinding")
        self.assertEqual(evidence, "supported_raw_coded_value_decode_later")

    def test_build_cells_rank_rebinds_coded_gt_ph_from_physical_system_levels(self):
        gt_rows = [
            {"paper_key": "DOE_PH", "doi": "d", "gt_formulation_id": "G1", "formulation_label": "F1", "pH_raw": "−1"},
            {"paper_key": "DOE_PH", "doi": "d", "gt_formulation_id": "G2", "formulation_label": "F2", "pH_raw": "0"},
            {"paper_key": "DOE_PH", "doi": "d", "gt_formulation_id": "G3", "formulation_label": "F3", "pH_raw": "1"},
        ]
        system_rows = [
            {"key": "DOE_PH", "formulation_id": "F1", "representative_source_formulation_id": "G1", "pH_raw_value_text": "3.50", "raw_formulation_label": "F1"},
            {"key": "DOE_PH", "formulation_id": "F2", "representative_source_formulation_id": "G2", "pH_raw_value_text": "4.50", "raw_formulation_label": "F2"},
            {"key": "DOE_PH", "formulation_id": "F3", "representative_source_formulation_id": "G3", "pH_raw_value_text": "5.50", "raw_formulation_label": "F3"},
        ]
        cells, _ = build_cells(gt_rows, system_rows)
        ph_cells = [row for row in cells if row["field_name"] == "pH_raw"]
        self.assertEqual([row["system_value_raw"] for row in ph_cells], ["−1", "0", "1"])
        self.assertTrue(all(row["canonicalized_match"] == "yes" for row in ph_cells))
        self.assertTrue(all(row["system_value_source_type"] == "coded_doe_rank_rebinding" for row in ph_cells))

    def test_get_system_value_recovers_polymer_to_drug_ratio_from_named_ratio_label(self):
        value, source, evidence = get_system_value(
            "polymer_to_drug_ratio_raw",
            {
                "key": "GENERIC_RATIO",
                "raw_formulation_label": "PLGA:ITZ ratio=10:1",
                "representative_source_raw_formulation_label": "PLGA:ITZ ratio=10:1",
            },
            paper_key="GENERIC_RATIO",
        )
        self.assertEqual(value, "PLGA:ITZ ratio=10:1")
        self.assertEqual(source, "ratio_label_token_rebinding")
        self.assertEqual(evidence, "supported")

    def test_get_system_value_recovers_drug_to_polymer_ratio_from_compact_table_label(self):
        value, source, evidence = get_system_value(
            "drug_to_polymer_ratio_raw",
            {
                "key": "GENERIC_RATIO",
                "raw_formulation_label": "1:20 / 50:50",
                "representative_source_raw_formulation_label": "1:20 / 50:50",
            },
            paper_key="GENERIC_RATIO",
        )
        self.assertEqual(value, "1:20")
        self.assertEqual(source, "ratio_label_token_rebinding")
        self.assertEqual(evidence, "supported")

    def test_get_system_value_does_not_recover_reversed_named_ratio_direction(self):
        value, source, evidence = get_system_value(
            "polymer_to_drug_ratio_raw",
            {
                "key": "GENERIC_RATIO",
                "raw_formulation_label": "ITZ:PLGA ratio=10:1",
            },
            paper_key="GENERIC_RATIO",
        )
        self.assertEqual(value, "")
        self.assertEqual(source, "missing_system_field_surface")
        self.assertEqual(evidence, "missing_system_field_surface")

    def test_compare_values_rejects_reversed_named_ratio_direction(self):
        strict, relaxed, canonicalized = compare_values(
            "polymer_to_drug_ratio_raw",
            "PLGA:ITZ ratio=5:1",
            "ITZ:PLGA ratio=5:1",
        )
        self.assertFalse(strict)
        self.assertFalse(relaxed)
        self.assertFalse(canonicalized)

    def test_compare_values_matches_contextual_ratio_text_to_compact_ratio_token(self):
        strict, relaxed, canonicalized = compare_values(
            "drug_to_polymer_ratio_raw",
            "Drug:Polymer ratio 1:10 (Nanoprecipitation method)",
            "1:10",
        )
        self.assertFalse(strict)
        self.assertTrue(relaxed)
        self.assertTrue(canonicalized)

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

    def test_get_system_value_recovers_la_ga_ratio_from_plga_family_label(self):
        raw_value, raw_source, raw_evidence = get_system_value(
            "la_ga_ratio_raw",
            {
                "key": "5GIF3D8W",
                "raw_formulation_label": "PLGA 75/25 / Drug loaded",
                "representative_source_raw_formulation_label": "PLGA 75/25 / Drug loaded",
            },
            paper_key="5GIF3D8W",
        )
        norm_value, norm_source, norm_evidence = get_system_value(
            "la_ga_ratio_normalized",
            {
                "key": "5GIF3D8W",
                "raw_formulation_label": "PLGA 75/25 / Drug loaded",
                "representative_source_raw_formulation_label": "PLGA 75/25 / Drug loaded",
            },
            paper_key="5GIF3D8W",
        )
        self.assertEqual(raw_value, "75/25")
        self.assertEqual(raw_source, "polymer_family_ratio_rebinding")
        self.assertEqual(raw_evidence, "supported")
        self.assertEqual(norm_value, "75:25")
        self.assertEqual(norm_source, "polymer_family_ratio_rebinding")
        self.assertEqual(norm_evidence, "supported")

    def test_get_system_value_recovers_la_ga_ratio_from_compact_table_label(self):
        value, source, evidence = get_system_value(
            "la_ga_ratio_raw",
            {
                "key": "PA3SPZ28",
                "raw_formulation_label": "1:10 / 50:50",
                "evidence_span_text": "Size (nm)=88.05 ± 2.7 | PDI=0.170 ± 0.05",
            },
            paper_key="PA3SPZ28",
        )
        self.assertEqual(value, "50:50")
        self.assertEqual(source, "polymer_family_ratio_rebinding")
        self.assertEqual(evidence, "supported")

    def test_get_system_value_recovers_la_ga_ratio_from_table_assignment_column(self):
        value, source, evidence = get_system_value(
            "la_ga_ratio_raw",
            {
                "key": "BB3JUVW7",
                "raw_formulation_label": "row_06__100",
                "evidence_span_text": "100 | 75:25 | 2x | Heat* | 15 | 329.6 ± 79.3 | 92.2 ± 14.9",
            },
            paper_key="BB3JUVW7",
        )
        self.assertEqual(value, "75:25")
        self.assertEqual(source, "polymer_family_ratio_rebinding")
        self.assertEqual(evidence, "supported")

    def test_get_system_value_recovers_la_ga_ratio_from_resomer_grade_token(self):
        value, source, evidence = get_system_value(
            "la_ga_ratio_normalized",
            {
                "key": "V99GKZEI",
                "raw_formulation_label": "MB loaded-PLGAb",
                "polymer_mw_kDa_value_text": "RG502H MW range 7000-17000 Da",
            },
            paper_key="V99GKZEI",
        )
        self.assertEqual(value, "50:50")
        self.assertEqual(source, "polymer_family_ratio_rebinding")
        self.assertEqual(evidence, "supported")

    def test_get_system_value_does_not_treat_drug_polymer_ratio_as_la_ga_ratio(self):
        value, source, evidence = get_system_value(
            "la_ga_ratio_raw",
            {
                "key": "GENERIC",
                "raw_formulation_label": "Drug:Polymer ratio 1:10",
            },
            paper_key="GENERIC",
        )
        self.assertEqual(value, "")
        self.assertEqual(source, "missing_system_field_surface")
        self.assertEqual(evidence, "missing_system_field_surface")

    def test_get_system_value_recovers_laga_from_direct_stage5_ratio_surface(self):
        value, source, evidence = get_system_value(
            "la_ga_ratio_raw",
            {
                "key": "BXCV5XWB",
                "raw_formulation_label": "KGN-loaded PLGA nanoparticles",
                "la_ga_ratio_value_text": "1:1 d,l-lactic to glycolic acid",
            },
            paper_key="BXCV5XWB",
        )
        self.assertEqual(value, "1:1 d,l-lactic to glycolic acid")
        self.assertEqual(source, "direct_extracted")
        self.assertEqual(evidence, "supported")

    def test_stage5_carries_single_global_polymer_la_ga_material_ratio_to_blank_plga_rows(self):
        material_text = (
            "Materials Lopinavir was obtained as a gift sample. "
            "Poly (lactide-co-glycolide) PLGA (50:50) (inherent viscosity 0.2 dl/g) "
            "was gift sample from Purac Biomaterials, The Netherlands."
        )
        final_row, applied_fields = apply_global_polymer_material_carrythrough(
            final_row={
                "key": "WFDTQ4VX",
                "polymer_identity_final": "PLGA",
                "polymer_identity": "PLGA",
                "la_ga_ratio_value": "",
                "la_ga_ratio_value_text": "",
                "la_ga_ratio_scope": "",
                "la_ga_ratio_membership_confidence": "",
                "la_ga_ratio_evidence_region_type": "",
                "la_ga_ratio_missing_reason": "not_reported",
            },
            source_text=material_text,
        )
        self.assertEqual(final_row["la_ga_ratio_value"], "50:50")
        self.assertEqual(final_row["la_ga_ratio_value_text"], "50:50")
        self.assertEqual(final_row["la_ga_ratio_scope"], "global_shared")
        self.assertEqual(final_row["la_ga_ratio_evidence_region_type"], "global_material_evidence")
        self.assertEqual(final_row["la_ga_ratio_missing_reason"], "")
        self.assertIn("la_ga_ratio", applied_fields)

    def test_stage5_carries_global_polymer_la_ga_material_ratio_to_unknown_table_rows(self):
        material_text = (
            "2.1. Materials Xanthone (XAN), PLGA (50:50) MW 50 000–75 000, "
            "Pluronic F-68 and soybean lecithin were purchased from Sigma-Aldrich."
        )
        final_row, applied_fields = apply_global_polymer_material_carrythrough(
            final_row={
                "key": "L3H2RS2H",
                "representative_source_raw_formulation_label": "XAN nanospheres (Theoretical concentration 50 mg/mL)",
                "polymer_identity_final": "unknown",
                "la_ga_ratio_value": "",
                "la_ga_ratio_value_text": "",
                "la_ga_ratio_missing_reason": "",
            },
            source_text=material_text,
        )
        self.assertEqual(final_row["la_ga_ratio_value"], "50:50")
        self.assertIn("la_ga_ratio", applied_fields)

    def test_stage5_global_polymer_la_ga_carrythrough_ignores_model_equation_ratios(self):
        source_text = (
            "Materials Poly (lactide-co-glycolide) PLGA (50:50) was used. "
            "YEE = 75:25 + 22:85X1 + 7:15X2."
        )
        final_row, applied_fields = apply_global_polymer_material_carrythrough(
            final_row={
                "key": "WFDTQ4VX",
                "polymer_identity_final": "PLGA",
                "la_ga_ratio_value": "",
                "la_ga_ratio_value_text": "",
                "la_ga_ratio_missing_reason": "not_reported",
            },
            source_text=source_text,
        )
        self.assertEqual(final_row["la_ga_ratio_value"], "50:50")
        self.assertNotEqual(final_row["la_ga_ratio_value"], "75:25")
        self.assertIn("la_ga_ratio", applied_fields)

    def test_get_system_value_decodes_wivucmyg_coded_polymer_concentration(self):
        value, source, evidence = get_system_value(
            "polymer_concentration_value",
            {
                "key": "WIVUCMYG",
                "raw_formulation_label": "F22",
                "evidence_span_text": "F22 | 0 | 0 | −2.0 | 0 | 340.40 ± 4.66 | 0.118 ± 0.01 | −8.06 ± 0.30 | 68.48 ± 2.45",
            },
            paper_key="WIVUCMYG",
        )
        unit, unit_source, unit_evidence = get_system_value(
            "polymer_concentration_unit",
            {
                "key": "WIVUCMYG",
                "raw_formulation_label": "F22",
                "evidence_span_text": "F22 | 0 | 0 | −2.0 | 0 | 340.40 ± 4.66 | 0.118 ± 0.01 | −8.06 ± 0.30 | 68.48 ± 2.45",
            },
            paper_key="WIVUCMYG",
        )
        self.assertEqual(value, "8")
        self.assertEqual(source, "coded_factor_table_rebinding")
        self.assertEqual(evidence, "supported")
        self.assertEqual(unit, "mg/mL")
        self.assertEqual(unit_source, "coded_factor_table_rebinding")
        self.assertEqual(unit_evidence, "supported")

    def test_get_system_value_decodes_wfdtq4vx_coded_polymer_concentration(self):
        value, source, evidence = get_system_value(
            "polymer_concentration_value",
            {
                "key": "WFDTQ4VX",
                "raw_formulation_label": "Sr. No. 23",
                "evidence_span_text": "23 | 1 | 0 | 0 | 91.4 ± 1.67 | 231.3 ± 2.59",
            },
            paper_key="WFDTQ4VX",
        )
        unit, unit_source, unit_evidence = get_system_value(
            "polymer_concentration_unit",
            {
                "key": "WFDTQ4VX",
                "raw_formulation_label": "Sr. No. 23",
                "evidence_span_text": "23 | 1 | 0 | 0 | 91.4 ± 1.67 | 231.3 ± 2.59",
            },
            paper_key="WFDTQ4VX",
        )
        self.assertEqual(value, "2.0")
        self.assertEqual(source, "coded_factor_table_rebinding")
        self.assertEqual(evidence, "supported")
        self.assertEqual(unit, "%w/v")
        self.assertEqual(unit_source, "coded_factor_table_rebinding")
        self.assertEqual(unit_evidence, "supported")

    def test_get_system_value_decodes_wfdtq4vx_ocr_cid_minus_polymer_concentration(self):
        value, source, evidence = get_system_value(
            "polymer_concentration_value",
            {
                "key": "WFDTQ4VX",
                "raw_formulation_label": "Sr. No. 12",
                "evidence_span_text": "12 | 0 | (cid:4)1 | 1 | 47.4 ± 1.04 | 152.6 ± 6.23",
            },
            paper_key="WFDTQ4VX",
        )
        self.assertEqual(value, "1.0")
        self.assertEqual(source, "coded_factor_table_rebinding")
        self.assertEqual(evidence, "supported")

    def test_get_system_value_recovers_v99gkzei_wow_organic_phase_ratio(self):
        value, source, evidence = get_system_value(
            "phase_ratio_raw",
            {
                "key": "V99GKZEI",
                "raw_formulation_label": "MB loaded-PLGA (W/O/W)c",
                "evidence_span_text": "MB loaded-PLGA (W/O/W)c | 266 ± 5 | 0.40 ± 0.10 | 39.98 ± 6.32 | 1.13 ± 0.26 | 6.75 ± 1.54",
            },
            paper_key="V99GKZEI",
        )
        self.assertEqual(value, "1:1 v/v")
        self.assertEqual(source, "paper_local_source_footnote_rebinding")
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

    def test_choose_system_row_does_not_use_ratio_token_bridge_for_reversed_named_ratio_direction(self):
        gt_row = {
            "paper_key": "P6B",
            "gt_formulation_id": "P6B_G001",
            "seed_pred_representative_source_formulation_id": "",
            "formulation_label": "PLGA:ITZ ratio=5:1",
            "polymer_name": "",
            "drug_name": "",
            "drug_mass_mg": "",
            "la_ga_ratio_normalized": "",
        }
        system_rows = [
            {
                "key": "P6B",
                "final_formulation_id": "P6B__fo__1",
                "formulation_id": "P6B__table_1__itz_plga_5_1",
                "representative_source_formulation_id": "P6B__table_1__itz_plga_5_1",
                "parent_core_row_id": "P6B__table_1__itz_plga_5_1",
                "raw_formulation_label": "ITZ:PLGA ratio=5:1",
                "source_candidate_labels": "[\"ITZ:PLGA ratio=5:1\"]",
                "polymer_identity_final": "",
                "polymer_name_raw": "",
                "drug_name_value_text": "",
                "drug_feed_amount_text_value_text": "",
                "la_ga_ratio_value_text": "",
            }
        ]
        row, rule, ok = choose_system_row(gt_row, system_rows)
        self.assertFalse(ok)
        self.assertEqual(rule, "no_unique_alignment")
        self.assertIsNone(row)

    def test_choose_system_row_does_not_align_on_trailing_laga_ratio_token_only(self):
        gt_row = {
            "paper_key": "P6C",
            "gt_formulation_id": "P6C_G001",
            "seed_pred_representative_source_formulation_id": "",
            "formulation_label": "1:10 / 50:50",
            "polymer_name": "",
            "drug_name": "",
            "drug_mass_mg": "",
            "la_ga_ratio_normalized": "",
        }
        system_rows = [
            {
                "key": "P6C",
                "final_formulation_id": "P6C__fo__1",
                "formulation_id": "P6C__table_1__2:10_/_50:50",
                "representative_source_formulation_id": "P6C__table_1__2:10_/_50:50",
                "parent_core_row_id": "P6C__table_1__2:10_/_50:50",
                "raw_formulation_label": "2:10 / 50:50",
                "source_candidate_labels": "[\"2:10 / 50:50\"]",
                "polymer_identity_final": "",
                "polymer_name_raw": "",
                "drug_name_value_text": "",
                "drug_feed_amount_text_value_text": "",
                "la_ga_ratio_value_text": "",
            }
        ]
        row, rule, ok = choose_system_row(gt_row, system_rows)
        self.assertFalse(ok)
        self.assertEqual(rule, "no_unique_alignment")
        self.assertIsNone(row)

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

    def test_finalize_llm_first_document_synthesizes_shrunken_contract_from_legacy_candidates(self):
        from src.stage2_sampling_labels.validate_stage2_semantic_authority_contract_v1 import validate_semantic_documents

        document = finalize_llm_first_document(
            {
                "document_key": "LEGACY1",
                "source_mode": "legacy_llm_raw_response_replay_to_stage2_v2",
                "formulation_candidates": [
                    {
                        "candidate_id": "F1",
                        "raw_label": "PLGA Empty",
                        "instance_kind": "new_formulation",
                        "formulation_role": "control",
                        "status": "reported",
                        "instance_context_tags": ["control"],
                    },
                    {
                        "candidate_id": "F2",
                        "raw_label": "PLGA Drug loaded",
                        "instance_kind": "variant_formulation",
                        "formulation_role": "optimized",
                        "status": "reported",
                    },
                ],
                "table_formulation_scopes": [
                    {
                        "scope_id": "LEGACY1__table_formulation_scope__table_1",
                        "table_id": "Table 1",
                        "is_formulation_table": True,
                        "table_type": "full_formulation",
                        "confidence": "high",
                        "marker_provenance": "llm_explicit",
                    }
                ],
                "boundary_markers": [{"table_id": "Table 1", "is_doe": False, "marker_provenance": "llm_explicit"}],
                "selection_markers": [{"selected_variable": "polymer type", "selected_value": "PLGA 50/50"}],
                "relation_hints": [],
                "variable_candidates": [{"variable_name": "polymer type", "variable_role": "process_setting"}],
                "measurement_candidates": [{"measurement_id": "M1", "formulation_candidate_id": "F2", "measurement_type": "loading_content_percent", "value_text": "1.45"}],
                "shared_semantics": {"preparation_method": "single emulsion"},
            }
        )

        self.assertEqual(document["paper_key"], "LEGACY1")
        self.assertEqual(document["table_scopes"][0]["table_id"], "Table 1")
        self.assertEqual(document["table_scopes"][0]["scope_kind"], "formulation_table")
        self.assertTrue(document["semantic_signals"]["has_variable_sweep"])
        self.assertEqual(document["semantic_signals"]["primary_preparation_method_hint"], "single emulsion")
        self.assertEqual(document["formulation_candidates"][0]["candidate_kind"], "single_formulation")
        self.assertEqual(document["formulation_candidates"][0]["instance_role"], "control")
        self.assertEqual(document["formulation_candidates"][1]["candidate_kind"], "variant_formulation")
        self.assertEqual(document["formulation_candidates"][1]["instance_role"], "synthesis_core")
        validation = validate_semantic_documents([document], require_mode=False)
        self.assertEqual(validation["errors"], [])

    def test_shrunken_contract_projection_preserves_measurement_candidates_when_present(self):
        document = {
            "document_key": "5GIF3D8W",
            "paper_key": "5GIF3D8W",
            "doi": "10.1000/example",
            "source_mode": "saved_raw_live_v2_replay_to_stage2_v2",
            "stage2_semantic_source_mode": "llm_first_composite",
            "semantic_universe_authority": "llm_semantic_discovery",
            "table_scopes": [
                {
                    "table_id": "Table 4",
                    "scope_kind": "formulation_table",
                    "is_formulation_bearing": True,
                    "is_doe": False,
                    "confidence": "medium",
                }
            ],
            "semantic_signals": {
                "has_variable_sweep": True,
                "has_sequential_optimization": False,
                "has_parent_child_table_relation": True,
                "has_downstream_non_synthesis_variants": False,
                "has_measurement_only_variants": False,
                "primary_preparation_method_hint": "single emulsion",
                "primary_variable_names": ["polymer_type"],
                "selected_condition_hints": [],
            },
            "formulation_candidates": [
                {
                    "candidate_id": "F2",
                    "candidate_kind": "variant_formulation",
                    "source_table_id": "Table 4",
                    "label_hint": "PLGA 50/50 Drug loaded",
                    "instance_role": "synthesis_core",
                    "parent_candidate_hint": "F1",
                    "core_change_hint": "loaded with etoposide",
                    "shared_context_hint": "",
                    "status": "reported",
                    "confidence": "medium",
                    "semantic_scope_ref": "5GIF3D8W__llm_document_scope__01|candidate:F2",
                }
            ],
            "measurement_candidates": [
                {
                    "measurement_id": "M9",
                    "formulation_candidate_id": "F2",
                    "measurement_name": "loading_content",
                    "value_text": "1.04 ± 0.06",
                    "unit_text": "%",
                    "evidence_span_ids": ["span_66"],
                }
            ],
            "evidence_spans": [
                {
                    "span_id": "span_66",
                    "source_region_type": "table_cell",
                    "source_locator_text": "Table 4",
                    "supporting_text": "Drug content (%) = 1.04 ± 0.06",
                }
            ],
            "shared_semantics": {
                "drug_name": "etoposide",
                "polymer_name_raw": "PLGA 50/50",
                "preparation_method": "single emulsion",
            },
            "semantic_scope_declarations": [],
        }
        projected = normalize_stage2_document_for_projection(document)
        measurements = projected["measurement_candidates"]
        self.assertEqual(len(measurements), 1)
        self.assertEqual(measurements[0]["measurement_name_raw"], "loading_content")
        self.assertEqual(measurements[0]["measurement_value_raw"], "1.04 ± 0.06")
        self.assertEqual(measurements[0]["measurement_unit_raw"], "%")
        self.assertEqual(len(projected["evidence_handoffs"]), 1)
        self.assertIn("measurement_candidate", projected["evidence_handoffs"][0]["target_object_ref"])

    def test_replayed_live_document_can_fallback_to_richer_legacy_raw_response(self):
        import json
        import shutil
        from pathlib import Path
        from src.stage2_sampling_labels.extract_semantic_stage2_objects_v2 import normalize_replayed_live_document

        shrunken = {
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
        }
        legacy_payload = {
            "paper_notes": "Recovered richer saved response.",
            "formulations": [
                {
                    "formulation_id": "UFXX9WXE_F1",
                    "raw_formulation_label": "Formulation F1",
                    "instance_kind": "new_formulation",
                    "formulation_role": "reported",
                    "fields": {
                        "polymer_name_raw": {"value_text": "PLGA"},
                        "drug_name": {"value_text": "Lorazepam"},
                        "loading_content_percent": {"value_text": "1.45"},
                    },
                }
            ],
        }
        legacy_dir = Path("/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/test_tmp_legacy_raw")
        shutil.rmtree(legacy_dir, ignore_errors=True)
        legacy_dir.mkdir(parents=True, exist_ok=True)
        try:
            legacy_path = legacy_dir / "UFXX9WXE__stage2_v2_raw_response.json"
            legacy_path.write_text(json.dumps(legacy_payload), encoding="utf-8")
            document = normalize_replayed_live_document(
                record={
                    "key": "UFXX9WXE",
                    "doi": "10.1/example",
                    "title": "Example",
                    "text_path": "data/cleaned/content/text/UFXX9WXE.html.txt",
                },
                parsed=shrunken,
                raw_response_path=Path("/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/fake_saved_live_raw.json"),
                fallback_legacy_raw_dir=legacy_dir,
            )
        finally:
            shutil.rmtree(legacy_dir, ignore_errors=True)
        self.assertEqual(document["source_mode"], "legacy_llm_raw_response_replay_to_stage2_v2")
        self.assertTrue(document["measurement_candidates"])
        self.assertEqual(document["measurement_candidates"][0]["measurement_name"], "loading_content")
        self.assertEqual(document["measurement_candidates"][0]["value_text"], "1.45")
        self.assertTrue(document["source_raw_response_path"].endswith("UFXX9WXE__stage2_v2_raw_response.json"))

    def test_convert_legacy_raw_response_to_v2_uses_fallback_for_shrunken_live_payload(self):
        import json
        import shutil
        from pathlib import Path
        from src.stage2_sampling_labels.extract_semantic_stage2_objects_v2 import convert_legacy_raw_response_to_v2

        shrunken_live_payload = {
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
        }
        legacy_payload = {
            "paper_notes": "Recovered richer saved response.",
            "formulations": [
                {
                    "formulation_id": "UFXX9WXE_F1",
                    "raw_formulation_label": "Formulation F1",
                    "instance_kind": "new_formulation",
                    "formulation_role": "reported",
                    "fields": {
                        "polymer_name_raw": {"value_text": "PLGA"},
                        "drug_name": {"value_text": "Lorazepam"},
                        "loading_content_percent": {"value_text": "1.45"},
                    },
                }
            ],
        }
        legacy_dir = Path("/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/test_tmp_legacy_raw_convert")
        shutil.rmtree(legacy_dir, ignore_errors=True)
        legacy_dir.mkdir(parents=True, exist_ok=True)
        try:
            (legacy_dir / "UFXX9WXE__stage2_v2_raw_response.json").write_text(json.dumps(legacy_payload), encoding="utf-8")
            document = convert_legacy_raw_response_to_v2(
                record={
                    "key": "UFXX9WXE",
                    "doi": "10.1/example",
                    "title": "Example",
                    "text_path": "data/cleaned/content/text/UFXX9WXE.html.txt",
                },
                raw_response_path=Path("/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/fake_saved_live_raw.json"),
                raw_response_text=json.dumps(shrunken_live_payload),
                fallback_legacy_raw_dir=legacy_dir,
            )
        finally:
            shutil.rmtree(legacy_dir, ignore_errors=True)
        self.assertEqual(document["source_mode"], "legacy_llm_raw_response_replay_to_stage2_v2")
        self.assertTrue(any(m.get("measurement_name") == "loading_content" for m in document["measurement_candidates"]))

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

    def test_build_live_prompt_uses_compact_gemma_ollama_variant(self):
        evidence_artifact = {
            "input_contract": {"table_mode": "summary"},
            "evidence_blocks": [
                {"rendered_text": "dummy", "block_type": "table", "evidence_kind": "table", "source_type": "table_summary", "text_content": "dummy"}
            ],
        }
        default_prompt = build_live_prompt({"key": "L3H2RS2H", "title": "Example title", "doi": "10.1/example"}, evidence_artifact)
        compact_prompt = build_live_prompt(
            {"key": "L3H2RS2H", "title": "Example title", "doi": "10.1/example"},
            evidence_artifact,
            llm_backend="ollama",
            model="gemma4:e4b",
        )
        self.assertLess(len(compact_prompt), len(default_prompt))
        self.assertIn("Return JSON matching this compact skeleton", compact_prompt)
        self.assertTrue(should_use_compact_live_prompt("ollama", "gemma4:e4b"))
        self.assertFalse(should_use_compact_live_prompt("gemini", "gemma4:e4b"))

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


    def test_stage5_carries_unique_global_preparation_solvent_to_blank_doe_rows(self):
        row = {
            "key": "SOLV1",
            "formulation_id": "SOLV1_DOE_Row_F1",
            "raw_formulation_label": "F1",
            "preparation_method": "solvent displacement technique",
            "organic_solvent_value": "",
            "organic_solvent_value_text": "",
            "organic_solvent_scope": "",
            "organic_solvent_membership_confidence": "",
            "organic_solvent_evidence_region_type": "",
            "organic_solvent_missing_reason": "missing",
        }
        source_text = (
            "Preparation of nanoparticles. PLGA nanoparticles were prepared by the "
            "solvent displacement method. Briefly, PLGA and the drug were dissolved "
            "in acetone and the organic solution was added dropwise into an aqueous "
            "PVA solution."
        )
        materialized, applied = apply_global_preparation_material_carrythrough(
            final_row=row,
            source_text=source_text,
        )
        self.assertIn("organic_solvent", applied)
        self.assertEqual(materialized["organic_solvent_value_text"], "acetone")
        self.assertEqual(materialized["organic_solvent_scope"], "global_shared")
        self.assertEqual(materialized["organic_solvent_evidence_region_type"], "global_preparation_evidence")
        self.assertEqual(materialized["organic_solvent_missing_reason"], "")

    def test_stage5_global_preparation_solvent_carrythrough_requires_unique_preparation_context(self):
        row = {
            "key": "SOLV2",
            "formulation_id": "SOLV2_Row_1",
            "raw_formulation_label": "F1",
            "preparation_method": "nanoprecipitation",
            "organic_solvent_value": "",
            "organic_solvent_value_text": "",
        }
        source_text = (
            "Materials listed acetone, acetonitrile, methanol, and dichloromethane. "
            "Characterization used acetonitrile:methanol mobile phase."
        )
        materialized, applied = apply_global_preparation_material_carrythrough(
            final_row=row,
            source_text=source_text,
        )
        self.assertNotIn("organic_solvent", applied)
        self.assertEqual(materialized["organic_solvent_value_text"], "")

    def test_stage5_carries_unique_global_loaded_drug_name_to_blank_doe_rows(self):
        row = {
            "key": "DRUG1",
            "formulation_id": "DRUG1_DOE_Row_F1",
            "raw_formulation_label": "F1",
            "loaded_state_final": "unknown",
            "drug_name_value": "",
            "drug_name_value_text": "",
            "drug_name_scope": "",
            "drug_name_membership_confidence": "",
            "drug_name_evidence_region_type": "",
            "drug_name_missing_reason": "not_reported",
        }
        source_text = (
            "Effect of polymer viscosity on physicochemical properties and ocular tolerance "
            "of FB-loaded PLGA nanospheres. Table 1 reports cFB, concentration of "
            "flurbiprofen (mg/mL), and cP188."
        )
        materialized, applied = apply_global_preparation_material_carrythrough(
            final_row=row,
            source_text=source_text,
        )
        self.assertIn("drug_name", applied)
        self.assertEqual(materialized["drug_name_value_text"], "flurbiprofen")
        self.assertEqual(materialized["drug_name_scope"], "global_shared")
        self.assertEqual(materialized["drug_name_evidence_region_type"], "global_drug_identity_evidence")
        self.assertEqual(materialized["drug_name_missing_reason"], "")

    def test_stage5_global_drug_name_carrythrough_does_not_fill_blank_or_empty_rows(self):
        row = {
            "key": "DRUG2",
            "formulation_id": "DRUG2_blank",
            "raw_formulation_label": "Drug free nanoparticles",
            "loaded_state_final": "empty",
            "drug_name_value": "",
            "drug_name_value_text": "",
        }
        source_text = "Pranoprofen (PF) is a nonsteroidal anti-inflammatory drug. PF-loaded PLGA nanoparticles were prepared."
        materialized, applied = apply_global_preparation_material_carrythrough(
            final_row=row,
            source_text=source_text,
        )
        self.assertNotIn("drug_name", applied)
        self.assertEqual(materialized["drug_name_value_text"], "")

    def test_stage5_global_drug_name_carrythrough_requires_unique_loaded_drug_identity(self):
        row = {
            "key": "DRUG3",
            "formulation_id": "DRUG3_Row_1",
            "raw_formulation_label": "F1",
            "loaded_state_final": "unknown",
            "drug_name_value": "",
            "drug_name_value_text": "",
        }
        source_text = (
            "Gatifloxacin-loaded PLGA nanoparticles and Rhodamine-loaded PLGA nanoparticles "
            "were prepared as distinct formulation families."
        )
        materialized, applied = apply_global_preparation_material_carrythrough(
            final_row=row,
            source_text=source_text,
        )
        self.assertNotIn("drug_name", applied)
        self.assertEqual(materialized["drug_name_value_text"], "")

    def test_stage5_global_drug_name_carrythrough_ignores_helper_dye_uptake_context(self):
        row = {
            "key": "DRUG4",
            "formulation_id": "DRUG4_Row_1",
            "raw_formulation_label": "F1",
            "loaded_state_final": "unknown",
            "drug_name_value": "",
            "drug_name_value_text": "",
        }
        source_text = (
            "The study was aimed to formulate PLGA NPs of lopinavir. "
            "Formulation of lopinavir-loaded PLGA NPs used nanoprecipitation. "
            "For cell uptake assay, 6-coumarin-loaded PLGA NPs and dye solution were used."
        )
        materialized, applied = apply_global_preparation_material_carrythrough(
            final_row=row,
            source_text=source_text,
        )
        self.assertIn("drug_name", applied)
        self.assertEqual(materialized["drug_name_value_text"], "lopinavir")

    def test_stage5_global_drug_name_carrythrough_skips_coded_np_family_labels(self):
        row = {
            "key": "DRUG5",
            "formulation_id": "DRUG5__table_1__npg1",
            "raw_formulation_label": "NPG1",
            "loaded_state_final": "unknown",
            "drug_name_value": "",
            "drug_name_value_text": "",
        }
        source_text = "Rhodamine-loaded PLGA NPs were prepared for one formulation family."
        materialized, applied = apply_global_preparation_material_carrythrough(
            final_row=row,
            source_text=source_text,
        )
        self.assertNotIn("drug_name", applied)
        self.assertEqual(materialized["drug_name_value_text"], "")

    def test_project_document_carries_shared_context_values_into_each_row(self):
        document = {
            "document_key": "SHARED1",
            "paper_key": "SHARED1",
            "doi": "10.1/shared",
            "model_name": "test-model",
            "source_mode": "saved_raw_live_v2_replay_to_stage2_v2",
            "semantic_signals": {"primary_preparation_method_hint": "single emulsion"},
            "formulation_identity_candidates": [
                {
                    "formulation_candidate_id": "F1",
                    "raw_formulation_label": "Loaded nanoparticles",
                    "formulation_role": "reported",
                    "instance_kind": "single_formulation",
                }
            ],
            "component_candidates": [
                {
                    "component_id": "c_poly",
                    "formulation_candidate_id": "",
                    "component_role_raw": "polymer",
                    "component_name_raw": "PLGA",
                    "component_properties_raw": [{"name": "la ga ratio", "value": "50:50"}],
                },
                {
                    "component_id": "c_pva",
                    "formulation_candidate_id": "",
                    "component_role_raw": "additive",
                    "component_name_raw": "PVA",
                    "amount_expression_raw": "1 %",
                    "parsed_value_raw": "1",
                },
            ],
            "variable_or_factor_candidates": [
                {
                    "factor_id": "v_ph",
                    "formulation_candidate_id": "",
                    "factor_name_raw": "aqueous phase pH",
                    "factor_expression_raw": "7.4",
                    "identity_defining_signal": "yes",
                },
                {
                    "factor_id": "v_method",
                    "formulation_candidate_id": "",
                    "factor_name_raw": "preparation method",
                    "factor_expression_raw": "single emulsion",
                    "identity_defining_signal": "no",
                },
            ],
            "measurement_candidates": [],
            "evidence_handoffs": [],
            "table_formulation_scopes": [],
            "table_variable_roles": [],
            "selection_markers": [],
            "inheritance_markers": [],
            "boundary_markers": [],
        }
        rows, traces, jsonl_rows, recovery_summary, _ = project_document(document)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["polymer_name_raw"], "PLGA")
        self.assertEqual(row["la_ga_ratio_value_text"], "50:50")
        self.assertEqual(row["surfactant_name_value_text"], "PVA")
        self.assertEqual(row["surfactant_concentration_text_value_text"], "1 %")
        self.assertEqual(row["preparation_method"], "single emulsion")
        self.assertIn("aqueous phase pH", row["identity_variables_json"])
        self.assertIn("7.4", row["identity_variables_json"])

    def test_project_document_does_not_apply_shared_drug_to_control_rows(self):
        document = {
            "document_key": "SHARED2",
            "paper_key": "SHARED2",
            "doi": "10.1/shared2",
            "model_name": "test-model",
            "source_mode": "saved_raw_live_v2_replay_to_stage2_v2",
            "semantic_signals": {},
            "formulation_identity_candidates": [
                {
                    "formulation_candidate_id": "CTRL",
                    "raw_formulation_label": "Empty nanoparticles",
                    "formulation_role": "control",
                    "instance_kind": "single_formulation",
                },
                {
                    "formulation_candidate_id": "LOAD",
                    "raw_formulation_label": "Drug-loaded nanoparticles",
                    "formulation_role": "reported",
                    "instance_kind": "single_formulation",
                },
            ],
            "component_candidates": [
                {
                    "component_id": "c_drug",
                    "formulation_candidate_id": "",
                    "component_role_raw": "drug",
                    "component_name_raw": "Doxorubicin",
                    "amount_expression_raw": "5 mg",
                    "parsed_value_raw": "5",
                }
            ],
            "variable_or_factor_candidates": [],
            "measurement_candidates": [],
            "evidence_handoffs": [],
            "table_formulation_scopes": [],
            "table_variable_roles": [],
            "selection_markers": [],
            "inheritance_markers": [],
            "boundary_markers": [],
        }
        rows, _, _, _, _ = project_document(document)
        by_id = {row["formulation_id"]: row for row in rows}
        self.assertEqual(by_id["CTRL"]["drug_name_value_text"], "")
        self.assertEqual(by_id["LOAD"]["drug_name_value_text"], "Doxorubicin")

    def test_project_document_projects_factor_level_explicit_value_fields(self):
        document = {
            "document_key": "FACTORS1",
            "paper_key": "FACTORS1",
            "doi": "10.1/factors1",
            "model_name": "test-model",
            "source_mode": "saved_raw_live_v2_replay_to_stage2_v2",
            "semantic_signals": {},
            "formulation_identity_candidates": [
                {
                    "formulation_candidate_id": "F1",
                    "raw_formulation_label": "F1",
                    "formulation_role": "reported",
                    "instance_kind": "single_formulation",
                }
            ],
            "component_candidates": [
                {
                    "component_id": "c_poly",
                    "formulation_candidate_id": "F1",
                    "component_role_raw": "polymer",
                    "component_name_raw": "PLGA",
                }
            ],
            "variable_or_factor_candidates": [
                {"factor_id": "v1", "formulation_candidate_id": "F1", "factor_name_raw": "polymer concentration in organic phase", "factor_expression_raw": "25 mg/mL", "identity_defining_signal": "yes"},
                {"factor_id": "v2", "formulation_candidate_id": "F1", "factor_name_raw": "aqueous phase pH", "factor_expression_raw": "7.4", "identity_defining_signal": "yes"},
                {"factor_id": "v3", "formulation_candidate_id": "F1", "factor_name_raw": "organic:aqueous phase ratio", "factor_expression_raw": "1:2", "identity_defining_signal": "yes"},
                {"factor_id": "v4", "formulation_candidate_id": "F1", "factor_name_raw": "drug/polymer ratio", "factor_expression_raw": "1:10", "identity_defining_signal": "yes"},
                {"factor_id": "v5", "formulation_candidate_id": "F1", "factor_name_raw": "polymer/drug ratio", "factor_expression_raw": "10:1", "identity_defining_signal": "yes"},
                {"factor_id": "v6", "formulation_candidate_id": "F1", "factor_name_raw": "lactide:glycolide ratio", "factor_expression_raw": "75:25", "identity_defining_signal": "yes"},
            ],
            "measurement_candidates": [],
            "evidence_handoffs": [],
            "table_formulation_scopes": [],
            "table_variable_roles": [],
            "selection_markers": [],
            "inheritance_markers": [],
            "boundary_markers": [],
        }
        rows, _, _, _, _ = project_document(document)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["polymer_concentration_value_value_text"], "25 mg/mL")
        self.assertEqual(row["polymer_concentration_unit_value_text"], "mg/mL")
        self.assertEqual(row["polymer_concentration_phase_value_text"], "O")
        self.assertEqual(row["pH_raw_value_text"], "7.4")
        self.assertEqual(row["phase_ratio_raw_value_text"], "1:2")
        self.assertEqual(row["drug_to_polymer_ratio_raw_value_text"], "1:10")
        self.assertEqual(row["polymer_to_drug_ratio_raw_value_text"], "10:1")
        self.assertEqual(row["la_ga_ratio_raw_value_text"], "75:25")
        self.assertEqual(row["la_ga_ratio_normalized_value_text"], "75:25")

    def test_project_document_falls_back_to_polymer_name_for_la_ga_ratio(self):
        document = {
            "document_key": "LAGA1",
            "paper_key": "LAGA1",
            "doi": "10.1/laga1",
            "model_name": "test-model",
            "source_mode": "saved_raw_live_v2_replay_to_stage2_v2",
            "semantic_signals": {},
            "formulation_identity_candidates": [
                {
                    "formulation_candidate_id": "F1",
                    "raw_formulation_label": "F1",
                    "formulation_role": "reported",
                    "instance_kind": "single_formulation",
                }
            ],
            "component_candidates": [
                {
                    "component_id": "c_poly",
                    "formulation_candidate_id": "F1",
                    "component_role_raw": "polymer",
                    "component_name_raw": "PLGA 75:25",
                }
            ],
            "variable_or_factor_candidates": [],
            "measurement_candidates": [],
            "evidence_handoffs": [],
            "table_formulation_scopes": [],
            "table_variable_roles": [],
            "selection_markers": [],
            "inheritance_markers": [],
            "boundary_markers": [],
        }
        rows, _, _, _, _ = project_document(document)
        self.assertEqual(rows[0]["la_ga_ratio_raw_value_text"], "75:25")
        self.assertEqual(rows[0]["la_ga_ratio_normalized_value_text"], "75:25")

    def test_project_document_uses_row_specific_polymer_name_for_la_ga_ratio_and_keeps_pcl_blank(self):
        document = {
            "document_key": "LAGA2",
            "paper_key": "LAGA2",
            "doi": "10.1/laga2",
            "model_name": "test-model",
            "source_mode": "saved_raw_live_v2_replay_to_stage2_v2",
            "semantic_signals": {},
            "formulation_identity_candidates": [
                {"formulation_candidate_id": "F1", "raw_formulation_label": "PLGA 50/50 Empty", "formulation_role": "reported", "instance_kind": "single_formulation"},
                {"formulation_candidate_id": "F2", "raw_formulation_label": "PLGA 75/25 Loaded", "formulation_role": "reported", "instance_kind": "single_formulation"},
                {"formulation_candidate_id": "F3", "raw_formulation_label": "PCL Empty", "formulation_role": "reported", "instance_kind": "single_formulation"},
            ],
            "component_candidates": [
                {"component_id": "c1", "formulation_candidate_id": "", "component_role_raw": "polymer", "component_name_raw": "PLGA 50/50"},
                {"component_id": "c2", "formulation_candidate_id": "", "component_role_raw": "polymer", "component_name_raw": "PLGA 75/25"},
                {"component_id": "c3", "formulation_candidate_id": "", "component_role_raw": "polymer", "component_name_raw": "PCL"},
            ],
            "variable_or_factor_candidates": [],
            "measurement_candidates": [],
            "evidence_handoffs": [],
            "table_formulation_scopes": [],
            "table_variable_roles": [],
            "selection_markers": [],
            "inheritance_markers": [],
            "boundary_markers": [],
        }
        rows, _, _, _, _ = project_document(document)
        by_id = {row["formulation_id"]: row for row in rows}
        self.assertEqual(by_id["F1"]["la_ga_ratio_raw_value_text"], "50:50")
        self.assertEqual(by_id["F2"]["la_ga_ratio_raw_value_text"], "75:25")
        self.assertEqual(by_id["F3"]["la_ga_ratio_raw_value_text"], "")

    def test_project_document_excludes_release_ph_from_formulation_ph_surface(self):
        document = {
            "document_key": "PHSCOPE1",
            "paper_key": "PHSCOPE1",
            "doi": "10.1/phscope1",
            "model_name": "test-model",
            "source_mode": "saved_raw_live_v2_replay_to_stage2_v2",
            "semantic_signals": {},
            "formulation_identity_candidates": [
                {
                    "formulation_candidate_id": "F1",
                    "raw_formulation_label": "F1",
                    "formulation_role": "reported",
                    "instance_kind": "single_formulation",
                }
            ],
            "component_candidates": [],
            "variable_or_factor_candidates": [
                {"factor_id": "v1", "formulation_candidate_id": "F1", "factor_name_raw": "drug release pH", "factor_expression_raw": "7.4", "identity_defining_signal": "yes"},
                {"factor_id": "v2", "formulation_candidate_id": "F1", "factor_name_raw": "aqueous phase pH", "factor_expression_raw": "5.5", "identity_defining_signal": "yes"},
            ],
            "measurement_candidates": [],
            "evidence_handoffs": [],
            "table_formulation_scopes": [],
            "table_variable_roles": [],
            "selection_markers": [],
            "inheritance_markers": [],
            "boundary_markers": [],
        }
        rows, _, _, _, _ = project_document(document)
        self.assertEqual(rows[0]["pH_raw_value_text"], "5.5")

    def test_compare_values_allows_family_ratio_list_containment(self):
        strict, relaxed, canonicalized = compare_values(
            "polymer_to_drug_ratio_raw",
            "10:1",
            "5:1 | 10:1 | 15:1",
        )
        self.assertFalse(strict)
        self.assertTrue(relaxed)
        self.assertTrue(canonicalized)


class Stage2LiveBackendDispatchTests(unittest.TestCase):
    @patch("src.stage2_sampling_labels.extract_semantic_stage2_objects_v2.call_gemini", return_value="gemini-ok")
    def test_call_live_backend_dispatches_gemini(self, mock_call):
        result = call_live_backend("gemini", "m", "p", 1, 0.1, timeout_seconds=7)
        self.assertEqual(result, "gemini-ok")
        mock_call.assert_called_once_with(
            "m",
            "p",
            1,
            0.1,
            progress_label="",
            timeout_seconds=7,
        )

    @patch("src.stage2_sampling_labels.extract_semantic_stage2_objects_v2.call_nvidia_hosted", return_value="nvidia-ok")
    def test_call_live_backend_dispatches_nvidia(self, mock_call):
        result = call_live_backend("nvidia", "m", "p", 2, 0.2)
        self.assertEqual(result, "nvidia-ok")
        mock_call.assert_called_once_with(
            "m",
            "p",
            2,
            0.2,
            progress_label="",
        )

    @patch("src.stage2_sampling_labels.extract_semantic_stage2_objects_v2.call_ollama_hosted", return_value="ollama-ok")
    def test_call_live_backend_dispatches_ollama(self, mock_call):
        result = call_live_backend(
            "ollama",
            "m",
            "p",
            3,
            0.3,
            timeout_seconds=11,
            system_prompt=ollama_live_system_prompt(),
            response_format=OLLAMA_SHRUNKEN_LIVE_SCHEMA,
        )
        self.assertEqual(result, "ollama-ok")
        mock_call.assert_called_once_with(
            "m",
            "p",
            3,
            0.3,
            progress_label="",
            timeout_seconds=11,
            system_prompt=ollama_live_system_prompt(),
            response_format=OLLAMA_SHRUNKEN_LIVE_SCHEMA,
        )


if __name__ == "__main__":
    unittest.main()
