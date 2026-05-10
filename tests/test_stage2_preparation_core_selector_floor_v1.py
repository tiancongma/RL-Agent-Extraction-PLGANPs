import unittest

from src.stage2_sampling_labels.evaluate_s2_4a_hard_gate_v1 import table_summary_satisfies_pre_live_evidence
from pathlib import Path

from src.stage2_sampling_labels.extract_semantic_stage2_objects_v2 import (
    SEGMENT_MATERIALS_CUES,
    TABLE_INCLUSION_HARD_DROP,
    TABLE_INCLUSION_MUST_INCLUDE,
    TABLE_INCLUSION_OPTIONAL_CONTEXT,
    apply_minimal_evidence_floor,
    build_evidence_priority_selection,
    build_inline_formulation_table_item,
    build_table_authority_score_breakdown,
    classify_heading_line,
    classify_payload_authority_status,
    compute_reconstruction_confidence,
    derive_stable_table_id,
    derive_table_identity_aliases,
    classify_execution_table_type,
    classify_table_source_role,
    infer_authority_table_type,
    infer_table_role_hint,
    payload_inclusion_class,
    payload_usage_role,
    render_value_evidence_payload_blocks,
    table_inclusion_class,
)


class Stage2PreparationCoreSelectorFloorTests(unittest.TestCase):
    def test_generic_selector_cues_do_not_depend_on_paper_local_material_names(self):
        banned = {"labrafil", "polysorbate"}
        self.assertTrue(banned.isdisjoint({cue.lower() for cue in SEGMENT_MATERIALS_CUES}))

    def test_heading_classifier_does_not_promote_paper_local_loaded_drug_headings(self):
        self.assertIsNone(classify_heading_line("Gatifloxacin-loaded PLGA NPs"))
        self.assertIsNone(classify_heading_line("Rhodamine-loaded PLGA NPs"))
        self.assertEqual(classify_heading_line("Preparation and characterization of NPs"), "heading")

    def test_inline_table_recovery_requires_generic_schema_not_paper_local_names_only(self):
        paper_local_only = (
            "Table 1 Gatifloxacin Rhodamine Labrafil Polysorbate 80 "
            "NPG1 5 10 15 20 NPG2 6 11 16 21 NPG3 7 12 17 22"
        )
        self.assertIsNone(
            build_inline_formulation_table_item(
                paper_local_only,
                text_path=Path("paper.txt"),
                paragraph_index=1,
                segment_index=1,
            )
        )
        generic_schema = (
            "Table 1 Formulation drug loading surfactant particle size "
            "NP1 5 1 120 NP2 10 2 140 NP3 15 3 160"
        )
        self.assertIsNotNone(
            build_inline_formulation_table_item(
                generic_schema,
                text_path=Path("paper.txt"),
                paragraph_index=1,
                segment_index=2,
            )
        )

    def _method_candidate(self, candidate_id: str, text: str, *, paragraph_index: int = 1) -> dict:
        return {
            "candidate_id": candidate_id,
            "candidate_kind": "text",
            "section_kind": "preparation",
            "section_label": "Preparation of nanoparticles",
            "block_type": "synthesis_method",
            "text_content": text,
            "origin_locator": f"paragraph:{paragraph_index}",
            "evidence_kind": "method",
            "priority_score": 10.0,
        }

    def test_preparation_core_floor_adds_source_backed_core_even_when_generic_method_already_selected(self):
        selected = [
            self._method_candidate(
                "method-overview",
                "Nanoparticles were prepared by nanoprecipitation. The organic phase was added dropwise to the aqueous phase under stirring and the solvent was evaporated under vacuum before filtration and centrifugation. This overview describes the procedure but contains no material-bound value.",
                paragraph_index=1,
            )
        ]
        preparation_core = self._method_candidate(
            "method-core",
            "For nanoparticle preparation, PLGA 100 mg and curcumin 10 mg were dissolved in 5 mL acetone. The organic phase was added dropwise into 20 mL aqueous PVA solution at 1% w/v under stirring before solvent evaporation and centrifugation.",
            paragraph_index=2,
        )
        events = []

        result = apply_minimal_evidence_floor(
            selected_candidates=selected,
            ranked_candidates=[selected[0], preparation_core],
            suppression_events=events,
        )

        selected_ids = {candidate["candidate_id"] for candidate in result["selected_candidates"]}
        self.assertIn("method-core", selected_ids)
        self.assertEqual(result.get("floor_added_preparation_core", "no"), "yes")
        self.assertIn("added_source_backed_preparation_core", result["floor_rationale"])
        self.assertIn(
            {"candidate_id": "method-core", "reason": "minimal_evidence_floor_added_preparation_core"},
            events,
        )

    def test_solvent_diffusion_experimental_design_prose_counts_as_preparation_core(self):
        selected = [
            self._method_candidate(
                "figure-caption-method",
                "Figure 2 The schematic diagram showing the preparation of AP-PLGA-NPs. Figure 3 In-vitro release profiles from PLGA-NPs in buffer at 37C.",
                paragraph_index=28,
            )
        ]
        selected[0]["section_kind"] = "preparation"
        selected[0]["section_label"] = "Results Characterization"
        core = self._method_candidate(
            "solvent-diffusion-core",
            "Preparation and characterization of AP-PLGA-NPs AP-PLGA-NPs were prepared by a solvent diffusion methodology. Briefly, 18 mg of PLGA and 7 mg of AP were dissolved in 1 ml of acetone to form an organic phase; then the organic phase was poured into 4 ml of stirred aqueous phase containing 1% polysorbate 80.",
            paragraph_index=13,
        )
        core["section_kind"] = "experimental_design"
        events = []

        result = apply_minimal_evidence_floor(
            selected_candidates=selected,
            ranked_candidates=[selected[0], core],
            suppression_events=events,
        )

        selected_ids = {candidate["candidate_id"] for candidate in result["selected_candidates"]}
        self.assertIn("solvent-diffusion-core", selected_ids)
        self.assertEqual(result.get("minimal_evidence_floor_applied", "no"), "yes")

    def test_selector_suppresses_noisy_method_spillover_after_source_core_method(self):
        core = self._method_candidate(
            "source-core-method",
            "Preparation and characterization of AP-PLGA-NPs AP-PLGA-NPs were prepared by a solvent diffusion methodology. Briefly, 18 mg of PLGA and 7 mg of AP were dissolved in 1 ml of acetone to form an organic phase; then the organic phase was poured into 4 ml of stirred aqueous phase containing 1% polysorbate 80.",
            paragraph_index=13,
        )
        core["section_kind"] = "experimental_design"
        noisy = self._method_candidate(
            "figure-spillover-method",
            "1654 PLGA-NPs improve AP permeability Deqing Sun et al. © 2015 Royal Pharmaceutical Society, Journal of Pharmacy and Pharmacology, 67, pp. 1650–1662 solvent diffusion/evaporation Encapsulated drug Polysorbate 80 Acetylpuerarin PLGA 50:50 200nm Dissolved in organic phase Dissolved in aqueous phase PLGA nanoparticle Polysorbate 80 w+x+y+z=20 Cumulative release rate (%) Time (h) AP solution AP-PLGA-NPs Downloaded from journal site. Tissue distributions in mice The standard calibrations were plotted.",
            paragraph_index=30,
        )
        noisy["quality_flags"] = ["residual_noise"]

        result = build_evidence_priority_selection(segmented_candidates=[core, noisy], signals={})
        selected_ids = {candidate["candidate_id"] for candidate in result["selected_candidates"]}
        suppression_reasons = {event["candidate_id"]: event["reason"] for event in result["suppression_events"]}

        self.assertIn("source-core-method", selected_ids)
        self.assertNotIn("figure-spillover-method", selected_ids)
        self.assertEqual(suppression_reasons.get("figure-spillover-method"), "noisy_method_spillover_after_core_method")

    def test_variant_preparation_same_procedure_block_is_selected_with_core_method(self):
        core = self._method_candidate(
            "same-procedure-core",
            "Drug-loaded polymer nanoparticles (NPV1; Table 1) were prepared by nanoprecipitation using an acetone-water system. Briefly, 50 mg of polymer and 2.5 mg model drug were dissolved in 4 mL acetone and mixed by vortexing. This mixture was added dropwise into 12 mL of 1% stabilizer under continuous stirring for 15 minutes.",
            paragraph_index=10,
        )
        variant = self._method_candidate(
            "same-procedure-variant-prep",
            "Drug-loaded polymer-modifier NPs were prepared using the same protocol, but incorporating 3.5 mg surface modifier into the inner phase of the emulsion (NPV3). When the modifier was incorporated, the desiccation process was performed under vacuum. Active-loaded polymer NPs, active-loaded polymer-surfactant NPs, and active-loaded polymer-modifier NPs were prepared using the same procedure but incorporating 5 mg of active compound into the inner phase. Table 1 shows the different formulations prepared.",
            paragraph_index=13,
        )
        variant["section_kind"] = "variant_preparation"
        variant["section_label"] = "Preparation and characterization of nanoparticles"
        variant["block_type"] = ""

        assay = self._method_candidate(
            "analytical-extraction-assay",
            "Drug-encapsulation efficiency (EE%) was determined by weighing 10 mg of nanoparticles, which were dissolved in 1 mL solvent. Then, ethanol (15 mL) was added and centrifuged for 5 minutes at 5,000 rpm. This procedure was repeated five times to extract the drug completely. Then, samples were filtered through 0.45 µm filters and the drug content of each formulation quantified by high-performance liquid chromatography (HPLC).",
            paragraph_index=14,
        )
        assay["section_kind"] = "optimization"
        assay["block_type"] = "paragraph"

        result = build_evidence_priority_selection(segmented_candidates=[core, variant, assay], signals={})
        selected_ids = {candidate["candidate_id"] for candidate in result["selected_candidates"]}
        prompt_ids = {candidate["candidate_id"] for candidate in result["prompt_selected_candidates"]}

        self.assertIn("same-procedure-core", selected_ids)
        self.assertIn("same-procedure-variant-prep", selected_ids)
        self.assertIn("same-procedure-variant-prep", prompt_ids)
        self.assertNotIn("analytical-extraction-assay", prompt_ids)

    def test_preparation_core_floor_rejects_caption_locator_even_with_local_cues(self):
        selected = [
            self._method_candidate(
                "method-overview",
                "Nanoparticles were prepared by nanoprecipitation. The organic phase was added dropwise to the aqueous phase under stirring and the solvent was evaporated under vacuum before filtration and centrifugation. This overview describes the procedure but contains no material-bound value.",
            )
        ]
        caption_like = self._method_candidate(
            "caption-like-core",
            "For preparation, PLGA 100 mg was dissolved in 5 mL acetone and added dropwise into aqueous PVA solution under stirring.",
            paragraph_index=2,
        )
        caption_like["origin_locator"] = "caption:figure-2"
        events = []

        result = apply_minimal_evidence_floor(
            selected_candidates=selected,
            ranked_candidates=[selected[0], caption_like],
            suppression_events=events,
        )

        selected_ids = {candidate["candidate_id"] for candidate in result["selected_candidates"]}
        self.assertNotIn("caption-like-core", selected_ids)
        self.assertEqual(result.get("floor_added_preparation_core", "no"), "no")

    def test_preparation_core_floor_requires_local_action_material_value_binding(self):
        selected = [
            self._method_candidate(
                "method-overview",
                "Nanoparticles were prepared by nanoprecipitation. The organic phase was added dropwise to the aqueous phase under stirring and the solvent was evaporated under vacuum before filtration and centrifugation. This overview describes the procedure but contains no material-bound value.",
            )
        ]
        material_value_without_action = self._method_candidate(
            "material-value-no-action",
            "Nanoparticles were prepared by nanoprecipitation and purified by centrifugation. The PLGA polymer amount was 100 mg and aqueous volume was 20 mL for the experiment.",
            paragraph_index=2,
        )
        events = []

        result = apply_minimal_evidence_floor(
            selected_candidates=selected,
            ranked_candidates=[selected[0], material_value_without_action],
            suppression_events=events,
        )

        selected_ids = {candidate["candidate_id"] for candidate in result["selected_candidates"]}
        self.assertNotIn("material-value-no-action", selected_ids)
        self.assertEqual(result.get("floor_added_preparation_core", "no"), "no")

    def test_preparation_core_floor_rejects_missing_source_locator(self):
        selected = [
            self._method_candidate(
                "method-overview",
                "Nanoparticles were prepared by nanoprecipitation. The organic phase was added dropwise to the aqueous phase under stirring and the solvent was evaporated under vacuum before filtration and centrifugation. This overview describes the procedure but contains no material-bound value.",
            )
        ]
        missing_locator = self._method_candidate(
            "missing-locator-core",
            "For preparation, PLGA 100 mg was dissolved in 5 mL acetone and added dropwise into aqueous PVA solution under stirring.",
            paragraph_index=2,
        )
        missing_locator["origin_locator"] = ""
        events = []

        result = apply_minimal_evidence_floor(
            selected_candidates=selected,
            ranked_candidates=[selected[0], missing_locator],
            suppression_events=events,
        )

        selected_ids = {candidate["candidate_id"] for candidate in result["selected_candidates"]}
        self.assertNotIn("missing-locator-core", selected_ids)
        self.assertEqual(result.get("floor_added_preparation_core", "no"), "no")

    def test_preparation_core_floor_rejects_toc_or_caption_markers_in_text(self):
        selected = [
            self._method_candidate(
                "method-overview",
                "Nanoparticles were prepared by nanoprecipitation. The organic phase was added dropwise to the aqueous phase under stirring and the solvent was evaporated under vacuum before filtration and centrifugation. This overview describes the procedure but contains no material-bound value.",
            )
        ]
        text_marked_noise = self._method_candidate(
            "text-marked-caption-core",
            "Figure caption: PLGA 100 mg was dissolved in 5 mL acetone and added dropwise into aqueous PVA solution under stirring.",
            paragraph_index=2,
        )
        events = []

        result = apply_minimal_evidence_floor(
            selected_candidates=selected,
            ranked_candidates=[selected[0], text_marked_noise],
            suppression_events=events,
        )

        selected_ids = {candidate["candidate_id"] for candidate in result["selected_candidates"]}
        self.assertNotIn("text-marked-caption-core", selected_ids)
        self.assertEqual(result.get("floor_added_preparation_core", "no"), "no")

    def test_preparation_core_floor_rejects_dispersed_toc_and_caption_cues(self):
        selected = [
            self._method_candidate(
                "method-overview",
                "Nanoparticles were prepared by nanoprecipitation. The organic phase was added dropwise to the aqueous phase under stirring and the solvent was evaporated under vacuum before filtration and centrifugation. This overview describes the procedure but contains no material-bound value.",
            )
        ]
        dispersed_noise = self._method_candidate(
            "toc-caption-noise",
            "Contents: preparation of particles. Figure captions mention PLGA nanoparticles. Table captions list formulation labels. References include 100 mg in an analytical assay and 5 mL sample preparation, but no local preparation sentence binds a material to a value.",
            paragraph_index=2,
        )
        events = []

        result = apply_minimal_evidence_floor(
            selected_candidates=selected,
            ranked_candidates=[selected[0], dispersed_noise],
            suppression_events=events,
        )

        selected_ids = {candidate["candidate_id"] for candidate in result["selected_candidates"]}
        self.assertNotIn("toc-caption-noise", selected_ids)
        self.assertEqual(result.get("floor_added_preparation_core", "no"), "no")
        self.assertNotIn("minimal_evidence_floor_added_preparation_core", {event["reason"] for event in events})
    def _table_candidate(
        self,
        candidate_id: str,
        text: str,
        *,
        inclusion_class: str,
        table_role_hint: str = "results",
        source_role: str = "characterization_result_table",
        score: float = 100.0,
    ) -> dict:
        return {
            "candidate_id": candidate_id,
            "candidate_kind": "table",
            "source_type": "table_excerpt",
            "origin_locator": f"data/cleaned/dev/tables/PAPER/{candidate_id}.csv",
            "text_content": text,
            "table_inclusion_class": inclusion_class,
            "table_role_hint": table_role_hint,
            "table_source_role": source_role,
            "authority_score": score,
            "selector_readiness_label": "ready",
            "representation_status": "raw_summary",
            "rows": [["Formulation", "Signal"], [candidate_id, text]],
            "meta": {"caption_or_title": text[:120]},
        }

    def test_optional_numeric_only_table_is_preserved_but_not_prompt_selected(self):
        must_table = self._table_candidate(
            "must_formulation",
            "Table 1. Formulation composition of PLGA nanoparticles prepared with PVA: F1 PLGA 100 mg drug 10 mg.",
            inclusion_class=TABLE_INCLUSION_MUST_INCLUDE,
            table_role_hint="formulation",
            source_role="formulation_composition_table",
        )
        numeric_only_optional = self._table_candidate(
            "optional_numeric_only",
            "1 2 3 4 5 6 7 8 9 10",
            inclusion_class=TABLE_INCLUSION_OPTIONAL_CONTEXT,
        )

        result = build_evidence_priority_selection(segmented_candidates=[must_table, numeric_only_optional], signals={})

        selected_ids = {candidate["candidate_id"] for candidate in result["selected_candidates"]}
        prompt_ids = {candidate["candidate_id"] for candidate in result["prompt_selected_candidates"]}
        self.assertIn("optional_numeric_only", selected_ids)
        self.assertIn("must_formulation", prompt_ids)
        self.assertNotIn("optional_numeric_only", prompt_ids)
        self.assertNotIn(
            {"candidate_id": "optional_numeric_only", "reason": "hard_drop_table_noise"},
            result["suppression_events"],
        )

    def test_optional_rowlocal_result_summary_is_preserved_but_not_prompt_selected(self):
        rowlocal_table = self._table_candidate(
            "optional_rowlocal",
            "[TABLE_SUMMARY] key_columns: Cytotoxicity studies of drug-loaded NPs; semantic_summary: row-local measurement/context evidence only; not formulation-universe or full numeric/table authority.",
            inclusion_class=TABLE_INCLUSION_OPTIONAL_CONTEXT,
            table_role_hint="results",
        )

        result = build_evidence_priority_selection(segmented_candidates=[rowlocal_table], signals={})

        selected_ids = {candidate["candidate_id"] for candidate in result["selected_candidates"]}
        prompt_ids = {candidate["candidate_id"] for candidate in result["prompt_selected_candidates"]}
        self.assertIn("optional_rowlocal", selected_ids)
        self.assertNotIn("optional_rowlocal", prompt_ids)

    def test_optional_semantic_design_table_can_be_prompt_selected(self):
        design_table = self._table_candidate(
            "optional_design",
            "Table 2. Optimization design matrix for PLGA nanoparticles prepared with different PVA concentrations and drug/polymer ratios.",
            inclusion_class=TABLE_INCLUSION_OPTIONAL_CONTEXT,
            table_role_hint="design matrix",
            source_role="formulation_composition_table",
        )

        result = build_evidence_priority_selection(segmented_candidates=[design_table], signals={})

        selected_ids = {candidate["candidate_id"] for candidate in result["selected_candidates"]}
        prompt_ids = {candidate["candidate_id"] for candidate in result["prompt_selected_candidates"]}
        self.assertIn("optional_design", selected_ids)
        self.assertIn("optional_design", prompt_ids)

    def test_table_authority_keeps_pharmacokinetic_result_table_out_of_formulation_authority(self):
        item = {
            "path": "synthetic__table_7__.csv",
            "rows": [
                ["Run", "Dose", "AUC", "Cmax", "Tmax"],
                ["1", "10 mg/kg", "120", "34", "2"],
                ["2", "10 mg/kg", "118", "33", "2"],
                ["3", "10 mg/kg", "124", "35", "3"],
            ],
            "meta": {
                "caption_or_title": "Pharmacokinetic parameters and plasma concentration-time profiles in rats",
                "n_rows": 4,
                "n_cols": 5,
            },
            "selector_readiness_label": "ready",
            "representation_status": "raw_summary",
            "repair_confidence": 1.0,
            "repair_primary_source": "stage1_selected_table_asset",
            "score": 160,
        }

        breakdown = build_table_authority_score_breakdown(item)
        role_hint = infer_table_role_hint(item["rows"][0], {**item["meta"], "_signal_text": " ".join(" ".join(row) for row in item["rows"])})

        self.assertEqual(role_hint, "results")
        self.assertEqual(infer_authority_table_type(item, role_hint=role_hint, breakdown=breakdown), "non_formulation_table")
        self.assertEqual(table_inclusion_class(item, breakdown=breakdown), TABLE_INCLUSION_OPTIONAL_CONTEXT)
    def test_table_authority_negative_taxonomy_keeps_release_profile_out_of_formulation_priority(self):
        item = {
            "path": "synthetic__table_8__.csv",
            "rows": [
                ["Formulation", "Time (h)", "Cumulative release (%)"],
                ["F1", "1", "18"],
                ["F1", "4", "47"],
                ["F1", "24", "82"],
            ],
            "meta": {
                "caption_or_title": "In vitro release profile of drug-loaded PLGA nanoparticles",
                "n_rows": 4,
                "n_cols": 3,
            },
            "selector_readiness_label": "ready",
            "representation_status": "raw_summary",
            "repair_confidence": 1.0,
            "repair_primary_source": "stage1_selected_table_asset",
            "score": 160,
        }

        source_role = classify_table_source_role(
            item["rows"][0],
            {**item["meta"], "_signal_text": " ".join(" ".join(row) for row in item["rows"])},
        )
        breakdown = build_table_authority_score_breakdown(item)

        self.assertEqual(source_role, "release_profile_table")
        self.assertEqual(table_inclusion_class(item, breakdown=breakdown), TABLE_INCLUSION_OPTIONAL_CONTEXT)

    def test_table_authority_negative_taxonomy_allows_strong_composition_override(self):
        item = {
            "path": "synthetic__table_9__.csv",
            "rows": [
                ["Formulation", "PLGA mass (mg)", "Drug mass (mg)", "PVA concentration (% w/v)", "Organic phase volume (mL)"],
                ["F1", "100", "10", "1", "5"],
                ["F2", "150", "15", "1", "5"],
                ["F3", "200", "20", "2", "5"],
            ],
            "meta": {
                "caption_or_title": "Formulation composition and in vitro release study groups",
                "n_rows": 4,
                "n_cols": 5,
            },
            "selector_readiness_label": "ready",
            "representation_status": "raw_summary",
            "repair_confidence": 1.0,
            "repair_primary_source": "stage1_selected_table_asset",
            "score": 160,
        }

        source_role = classify_table_source_role(
            item["rows"][0],
            {**item["meta"], "_signal_text": " ".join(" ".join(row) for row in item["rows"])},
        )
        breakdown = build_table_authority_score_breakdown(item)

        self.assertEqual(source_role, "formulation_composition_table")
        self.assertEqual(table_inclusion_class(item, breakdown=breakdown), TABLE_INCLUSION_MUST_INCLUDE)

    def test_table_authority_compact_composition_headers_override_characterization_columns(self):
        item = {
            "path": "synthetic__table_10__.csv",
            "rows": [
                ["Formulation", "PLGA (mg)", "Drug (mg)", "PVA (%)", "Size (nm)", "PDI"],
                ["F1", "100", "10", "1", "182", "0.10"],
                ["F2", "150", "15", "1", "190", "0.12"],
                ["F3", "200", "20", "2", "205", "0.14"],
            ],
            "meta": {"caption_or_title": "Composition and characterization of nanoparticle formulations"},
            "selector_readiness_label": "ready",
            "representation_status": "raw_summary",
            "repair_confidence": 1.0,
            "repair_primary_source": "stage1_selected_table_asset",
            "score": 160,
        }

        source_role = classify_table_source_role(
            item["rows"][0],
            {**item["meta"], "_signal_text": " ".join(" ".join(row) for row in item["rows"])},
        )
        breakdown = build_table_authority_score_breakdown(item)

        self.assertEqual(source_role, "formulation_composition_table")
        self.assertEqual(table_inclusion_class(item, breakdown=breakdown), TABLE_INCLUSION_MUST_INCLUDE)

    def test_execution_payload_negative_taxonomy_preserves_release_table_as_optional_context(self):
        normalized_matrix = [
            ["Formulation", "Time (h)", "Cumulative release (%)"],
            ["F1", "1", "18"],
            ["F1", "4", "47"],
            ["F1", "24", "82"],
        ]
        meta = {"caption_or_title": "In vitro release profile of drug-loaded PLGA nanoparticles"}

        table_type = classify_execution_table_type(
            normalized_matrix,
            meta=meta,
            table_role_hint="results",
            normalization_metadata={"numbered_row_count": 0},
        )
        payload = {
            "table_type": table_type,
            "table_role_hint": "results",
            "table_source_role": "release_profile_table",
            "raw_cells": normalized_matrix,
            "header_structure": {"header_rows": [normalized_matrix[0]]},
            "source_caption_or_title": meta["caption_or_title"],
            "row_identity_signals": {"row_pattern": "mixed identifiers", "first_column_labels": ["F1", "F1", "F1"]},
            "data_row_count": 3,
            "representation_status": "raw_summary",
        }

        self.assertEqual(table_type, "non_formulation_table")
        self.assertEqual(payload_inclusion_class(payload), TABLE_INCLUSION_OPTIONAL_CONTEXT)
        self.assertEqual(payload_usage_role(payload), "row_local_value_evidence")

    def test_execution_payload_primary_formulation_table_is_universe_authority(self):
        payload = {
            "table_type": "formulation_table",
            "table_role_hint": "formulation",
            "table_source_role": "formulation_composition_table",
            "table_inclusion_class": TABLE_INCLUSION_MUST_INCLUDE,
            "raw_cells": [
                ["Formulation", "Drug", "Polymer"],
                ["F1", "Drug A", "PLGA"],
                ["F2", "Drug A", "PLGA"],
            ],
            "header_structure": {"header_rows": [["Formulation", "Drug", "Polymer"]]},
        }

        self.assertEqual(payload_usage_role(payload), "formulation_universe_authority")

    def test_characterization_payload_is_value_evidence_even_with_formulation_labels(self):
        payload = {
            "table_type": "non_formulation_table",
            "table_role_hint": "characterization",
            "table_source_role": "characterization_result_table",
            "table_inclusion_class": TABLE_INCLUSION_OPTIONAL_CONTEXT,
            "raw_cells": [
                ["Formulation", "Size (nm)", "PDI", "Zeta potential (mV)", "EE (%)"],
                ["F1", "180.2", "0.14", "-21.1", "78.5"],
                ["F2", "192.4", "0.18", "-18.6", "81.2"],
            ],
            "header_structure": {"header_rows": [["Formulation", "Size (nm)", "PDI", "Zeta potential (mV)", "EE (%)"]]},
        }

        self.assertEqual(payload_inclusion_class(payload), TABLE_INCLUSION_OPTIONAL_CONTEXT)
        self.assertEqual(payload_usage_role(payload), "row_local_value_evidence")

    def test_table_authority_negative_taxonomy_blocks_targeting_and_intravenous_result_tables(self):
        rows = [
            ["Formulation", "Dose (mg/kg)", "Intravenous administration", "Targeting efficiency (%)"],
            ["F1", "10", "tail vein", "18"],
            ["F2", "10", "tail vein", "23"],
        ]
        meta = {"caption_or_title": "Targeting parameters after intravenous administration of nanoparticle formulations"}

        source_role = classify_table_source_role(
            rows[0],
            {**meta, "_signal_text": " ".join(" ".join(row) for row in rows)},
        )

        self.assertEqual(source_role, "noise_or_nonformulation_table")

    def test_table_identity_aliases_preserve_conflicting_caption_and_file_ids(self):
        source_csv_path = "data/cleaned/dev/tables/PAPER/PAPER__table_2__.csv"
        meta = {
            "table_id": "Table 1",
            "caption_or_title": "Table 3 Composition of PLGA nanoparticle formulations",
        }

        self.assertEqual(derive_stable_table_id(source_csv_path, meta), "Table 2")
        self.assertEqual(
            derive_table_identity_aliases(source_csv_path, meta),
            ["Table 1", "Table 3", "Table 2"],
        )

    def test_payload_reconstruction_confidence_marks_unresolved_representations_below_visible_floor(self):
        confidence = compute_reconstruction_confidence(
            representation_status="unrepaired_corrupted",
            selector_readiness_label="unresolved",
            normalization_actions=[],
            normalized_row_count=1,
            raw_row_count=6,
        )
        status, reasons = classify_payload_authority_status(
            normalized_rows=[["orphan value"]],
            header_structure={"header_row_count": 0},
            reconstruction_confidence=confidence,
        )

        self.assertEqual(status, "unusable_broken_payload")
        self.assertIn("normalized_matrix_too_small", reasons)
        self.assertIn("missing_header_structure", reasons)

    def test_payload_authority_status_keeps_healthy_grid_visible_and_aliases_deduped(self):
        status, reasons = classify_payload_authority_status(
            normalized_rows=[["Formulation", "Drug", "Polymer"], ["F1", "Curcumin", "PLGA"]],
            header_structure={"header_row_count": 1},
            reconstruction_confidence=0.95,
        )

        self.assertEqual(status, "authority_visible")
        self.assertEqual(reasons, [])
        self.assertEqual(
            derive_table_identity_aliases(
                "data/cleaned/dev/tables/PAPER/PAPER__table_2__.csv",
                {"table_id": "Table 2", "caption_or_title": "Table 2 Formulation composition"},
            ),
            ["Table 2"],
        )
    def test_s2_4a_gate_path1_uses_visible_table_evidence_not_formulation_authority(self):
        payload = {
            "table_id": "Table 1",
            "table_inclusion_class": TABLE_INCLUSION_OPTIONAL_CONTEXT,
            "payload_authority_status": "authority_visible",
            "table_source_role": "characterization_result_table",
            "data_row_count": 7,
            "raw_cells": [
                ["Formulation", "Artemether (mg)", "PLGA (mg)", "PVA (mg)"],
                ["Batch 1", "20", "200", "50"],
                ["Batch 2", "30", "200", "50"],
            ],
        }
        block = {"table_id": "Table 1", "source_type": "table_summary", "is_table_derived": True}

        self.assertTrue(table_summary_satisfies_pre_live_evidence(block, payload))

    def test_s2_4a_gate_path1_rejects_hard_dropped_or_unusable_tables(self):
        block = {"table_id": "Table 1", "source_type": "table_summary", "is_table_derived": True}
        hard_drop_payload = {
            "table_id": "Table 1",
            "table_inclusion_class": TABLE_INCLUSION_HARD_DROP,
            "payload_authority_status": "authority_visible",
            "data_row_count": 3,
        }
        unusable_payload = {
            "table_id": "Table 1",
            "table_inclusion_class": TABLE_INCLUSION_OPTIONAL_CONTEXT,
            "payload_authority_status": "unusable_broken_payload",
            "data_row_count": 3,
        }

        self.assertFalse(table_summary_satisfies_pre_live_evidence(block, hard_drop_payload))
        self.assertFalse(table_summary_satisfies_pre_live_evidence(block, unusable_payload))

    def test_value_evidence_prompt_echo_is_budgeted_to_doe_design_payloads(self):
        evidence_artifact = {
            "paper_key": "UFXX9WXE",
            "authority_payload_root": "unused",
        }
        design_payload = {
            "table_id": "Table 13",
            "source_table_reference": "table_13.csv",
            "source_caption_or_title": "Table 2: Effect of independent process variables on dependent variable.",
            "payload_usage_role": "row_local_value_evidence",
            "value_evidence_only": True,
            "table_source_role": "characterization_result_table",
            "table_inclusion_class": TABLE_INCLUSION_OPTIONAL_CONTEXT,
            "authority_rank": 1,
            "normalized_matrix": [
                ["Formulation", "PLGA", "Poloxamer", "w/o phase", "Drug conc.", "z-Average", "% entrapment"],
                ["1", "35", "2", "6", "1", "211", "70"],
                ["2", "35", "2", "6", "5", "220", "88.48"],
            ],
        }
        noise_payload = {
            "table_id": "Table 4",
            "source_table_reference": "table_04.csv",
            "source_caption_or_title": "Biodistribution parameters in tissues",
            "payload_usage_role": "row_local_value_evidence",
            "value_evidence_only": True,
            "table_source_role": "tissue_distribution_table",
            "table_inclusion_class": TABLE_INCLUSION_OPTIONAL_CONTEXT,
            "authority_rank": 14,
            "normalized_matrix": [["Organ", "0.5 h", "1 h"], ["Blood", "2.9", "2.7"]],
        }

        original_loader = render_value_evidence_payload_blocks.__globals__["load_value_evidence_payloads_for_prompt"]
        render_value_evidence_payload_blocks.__globals__["load_value_evidence_payloads_for_prompt"] = lambda _artifact: [noise_payload, design_payload]
        try:
            blocks = render_value_evidence_payload_blocks(evidence_artifact)
        finally:
            render_value_evidence_payload_blocks.__globals__["load_value_evidence_payloads_for_prompt"] = original_loader

        rendered = "\n\n".join(blocks)
        self.assertIn("Table 13", rendered)
        self.assertIn("PLGA", rendered)
        self.assertNotIn("Table 4", rendered)
        self.assertNotIn("Biodistribution", rendered)


if __name__ == "__main__":
    unittest.main()
