import unittest

from src.stage2_sampling_labels.extract_semantic_stage2_objects_v2 import (
    TABLE_INCLUSION_MUST_INCLUDE,
    TABLE_INCLUSION_OPTIONAL_CONTEXT,
    apply_minimal_evidence_floor,
    build_table_authority_score_breakdown,
    classify_execution_table_type,
    classify_table_source_role,
    infer_authority_table_type,
    infer_table_role_hint,
    payload_inclusion_class,
    table_inclusion_class,
)


class Stage2PreparationCoreSelectorFloorTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
