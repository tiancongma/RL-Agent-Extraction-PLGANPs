import unittest

from src.stage2_sampling_labels.evaluate_s2_4a_hard_gate_v1 import (
    matches_path2_text,
    matches_path3_text,
    matches_path4_preparation_core_text,
    table_summary_satisfies_pre_live_evidence,
)
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
    evidence_text_has_noise,
    formulation_method_candidate_is_floor_eligible,
    infer_authority_table_type,
    infer_table_role_hint,
    infer_section_kind,
    is_reference_like_text,
    payload_inclusion_class,
    payload_usage_role,
    render_value_evidence_payload_blocks,
    resolve_tables_dir_for_record,
    select_prompt_table_floor_candidates,
    should_drop_segment,
    source_backed_preparation_statement_candidate_is_floor_eligible,
    source_body_preparation_anchor_candidate,
    preparation_core_locator_is_source_body,
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

    def test_formulation_method_floor_recovers_preparation_with_low_generic_procedure_count(self):
        materials = {
            "candidate_id": "materials-only",
            "candidate_kind": "text",
            "section_kind": "materials",
            "section_label": "2.1. Materials",
            "block_type": "materials_procurement",
            "text_content": "PLGA and PVA were purchased from commercial suppliers and ethyl acetate was obtained from a chemical vendor.",
            "origin_locator": "paragraph:3",
            "evidence_kind": "materials",
            "priority_score": 8.0,
        }
        formulation_method = self._method_candidate(
            "emulsion-preparation",
            "Different double emulsions were prepared to optimize the composition of each phase: 5 or 10 mg/mL of live cells were pre-suspended in aqueous PVA solution and EA/PLGA organic phase was then sonicated to form emulsion droplets before solvent removal.",
            paragraph_index=8,
        )
        formulation_method["evidence_kind"] = "supporting"
        formulation_method["priority_score"] = 2.0

        events = []
        result = apply_minimal_evidence_floor(
            selected_candidates=[materials],
            ranked_candidates=[formulation_method, materials],
            suppression_events=events,
        )
        selected_ids = {candidate["candidate_id"] for candidate in result["selected_candidates"]}

        self.assertIn("emulsion-preparation", selected_ids)
        self.assertEqual(result.get("floor_added_formulation_method", "no"), "yes")
        self.assertIn("added_source_backed_formulation_method", result["floor_rationale"])

    def test_short_source_body_preparation_anchor_is_method_evidence_without_numeric_values(self):
        candidate = {
            "candidate_id": "plga-np-anchor",
            "candidate_kind": "paragraph",
            "section_kind": "preparation",
            "section_label": "2.2. Preparation of PLGA-NPs",
            "block_type": "paragraph",
            "text_content": (
                "2.2. Preparation of PLGA-NPs PLGA-NPs were obtained using "
                "the oil-in-water (o/w) emulsion solvent extraction method. "
                "A scheme of the preparation method is reported in the "
                "Supporting Information."
            ),
            "origin_locator": "paper#paragraph:3#segment:0",
            "evidence_kind": "method",
            "priority_score": 8.0,
        }

        self.assertTrue(source_body_preparation_anchor_candidate(candidate))

    def test_numeric_chemical_parentheses_do_not_make_preparation_segment_reference_like(self):
        text = (
            "2.2. Preparation of PLGA-NPs PLGA-NPs were obtained using the "
            "oil-in-water (o/w) emulsion solvent extraction method.13 A scheme "
            "of the preparation method is reported in the Supporting Information "
            "(Scheme S2). Briefly, phase 1 emulsion was prepared by dissolving "
            "25 mg of PLGA, 3.2 mg of Gd-DOTAMA (C18H37)2, 1 mg of "
            "DSPE-PEG (2000) methoxy, and 1.2 mg of DPPE-PEG (2000)NHS "
            "in 0.5 mL of chloroform."
        )
        section_label = "2.2. Preparation of PLGA-NPs"
        section_kind = infer_section_kind(text, section_label)

        self.assertFalse(is_reference_like_text(text))
        self.assertEqual(section_kind, "preparation")
        self.assertFalse(should_drop_segment(text, section_kind, section_label))

    def test_reference_named_run_directory_does_not_block_source_body_locator(self):
        locator = (
            "data/results/20260511_b069802/"
            "102_stage2_s2_2_reference_aware_multi_evidence_tablebudget_campaign_no_live/"
            "semantic_stage2_objects/s2_1b_denoised_text/C4UK4CJI.txt#paragraph:2#segment:0"
        )

        self.assertTrue(
            preparation_core_locator_is_source_body(
                locator,
                "2.3. Emulsions preparation",
            )
        )

    def test_manifest_table_dir_accepts_backslash_remapped_relative_path(self):
        table_dir = Path("data/cleaned/goren_2025/tables/5GIF3D8W")
        if not table_dir.exists():
            self.skipTest("canonical DEV15 table directory is not present")

        resolved = resolve_tables_dir_for_record(
            {
                "key": "5GIF3D8W",
                "table_available": "yes",
                "table_dir": "data\\cleaned\\goren_2025\\tables\\5GIF3D8W",
            },
            Path("data/cleaned/content/5GIF3D8W.txt"),
            "5GIF3D8W",
        )

        self.assertEqual(resolved, table_dir.resolve())

    def test_selector_preserves_multiple_distinct_evidence_anchors_without_category_quota(self):
        materials_a = {
            "candidate_id": "materials-a",
            "candidate_kind": "paragraph",
            "section_kind": "materials",
            "section_label": "2.1. Materials",
            "block_type": "materials_procurement",
            "text_content": (
                "PLGA and PVA were purchased from commercial suppliers, and "
                "acetone and dichloromethane were obtained as analytical grade "
                "solvents for nanoparticle preparation."
            ),
            "origin_locator": "paper#paragraph:1",
            "evidence_kind": "materials",
            "priority_score": 9.0,
        }
        materials_b = {
            "candidate_id": "materials-b",
            "candidate_kind": "paragraph",
            "section_kind": "materials",
            "section_label": "2.1. Materials",
            "block_type": "materials_procurement",
            "text_content": (
                "The targeting ligand, albumin, and gadolinium complex were "
                "purchased from suppliers, and methanol and chloroform were "
                "obtained from Sigma for later functionalization steps."
            ),
            "origin_locator": "paper#paragraph:2",
            "evidence_kind": "materials",
            "priority_score": 8.5,
        }
        preparation_a = {
            "candidate_id": "preparation-a",
            "candidate_kind": "paragraph",
            "section_kind": "preparation",
            "section_label": "2.2. Preparation of PLGA-NPs",
            "block_type": "paragraph",
            "text_content": (
                "PLGA-NPs were obtained using the oil-in-water (o/w) emulsion "
                "solvent extraction method with a PLGA nanoparticle preparation "
                "scheme reported in the supporting information."
            ),
            "origin_locator": "paper#paragraph:3",
            "evidence_kind": "method",
            "priority_score": 8.0,
        }
        preparation_b = {
            "candidate_id": "preparation-b",
            "candidate_kind": "paragraph",
            "section_kind": "preparation",
            "section_label": "2.3. Conjugation of BSA to PLGA-NPs",
            "block_type": "paragraph",
            "text_content": (
                "A solution of BSA in buffer was added to PLGA-NP solution "
                "after NP preparation using an NHS to ligand molar ratio, and "
                "the conjugation reaction was carried out under stirring."
            ),
            "origin_locator": "paper#paragraph:4",
            "evidence_kind": "method",
            "priority_score": 7.8,
        }

        result = build_evidence_priority_selection(
            segmented_candidates=[materials_a, materials_b, preparation_a, preparation_b],
            signals={},
        )
        selected_ids = {
            candidate["candidate_id"]
            for candidate in result["prompt_selected_candidates"]
        }

        self.assertTrue({"materials-a", "materials-b", "preparation-a", "preparation-b"}.issubset(selected_ids))
        suppression_reasons = {event["reason"] for event in result["suppression_events"]}
        self.assertNotIn("materials_already_selected", suppression_reasons)
        self.assertNotIn("method_budget_reached", suppression_reasons)

    def test_prompt_budget_keeps_bounded_structural_table_floor(self):
        long_method = self._method_candidate(
            "long-method",
            "Nanoparticles were prepared by solvent evaporation using PLGA, "
            "drug, organic solvent, and aqueous surfactant. " * 170,
            paragraph_index=1,
        )
        long_materials = {
            "candidate_id": "long-materials",
            "candidate_kind": "paragraph",
            "section_kind": "materials",
            "section_label": "2.1 Materials",
            "block_type": "materials_procurement",
            "text_content": (
                "PLGA, drug, PVA, acetone, dichloromethane, and buffers were "
                "purchased from commercial suppliers. "
            )
            * 80,
            "origin_locator": "paper#paragraph:2",
        }
        table_candidates = []
        for idx in range(1, 6):
            table_candidates.append(
                {
                    "candidate_id": f"table-{idx}",
                    "candidate_kind": "table",
                    "evidence_kind": "table",
                    "section_label": f"Table {idx}. Formulation composition",
                    "text_content": (
                        f"Table {idx} Formulation Drug PLGA PVA particle size "
                        f"F{idx} 10 mg 100 mg 1% 150 nm"
                    )
                    * 38,
                    "origin_locator": f"tables/table_{idx}.csv",
                    "table_inclusion_class": TABLE_INCLUSION_MUST_INCLUDE,
                    "authority_score": 10.0 - idx,
                    "item": {
                        "meta": {"caption_or_title": f"Table {idx}. Formulation composition"},
                        "rows": [
                            ["Formulation", "Drug", "PLGA", "PVA", "Size"],
                            [f"F{idx}", f"{idx * 10} mg", f"{idx * 100} mg", f"{idx}%", f"{140 + idx} nm"],
                        ],
                    },
                }
            )

        direct_floor = select_prompt_table_floor_candidates(
            [long_method, long_materials, *table_candidates],
            max_tables=4,
        )
        self.assertEqual([candidate["candidate_id"] for candidate in direct_floor], ["table-1", "table-2", "table-3", "table-4"])

        result = build_evidence_priority_selection(
            segmented_candidates=[long_method, long_materials, *table_candidates],
            signals={},
        )
        selected_ids = {
            candidate["candidate_id"]
            for candidate in result["prompt_selected_candidates"]
        }

        self.assertTrue({"table-1", "table-2", "table-3", "table-4", "table-5"}.issubset(selected_ids))
        self.assertEqual(result["prompt_structural_table_floor_applied"], "yes")

    def test_formulation_method_floor_adds_plga_method_when_selected_method_is_non_plga(self):
        selected_non_plga = self._method_candidate(
            "chitosan-method",
            "The nanoparticles were composed of chitosan and TPP. A 0.5% w/v "
            "chitosan solution was prepared in PVA acetate buffer and 5 mg "
            "ranibizumab was added dropwise before TPP solution was added, "
            "emulsified, stirred, centrifuged, and washed.",
            paragraph_index=12,
        )
        selected_non_plga["priority_score"] = 8.0
        plga_method = self._method_candidate(
            "plga-method",
            "Preparation of PLGA microparticles. Chitosan-based nanoparticles "
            "were added to PLGA microparticles using the w/o/w double emulsion "
            "method. Briefly, 100 mg of PLGA was dissolved in 2.5 mL DCM and "
            "1 mL aqueous phase was added to the organic phase.",
            paragraph_index=18,
        )
        plga_method["evidence_kind"] = "supporting"
        plga_method["priority_score"] = 4.0

        events = []
        result = apply_minimal_evidence_floor(
            selected_candidates=[selected_non_plga],
            ranked_candidates=[selected_non_plga, plga_method],
            suppression_events=events,
        )
        selected_ids = {candidate["candidate_id"] for candidate in result["selected_candidates"]}

        self.assertIn("plga-method", selected_ids)
        self.assertIn("added_source_backed_plga_formulation_method", result["floor_rationale"])

    def test_preparation_statement_floor_recovers_strong_plga_preparation_context(self):
        materials = {
            "candidate_id": "materials-only",
            "candidate_kind": "text",
            "section_kind": "materials",
            "section_label": "Materials",
            "block_type": "materials_procurement",
            "text_content": "PLGA was purchased from a commercial supplier.",
            "origin_locator": "paragraph:4",
            "evidence_kind": "materials",
            "priority_score": 8.0,
        }
        statement = {
            "candidate_id": "preparation-statement",
            "candidate_kind": "text",
            "section_kind": "optimization",
            "section_label": "",
            "block_type": "paragraph",
            "text_content": (
                "In this work, porous and nonporous PLGA nanoparticles containing "
                "BSA were prepared by the w/o/w double emulsion method using "
                "sodium bicarbonate as the extractable porogen."
            ),
            "origin_locator": "paragraph:7",
            "evidence_kind": "supporting",
            "priority_score": 4.0,
        }

        events = []
        result = apply_minimal_evidence_floor(
            selected_candidates=[materials],
            ranked_candidates=[materials, statement],
            suppression_events=events,
        )
        selected_ids = {candidate["candidate_id"] for candidate in result["selected_candidates"]}

        self.assertIn("preparation-statement", selected_ids)
        self.assertEqual(result.get("floor_added_preparation_statement", "no"), "yes")

    def test_formulation_method_floor_accepts_optimization_typical_procedure(self):
        candidate = {
            "candidate_id": "typical-procedure",
            "candidate_kind": "paragraph",
            "section_kind": "optimization",
            "section_label": "",
            "block_type": "paragraph",
            "text_content": (
                "Cisplatin loaded nanocapsules were prepared at theoretical "
                "loading of 6.67 wt% relative to polymer. In a typical procedure, "
                "magnetic nanoparticles were dispersed in 2 mL aqueous solution "
                "and emulsified in DCM solution of 90 mg PLGA to obtain a W/O/W "
                "double emulsion solvent evaporation system."
            ),
            "origin_locator": "paper#paragraph:27",
            "evidence_kind": "supporting",
            "priority_score": 4.0,
        }

        self.assertTrue(formulation_method_candidate_is_floor_eligible(candidate))

    def test_formulation_method_floor_accepts_spray_dried_formulation_method(self):
        candidate = {
            "candidate_id": "spray-dried-method",
            "candidate_kind": "paragraph",
            "section_kind": "preparation",
            "section_label": "Preparation of spray dried nanoparticles",
            "block_type": "paragraph",
            "text_content": (
                "Two different nanoparticle formulations, amifostine-PLGA "
                "0.4:1.0 and 1.0:1.0, were prepared using a Buchi B191 Mini "
                "Spray Dryer with a standard nozzle."
            ),
            "origin_locator": "paper#paragraph:2",
            "evidence_kind": "method",
            "priority_score": 5.0,
        }

        self.assertTrue(formulation_method_candidate_is_floor_eligible(candidate))

    def test_formulation_method_floor_recovers_section_split_from_same_origin_paragraph(self):
        chemistry_method = {
            "candidate_id": "chemistry-method",
            "candidate_kind": "paragraph",
            "section_kind": "preparation",
            "section_label": "",
            "block_type": "paragraph",
            "text_content": (
                "4.6H2O (0.64 mmol, 10 mL) was added dropwise to a solution "
                "of cyclen in methanol. The dark blue solution was refluxed and "
                "precipitated by addition of diethyl ether."
            ),
            "origin_locator": "paper#paragraph:78#segment:0",
            "evidence_kind": "method",
            "priority_score": 12.0,
        }
        nanoparticle_method = {
            "candidate_id": "nanoparticle-method",
            "candidate_kind": "paragraph",
            "section_kind": "preparation",
            "section_label": "2.4. Nanoparticles preparation",
            "block_type": "paragraph",
            "text_content": (
                "2.4. Nanoparticles preparation. The formation of nanoparticles "
                "was achieved by adjusting the multiple emulsion technique, "
                "water-in-oil-in-water. Batches of nanoparticles were prepared "
                "using an ultrasonic probe. The first emulsion contained Pluronic "
                "F-68, metal complexes, and an organic solution of PLGA in 5 mL "
                "triacetin, and the emulsion was sonicated for 30 s."
            ),
            "origin_locator": "paper#paragraph:78#segment:0",
            "evidence_kind": "method",
            "priority_score": 8.0,
        }

        result = apply_minimal_evidence_floor(
            selected_candidates=[chemistry_method],
            ranked_candidates=[chemistry_method, nanoparticle_method],
            suppression_events=[],
        )
        selected_ids = {candidate["candidate_id"] for candidate in result["selected_candidates"]}

        self.assertIn("nanoparticle-method", selected_ids)
        self.assertEqual(result.get("floor_added_formulation_method", "no"), "yes")

    def test_preparation_statement_floor_prefers_double_emulsion_over_in_vivo_treatment(self):
        in_vivo_treatment = {
            "candidate_id": "in-vivo-treatment",
            "candidate_kind": "paragraph",
            "section_kind": "preparation",
            "section_label": "",
            "block_type": "paragraph",
            "text_content": (
                "The mice were treated five times with formulations containing "
                "drug at a dose of 5 mg/kg, followed by radiation. Tumor volumes "
                "and body weights were recorded after in vivo treatment."
            ),
            "origin_locator": "paper#paragraph:45",
            "evidence_kind": "method",
            "priority_score": 12.0,
        }
        double_emulsion = {
            "candidate_id": "double-emulsion-method",
            "candidate_kind": "paragraph",
            "section_kind": "optimization",
            "section_label": "Material/Methods",
            "block_type": "paragraph",
            "text_content": (
                "Tf-NPs-DOX-THC were prepared via the double-emulsion method. "
                "The morphologies and particle sizes of the prepared nanoparticles "
                "were examined by TEM and DLS for the drug delivery system."
            ),
            "origin_locator": "paper#paragraph:28",
            "evidence_kind": "supporting",
            "priority_score": 8.0,
        }

        self.assertFalse(formulation_method_candidate_is_floor_eligible(in_vivo_treatment))
        self.assertTrue(source_backed_preparation_statement_candidate_is_floor_eligible(double_emulsion))

    def test_formulation_result_context_floor_recovers_table_pointer_when_no_table_authority_exists(self):
        materials = {
            "candidate_id": "materials-only",
            "candidate_kind": "text",
            "section_kind": "materials",
            "section_label": "Materials",
            "block_type": "materials_procurement",
            "text_content": "PLGA and PVA were purchased from commercial suppliers and growth factors were obtained from a vendor.",
            "origin_locator": "paragraph:4",
            "evidence_kind": "materials",
            "priority_score": 8.0,
        }
        result_context = {
            "candidate_id": "formulation-table-pointer",
            "candidate_kind": "text",
            "section_kind": "table_related",
            "section_label": "2.2. Fabrication of scaffolds",
            "block_type": "paragraph",
            "text_content": "The formulation of emulsions for producing different mono- and bicomponent PLGA scaffolds was summarized in table 1 with component ratios and loading conditions.",
            "origin_locator": "paragraph:16",
            "evidence_kind": "supporting",
            "priority_score": 3.5,
        }

        result = build_evidence_priority_selection(segmented_candidates=[materials, result_context], signals={})
        prompt_ids = {candidate["candidate_id"] for candidate in result["prompt_selected_candidates"]}

        self.assertIn("formulation-table-pointer", prompt_ids)
        self.assertEqual(result.get("floor_added_result_context", "no"), "yes")
        self.assertIn("added_formulation_result_context", result["floor_rationale"])

    def test_hard_gate_accepts_microsphere_preparation_and_formulation_table_pointer(self):
        microsphere_method = (
            "Vancomycin-loaded microspheres were prepared with DEEM. First, "
            "0.1 mL of vancomycin solution was mixed in dichloromethane "
            "containing 250 mg of PLGA-mPEG and emulsified at 14,500 rpm."
        )
        self.assertTrue(matches_path3_text(microsphere_method))
        table_pointer = (
            "The formulation of emulsions for producing different mono- and "
            "bicomponent PLGA scaffolds was summarized in table 1 with "
            "component ratios and loading conditions."
        )
        self.assertTrue(matches_path2_text(table_pointer))

    def test_hard_gate_accepts_encapsulated_np_method_without_method_block_id(self):
        text = (
            "DNA with either TB10.4 or CpG was encapsulated in PLGA NPs using "
            "W/O/W double emulsion method and characterized for size and PDI."
        )
        self.assertTrue(matches_path3_text(text))

    def test_noise_gate_does_not_flag_material_references_word(self):
        self.assertFalse(
            evidence_text_has_noise(
                "Five PLGA references with different molecular weight and "
                "lactic acid ratio were assayed to select the polymer."
            )
        )
        self.assertTrue(evidence_text_has_noise("References\n1. Smith et al."))

    def test_hard_gate_accepts_single_row_must_include_formulation_table(self):
        block = {"source_type": "table_summary"}
        payload = {
            "payload_authority_status": "authority_visible",
            "data_row_count": 1,
            "table_type": "formulation_table",
            "table_inclusion_class": "must_include",
            "representation_status": "repaired_summary",
        }
        self.assertTrue(table_summary_satisfies_pre_live_evidence(block, payload))

    def test_hard_gate_rejects_single_row_optional_nonformulation_table(self):
        block = {"source_type": "table_summary"}
        payload = {
            "payload_authority_status": "authority_visible",
            "data_row_count": 1,
            "table_type": "non_formulation_table",
            "table_inclusion_class": "optional_context",
            "representation_status": "repaired_summary",
        }
        self.assertFalse(table_summary_satisfies_pre_live_evidence(block, payload))

    def test_hard_gate_rejects_pharmacokinetic_table_as_sole_path1_surface(self):
        block = {"source_type": "table_summary"}
        payload = {
            "payload_authority_status": "authority_visible",
            "data_row_count": 3,
            "table_type": "non_formulation_table",
            "table_role_hint": "results",
            "table_source_role": "pharmacokinetic_table",
            "table_inclusion_class": "optional_context",
            "representation_status": "raw_summary",
        }
        self.assertFalse(table_summary_satisfies_pre_live_evidence(block, payload))

    def test_orthogonal_factor_level_table_is_preparation_parameter_surface(self):
        source_role = classify_table_source_role(
            ["Factor", "Level -1", "Level 0", "Level 1"],
            {
                "caption_or_title": "The factors and levels of orthogonal test",
                "_signal_text": "A weight of drug B PVA concentration C ratio oil/water D stirring speed",
            },
        )
        self.assertEqual(source_role, "preparation_parameter_table")

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

    def test_protocol_inheritance_trigger_survives_method_budget(self):
        core_overview = self._method_candidate(
            "core-overview",
            "Rh-loaded PLGA NPs (NPR1; Table 1) were prepared by nanoprecipitation using an acetone-water system. The formulation procedure defines the base preparation method for the coded nanoparticles.",
            paragraph_index=10,
        )
        core_details = self._method_candidate(
            "core-details",
            "Briefly, 50 mg of PLGA and 2.5 mg Rh were dissolved in 4 mL acetone and mixed by vortexing. This mixture was added dropwise into 12 mL of 1% PVA under continuous stirring for 15 minutes. The suspension was evaporated for 2 hours, washed, centrifuged, and freeze-dried.",
            paragraph_index=11,
        )
        same_protocol_modifier = self._method_candidate(
            "same-protocol-modifier",
            "Rh-loaded PLGA-modifier NPs were prepared using the same protocol, but incorporating 3.5 mg surface modifier into the inner phase of the emulsion (NPR3). Table 1 shows the different formulations prepared.",
            paragraph_index=12,
        )
        same_procedure_drug = self._method_candidate(
            "same-procedure-drug",
            "Gat-loaded PLGA NPs, Gat-loaded PLGA-polysorbate 80 NPs, and Gat-loaded PLGA-Labrafil NPs were prepared using the same procedure but incorporating 5 mg of Gat into the inner phase. These active-loaded NPG1, NPG2, and NPG3 formulations represent the later drug-loaded family.",
            paragraph_index=13,
        )
        for candidate in (same_protocol_modifier, same_procedure_drug):
            candidate["section_kind"] = "variant_preparation"
            candidate["section_label"] = "Preparation and characterization of nanoparticles"

        result = build_evidence_priority_selection(
            segmented_candidates=[core_overview, core_details, same_protocol_modifier, same_procedure_drug],
            signals={},
        )
        prompt_ids = {candidate["candidate_id"] for candidate in result["prompt_selected_candidates"]}
        suppression_reasons = [
            (event["candidate_id"], event["reason"])
            for event in result["suppression_events"]
            if event["reason"] == "protocol_inheritance_trigger_selected"
        ]

        self.assertIn("core-details", prompt_ids)
        self.assertIn("same-protocol-modifier", prompt_ids)
        self.assertIn("same-procedure-drug", prompt_ids)
        self.assertEqual(
            suppression_reasons,
            [
                ("same-protocol-modifier", "protocol_inheritance_trigger_selected"),
                ("same-procedure-drug", "protocol_inheritance_trigger_selected"),
            ],
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
    def test_s2_4a_gate_path1_uses_positive_source_table_evidence(self):
        payload = {
            "table_id": "Table 1",
            "table_inclusion_class": TABLE_INCLUSION_OPTIONAL_CONTEXT,
            "payload_authority_status": "authority_visible",
            "table_source_role": "formulation_composition_table",
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

    def test_s2_4a_gate_path4_accepts_source_backed_plga_preparation_core(self):
        text = (
            "Preparation of CAPE loaded PLGA nanoparticles. PLGA-NPs were fabricated "
            "using an oil-water single-emulsion solvent evaporation method. PVA "
            "(3% w/w) was dissolved in water and 55 mg CAPE and 100 mg PLGA were "
            "dissolved to prepare the organic phase."
        )

        self.assertTrue(matches_path4_preparation_core_text(text))

    def test_s2_4a_gate_path4_accepts_plga_phase_recipe_when_target_is_abbreviated(self):
        text = (
            "Preparation of CCPN PLGA (100 mg) was dissolved in 2 mL acetone, "
            "and 280 mg CEP was added as the oil phase. Then 200 uL water was "
            "added to achieve an internal water phase-to-oil phase ratio of 1:9."
        )

        self.assertTrue(matches_path4_preparation_core_text(text))

    def test_s2_4a_gate_path4_accepts_co_loaded_nanocarrier_recipe(self):
        text = (
            "NaYF4 NPs and Rose Bengal were co-laded into colloidal nanocarriers "
            "obtained by an optimized double emulsion w/o/w evaporation process. "
            "The dual nanoplatform was stabilized by PLGA and 1 mL aqueous phase "
            "was emulsified for 5 min in 4 mL dichloromethane."
        )

        self.assertTrue(matches_path4_preparation_core_text(text))

    def test_s2_4a_gate_path4_accepts_splga_phase_recipe(self):
        text = (
            "Briefly, 160 mg sPLGA-COOH and 40 mg CuB were dissolved in "
            "2 mL dichloromethane as the oil phase, and GA-CS was dissolved "
            "in 20 mL 2% PVA solution as the water phase."
        )

        self.assertTrue(matches_path4_preparation_core_text(text))

    def test_s2_4a_gate_path4_accepts_polymer_phase_recipe_with_target(self):
        text = (
            "Insulin-loaded microspheres were prepared by double emulsion. "
            "The organic phase containing 300 mg polymer dissolved in 2 ml "
            "methylene chloride was mixed with aqueous PVA and stirred for 2 h."
        )

        self.assertTrue(matches_path4_preparation_core_text(text))

    def test_s2_4a_gate_accepts_structural_design_table_summary_without_payload(self):
        block = {
            "source_type": "table_summary",
            "text_content": (
                "[TABLE_SUMMARY] column_schema: Coded levels of factors || "
                "Measured responses cFB (mg/mL) || cP188 (mg/mL) || pH || "
                "Mean size (nm) || EE (%)"
            ),
        }

        self.assertTrue(table_summary_satisfies_pre_live_evidence(block, {}))

    def test_s2_4a_gate_accepts_formulation_ab_table_even_if_payload_role_is_characterization(self):
        block = {
            "source_type": "table_summary",
            "text_content": (
                "[TABLE_SUMMARY] Formulation A Formulation B Efficiency of "
                "encapsulation; Formulation A: 400 mg drug in 1 mL water and "
                "1 g PLGA."
            ),
        }
        payload = {
            "payload_authority_status": "authority_visible",
            "data_row_count": 2,
            "table_source_role": "characterization_result_table",
            "table_type": "non_formulation_table",
            "table_inclusion_class": "optional_context",
            "representation_status": "repaired_summary",
        }

        self.assertTrue(table_summary_satisfies_pre_live_evidence(block, payload))

    def test_s2_4a_gate_accepts_inline_formulation_metric_table_summary(self):
        block = {
            "source_type": "table_summary",
            "text_content": (
                "[TABLE_SUMMARY] title_or_caption: Inline formulation table - "
                "column_schema: Encapsulation efficiency || EE || Particle size || Sample"
            ),
        }

        self.assertTrue(table_summary_satisfies_pre_live_evidence(block, {}))

    def test_s2_4a_gate_accepts_scheme_preparation_table_summary(self):
        block = {
            "source_type": "table_summary",
            "text_content": (
                "[TABLE_SUMMARY] column_schema: Scheme 1 Schematic illustration "
                "for the preparation of PlgNPs and EPD of PlgNPs onto ITO electrode."
            ),
        }

        self.assertTrue(table_summary_satisfies_pre_live_evidence(block, {}))

    def test_s2_4a_gate_path4_rejects_non_plga_particle_synthesis(self):
        text = (
            "TOPO stabilized NaYF4 nanoparticles were synthesized by mixing 1 mmol "
            "of lanthanide oxides with 15 ml octadecane and 20 g TOPO before "
            "centrifugation at 11000 rpm."
        )

        self.assertFalse(matches_path4_preparation_core_text(text))

    def test_evidence_noise_tolerates_footer_token_inside_scientific_paragraph(self):
        text = (
            "DPPC/PLGA hNPs were prepared by an emulsion solvent diffusion technique "
            "with 10 mg PLGA and 1 mL methylene chloride. For personal use only."
        )

        self.assertFalse(evidence_text_has_noise(text))
        self.assertTrue(evidence_text_has_noise("For personal use only."))

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
