import unittest

from src.stage5_benchmark.material_value_binding_v1 import (
    build_material_alias_graph,
    evaluate_canonical_promotions,
    extract_entity_bound_values,
    infer_value_scope,
    propose_canonical_promotions,
    validate_direct_value,
)


class MaterialValueBindingTests(unittest.TestCase):
    def test_direct_mass_validator_requires_numeric_mass_unit_and_rejects_identity_or_concentration(self):
        valid = validate_direct_value("50 mg", "mass")
        self.assertEqual(valid["status"], "valid")
        self.assertEqual(valid["normalized_value"], "50")
        self.assertEqual(valid["normalized_unit"], "mg")

        identity = validate_direct_value("PLGA", "mass")
        self.assertEqual(identity["status"], "invalid")
        self.assertEqual(identity["reason"], "invalid_mass_no_numeric_value")

        concentration = validate_direct_value("2 mg/mL", "mass")
        self.assertEqual(concentration["status"], "invalid")
        self.assertEqual(concentration["reason"], "invalid_mass_concentration_only")

    def test_concentration_validator_accepts_simple_percent_forms(self):
        for expression in ["1%", "1 %", "1 % w/v"]:
            with self.subTest(expression=expression):
                result = validate_direct_value(expression, "concentration")
                self.assertEqual(result["status"], "valid")
                self.assertEqual(result["normalized_value"], "1")
                self.assertEqual(result["normalized_unit"], "%")

    def test_alias_graph_resolves_source_backed_abbreviation_roles_and_leaves_unknown_tokens_unknown(self):
        graph = build_material_alias_graph(
            "The drug curcumin (CUR) was dissolved with poly(lactic-co-glycolic acid) (PLGA). "
            "Polyvinyl alcohol (PVA) was used as stabilizer. ABC was mentioned without definition."
        )

        self.assertEqual(graph.resolve_role("CUR"), "drug")
        self.assertEqual(graph.resolve_role("curcumin"), "drug")
        self.assertEqual(graph.resolve_role("PLGA"), "polymer")
        self.assertEqual(graph.resolve_role("PVA"), "surfactant")
        self.assertEqual(graph.resolve_role("ABC"), "unknown")

    def test_entity_bound_extraction_binds_preparation_masses_and_rejects_downstream_dose_context(self):
        graph = build_material_alias_graph(
            "Curcumin (CUR) drug and PLGA polymer were prepared as nanoparticles."
        )
        text = (
            "For preparation, 100 mg of PLGA and CUR (10 mg) were dissolved in acetone. "
            "In the animal study, rats received CUR 25 mg/kg by intravenous injection."
        )

        candidates = extract_entity_bound_values(text, graph)
        bound = {(c["entity_role"], c["material_alias"], c["normalized_value"], c["normalized_unit"]) for c in candidates}

        self.assertIn(("polymer", "PLGA", "100", "mg"), bound)
        self.assertIn(("drug", "CUR", "10", "mg"), bound)
        self.assertNotIn(("drug", "CUR", "25", "mg/kg"), bound)
        self.assertTrue(all(c["source_provenance"] == "direct_text" for c in candidates))

    def test_scope_and_canonical_promotions_apply_only_to_admitted_rows_without_overwriting_row_local_values(self):
        graph = build_material_alias_graph("Paclitaxel (PTX) drug and PLGA polymer were used.")
        candidates = extract_entity_bound_values(
            "Nanoparticles were prepared from PLGA (50 mg) and PTX (5 mg) in organic phase.",
            graph,
        )
        for candidate in candidates:
            scoped = infer_value_scope(candidate, admitted_rows=[{"final_formulation_id": "row-1"}, {"final_formulation_id": "row-2"}])
            self.assertEqual(scoped["scope_type"], "method_shared")
            candidate.update(scoped)

        admitted_rows = [
            {"final_formulation_id": "row-1", "polymer_mass_mg": "60"},
            {"final_formulation_id": "row-2", "polymer_mass_mg": ""},
        ]
        proposals = propose_canonical_promotions(candidates, admitted_rows)
        proposal_keys = {(p["final_formulation_id"], p["canonical_field"], p["normalized_value"]) for p in proposals}

        self.assertNotIn(("row-1", "polymer_mass_mg", "50"), proposal_keys)
        self.assertIn(("row-2", "polymer_mass_mg", "50"), proposal_keys)
        self.assertIn(("row-1", "drug_mass_mg", "5"), proposal_keys)
        self.assertIn(("row-2", "drug_mass_mg", "5"), proposal_keys)
        self.assertTrue(all(p["promotion_status"] == "proposed_direct" for p in proposals))

    def test_promotion_review_rejects_conflicting_shared_values_instead_of_donor_filling_all_rows(self):
        candidates = [
            {
                "material_alias": "PLGA",
                "entity_role": "polymer",
                "value_type": "mass",
                "normalized_value": "50",
                "normalized_unit": "mg",
                "source_provenance": "direct_text",
                "source_span": "PLGA (50 mg) was used in one preparation condition.",
                "scope_type": "method_shared",
            },
            {
                "material_alias": "PLGA",
                "entity_role": "polymer",
                "value_type": "mass",
                "normalized_value": "100",
                "normalized_unit": "mg",
                "source_provenance": "direct_text",
                "source_span": "PLGA (100 mg) was used in another preparation condition.",
                "scope_type": "method_shared",
            },
        ]
        review = evaluate_canonical_promotions(candidates, [{"final_formulation_id": "row-1", "polymer_mass_mg": ""}])

        self.assertEqual(review["proposals"], [])
        self.assertEqual(review["rejections"][0]["rejection_reason"], "conflicting_shared_values_for_field")

    def test_scope_inference_keeps_bare_direct_text_ambiguous_without_shared_preparation_basis(self):
        candidate = {
            "material_alias": "PLGA",
            "entity_role": "polymer",
            "value_type": "mass",
            "normalized_value": "50",
            "normalized_unit": "mg",
            "source_provenance": "direct_text",
        }

        scoped = infer_value_scope(candidate, admitted_rows=[{"final_formulation_id": "row-1"}])

        self.assertEqual(scoped["scope_type"], "ambiguous")
        self.assertEqual(scoped["scope_reason"], "direct_text_without_shared_scope_basis")

    def test_typed_row_local_assignment_promotes_only_to_target_admitted_row(self):
        candidates = [
            {
                "final_formulation_id": "row-1",
                "material_alias": "CUR",
                "entity_role": "drug",
                "value_type": "mass",
                "normalized_value": "5",
                "normalized_unit": "mg",
                "source_provenance": "direct_text",
                "scope_type": "typed_row_local_assignment",
            }
        ]

        review = evaluate_canonical_promotions(
            candidates,
            [
                {"final_formulation_id": "row-1", "drug_mass_mg": ""},
                {"final_formulation_id": "row-2", "drug_mass_mg": ""},
            ],
        )

        self.assertEqual([proposal["final_formulation_id"] for proposal in review["proposals"]], ["row-1"])

    def test_row_local_candidate_without_admitted_target_is_rejected_auditably(self):
        missing_target = {
            "material_alias": "CUR",
            "entity_role": "drug",
            "value_type": "mass",
            "normalized_value": "5",
            "normalized_unit": "mg",
            "source_provenance": "direct_text",
            "scope_type": "row_local",
        }
        unmatched_target = dict(missing_target, final_formulation_id="row-x")

        missing_review = evaluate_canonical_promotions([missing_target], [{"final_formulation_id": "row-1", "drug_mass_mg": ""}])
        unmatched_review = evaluate_canonical_promotions([unmatched_target], [{"final_formulation_id": "row-1", "drug_mass_mg": ""}])

        self.assertEqual(missing_review["proposals"], [])
        self.assertEqual(missing_review["rejections"][0]["rejection_reason"], "row_local_missing_row_identifier")
        self.assertEqual(unmatched_review["proposals"], [])
        self.assertEqual(unmatched_review["rejections"][0]["rejection_reason"], "row_local_target_not_admitted")

    def test_alias_graph_uses_value_suffix_row_hints_and_sample_prep_negative_context(self):
        graph = build_material_alias_graph(
            "",
            row_hints=[{"drug_name_value": "Rifampicin", "polymer_name_value": "PLA", "emulsifier_stabilizer_name_value": "PVA"}],
        )
        self.assertEqual(graph.resolve_role("Rifampicin"), "drug")
        self.assertEqual(graph.resolve_role("PLA"), "polymer")
        self.assertEqual(graph.resolve_role("PVA"), "surfactant")

        candidates = extract_entity_bound_values(
            "Sample-prep solution contained Rifampicin (10 mg) before assay injection.",
            graph,
        )
        self.assertEqual(candidates, [])


if __name__ == "__main__":
    unittest.main()
