import json
import unittest
from unittest.mock import patch

from src.stage2_sampling_labels.table_row_expansion_v1 import (
    TABLE_SCOPE_FIELD,
    TABLE_VARIABLE_ROLE_FIELD,
    _extract_row_assignments_from_authority,
    build_single_variable_recovery_contract,
    emit_single_variable_recovery_rows,
    extract_empty_control_characterization_pair_rows_from_source_text,
    mark_llm_summary_rows_as_helpers,
    run_table_row_expansion,
)
from src.stage2_sampling_labels.build_stage2_compatibility_projection_v1 import (
    normalize_stage2_document_for_projection,
)


class Stage2TableRowExpansionScopeAliasRolesTest(unittest.TestCase):
    def test_reopened_authority_table_keeps_semantic_scope_role_lookup(self):
        """Role lookup must use LLM semantic scope id when payload reopen changes table_id."""
        document = {
            "document_key": "ALIASDOC",
            "doi": "10.0000/alias",
            "stage2_semantic_source_mode": "llm_first_composite",
            "semantic_universe_authority": "llm_semantic_discovery",
            "table_formulation_scopes": [
                {
                    "scope_id": "ALIASDOC__scope__table_2",
                    "table_id": "Table 2",
                    "is_formulation_table": True,
                    "table_type": "full_formulation",
                    "marker_provenance": "llm_parsed",
                    "candidate_values": ["1:1", "1:2"],
                    "table_scope_locators": [
                        {
                            "table_id": "Table 9",
                            "source_table_asset_id": "ALIASDOC__table_09__pdf_table",
                        }
                    ],
                }
            ],
            "table_variable_roles": [
                {
                    "table_id": "Table 2",
                    "varying_variables": ["drug:polymer ratio"],
                    "constant_variables": [],
                    "new_variables_introduced": [],
                    "marker_provenance": "llm_parsed",
                }
            ],
            "boundary_markers": [
                {"table_id": "Table 2", "is_doe": False, "marker_provenance": "llm_parsed"}
            ],
            "selection_markers": [],
            "inheritance_markers": [],
        }
        authority_payload = {
            "table_id": "Table 9",
            "source_table_asset_id": "ALIASDOC__table_09__pdf_table",
            "normalized_csv_path": "data/cleaned/example/Table_9.csv",
            "normalized_rows": [
                {"row_index": "1", "row_number": "1", "cells": ["ratio", "1:1"], "row_text": "ratio 1:1"},
                {"row_index": "2", "row_number": "2", "cells": ["ratio", "1:2"], "row_text": "ratio 1:2"},
            ],
        }

        with patch(
            "src.stage2_sampling_labels.table_row_expansion_v1._load_normalized_table_payloads",
            return_value=([authority_payload], {"reopen_resolution_status": "resolved", "normalized_payload_used": "yes"}),
        ), patch(
            "src.stage2_sampling_labels.table_row_expansion_v1.extract_direct_formulation_rows_from_authority",
            return_value=([], "no_direct_rows"),
        ), patch(
            "src.stage2_sampling_labels.table_row_expansion_v1.extract_rowwise_formulation_rows_from_authority",
            return_value=([], "no_rowwise_rows"),
        ), patch(
            "src.stage2_sampling_labels.table_row_expansion_v1.extract_first_column_identity_rows_from_authority",
            return_value=([], "no_first_column_rows"),
        ), patch(
            "src.stage2_sampling_labels.table_row_expansion_v1.extract_column_anchor_rows_from_authority",
            return_value=([], "no_column_rows"),
        ):
            rows, traces, jsonl_rows, summary = run_table_row_expansion(
                document=document,
                compatibility_columns=[
                    "key",
                    "doi",
                    "model",
                    "local_instance_id",
                    "formulation_id",
                    "raw_formulation_label",
                    "instance_kind",
                    "instance_kind_raw",
                    "instance_kind_inferred",
                    "instance_confidence",
                    "candidate_source",
                ],
                doe_summary={"doe_rows_emitted": 0},
            )

        self.assertEqual(2, len(rows))
        self.assertNotEqual("missing_llm_variable_roles", summary["skip_reason"])
        activation_rows = summary["table_activation_rows"]
        self.assertEqual("Table 9", activation_rows[0]["table_id"])
        self.assertEqual("1", activation_rows[0]["varying_variable_count"])
        role_info = json.loads(rows[0][TABLE_VARIABLE_ROLE_FIELD])
        self.assertEqual("Table 2", role_info["table_id"])
        self.assertEqual(["drug:polymer ratio"], role_info["varying_variables"])
        scope_info = json.loads(rows[0][TABLE_SCOPE_FIELD])
        self.assertEqual("Table 9", scope_info["table_id"])
        self.assertEqual("Table 2", scope_info["semantic_table_id"])
        self.assertEqual("Table 2", scope_info["original_scope_table_id"])

    def test_row_assignment_skips_prose_spillover_optimization_sentence(self):
        rows = [
            {
                "row_index": "9",
                "row_number": "9",
                "cells": ["optimized by modifying", "PLGA:ITZ ratios 5:1"],
                "row_text": "optimized by modifying the surfactant poloxamer 188 concentration and PLGA:ITZ ratio. Their PLGA:ITZ ratios 5:1",
            },
            {
                "row_index": "10",
                "row_number": "10",
                "cells": ["PLGA:ITZ ratio", "10:1"],
                "row_text": "PLGA:ITZ ratio 10:1",
            },
        ]
        assignments = _extract_row_assignments_from_authority(rows, ["5:1", "10:1"])
        self.assertEqual([{"row_ordinal": "10", "variable_value": "10:1", "row_text": "PLGA:ITZ ratio 10:1"}], assignments)

    def test_drug_amount_sweep_uses_observed_family_endpoints_without_range_overcount(self):
        source_text = """
        Different formulation variables like polymer amount (25, 50, 100, and 200 mg),
        stabilizer concentration (0.5, 0.75, 1.0, and 2.0% w/v), and drug amount
        (2.5, 5, 10, and 20 mg) were studied. On the basis of preliminary studies,
        amount of drug, amount of polymer, and concentration of stabilizer were optimized
        to 5 mg, 50 mg, and 1.0% w/v. Amount of Drug Maintaining a constant initial
        mass of polymers (50 mg), the mass of etoposide was varied between 2.5 and
        20 mg. It was observed that the increase in the amount of etoposide from 2.5
        to 10 mg increased the nanoparticle mean diameter from 82.7 to 92.4 nm for
        PLGA 50/50 and 221.4 to 255.7 nm for PCL. The encapsulation efficiency was
        increased for all batches when etoposide amount was increased from 2.5 to
        10 mg, beyond which there was no effect.
        """
        document = {
            "document_key": "5GIF3D8W",
            "doi": "10.1080/10717540802174662",
            "semantic_signals": {
                "has_variable_sweep": True,
                "has_sequential_optimization": True,
                "primary_variable_names": ["polymer amount", "stabilizer concentration", "drug amount"],
                "selected_condition_hints": [
                    "polymer amount varied: 25, 50, 100, 200 mg",
                    "stabilizer concentration varied: 0.5, 0.75, 1.0, 2.0% w/v",
                    "drug amount varied: 2.5, 5, 10, 20 mg",
                ],
            },
        }

        with patch("src.stage2_sampling_labels.table_row_expansion_v1.load_document_source_text", return_value=source_text):
            contract = build_single_variable_recovery_contract(document=document, require_anchor_rows=False)

        self.assertTrue(contract["detected"])
        self.assertEqual("anchorless_sequential_selected_chain", contract["source_type"])
        groups = [group for group in contract["groups"] if group["variable_name"] == "drug amount"]
        self.assertEqual(1, len(groups))
        self.assertEqual("drug amount", groups[0]["variable_name"])
        self.assertEqual(["2.5 mg", "10 mg"], groups[0]["levels"])
        self.assertEqual(["PLGA 50/50", "PCL"], groups[0]["family_contexts"])
        self.assertEqual(["drug_loaded"], groups[0]["payload_states"])
        self.assertFalse(groups[0]["emit_generic_row"])
        drug_only_contract = dict(contract)
        drug_only_contract["groups"] = groups

        rows, _jsonl_rows, _traces, emitted = emit_single_variable_recovery_rows(
            document=document,
            compatibility_columns=[
                "key",
                "doi",
                "model",
                "local_instance_id",
                "formulation_id",
                "raw_formulation_label",
                "instance_kind",
                "instance_kind_raw",
                "instance_kind_inferred",
                "instance_confidence",
                "candidate_source",
            ],
            contract=drug_only_contract,
            scope={},
            scope_id="5GIF3D8W__scope__single_variable",
            table_id="single_variable_context",
            group_hint_prefix="5GIF3D8W__single_variable",
        )

        self.assertEqual(4, emitted)
        labels = {row["raw_formulation_label"] for row in rows}
        self.assertEqual(
            {
                "PLGA 50/50 [drug amount=2.5 mg] / Drug loaded",
                "PLGA 50/50 [drug amount=10 mg] / Drug loaded",
                "PCL [drug amount=2.5 mg] / Drug loaded",
                "PCL [drug amount=10 mg] / Drug loaded",
            },
            labels,
        )
        self.assertFalse(any("20 mg" in label for label in labels))

    def test_polymer_amount_sweep_drops_underanchored_lower_bound_and_keeps_above_baseline(self):
        source_text = """
        Different formulation variables like polymer amount (25, 50, 100, and 200 mg),
        stabilizer concentration (0.5, 0.75, 1.0, and 2.0% w/v), and drug amount
        (2.5, 5, 10, and 20 mg) were studied. On the basis of preliminary studies,
        amount of drug, amount of polymer, and concentration of stabilizer were optimized
        to 5 mg, 50 mg, and 1.0% w/v. When polymer amount was increased from 25 to
        200 mg, there was increase in entrapment efficiency for batches prepared with
        PLGA-copolymers and PCL. For batches prepared with PCL, entrapment efficiency
        was increased with polymer content from 25 to 50 mg. Increasing content of
        polymer above 50 mg leads to precipitation and aggregate formation. Therefore,
        entrapment efficiency, drug content, and recovery for these preparations were
        not determined.
        """
        document = {
            "document_key": "5GIF3D8W",
            "doi": "10.1080/10717540802174662",
            "semantic_signals": {
                "has_variable_sweep": True,
                "has_sequential_optimization": True,
                "primary_variable_names": ["polymer amount", "stabilizer concentration", "drug amount"],
                "selected_condition_hints": [
                    "polymer amount varied: 25, 50, 100, 200 mg",
                    "stabilizer concentration varied: 0.5, 0.75, 1.0, 2.0% w/v",
                    "drug amount varied: 2.5, 5, 10, 20 mg",
                ],
            },
        }

        with patch("src.stage2_sampling_labels.table_row_expansion_v1.load_document_source_text", return_value=source_text):
            contract = build_single_variable_recovery_contract(document=document, require_anchor_rows=False)

        self.assertTrue(contract["detected"])
        polymer_groups = [group for group in contract["groups"] if group["variable_name"] == "polymer amount"]
        self.assertEqual(1, len(polymer_groups))
        self.assertEqual(["100 mg", "200 mg"], polymer_groups[0]["levels"])
        polymer_only_contract = dict(contract)
        polymer_only_contract["groups"] = polymer_groups
        rows, _jsonl_rows, _traces, emitted = emit_single_variable_recovery_rows(
            document=document,
            compatibility_columns=[
                "key",
                "doi",
                "model",
                "local_instance_id",
                "formulation_id",
                "raw_formulation_label",
                "instance_kind",
                "instance_kind_raw",
                "instance_kind_inferred",
                "instance_confidence",
                "candidate_source",
            ],
            contract=polymer_only_contract,
            scope={},
            scope_id="5GIF3D8W__scope__single_variable",
            table_id="single_variable_context",
            group_hint_prefix="5GIF3D8W__single_variable",
        )

        self.assertEqual(2, emitted)
        labels = {row["raw_formulation_label"] for row in rows}
        self.assertEqual({"polymer amount=100 mg", "polymer amount=200 mg"}, labels)
        self.assertFalse(any("25 mg" in label for label in labels))

    def test_optimal_axis_hint_does_not_duplicate_existing_primary_axis(self):
        source_text = """
        5 mL of PLGA (1% w/v) solution was slowly added to four aqueous solutions
        containing 2.5, 3, 4, and 10 mg/mL of nonionic surfactant (poloxamer 188)
        with stirring. After the optimal surfactant concentration had been determined,
        various amounts of ITZ were dissolved into three solutions of 1% PLGA in acetone
        to obtain PLGA:ITZ ratios of 5:1, 10:1, and 15:1, respectively.
        Table 2 Physicochemical properties of PLGA-ITZ-NS with different PLGA:ITZ initial ratios.
        Note: Ratio of 10:1 was then selected to prepare PLGA-ITZ-NS for the remaining studies.
        """
        document = {
            "document_key": "QLYKLPKT",
            "doi": "10.2147/IJN.S54040",
            "semantic_signals": {
                "has_variable_sweep": True,
                "has_sequential_optimization": True,
                "primary_variable_names": ["surfactant concentration", "PLGA:ITZ ratio"],
                "selected_condition_hints": [
                    "optimal surfactant concentration determined first",
                    "PLGA:ITZ ratios 5:1, 10:1, 15:1 tested",
                    "optimal PLGA:ITZ ratio selected for further studies",
                ],
            },
        }

        with patch("src.stage2_sampling_labels.table_row_expansion_v1.load_document_source_text", return_value=source_text):
            contract = build_single_variable_recovery_contract(document=document, require_anchor_rows=False)

        self.assertTrue(contract["detected"])
        groups = {group["variable_name"]: group["levels"] for group in contract["groups"]}
        self.assertIn("PLGA:ITZ ratio", groups)
        self.assertNotIn("optimal PLGA:ITZ ratio", groups)
        self.assertEqual(["5:1", "10:1", "15:1"], groups["PLGA:ITZ ratio"])

    def test_single_variable_recovery_does_not_duplicate_explicit_table_row_axis(self):
        """Axis-level recovery must not duplicate row-level table materialization for the same variable/value identity."""
        document = {
            "document_key": "AXISDUP",
            "doi": "10.0000/axisdup",
            "stage2_semantic_source_mode": "llm_first_composite",
            "semantic_universe_authority": "llm_semantic_discovery",
            "semantic_signals": {
                "has_variable_sweep": True,
                "has_sequential_optimization": True,
                "primary_variable_names": ["PLGA:ITZ ratio"],
                "selected_condition_hints": ["PLGA:ITZ ratio optimized"],
            },
            "table_formulation_scopes": [
                {
                    "scope_id": "AXISDUP__scope__table_2",
                    "table_id": "Table 2",
                    "is_formulation_table": True,
                    "table_type": "full_formulation",
                    "marker_provenance": "llm_parsed",
                    "candidate_values": ["5:1", "10:1"],
                }
            ],
            "table_variable_roles": [
                {
                    "table_id": "Table 2",
                    "varying_variables": ["PLGA:ITZ ratio"],
                    "constant_variables": [],
                    "new_variables_introduced": [],
                    "marker_provenance": "llm_parsed",
                }
            ],
            "boundary_markers": [
                {"table_id": "Table 2", "is_doe": False, "marker_provenance": "llm_parsed"}
            ],
            "selection_markers": [],
            "inheritance_markers": [],
        }
        authority_payload = {
            "table_id": "Table 2",
            "source_table_asset_id": "AXISDUP__table_02__pdf_table",
            "normalized_rows": [
                {"row_index": "1", "row_number": "1", "cells": ["PLGA:ITZ (w/w)", "5:1"], "row_text": "5:1 175±5 0.41±0.03 -17±1 61±4"},
                {"row_index": "2", "row_number": "2", "cells": ["PLGA:ITZ (w/w)", "10:1"], "row_text": "10:1 178±6 0.19±0.03 -20±1 72±1"},
                {"row_index": "3", "row_number": "3", "cells": ["PLGA:ITZ (w/w)", "15:1"], "row_text": "15:1 191±2 0.14±0.02 -30±1 73±1"},
            ],
        }
        duplicate_axis_contract = {
            "detected": True,
            "source_type": "anchorless_sequential_selected_chain",
            "groups": [
                {
                    "variable_name": "PLGA:ITZ ratio",
                    "levels": ["5:1", "10:1"],
                    "baseline_value": "10:1",
                    "family_contexts": [],
                    "payload_states": [],
                    "emit_generic_row": True,
                }
            ],
            "baseline_assignments": {},
            "held_constant_context_source": "selected_condition_hints_or_stagewise_text",
            "evidence_span": "PLGA:ITZ ratios 5:1, 10:1, and 15:1 were measured in Table 2.",
            "source_text": "PLGA:ITZ ratios 5:1, 10:1, and 15:1 were measured in Table 2.",
        }

        with patch(
            "src.stage2_sampling_labels.table_row_expansion_v1._load_normalized_table_payloads",
            return_value=([authority_payload], {"reopen_resolution_status": "resolved", "normalized_payload_used": "yes"}),
        ), patch(
            "src.stage2_sampling_labels.table_row_expansion_v1.extract_direct_formulation_rows_from_authority",
            return_value=([], "no_direct_rows"),
        ), patch(
            "src.stage2_sampling_labels.table_row_expansion_v1.extract_rowwise_formulation_rows_from_authority",
            return_value=([], "no_rowwise_rows"),
        ), patch(
            "src.stage2_sampling_labels.table_row_expansion_v1.extract_first_column_identity_rows_from_authority",
            return_value=([], "no_first_column_rows"),
        ), patch(
            "src.stage2_sampling_labels.table_row_expansion_v1.extract_column_anchor_rows_from_authority",
            return_value=([], "no_column_rows"),
        ), patch(
            "src.stage2_sampling_labels.table_row_expansion_v1.build_single_variable_recovery_contract",
            return_value=duplicate_axis_contract,
        ):
            rows, _traces, _jsonl_rows, summary = run_table_row_expansion(
                document=document,
                compatibility_columns=[
                    "key",
                    "doi",
                    "model",
                    "local_instance_id",
                    "formulation_id",
                    "raw_formulation_label",
                    "instance_kind",
                    "instance_kind_raw",
                    "instance_kind_inferred",
                    "instance_confidence",
                    "candidate_source",
                ],
                doe_summary={"doe_rows_emitted": 0},
            )

        labels = [row["raw_formulation_label"] for row in rows]
        self.assertEqual(2, len(rows))
        self.assertEqual(2, summary["explicit_table_rows_emitted"])
        self.assertEqual(0, summary["single_variable_rows_emitted"])
        self.assertTrue(all(label.startswith("Table 2 row") for label in labels))
        self.assertFalse(any(label.startswith("PLGA:ITZ ratio=") for label in labels))

    def test_result_bearing_sequential_child_table_is_authorized_for_projection(self):
        """Sequential child tables with row-local measurements must not be dropped before row expansion."""
        document = {
            "document_key": "BB3JUVW7",
            "doi": "10.1016/j.ijpharm.2021.120820",
            "source_raw_response_schema": "minimal_contract_v1",
            "semantic_signals": {},
            "formulation_candidates": [],
            "table_scopes": [
                {
                    "table_id": "Table 2",
                    "scope_kind": "sequential_child",
                    "is_formulation_bearing": False,
                    "is_doe": False,
                    "parent_table_hint": "Table 1",
                    "confidence": "high",
                    "table_scope_locators": {
                        "table_id": "Table 2",
                        "source_table_asset_id": "BB3JUVW7__table_02__html_table",
                        "source_table_reference": "data/cleaned/goren_2025/tables/BB3JUVW7/BB3JUVW7__table_02__html_table.csv",
                    },
                    "source_table_asset_id": "BB3JUVW7__table_02__html_table",
                    "source_table_reference": "data/cleaned/goren_2025/tables/BB3JUVW7/BB3JUVW7__table_02__html_table.csv",
                }
            ],
            "semantic_scope_declarations": [],
        }

        normalized = normalize_stage2_document_for_projection(document)

        scopes = normalized["table_formulation_scopes"]
        self.assertEqual(1, len(scopes))
        self.assertEqual("Table 2", scopes[0]["table_id"])
        self.assertTrue(scopes[0]["is_formulation_table"])
        self.assertEqual("sequential_child", scopes[0]["table_type"])
    def test_missing_payload_llm_authorized_table_falls_back_to_source_text_compact_rows(self):
        source_text = """
        Table 1 Nanoparticle formulations developed Formulation Rhodamine (mg) Gatifloxacin (mg)
        Polysorbate 80 (%) Labrafil (mg) NPR1 2.5 – – – NPR2 2.5 – 1 – NPB1 – – – –
        NPG1 – 5 – –
        """
        document = {
            "document_key": "INLINEAUTH",
            "doi": "10.0000/inlineauth",
            "stage2_semantic_source_mode": "llm_first_composite",
            "semantic_universe_authority": "llm_semantic_discovery",
            "table_formulation_scopes": [
                {
                    "scope_id": "INLINEAUTH__table_formulation_scope__01",
                    "table_id": "Table 1",
                    "is_formulation_table": True,
                    "table_type": "full_formulation",
                    "marker_provenance": "llm_parsed",
                }
            ],
            "table_variable_roles": [],
            "boundary_markers": [{"table_id": "Table 1", "is_doe": False, "marker_provenance": "llm_parsed"}],
            "selection_markers": [],
            "inheritance_markers": [],
        }
        with patch(
            "src.stage2_sampling_labels.table_row_expansion_v1._load_normalized_table_payloads",
            return_value=([], {"reopen_resolution_status": "resolved", "normalized_payload_used": "yes"}),
        ), patch("src.stage2_sampling_labels.table_row_expansion_v1.load_document_source_text", return_value=source_text):
            rows, _traces, _jsonl_rows, summary = run_table_row_expansion(
                document=document,
                compatibility_columns=["key", "doi", "model", "local_instance_id", "formulation_id", "raw_formulation_label", "instance_kind", "instance_kind_raw", "instance_kind_inferred", "instance_confidence", "candidate_source"],
                doe_summary={"doe_rows_emitted": 0},
            )

        self.assertEqual(4, len(rows))
        self.assertEqual("", summary["skip_reason"])
        self.assertEqual({"NPR1", "NPR2", "NPB1", "NPG1"}, {row["raw_formulation_label"] for row in rows})

    def test_anchorless_selected_sweep_runs_even_when_llm_marks_tables_non_formulation(self):
        source_text = """
        5 mL of PLGA (1% w/v) solution was slowly added to four aqueous solutions
        containing 2.5, 3, 4, and 10 mg/mL of nonionic surfactant (poloxamer 188)
        with stirring. After the optimal surfactant concentration had been determined,
        various amounts of ITZ were dissolved.
        Table 2 Physicochemical properties of PLGA-ITZ-NS with different PLGA:ITZ initial ratios
        PLGA:ITZ (w/w) Particle size (nm) PDI Zeta potential (mV) EE%
        5:1 175±5 0.41±0.03 -17±1 61±4
        10:1 178±6 0.19±0.03 -20±1 72±1
        15:1 191±2 0.14±0.02 -30±1 73±1
        Note: Ratio of 10:1 was then selected to prepare PLGA-ITZ-NS for the remaining studies.
        """
        document = {
            "document_key": "ANCHORLESSSEQ",
            "doi": "10.0000/anchorlessseq",
            "stage2_semantic_source_mode": "llm_first_composite",
            "semantic_universe_authority": "llm_semantic_discovery",
            "semantic_signals": {
                "has_variable_sweep": True,
                "has_sequential_optimization": False,
                "primary_variable_names": ["PLGA:ITZ ratio", "surfactant concentration"],
                "selected_condition_hints": ["PLGA:ITZ ratios of 5:1, 10:1, 15:1", "optimal surfactant concentration determined earlier"],
            },
            "table_formulation_scopes": [
                {"scope_id": "ANCHORLESSSEQ__scope__t2", "table_id": "Table 2", "is_formulation_table": False, "table_type": "non_formulation", "marker_provenance": "llm_parsed"}
            ],
            "table_variable_roles": [],
            "boundary_markers": [],
            "selection_markers": [],
            "inheritance_markers": [],
        }
        with patch(
            "src.stage2_sampling_labels.table_row_expansion_v1._load_normalized_table_payloads",
            return_value=([], {"reopen_resolution_status": "resolved", "normalized_payload_used": "yes"}),
        ), patch("src.stage2_sampling_labels.table_row_expansion_v1.load_document_source_text", return_value=source_text):
            rows, _traces, _jsonl_rows, summary = run_table_row_expansion(
                document=document,
                compatibility_columns=["key", "doi", "model", "local_instance_id", "formulation_id", "raw_formulation_label", "instance_kind", "instance_kind_raw", "instance_kind_inferred", "instance_confidence", "candidate_source"],
                doe_summary={"doe_rows_emitted": 0},
            )

        labels = {row["raw_formulation_label"] for row in rows}
        self.assertEqual(4, len(rows))
        self.assertIn("surfactant concentration=2.5 mg/mL", labels)
        self.assertIn("surfactant concentration=10 mg/mL", labels)
        self.assertEqual(4, summary["single_variable_rows_emitted"])

    def test_downstream_cryoprotectant_measurement_table_is_not_materialized_as_formulation_rows(self):
        document = {
            "document_key": "CRYODOWN",
            "doi": "10.0000/cryodown",
            "stage2_semantic_source_mode": "llm_first_composite",
            "semantic_universe_authority": "llm_semantic_discovery",
            "semantic_signals": {
                "has_variable_sweep": False,
                "has_measurement_only_variants": True,
                "primary_variable_names": ["cryoprotectant concentration"],
            },
            "table_formulation_scopes": [
                {"scope_id": "CRYODOWN__scope__t7", "table_id": "Table 7", "is_formulation_table": True, "table_type": "full_formulation", "marker_provenance": "llm_parsed"}
            ],
            "table_variable_roles": [],
            "boundary_markers": [{"table_id": "Table 7", "is_doe": False, "marker_provenance": "llm_parsed"}],
            "selection_markers": [],
            "inheritance_markers": [],
        }
        authority_payload = {
            "table_id": "Table 7",
            "normalized_rows": [
                {"row_index": "1", "row_number": "", "cells": ["cryoprotectant", "concentration (% w/v)", "mean diameter (nm)"], "row_text": "cryoprotectant concentration mean diameter"},
                {"row_index": "2", "row_number": "", "cells": ["sucrose", "4%", "162 ± 12"], "row_text": "sucrose 4% 162 ± 12"},
                {"row_index": "3", "row_number": "", "cells": ["mannitol", "4%", "210 ± 31"], "row_text": "mannitol 4% 210 ± 31"},
                {"row_index": "4", "row_number": "", "cells": ["is the ratio of particle size after", "freeze-drying", ""], "row_text": "is the ratio of particle size after freeze-drying"},
            ],
        }
        with patch(
            "src.stage2_sampling_labels.table_row_expansion_v1._load_normalized_table_payloads",
            return_value=([authority_payload], {"reopen_resolution_status": "resolved", "normalized_payload_used": "yes"}),
        ), patch(
            "src.stage2_sampling_labels.table_row_expansion_v1.extract_direct_formulation_rows_from_authority",
            return_value=(
                [
                    {
                        "label": "sucrose 4%",
                        "row_text": "sucrose 4% 162 ± 12 after freeze-drying",
                        "assignments": [
                            {"name": "cryoprotectant", "value": "sucrose"},
                            {"name": "concentration", "value": "4%"},
                            {"name": "mean diameter", "value": "162 nm"},
                        ],
                    }
                ],
                "",
            ),
        ), patch("src.stage2_sampling_labels.table_row_expansion_v1.load_document_source_text", return_value="MF NPs were lyophilized with cryoprotectants before characterization."):
            rows, _traces, _jsonl_rows, summary = run_table_row_expansion(
                document=document,
                compatibility_columns=["key", "doi", "model", "local_instance_id", "formulation_id", "raw_formulation_label", "instance_kind", "instance_kind_raw", "instance_kind_inferred", "instance_confidence", "candidate_source"],
                doe_summary={"doe_rows_emitted": 0},
            )

        self.assertEqual([], rows)
        activation = summary["table_activation_rows"][0]
        self.assertEqual("downstream_cryoprotectant_measurement_table", activation["skip_reason"])

    def test_source_measured_empty_control_pair_recovers_only_blank_control(self):
        document = {
            "document_key": "EMPTYPAIR",
            "doi": "10.0000/emptypair",
            "stage2_semantic_source_mode": "llm_first_composite",
            "semantic_universe_authority": "llm_semantic_discovery",
            "formulation_identity_candidates": [
                {
                    "raw_formulation_label": "AP-PLGA-NPs",
                    "instance_kind": "formulation_family",
                }
            ],
            "table_formulation_scopes": [],
            "table_variable_roles": [],
            "boundary_markers": [],
            "selection_markers": [],
            "inheritance_markers": [],
        }
        source_text = (
            "The zeta potential of AP-PLGA-NPs was found to be −14.81 ± 1.39 mV, "
            "whereas that of empty NPs was −36.13 ± 3.35 mV."
        )
        with patch("src.stage2_sampling_labels.table_row_expansion_v1.load_document_source_text", return_value=source_text):
            recovered, reason = extract_empty_control_characterization_pair_rows_from_source_text(document=document)
        self.assertEqual("", reason)
        self.assertEqual(1, len(recovered))
        self.assertEqual("AP-PLGA-NPs / Empty", recovered[0]["label"])
        self.assertEqual("control", recovered[0]["instance_role"])
        assignment_map = {item["name"]: item["value"] for item in recovered[0]["assignments"]}
        self.assertEqual("−36.13 ± 3.35 mV", assignment_map["zeta potential"])

    def test_anchorless_empty_control_pair_materializes_without_loaded_duplicate(self):
        document = {
            "document_key": "EMPTYPAIR",
            "doi": "10.0000/emptypair",
            "model_name": "unit",
            "stage2_semantic_source_mode": "llm_first_composite",
            "semantic_universe_authority": "llm_semantic_discovery",
            "formulation_identity_candidates": [
                {
                    "raw_formulation_label": "AP-PLGA-NPs",
                    "instance_kind": "formulation_family",
                }
            ],
            "table_formulation_scopes": [],
            "table_variable_roles": [],
            "boundary_markers": [],
            "selection_markers": [],
            "inheritance_markers": [],
        }
        source_text = (
            "The zeta potential of AP-PLGA-NPs was found to be −14.81 ± 1.39 mV, "
            "whereas that of empty NPs was −36.13 ± 3.35 mV."
        )
        with patch(
            "src.stage2_sampling_labels.table_row_expansion_v1._load_normalized_table_payloads",
            return_value=([], {"reopen_resolution_status": "", "normalized_payload_used": "no"}),
        ), patch("src.stage2_sampling_labels.table_row_expansion_v1.load_document_source_text", return_value=source_text):
            rows, _traces, _jsonl_rows, summary = run_table_row_expansion(
                document=document,
                compatibility_columns=[
                    "key",
                    "doi",
                    "model",
                    "local_instance_id",
                    "formulation_id",
                    "raw_formulation_label",
                    "instance_kind",
                    "instance_kind_raw",
                    "instance_kind_inferred",
                    "instance_confidence",
                    "candidate_source",
                    "polymer_identity",
                    "zeta_mV_value",
                    "zeta_mV_value_text",
                    "zeta_mV_membership_confidence",
                    "zeta_mV_evidence_region_type",
                ],
                doe_summary={"doe_rows_emitted": 0},
            )
        self.assertEqual(1, len(rows))
        self.assertEqual("AP-PLGA-NPs / Empty", rows[0]["raw_formulation_label"])
        self.assertEqual("control", rows[0]["formulation_role"])
        self.assertEqual("PLGA", rows[0]["polymer_identity"])
        self.assertEqual("−36.13 ± 3.35 mV", rows[0]["zeta_mV_value_text"])
        self.assertEqual(1, summary["emitted_row_count"])
    def test_loaded_family_summary_survives_empty_control_pair_recovery(self):
        rows = [
            {
                "key": "RHMJWZX8",
                "formulation_id": "form_1",
                "raw_formulation_label": "AP-PLGA-NPs",
                "instance_kind": "formulation_family",
                "instance_kind_raw": "formulation_family",
                "instance_kind_inferred": "formulation_family",
                "candidate_source": "saved_raw_live_v2_replay_to_stage2_v2",
                "formulation_role": "unclear",
                "instance_context_tags": json.dumps(["synthesis_core"]),
                "identity_variables_json": json.dumps(
                    [
                        {"name": "PLGA amount", "value": "optimized formulation"},
                        {"name": "AP amount", "value": "optimized formulation"},
                        {"name": "polysorbate 80 concentration", "value": "optimized formulation"},
                    ]
                ),
                "drug_name_value_text": "acetylpuerarin (AP)",
            },
            {
                "key": "RHMJWZX8",
                "formulation_id": "RHMJWZX8__source_text_characterization_pair__ap-plga-nps_/_empty",
                "raw_formulation_label": "AP-PLGA-NPs / Empty",
                "instance_kind": "new_formulation",
                "candidate_source": "table_row_expansion_v1",
                "formulation_role": "control",
                "instance_context_tags": json.dumps(
                    ["table_row_expansion", "empty_control_characterization_pair_recovery"]
                ),
                "change_context_tags": json.dumps(["empty_control_characterization_pair_recovery"]),
            },
        ]
        jsonl_rows = [dict(row) for row in rows]
        mark_llm_summary_rows_as_helpers(rows, jsonl_rows, "RHMJWZX8__table_formulation_group__01")
        self.assertEqual("formulation_family", rows[0]["instance_kind"])
        self.assertEqual("formulation_family", jsonl_rows[0]["instance_kind"])
        self.assertEqual("new_formulation", rows[1]["instance_kind"])


if __name__ == "__main__":
    unittest.main()
