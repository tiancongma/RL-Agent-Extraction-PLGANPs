import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.stage2_sampling_labels.table_row_expansion_v1 import (
    TABLE_SCOPE_FIELD,
    TABLE_VARIABLE_ROLE_FIELD,
    _extract_row_assignments_from_authority,
    augment_with_packed_prefixed_sample_ids_from_source_csv,
    build_single_variable_recovery_contract,
    evaluate_simple_table_enumeration_contract,
    emit_single_variable_recovery_rows,
    extract_column_anchor_rows_from_authority,
    extract_empty_control_characterization_pair_rows_from_source_text,
    extract_rowwise_formulation_rows_from_authority,
    extract_source_backed_condition_rows_from_authority,
    extract_source_backed_prefixed_identity_rows_from_authority,
    extract_split_column_concentration_sweep_rows_from_source_csv,
    formulation_label_identity_key,
    formulation_identity_label_looks_primary,
    mark_llm_summary_rows_as_helpers,
    non_primary_direct_rows_look_measurement_only,
    parse_formulation_row_label_info,
    resolve_table_authority_payload_for_scope,
    row_identity_surface_kind,
    run_table_row_expansion,
)
from src.stage2_sampling_labels.table_structure_dictionary_v1 import (
    infer_header_structure,
    infer_table_structure_profile,
)
from src.stage2_sampling_labels.build_stage2_compatibility_projection_v1 import (
    merge_table_scope_locators,
    normalize_stage2_document_for_projection,
    table_expansion_can_replace_llm_summary_rows,
)
from src.stage2_sampling_labels.extract_semantic_stage2_objects_v2 import (
    finalize_llm_first_document,
    normalize_executable_scope_authorization_to_shrunken,
)
from src.stage2_sampling_labels.function_units.doe_row_expansion_function_unit_v1 import (
    resolve_authorized_doe_targets,
    run_doe_row_expansion_function_unit,
)
from src.stage2_sampling_labels.build_numbered_doe_row_candidates_v1 import (
    explicit_table_candidate,
    row_label_info,
)


class Stage2TableRowExpansionScopeAliasRolesTest(unittest.TestCase):
    def test_executable_scope_composition_hints_feed_non_doe_table_roles(self):
        parsed = {
            "paper_key": "COMPSCOPE",
            "formulation_scope_authorizations": [
                {
                    "source_scope_type": "table",
                    "semantic_role": "primary_formulation_universe",
                    "formulation_bearing_status": "formulation_bearing",
                    "row_universe_signal": "explicit_formulation_rows",
                    "downstream_expansion_required": "yes",
                    "expected_expansion_unit": "table_row",
                    "authority_locator": {
                        "primary_table_ref": "COMPSCOPE__table_04__pdf_table.csv",
                        "composition_column_hints": ["PLGA 50/50", "PLGA 75/25"],
                        "factor_column_hints": [],
                    },
                    "confidence": "high",
                }
            ],
        }

        shrunken = normalize_executable_scope_authorization_to_shrunken(parsed)
        self.assertEqual(["PLGA 50/50", "PLGA 75/25"], shrunken["semantic_signals"]["primary_variable_names"])

        document = finalize_llm_first_document(
            {
                "document_key": "COMPSCOPE",
                "table_scopes": shrunken["table_scopes"],
                "semantic_signals": shrunken["semantic_signals"],
                "formulation_candidates": shrunken["formulation_candidates"],
                "stage2_semantic_source_mode": "llm_first_composite",
            }
        )

        self.assertEqual(
            [
                {
                    "table_id": "COMPSCOPE__table_04__pdf_table.csv",
                    "varying_variables": ["PLGA 50/50", "PLGA 75/25"],
                    "marker_provenance": "llm_parsed",
                }
            ],
            document["table_variable_roles"],
        )

    def test_native_prefixed_formulation_labels_are_stable_row_identity(self):
        self.assertEqual(
            {"number": 1, "label": "NP1", "label_style": "prefixed_numeric"},
            row_label_info(["NP 1", "100", "76 ± 4"]),
        )
        self.assertEqual(
            {"number": 12, "label": "HbNPs12", "label_style": "prefixed_numeric"},
            parse_formulation_row_label_info("HbNPs-12"),
        )

        explicit_rows = [
            {"row_label_info": parse_formulation_row_label_info("NP 1")},
            {"row_label_info": parse_formulation_row_label_info("NP 2")},
        ]
        self.assertEqual("prefixed_numeric_first_column", row_identity_surface_kind(explicit_rows))
        allowed, reason, surface = evaluate_simple_table_enumeration_contract(
            scope={"table_type": "full_formulation"},
            boundary={"is_doe": False},
            explicit_rows=explicit_rows,
            direct_rows=[{"label": "NP 1"}, {"label": "NP 2"}],
        )
        self.assertTrue(allowed)
        self.assertEqual("", reason)
        self.assertEqual("prefixed_numeric_first_column", surface)

    def test_hbnps_sample_ids_are_primary_in_nonprimary_table_guard(self):
        self.assertTrue(formulation_identity_label_looks_primary("HbNPs-5"))
        self.assertTrue(formulation_identity_label_looks_primary("HbNPs-4 TRE3"))

        blocked, reason = non_primary_direct_rows_look_measurement_only(
            scope={"table_type": "partial_formulation"},
            role_info={},
            direct_rows=[
                {
                    "label": "HbNPs-5",
                    "assignments": [
                        {"name": "formulation_identity_label", "value": "HbNPs-5"},
                        {"name": "EE (%)", "value": "37.7"},
                    ],
                }
            ],
        )

        self.assertFalse(blocked)
        self.assertEqual("", reason)

    def test_rowwise_formulation_rows_preserve_primary_sample_id_label(self):
        rows, reason = extract_rowwise_formulation_rows_from_authority(
            authority_payload={
                "header_structure": {
                    "flattened_headers": [
                        "Sample ID",
                        "Hb (mg mL -1 )",
                        "PLGA (mg mL -1 )",
                        "Size (nm)",
                        "PDI",
                    ]
                }
            },
            row_entries=[
                {"row_index": 1, "cells": ["HbNPs-7", "50", "1", "466.7", "0.391"], "row_text": "HbNPs-7 | 50 | 1 | 466.7 | 0.391"},
                {"row_index": 2, "cells": ["HbNPs-8", "50", "3", "421.8", "0.320"], "row_text": "HbNPs-8 | 50 | 3 | 421.8 | 0.320"},
            ],
        )

        self.assertEqual("", reason)
        self.assertEqual(["HbNPs7", "HbNPs8"], [row["label"] for row in rows])
        self.assertEqual(["7", "8"], [row["label_number"] for row in rows])
        self.assertEqual("formulation_identity_label", rows[0]["assignments"][0]["name"])

    def test_packed_prefixed_sample_ids_augment_authorized_source_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "packed.csv"
            csv_path.write_text(
                "HbNPs-1 HbNPs-14 HbNPs-22 HbNPs-2 HbNPs-5,5 75 50 10 75,262.5 5 5 273.8 12.5\n",
                encoding="utf-8",
            )
            rows = augment_with_packed_prefixed_sample_ids_from_source_csv(
                authority_payload={"source_csv_path": str(csv_path)},
                direct_rows=[{"label": "HbNPs5", "assignments": [{"name": "formulation_identity_label", "value": "HbNPs-5"}]}],
            )

        self.assertEqual(["HbNPs5", "HbNPs1", "HbNPs14", "HbNPs22", "HbNPs2"], [row["label"] for row in rows])
        self.assertEqual("packed_prefixed_sample_id_source_row", rows[1]["change_context_tag"])
        self.assertEqual(formulation_label_identity_key("HbNPs-5"), formulation_label_identity_key("HbNPs5"))

    def test_source_backed_contextual_prefixed_rows_survive_missing_headers(self):
        self.assertEqual(
            {
                "number": 1,
                "label": "NS-01 (10 min homogenization)",
                "label_style": "prefixed_numeric_context",
            },
            parse_formulation_row_label_info("NS-01 (10 min homogenization)"),
        )
        rows, reason = extract_source_backed_prefixed_identity_rows_from_authority(
            authority_payload={"normalized_csv_path": "table.csv"},
            row_entries=[
                {"row_index": 1, "cells": ["NS-01 (5 min homogenization)", ""], "row_text": "NS-01 (5 min homogenization)"},
                {"row_index": 2, "cells": ["NS-01 (10 min homogenization)", "52.14"], "row_text": "NS-01 (10 min homogenization) | 52.14"},
                {"row_index": 3, "cells": ["NF-06 (15 min homogenization)", "42.18"], "row_text": "NF-06 (15 min homogenization) | 42.18"},
            ],
        )

        self.assertEqual("", reason)
        self.assertEqual(["NS-01 (10 min homogenization)", "NF-06 (15 min homogenization)"], [row["label"] for row in rows])
        self.assertEqual("source_backed_prefixed_identity_table_row", rows[0]["change_context_tag"])

    def test_source_backed_condition_rows_keep_authorized_factor_level_matrix(self):
        rows, reason = extract_source_backed_condition_rows_from_authority(
            authority_payload={"normalized_csv_path": "table.csv"},
            row_entries=[
                {"row_index": 1, "cells": ["Polymer amount", "50 mg", "111 ± 38"], "row_text": "Polymer amount | 50 mg | 111 ± 38"},
                {"row_index": 2, "cells": ["", "70 mg", "112 ± 36"], "row_text": "70 mg | 112 ± 36"},
                {"row_index": 3, "cells": ["Surfactant amount", "50 mg", "132 ± 44"], "row_text": "Surfactant amount | 50 mg | 132 ± 44"},
            ],
            scope={"table_type": "doe_table"},
            role_info={"varying_variables": ["polymer amount", "surfactant amount"]},
            boundary={"is_doe": True},
        )

        self.assertEqual("", reason)
        self.assertEqual(["Polymer amount: 50 mg", "Polymer amount: 70 mg", "Surfactant amount: 50 mg"], [row["label"] for row in rows])
        self.assertEqual("source_backed_condition_table_row", rows[0]["change_context_tag"])

    def test_partial_table_expansion_cannot_replace_llm_summary_rows(self):
        self.assertFalse(
            table_expansion_can_replace_llm_summary_rows(
                {
                    "emitted_row_count": 2,
                    "single_variable_rows_emitted": 2,
                    "simple_table_enumeration_activated": "no",
                    "simple_table_block_reason": "table_type_not_simple:partial_formulation",
                }
            )
        )
        self.assertTrue(
            table_expansion_can_replace_llm_summary_rows(
                {
                    "simple_table_enumeration_activated": "yes",
                    "simple_table_rows_emitted": 5,
                    "row_identity_surface_used": "prefixed_numeric_first_column",
                }
            )
        )

    def test_t_style_scope_reattaches_payload_by_identity_aliases(self):
        payload, reason = resolve_table_authority_payload_for_scope(
            {"table_id": "t004", "is_formulation_table": True},
            normalized_payloads=[
                {
                    "table_id": "Table 4",
                    "source_table_id": "Table 4",
                    "table_identity_aliases": ["t004", "Table 4"],
                    "normalized_csv_path": "table4.csv",
                }
            ],
        )

        self.assertEqual("", reason)
        self.assertEqual("Table 4", payload["table_id"])

    def test_explicit_t_style_scope_does_not_bind_broad_evidence_span_table(self):
        payload, reason = resolve_table_authority_payload_for_scope(
            {
                "table_id": "t001",
                "is_formulation_table": True,
                "evidence_span": "The formulation-bearing sweep is described across Tables 1-4.",
            },
            normalized_payloads=[
                {
                    "table_id": "Table 4",
                    "source_table_id": "Table 4",
                    "table_identity_aliases": ["t004", "Table 4"],
                    "authority_rank": 1,
                    "normalized_csv_path": "table4.csv",
                },
                {
                    "table_id": "Table 1",
                    "source_table_id": "Table 1",
                    "table_identity_aliases": ["t001", "Table 1"],
                    "authority_rank": 4,
                    "normalized_csv_path": "table1.csv",
                },
            ],
        )

        self.assertEqual("", reason)
        self.assertEqual("Table 1", payload["table_id"])

    def test_scope_locator_asset_wins_over_shared_sidecar_reference(self):
        payload, reason = resolve_table_authority_payload_for_scope(
            {
                "table_id": "t001",
                "table_scope_locators": {
                    "table_id": "Table 1",
                    "source_table_asset_id": "BHIKJJFF__sidecar_table_01",
                    "source_table_reference": "data/cleaned/content/stage1_table_cell_sidecars_current_v1/tables_cell_sidecar/BHIKJJFF/stage1_table_cells_v1.jsonl",
                },
            },
            normalized_payloads=[
                {
                    "table_id": "Table 4",
                    "source_table_asset_id": "BHIKJJFF__sidecar_table_04",
                    "source_table_reference": "data/cleaned/content/stage1_table_cell_sidecars_current_v1/tables_cell_sidecar/BHIKJJFF/stage1_table_cells_v1.jsonl",
                },
                {
                    "table_id": "Table 1",
                    "source_table_asset_id": "BHIKJJFF__sidecar_table_01",
                    "source_table_reference": "data/cleaned/content/stage1_table_cell_sidecars_current_v1/tables_cell_sidecar/BHIKJJFF/stage1_table_cells_v1.jsonl",
                },
            ],
        )

        self.assertEqual("", reason)
        self.assertEqual("Table 1", payload["table_id"])

    def test_doe_target_resolution_uses_t_style_payload_identity_aliases(self):
        document = {
            "document_key": "DOEALIASES",
            "stage2_semantic_source_mode": "llm_first_composite",
            "semantic_scope_declarations": [
                {
                    "scope_kind": "doe_table_row_enumeration_scope",
                    "declared_by": "llm_semantic_discovery",
                    "row_enumeration_required": "yes",
                    "authorizes_row_materialization_modes": [
                        "deterministic_row_expansion_within_llm_scope"
                    ],
                    "table_scope_refs": ["t012"],
                }
            ],
            "table_formulation_scopes": [
                {
                    "table_id": "t012",
                    "is_formulation_table": True,
                    "table_identity_aliases": ["t012", "Table 12"],
                    "table_type": "doe_run_matrix",
                }
            ],
            "boundary_markers": [{"table_id": "t012", "is_doe": True}],
        }
        payloads = [
            {
                "table_id": "Table 12",
                "source_table_id": "Table 12",
                "table_identity_aliases": ["t012", "Table 12"],
                "normalized_csv_path": "table12.csv",
            }
        ]

        def fake_candidate(csv_path, min_numbered_rows, table_id, source_type):
            if str(csv_path) == "table12.csv":
                return {"numbered_rows": [{} for _ in range(13)]}
            return None

        with patch(
            "src.stage2_sampling_labels.function_units.doe_row_expansion_function_unit_v1._load_normalized_table_payloads",
            return_value=(payloads, {"reopen_resolution_status": "resolved", "normalized_payload_used": "yes"}),
        ), patch(
            "src.stage2_sampling_labels.function_units.doe_row_expansion_function_unit_v1.explicit_table_candidate",
            side_effect=fake_candidate,
        ):
            targets, binding = resolve_authorized_doe_targets(
                document,
                document["semantic_scope_declarations"][0],
            )

        self.assertTrue(binding["binding_success"])
        self.assertEqual("t012", targets[0]["table_id"])
        self.assertEqual("table12.csv", targets[0]["table_path"])

    def test_doe_target_resolution_deduplicates_alias_refs_to_same_payload(self):
        document = {
            "document_key": "DOEALIASES2",
            "stage2_semantic_source_mode": "llm_first_composite",
            "semantic_scope_declarations": [
                {
                    "scope_kind": "doe_table_row_enumeration_scope",
                    "declared_by": "llm_semantic_discovery",
                    "row_enumeration_required": "yes",
                    "authorizes_row_materialization_modes": [
                        "deterministic_row_expansion_within_llm_scope"
                    ],
                    "table_scope_refs": ["t001", "Table 1"],
                }
            ],
            "table_formulation_scopes": [
                {
                    "table_id": "t001",
                    "is_formulation_table": True,
                    "table_identity_aliases": ["t001", "Table 1"],
                    "table_type": "doe_table",
                },
                {
                    "table_id": "Table 1",
                    "is_formulation_table": True,
                    "table_identity_aliases": ["t001", "Table 1"],
                    "table_type": "doe_table",
                },
            ],
            "boundary_markers": [
                {"table_id": "t001", "is_doe": True},
                {"table_id": "Table 1", "is_doe": True},
            ],
        }
        payloads = [
            {
                "table_id": "Table 1",
                "source_table_id": "Table 1",
                "source_table_asset_id": "DOEALIASES2__sidecar_table_01",
                "table_identity_aliases": ["t001", "Table 1"],
                "normalized_csv_path": "table1.csv",
            }
        ]

        def fake_candidate(csv_path, min_numbered_rows, table_id, source_type):
            if str(csv_path) == "table1.csv":
                return {"numbered_rows": [{} for _ in range(14)]}
            return None

        with patch(
            "src.stage2_sampling_labels.function_units.doe_row_expansion_function_unit_v1._load_normalized_table_payloads",
            return_value=(payloads, {"reopen_resolution_status": "resolved", "normalized_payload_used": "yes"}),
        ), patch(
            "src.stage2_sampling_labels.function_units.doe_row_expansion_function_unit_v1.explicit_table_candidate",
            side_effect=fake_candidate,
        ):
            targets, binding = resolve_authorized_doe_targets(
                document,
                document["semantic_scope_declarations"][0],
            )

        self.assertTrue(binding["binding_success"])
        self.assertEqual(1, len(targets))
        self.assertEqual("table1.csv", targets[0]["table_path"])

    def test_merge_table_scope_locators_ignores_stage1_table_cells_path_number(self):
        document = {
            "table_formulation_scopes": [
                {"table_id": "t001", "is_formulation_table": True},
                {"table_id": "t004", "is_formulation_table": True},
            ]
        }
        merge_table_scope_locators(
            document,
            [
                {
                    "table_id": "Table 4",
                    "source_table_asset_id": "BHIKJJFF__sidecar_table_04",
                    "source_table_reference": "data/cleaned/content/stage1_table_cell_sidecars_current_v1/tables_cell_sidecar/BHIKJJFF/stage1_table_cells_v1.jsonl",
                },
                {
                    "table_id": "Table 1",
                    "source_table_asset_id": "BHIKJJFF__sidecar_table_01",
                    "source_table_reference": "data/cleaned/content/stage1_table_cell_sidecars_current_v1/tables_cell_sidecar/BHIKJJFF/stage1_table_cells_v1.jsonl",
                },
            ],
        )

        by_id = {scope["table_id"]: scope for scope in document["table_formulation_scopes"]}
        self.assertEqual(
            "BHIKJJFF__sidecar_table_01",
            by_id["t001"]["source_table_asset_id"],
        )
        self.assertEqual(
            "BHIKJJFF__sidecar_table_04",
            by_id["t004"]["source_table_asset_id"],
        )

    def test_table_structure_profile_emits_hints_without_authorizing_rows(self):
        matrix = [
            ["Parameters Drug:Polymer ratio", "Nanoprecipitation method 1:20", "1:10", "1:6.66", "After storage at 4 °C for 3 months 1:10"],
            ["PLGA grade", "50:50", "50:50", "50:50", "50:50"],
            ["Size (nm)", "90.21 ± 2.2", "88.05 ± 2.7", "95.45 ± 2.4", "100 ± 4.2"],
            ["Encapsulation efficiency (%)", "73.4 ± 1.1", "75.2 ± 1.0", "70.6 ± 2.0", "68.0 ± 1.5"],
        ]
        header_structure = infer_header_structure(matrix)

        profile = infer_table_structure_profile(matrix, header_structure=header_structure, paper_key="PA3SPZ28")
        rows, reason = extract_column_anchor_rows_from_authority(
            authority_payload={
                "normalized_matrix": matrix,
                "header_structure": header_structure,
                "table_structure_profile": profile,
            },
            row_entries=[],
        )

        self.assertEqual("column_formulations", profile["table_orientation"])
        self.assertEqual("structural_alignment_hint", profile["profile_role"])
        self.assertFalse(profile["row_universe_authorized"])
        self.assertEqual("table_structure_profile_hint_only_no_row_authorization", reason)
        self.assertEqual([], rows)
        first_hint = profile["column_formulation_records"][0]
        self.assertEqual("column_record_hint", first_hint["record_role"])
        self.assertFalse(first_hint["row_universe_authorized"])
        self.assertIn({"row_index": "2", "name": "PLGA grade", "canonical_field": "", "value": "50:50"}, first_hint["attributes"])
        self.assertIn({"row_index": "3", "name": "Size (nm)", "canonical_field": "particle_size_nm", "value": "90.21 ± 2.2"}, first_hint["measurements"])

    def test_existing_main_formulation_scope_allows_supplemental_secondary_identity_scope(self):
        source_text = """
        Table 2 Mean diameter, polydispersity index (PI) and zeta potential (z) of
        PLGA empty and loaded nanospheres. Empty nanospheres XAN nanospheres
        3-MeOXAN nanospheres Diameter (nm) 154G6 164G8 164G9 PI 0.06G0.03
        0.06G0.03 0.06G0.01 z (mV) K36.2G5.2 K38.9G1.3 K36.0G3.0.
        a XAN nanosphere with theoretical concentration of 60 mg/mL.
        b 3-MeOXAN nanospheres with theoretical concentration of 60 mg/mL.
        """
        document = {
            "document_key": "SECONDARYID",
            "doi": "10.0000/secondary-id",
            "stage2_semantic_source_mode": "llm_first_composite",
            "semantic_universe_authority": "llm_semantic_discovery",
            "formulation_candidates": [
                {
                    "candidate_id": "NS_XAN",
                    "candidate_kind": "formulation_family",
                    "raw_formulation_label": "XAN nanospheres",
                }
            ],
            "semantic_signals": {"has_variable_sweep": True},
            "table_formulation_scopes": [
                {
                    "scope_id": "SECONDARYID__scope__table_1",
                    "table_id": "Table 1",
                    "is_formulation_table": True,
                    "table_type": "full_formulation",
                    "marker_provenance": "llm_parsed",
                }
            ],
            "table_variable_roles": [],
            "boundary_markers": [
                {"table_id": "Table 1", "is_doe": False, "marker_provenance": "llm_parsed"},
                {"table_id": "Table 2", "is_doe": False, "marker_provenance": "llm_parsed"},
            ],
            "selection_markers": [],
            "inheritance_markers": [],
        }

        with patch(
            "src.stage2_sampling_labels.table_row_expansion_v1.load_document_source_text",
            return_value=source_text,
        ), patch(
            "src.stage2_sampling_labels.table_row_expansion_v1._load_normalized_table_payloads",
            return_value=([], {"reopen_resolution_status": "missing", "normalized_payload_used": "no"}),
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
                    "formulation_role",
                    "instance_context_tags",
                    "change_context_tags",
                    "evidence_section",
                    "evidence_span_text",
                ],
                doe_summary={"doe_rows_emitted": 0},
            )

        self.assertEqual(["Empty nanospheres"], [row["raw_formulation_label"] for row in rows])
        self.assertEqual("Table 2", rows[0]["evidence_section"])
        self.assertIn("source_identity_table_recovery", json.loads(rows[0][TABLE_SCOPE_FIELD])["supplemental_scope_source"])
        activation_by_table = {row["table_id"]: row for row in summary["table_activation_rows"]}
        self.assertEqual("1", activation_by_table["Table 2"]["rows_emitted"])

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

    def test_reopened_doe_authority_table_blocks_secondary_table_expansion(self):
        """DOE gating must re-check after locator reopen changes the table id."""
        document = {
            "document_key": "DOEREOPEN",
            "doi": "10.0000/doe-reopen",
            "stage2_semantic_source_mode": "llm_first_composite",
            "semantic_universe_authority": "llm_semantic_discovery",
            "table_formulation_scopes": [
                {
                    "scope_id": "DOEREOPEN__scope__display_table",
                    "table_id": "Table X",
                    "is_formulation_table": True,
                    "table_type": "full_formulation",
                    "marker_provenance": "llm_parsed",
                    "table_scope_locators": [
                        {"table_id": "Table 1", "source_table_asset_id": "DOEREOPEN__table_01__pdf_table"}
                    ],
                }
            ],
            "table_variable_roles": [
                {"table_id": "Table X", "varying_variables": ["factor A"], "marker_provenance": "llm_parsed"}
            ],
            "boundary_markers": [
                {"table_id": "Table 1", "is_doe": True, "marker_provenance": "llm_parsed"}
            ],
            "selection_markers": [],
            "inheritance_markers": [],
        }
        authority_payload = {
            "table_id": "Table 1",
            "source_table_asset_id": "DOEREOPEN__table_01__pdf_table",
            "normalized_csv_path": "data/cleaned/example/Table_1.csv",
            "normalized_rows": [
                {"row_index": "1", "row_number": "1", "cells": ["F1", "-1"], "row_text": "F1 -1"},
                {"row_index": "2", "row_number": "2", "cells": ["F2", "1"], "row_text": "F2 1"},
            ],
        }

        with patch(
            "src.stage2_sampling_labels.table_row_expansion_v1._load_normalized_table_payloads",
            return_value=([authority_payload], {"reopen_resolution_status": "resolved", "normalized_payload_used": "yes"}),
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
                doe_summary={"doe_rows_emitted": 26},
            )

        self.assertEqual([], rows)
        activation_rows = summary["table_activation_rows"]
        self.assertEqual("Table 1", activation_rows[0]["table_id"])
        self.assertEqual("blocked_by_successful_doe_emission", activation_rows[0]["skip_reason"])
        self.assertEqual("no", activation_rows[0]["called"])

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

    def test_row_coded_llm_candidates_infer_table_scope_from_preserved_authority(self):
        payload = {
            "table_id": "Table 1",
            "source_table_id": "Table 1",
            "source_table_asset_id": "ROWCODE__table_01__pdf_table",
            "source_table_reference": "data/cleaned/goren_2025/tables/ROWCODE/ROWCODE__table_01__pdf_table.csv",
            "normalized_matrix": [
                ["Formulation", "Rhodamine (mg)", "Gatifloxacin (mg)", "Polysorbate 80 (%)", "Labrafil (mg)"],
                ["NPR1", "2.5", "-", "-", "-"],
                ["NPR2", "2.5", "-", "1", "-"],
                ["NPB1", "-", "-", "-", "-"],
                ["NPG1", "-", "5", "-", "-"],
            ],
            "header_structure": {"header_row_count": 1},
            "representation_status": "preserved_authority",
            "authority_rank": "1",
            "authority_score": "0.99",
        }
        document = {
            "document_key": "ROWCODE",
            "doi": "10.0000/rowcode",
            "stage2_semantic_source_mode": "llm_first_composite",
            "semantic_universe_authority": "llm_semantic_discovery",
            "formulation_candidates": [
                {"candidate_id": "NPR1", "candidate_kind": "single_formulation"},
                {"candidate_id": "NPG1", "candidate_kind": "single_formulation"},
            ],
            "protocol_inheritance_markers": [
                {
                    "marker_id": "proto_row_code",
                    "target_scope": {"formulation_ids": ["NPG1"], "target_group_label": "Gat-loaded PLGA NPs"},
                    "inheritance_trigger_text": "same procedure",
                    "marker_provenance": "llm_explicit",
                }
            ],
            "table_formulation_scopes": [],
            "table_variable_roles": [],
            "boundary_markers": [],
            "selection_markers": [],
            "inheritance_markers": [],
        }
        recovered_rows = [
            {"label": "NPR1", "assignments": [{"name": "Rhodamine (mg)", "value": "2.5"}], "row_text": "NPR1 2.5"},
            {"label": "NPR2", "assignments": [{"name": "Rhodamine (mg)", "value": "2.5"}, {"name": "Polysorbate 80 (%)", "value": "1"}], "row_text": "NPR2 2.5 1"},
            {"label": "NPB1", "assignments": [], "row_text": "NPB1"},
            {"label": "NPG1", "assignments": [{"name": "Gatifloxacin (mg)", "value": "5"}], "row_text": "NPG1 5"},
        ]
        with patch(
            "src.stage2_sampling_labels.table_row_expansion_v1._load_normalized_table_payloads",
            return_value=([payload], {"reopen_resolution_status": "resolved", "normalized_payload_used": "yes"}),
        ), patch(
            "src.stage2_sampling_labels.table_row_expansion_v1.extract_direct_formulation_rows_from_authority",
            return_value=(recovered_rows, ""),
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

        self.assertEqual({"NPR1", "NPR2", "NPB1", "NPG1"}, {row["raw_formulation_label"] for row in rows})
        self.assertEqual("", summary["skip_reason"])
        self.assertEqual(1, summary["table_count"])

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

    def test_authorized_loaded_family_with_descriptive_label_matches_empty_pair_loaded_token(self):
        document = {
            "document_key": "EMPTYPAIR",
            "doi": "10.0000/emptypair",
            "stage2_semantic_source_mode": "llm_first_composite",
            "semantic_universe_authority": "llm_semantic_discovery",
            "formulation_candidates": [
                {
                    "label_hint": "optimal AP-PLGA-NPs formulation from homogeneous design",
                    "candidate_kind": "single_formulation",
                }
            ],
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
    def test_shrunken_table_scopes_materialize_execution_markers(self):
        document = finalize_llm_first_document(
            {
                "document_key": "SHRUNKDOE",
                "paper_key": "SHRUNKDOE",
                "doi": "10.0000/shrunk",
                "title": "Shrunken DOE",
                "source_mode": "saved_raw_live_v2_replay_to_stage2_v2",
                "table_scopes": [
                    {
                        "table_id": "Table 2",
                        "scope_kind": "doe_table",
                        "is_formulation_bearing": True,
                        "is_doe": True,
                        "confidence": "high",
                    }
                ],
                "semantic_signals": {
                    "has_variable_sweep": True,
                    "primary_variable_names": ["drug concentration", "pH"],
                },
                "formulation_candidates": [],
                "shared_semantics": {},
            }
        )

        self.assertEqual("Table 2", document["table_formulation_scopes"][0]["table_id"])
        self.assertTrue(document["boundary_markers"][0]["is_doe"])
        self.assertEqual(["drug concentration", "pH"], document["table_variable_roles"][0]["varying_variables"])

    def test_doe_scope_factor_table_can_bind_single_companion_run_matrix(self):
        document = {
            "document_key": "COMPANIONDOE",
            "semantic_scope_declarations": [
                {
                    "scope_kind": "doe_table_row_enumeration_scope",
                    "declared_by": "llm_parsed",
                    "authorizes_row_materialization_modes": ["deterministic_row_expansion_within_llm_scope"],
                    "row_enumeration_required": "yes",
                    "table_scope_refs": ["Table 2"],
                }
            ],
            "table_formulation_scopes": [
                {
                    "table_id": "Table 2",
                    "is_formulation_table": True,
                    "table_type": "doe_table",
                    "marker_provenance": "llm_parsed",
                    "table_scope_locators": {"table_id": "Table 2", "source_table_asset_id": "factor_table"},
                }
            ],
            "boundary_markers": [{"table_id": "Table 2", "is_doe": True}],
        }
        payloads = [
            {
                "source_table_id": "Table 2",
                "source_table_asset_id": "factor_table",
                "normalized_csv_path": "factor.csv",
                "source_table_reference": "factor_source.csv",
            },
            {
                "source_table_id": "Table 3",
                "source_table_asset_id": "run_matrix",
                "normalized_csv_path": "run_matrix.csv",
                "source_table_reference": "run_matrix_source.csv",
            },
        ]

        def fake_candidate(csv_path, min_numbered_rows, table_id, source_type):
            if str(csv_path) == "run_matrix.csv":
                return {"numbered_rows": [{} for _ in range(17)]}
            return None

        with patch(
            "src.stage2_sampling_labels.function_units.doe_row_expansion_function_unit_v1._load_normalized_table_payloads",
            return_value=(payloads, {"reopen_resolution_status": "resolved", "normalized_payload_used": "yes"}),
        ), patch(
            "src.stage2_sampling_labels.function_units.doe_row_expansion_function_unit_v1.explicit_table_candidate",
            side_effect=fake_candidate,
        ):
            targets, binding = resolve_authorized_doe_targets(
                document,
                document["semantic_scope_declarations"][0],
            )

        self.assertTrue(binding["binding_success"])
        self.assertEqual("Table 3", targets[0]["table_id"])
        self.assertEqual("run_matrix.csv", targets[0]["table_path"])

    def test_doe_source_text_table_anchor_recovers_numbered_rows_without_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text_path = root / "paper.txt"
            text_path.write_text(
                "Table 1\n"
                "Physicochemical characteristics of PEG-ylated nanocarriers.\n"
                "System\nPE\nS1\nS2\nT = 0 days\nDH (nm) ± SD\nEETO (%) ± SD\n"
                + "\n".join(
                    f"{idx}\nPEG-PLGA\nCR A25\n{140 + idx} ± 5\n0.2{idx} ± 0.02\n{80 + idx} ± 4"
                    for idx in range(1, 10)
                )
                + "\nFig. 1. Next figure.\n",
                encoding="utf-8",
            )
            payload_root = root / "payloads"
            (payload_root / "DOCTEXT").mkdir(parents=True)
            (payload_root / "DOCTEXT" / "normalized_table_payloads_v1.json").write_text(
                json.dumps({"normalized_table_payloads": []}),
                encoding="utf-8",
            )
            document = {
                "document_key": "DOCTEXT",
                "doi": "10.0000/source-text-doe",
                "title": "Source text DOE table",
                "source_text_path": str(text_path),
                "authority_payload_root": str(payload_root),
                "stage2_semantic_source_mode": "llm_first_composite",
                "semantic_scope_declarations": [
                    {
                        "scope_id": "scope1",
                        "scope_kind": "doe_table_row_enumeration_scope",
                        "declared_by": "llm_parsed",
                        "authorizes_row_materialization_modes": ["deterministic_row_expansion_within_llm_scope"],
                        "row_enumeration_required": "yes",
                        "table_scope_refs": ["Table 1"],
                    }
                ],
                "table_formulation_scopes": [
                    {"table_id": "Table 1", "is_formulation_table": True, "table_type": "doe_table"}
                ],
                "boundary_markers": [{"table_id": "Table 1", "is_doe": True}],
            }

            rows, _traces, _jsonl_rows, summary = run_doe_row_expansion_function_unit(
                document=document,
                model_name="test_model",
                semantic_scope=document["semantic_scope_declarations"][0],
            )

        self.assertEqual(9, len(rows))
        self.assertEqual("source_text_table_anchor_fallback", summary["reopen_source_type"])
        self.assertEqual("DOCTEXT_DOE_Row_1", rows[0]["formulation_id"])

    def test_doe_companion_matrix_prefers_clean_text_rows_when_marker_payload_loses_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload_root = root / "payloads"
            paper_payload_dir = payload_root / "DOCCOMP"
            paper_payload_dir.mkdir(parents=True)
            factor_csv = root / "factor.csv"
            factor_csv.write_text("Factor,Low,Mid,High\nX1,10,35,60\n", encoding="utf-8")
            lossy_run_csv = root / "lossy_run.csv"
            lossy_run_csv.write_text(
                "Formulation,PLGA mg/mL,Poloxamer mg/mL,w/o phase volume ratio,Drug conc. Mg/mL,z-Average,% Drug entrapment,PDI\n"
                + "\n".join(f"{idx}.,35,2,6,1,21{idx},8{idx},0.{idx}" for idx in range(1, 9))
                + "\n",
                encoding="utf-8",
            )
            text_path = root / "paper.txt"
            text_path.write_text(
                "Table 2: Effect of independent process variables on dependent variable. "
                "Formulation PLGA mg/mL Poloxamer mg/mL w/o phase volume ratio Drug conc. Mg/mL "
                "z-Average d nm (±SD) % Drug entrapment (±SD) PDI (±SD) "
                + " ".join(
                    f"{idx}. 35 2 6 1 21{idx} ± 0.1 8{idx} ± 1.0 0.{idx} ± 0.002"
                    for idx in range(1, 10)
                ),
                encoding="utf-8",
            )
            (paper_payload_dir / "normalized_table_payloads_v1.json").write_text(
                json.dumps(
                    {
                        "normalized_table_payloads": [
                            {
                                "source_table_id": "Table 1",
                                "source_table_asset_id": "factor_table",
                                "normalized_csv_path": str(factor_csv),
                            },
                            {
                                "source_table_id": "Table 2",
                                "source_table_asset_id": "run_matrix",
                                "normalized_csv_path": str(lossy_run_csv),
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            document = {
                "document_key": "DOCCOMP",
                "doi": "10.0000/companion-doe",
                "title": "Companion DOE table",
                "source_text_path": str(text_path),
                "authority_payload_root": str(payload_root),
                "stage2_semantic_source_mode": "llm_first_composite",
                "semantic_scope_declarations": [
                    {
                        "scope_id": "scope1",
                        "scope_kind": "doe_table_row_enumeration_scope",
                        "declared_by": "llm_parsed",
                        "authorizes_row_materialization_modes": ["deterministic_row_expansion_within_llm_scope"],
                        "row_enumeration_required": "yes",
                        "table_scope_refs": ["Table 1"],
                        "table_scope_locators": [{"table_id": "Table 1", "source_table_asset_id": "factor_table"}],
                    }
                ],
                "table_formulation_scopes": [
                    {
                        "table_id": "Table 2",
                        "is_formulation_table": True,
                        "table_type": "full_formulation",
                        "table_scope_locators": {"table_id": "Table 2", "source_table_asset_id": "run_matrix"},
                    }
                ],
                "boundary_markers": [{"table_id": "Table 1", "is_doe": True}],
            }

            rows, _traces, _jsonl_rows, summary = run_doe_row_expansion_function_unit(
                document=document,
                model_name="test_model",
                semantic_scope=document["semantic_scope_declarations"][0],
            )

        self.assertEqual(9, len(rows))
        self.assertEqual("Table 2", summary["run_table_used"])
        self.assertEqual("DOCCOMP_DOE_Row_9", rows[-1]["formulation_id"])
    def test_source_excerpt_summary_sweep_reattaches_normalized_csv_when_source_reference_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing_source = Path(tmp) / "MISSING__source_excerpt_table_03__inline_table.csv"
            normalized = Path(tmp) / "L3__source_excerpt_table_03__inline_table__normalized.csv"
            normalized.write_text(
                "XAN,nanocapsules 3-MeOXAN nanocapsules\n"
                "Theoretical,concentration (μg/mL) XAN/Myritol 318% (w/v) Final concentration (μg/mL) Encapsulation efficiency (%) Theoretical concentration (μg/mL) 3-MeOXAN/Myritol 318% (w/v) Final concentration (μg/mL) Encapsulation efficiency (%)\n"
                "200,0.4 178±21 89±11 1000 2.0 887±51 89±5\n"
                "400,0.8 342±18 85±5 1200 2.4 918±9 77±1\n"
                "700,1.4 Crystals of XAN ND 1600 3.2 Crystals of 3-MeOXAN ND\n"
                "800,1.6 Crystals of XAN ND\n",
                encoding="utf-8",
            )
            rows, reason = extract_split_column_concentration_sweep_rows_from_source_csv(
                authority_payload={
                    "source_csv_path": str(missing_source),
                    "source_table_asset_id": "L3__source_excerpt_table_03__inline_table",
                    "normalized_csv_path": str(normalized),
                    "representation_status": "recovered_source_excerpt_summary",
                    "normalization_actions": ["preserve_coordinate_grid"],
                },
                document={"semantic_signals": {"has_variable_sweep": True}},
                scope={"table_id": "Table 3", "is_formulation_table": True},
            )
        self.assertEqual("", reason)
        labels = [row["label"] for row in rows]
        self.assertIn("XAN nanocapsules (Theoretical concentration 200 mg/mL)", labels)
        self.assertIn("3-MeOXAN nanocapsules (Theoretical concentration 1000 mg/mL)", labels)
        self.assertIn("XAN nanocapsules (Theoretical concentration 700 mg/mL)", labels)
        self.assertIn("3-MeOXAN nanocapsules (Theoretical concentration 1600 mg/mL)", labels)
        self.assertNotIn("XAN nanocapsules (Theoretical concentration 800 mg/mL)", labels)

    def test_semantic_authorized_companion_matrix_allows_coded_f_labels_without_keyword_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_matrix.csv"
            path.write_text(
                "Coded levels of factors,,,,Measured responses,,\n"
                "Empty Cell,cFB (mg/mL),cP188 (mg/mL),pH,Mean size (nm) ± SD,EE (%) ± SD,Zeta potential (mV) ± SD\n"
                "Factorial points,,,,,,\n"
                "F1,-1,-1,-1,240.00 ± 15.90,76.37 ± 0.46,-22.43 ± 0.40\n"
                "F2,1,-1,-1,205.30 ± 2.52,97.75 ± 0.01,-24.60 ± 0.95\n"
                "F3,-1,1,-1,236.00 ± 3.46,77.36 ± 0.05,-23.90 ± 0.30\n"
                "F4,1,1,-1,185.40 ± 4.10,97.05 ± 0.20,-24.00 ± 0.20\n"
                "F5,-1,-1,1,210.10 ± 2.00,81.10 ± 0.10,-22.10 ± 0.30\n"
                "F6,1,-1,1,190.10 ± 3.00,95.10 ± 0.30,-25.10 ± 0.40\n"
                "F7,-1,1,1,215.10 ± 2.50,79.10 ± 0.20,-23.10 ± 0.20\n"
                "F8,1,1,1,180.10 ± 1.50,96.10 ± 0.10,-24.10 ± 0.10\n",
                encoding="utf-8",
            )
            ordinary = explicit_table_candidate(
                csv_path=path,
                min_numbered_rows=8,
                table_id="Table 3",
                source_type="standalone_scan",
            )
            companion = explicit_table_candidate(
                csv_path=path,
                min_numbered_rows=8,
                table_id="Table 3",
                source_type="semantic_authorized_companion_table_target",
            )
        self.assertIsNone(ordinary)
        self.assertIsNotNone(companion)
        self.assertEqual(8, len(companion["numbered_rows"]))


if __name__ == "__main__":
    unittest.main()
