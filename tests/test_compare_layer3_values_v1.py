import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.stage2_sampling_labels.build_numbered_doe_row_candidates_v1 import (
    first_numbered_row_anchor,
)
from src.stage2_sampling_labels.build_stage2_compatibility_projection_v1 import (
    normalize_stage2_document_for_projection,
    project_document,
)
from src.stage2_sampling_labels.function_units.doe_row_expansion_function_unit_v1 import (
    resolve_authorized_doe_targets,
)
from src.stage2_sampling_labels.extract_semantic_stage2_objects_v2 import (
    OLLAMA_SHRUNKEN_LIVE_SCHEMA,
    build_live_prompt,
    call_live_backend,
    finalize_llm_first_document,
    ollama_live_system_prompt,
    should_use_compact_live_prompt,
)
from src.stage2_sampling_labels.build_stage2_compatibility_projection_v1 import apply_table_cell_grid_bindings_to_rows
from src.stage2_sampling_labels.table_cell_grid_v1 import (
    build_grid_cell_bindings_for_row,
    build_table_cell_grid_from_payload,
)
from src.stage2_sampling_labels.table_row_expansion_v1 import (
    augment_document_with_table_markers,
    canonical_field_for_header,
    compatibility_field_for_assignment,
    direct_rows_look_like_aggregate_variant_list,
    extract_column_anchor_rows_from_authority,
    extract_direct_formulation_rows_from_authority,
    extract_empty_control_characterization_row_from_source_text,
    extract_caption_sample_rows_from_source_text,
    extract_split_column_concentration_sweep_rows_from_source_csv,
    infer_table_scopes_from_source_sweep_tables,
    is_auxiliary_measurement_column_header,
    normalize_table_scope,
    resolve_table_authority_payload_for_scope,
)
from src.stage3_relation.build_formulation_relation_artifacts_v1 import (
    build_relation_artifacts,
    build_resolved_relation_fields_for_paper,
)
from src.stage5_benchmark.build_minimal_final_output_v1 import (
    apply_global_polymer_material_carrythrough,
    apply_global_preparation_material_carrythrough,
    apply_resolved_relation_fields,
    build_derived_mass_provenance_for_row,
    extract_unique_global_preparation_organic_phase_volume,
    extract_unique_shared_preparation_masses,
    load_resolved_relation_fields,
    should_filter_non_formulation,
)
from src.stage5_benchmark import build_minimal_final_output_v1 as final_output
from src.stage5_benchmark import build_s5_3a_llm_value_prompts_v1 as s5_3a
from src.stage5_benchmark import validate_s5_value_candidates_v1 as s5_4
from src.stage5_benchmark.compare_layer3_values_to_gt_v1 import (
    CORE_FIXED_FIELDS,
    NAMED_EXTENSIBLE_VARIABLE_FIELDS,
    PROVENANCE_ONLY_FIELDS,
    SYSTEM_FIELD_MAP,
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
    should_suppress_duplicate_concentration_unit_cell,
    validate_value_for_field,
)


class Stage2DoeGenericRepairTests(unittest.TestCase):
    def test_shifted_numbered_row_anchor_accepts_right_shifted_run_matrix(self):
        rows = [
            ["spill", "", "", "", "", "", "", "", "Sr. No.", "X1", "X2", "X3", "EE", "PS"],
            ["left narrative", "", "", "", "", "", "", "", "1", "-1", "0", "1", "85.0", "126.6"],
            ["left narrative", "", "", "", "", "", "", "", "2", "0", "1", "-1", "82.0", "150.0"],
        ]

        self.assertEqual(first_numbered_row_anchor(rows), (1, 8))

    def test_shifted_numbered_row_anchor_rejects_prose_number_without_table_tail(self):
        rows = [
            ["section", "", "", "", "", "", "", "", "1", "one value", "text only"],
            ["another", "", "", "", "", "", "", "", "2", "not a matrix", "more text"],
        ]

        self.assertIsNone(first_numbered_row_anchor(rows))

    def test_doe_target_resolution_prefers_formulation_child_over_noisy_carrier(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paper_dir = root / "TESTDOE"
            paper_dir.mkdir(parents=True)
            carrier_csv = root / "carrier.normalized.csv"
            child_csv = root / "child.normalized.csv"
            carrier_csv.write_text("Factor,Low,High\nX1,-1,1\n", encoding="utf-8")
            child_csv.write_text(
                "Sr. No.,X1,X2,X3,EE,PS\n"
                + "\n".join(f"{idx},-1,0,1,85.{idx},12{idx}.0" for idx in range(1, 9))
                + "\n",
                encoding="utf-8",
            )
            (paper_dir / "normalized_table_payloads_v1.json").write_text(
                json.dumps(
                    {
                        "normalized_table_payloads": [
                            {
                                "source_table_id": "Table 12",
                                "source_table_asset_id": "TESTDOE__table_12__html_table",
                                "source_csv_path": str(carrier_csv),
                                "normalized_csv_path": str(carrier_csv),
                            },
                            {
                                "source_table_id": "Table 13",
                                "source_table_asset_id": "TESTDOE__table_13__html_table",
                                "source_csv_path": str(child_csv),
                                "normalized_csv_path": str(child_csv),
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            document = {
                "document_key": "TESTDOE",
                "authority_payload_root": str(root),
                "boundary_markers": [{"table_id": "Table 12", "is_doe": True}],
                "table_formulation_scopes": [
                    {
                        "table_id": "Table 12",
                        "table_type": "doe",
                        "is_formulation_table": True,
                        "source_table_asset_id": "TESTDOE__table_12__html_table",
                    },
                    {
                        "table_id": "Table 13",
                        "table_type": "full_formulation",
                        "parent_table_hint": "Table 12",
                        "is_formulation_table": True,
                        "source_table_asset_id": "TESTDOE__table_13__html_table",
                    },
                ],
            }
            semantic_scope = {"table_scope_refs": ["Table 12"]}

            targets, binding = resolve_authorized_doe_targets(document, semantic_scope)

        self.assertEqual([target["table_id"] for target in targets], ["Table 13"])
        self.assertEqual(targets[0]["table_path"], str(child_csv))
        self.assertTrue(binding["binding_success"])
        self.assertIn(f"Table 13:{child_csv}", binding["resolved_execution_target"])
        self.assertNotIn(f"Table 12:{carrier_csv}", binding["resolved_execution_target"])

    def test_doe_target_resolution_falls_back_to_carrier_when_child_has_too_few_numbered_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paper_dir = root / "TESTDOE"
            paper_dir.mkdir(parents=True)
            carrier_csv = root / "carrier.normalized.csv"
            child_csv = root / "child.normalized.csv"
            carrier_csv.write_text(
                "Run,X1,X2,X3,EE,PS\n"
                + "\n".join(f"{idx},-1,0,1,85.{idx},12{idx}.0" for idx in range(1, 10))
                + "\n",
                encoding="utf-8",
            )
            child_csv.write_text(
                "Run,Size,PDI,Zeta\n1,120,0.1,-10\n2,121,0.2,-11\n3,122,0.3,-12\n",
                encoding="utf-8",
            )
            (paper_dir / "normalized_table_payloads_v1.json").write_text(
                json.dumps(
                    {
                        "normalized_table_payloads": [
                            {
                                "source_table_id": "Table 1",
                                "source_table_asset_id": "TESTDOE__table_01__html_table",
                                "source_csv_path": str(carrier_csv),
                                "normalized_csv_path": str(carrier_csv),
                            },
                            {
                                "source_table_id": "Table 6",
                                "source_table_asset_id": "TESTDOE__table_06__html_table",
                                "source_csv_path": str(child_csv),
                                "normalized_csv_path": str(child_csv),
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            document = {
                "document_key": "TESTDOE",
                "authority_payload_root": str(root),
                "boundary_markers": [{"table_id": "Table 1", "is_doe": True}],
                "table_formulation_scopes": [
                    {
                        "table_id": "Table 1",
                        "table_type": "doe_table",
                        "is_formulation_table": True,
                        "source_table_asset_id": "TESTDOE__table_01__html_table",
                    },
                    {
                        "table_id": "Table 6",
                        "table_type": "full_formulation",
                        "parent_table_hint": "Table 1",
                        "is_formulation_table": True,
                        "source_table_asset_id": "TESTDOE__table_06__html_table",
                    },
                ],
            }
            semantic_scope = {"table_scope_refs": ["Table 1"]}

            targets, binding = resolve_authorized_doe_targets(document, semantic_scope)

        self.assertEqual([target["table_id"] for target in targets], ["Table 1"])
        self.assertEqual(targets[0]["table_path"], str(carrier_csv))
        self.assertTrue(binding["binding_success"])
        self.assertIn(f"Table 1:{carrier_csv}", binding["resolved_execution_target"])
        self.assertNotIn(f"Table 6:{child_csv}", binding["resolved_execution_target"])

    def test_doe_target_resolution_falls_back_when_child_has_fewer_numbered_rows_than_carrier(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paper_dir = root / "TESTDOE"
            paper_dir.mkdir(parents=True)
            carrier_csv = root / "carrier.normalized.csv"
            child_csv = root / "child.normalized.csv"
            carrier_csv.write_text(
                "Run,X1,X2,X3,EE,PS\n"
                + "\n".join(f"{idx},-1,0,1,85.{idx},12{idx}.0" for idx in range(1, 10))
                + "\n",
                encoding="utf-8",
            )
            child_csv.write_text(
                "Run,X1,X2,X3,EE,PS\n"
                + "\n".join(f"{idx},-1,0,1,85.{idx},12{idx}.0" for idx in range(1, 9))
                + "\n",
                encoding="utf-8",
            )
            (paper_dir / "normalized_table_payloads_v1.json").write_text(
                json.dumps(
                    {
                        "normalized_table_payloads": [
                            {
                                "source_table_id": "Table 1",
                                "source_table_asset_id": "TESTDOE__table_01__html_table",
                                "source_csv_path": str(carrier_csv),
                                "normalized_csv_path": str(carrier_csv),
                            },
                            {
                                "source_table_id": "Table 6",
                                "source_table_asset_id": "TESTDOE__table_06__html_table",
                                "source_csv_path": str(child_csv),
                                "normalized_csv_path": str(child_csv),
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            document = {
                "document_key": "TESTDOE",
                "authority_payload_root": str(root),
                "boundary_markers": [{"table_id": "Table 1", "is_doe": True}],
                "table_formulation_scopes": [
                    {
                        "table_id": "Table 1",
                        "table_type": "doe_table",
                        "is_formulation_table": True,
                        "source_table_asset_id": "TESTDOE__table_01__html_table",
                    },
                    {
                        "table_id": "Table 6",
                        "table_type": "full_formulation",
                        "parent_table_hint": "Table 1",
                        "is_formulation_table": True,
                        "source_table_asset_id": "TESTDOE__table_06__html_table",
                    },
                ],
            }
            semantic_scope = {"table_scope_refs": ["Table 1"]}

            targets, binding = resolve_authorized_doe_targets(document, semantic_scope)

        self.assertEqual([target["table_id"] for target in targets], ["Table 1"])
        self.assertEqual(targets[0]["table_path"], str(carrier_csv))
        self.assertTrue(binding["binding_success"])
        self.assertIn(f"Table 1:{carrier_csv}", binding["resolved_execution_target"])
        self.assertNotIn(f"Table 6:{child_csv}", binding["resolved_execution_target"])

    def test_projection_normalization_preserves_parent_table_hint_for_child_handoff(self):
        document = {
            "document_key": "TESTDOE",
            "formulation_candidates": [],
            "table_formulation_scopes": [
                {
                    "scope_id": "TESTDOE__table_formulation_scope__01",
                    "scope_kind": "full_formulation",
                    "table_id": "Table 13",
                    "parent_table_hint": "Table 12",
                    "is_formulation_table": True,
                }
            ],
        }

        normalized = normalize_stage2_document_for_projection(document)

        scopes = normalized.get("table_formulation_scopes") or []
        self.assertEqual(scopes[0]["table_id"], "Table 13")
        self.assertEqual(scopes[0]["parent_table_hint"], "Table 12")

    def test_table_authority_locator_list_wins_over_conflicting_display_table_id(self):
        payloads = [
            {
                "table_id": "Table 2",
                "source_table_id": "Table 2",
                "source_table_asset_id": "TEST__table_02__pdf_table",
                "source_table_reference": "TEST__table_02__pdf_table.csv",
                "authority_rank": 1,
            },
            {
                "table_id": "Table 9",
                "source_table_id": "Table 9",
                "source_table_asset_id": "TEST__table_09__pdf_table",
                "source_table_reference": "TEST__table_09__pdf_table.csv",
                "authority_rank": 2,
            },
        ]
        scope = {
            "table_id": "Table 2",
            "table_scope_locators": [
                {
                    "table_id": "Table 9",
                    "source_table_asset_id": "TEST__table_09__pdf_table",
                    "source_table_reference": "TEST__table_09__pdf_table.csv",
                }
            ],
        }

        payload, reason = resolve_table_authority_payload_for_scope(scope, normalized_payloads=payloads)

        self.assertEqual(reason, "")
        self.assertEqual(payload["source_table_asset_id"], "TEST__table_09__pdf_table")

    def test_locatorless_llm_table_scope_reattaches_by_caption_alias(self):
        payloads = [
            {
                "table_id": "PAPER1__source_text_table_1",
                "source_table_id": "PAPER1__source_text_table_1",
                "source_caption_or_title": "Table 1. Composition of nanoparticle formulations",
                "source_table_reference": "data/cleaned/content/text/PAPER1.pdf.txt#Table 1",
                "row_identity_signals": {"first_column_labels": ["F1", "F2", "F3"]},
                "authority_rank": 1,
            }
        ]
        scope = {
            "table_id": "",
            "source_table_reference": "",
            "source_table_asset_id": "",
            "parent_table_hint": "Table 1 composition of nanoparticle formulations",
            "evidence_span": "The LLM authorized Table 1 as the formulation composition table but omitted the locator.",
        }

        payload, reason = resolve_table_authority_payload_for_scope(scope, normalized_payloads=payloads)

        self.assertEqual(reason, "")
        self.assertEqual(payload["source_caption_or_title"], "Table 1. Composition of nanoparticle formulations")

    def test_doe_target_resolution_uses_locator_list_numbered_payload_within_llm_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paper_dir = root / "TESTDOE"
            paper_dir.mkdir(parents=True)
            numbered_csv = root / "numbered.normalized.csv"
            numbered_csv.write_text(
                "Run,X1,X2,X3,EE,PS\n"
                + "\n".join(f"{idx},-1,0,1,85.{idx},12{idx}.0" for idx in range(1, 10))
                + "\n",
                encoding="utf-8",
            )
            (paper_dir / "normalized_table_payloads_v1.json").write_text(
                json.dumps(
                    {
                        "normalized_table_payloads": [
                            {
                                "source_table_id": "Table 13",
                                "source_table_asset_id": "TESTDOE__table_13__html_table",
                                "source_csv_path": str(numbered_csv),
                                "normalized_csv_path": str(numbered_csv),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            document = {
                "document_key": "TESTDOE",
                "authority_payload_root": str(root),
                "boundary_markers": [{"table_id": "Table 12", "is_doe": True}],
                "table_formulation_scopes": [
                    {
                        "table_id": "Table 12",
                        "table_type": "doe_table",
                        "is_formulation_table": True,
                        "table_scope_locators": [
                            {
                                "table_id": "Table 13",
                                "source_table_asset_id": "TESTDOE__table_13__html_table",
                            }
                        ],
                    }
                ],
            }
            semantic_scope = {"table_scope_refs": ["Table 12"]}

            targets, binding = resolve_authorized_doe_targets(document, semantic_scope)

        self.assertEqual([target["table_id"] for target in targets], ["Table 12"])
        self.assertEqual(targets[0]["table_path"], str(numbered_csv))
        self.assertTrue(binding["binding_success"])

    def test_projection_runs_sequential_interpreter_after_unrelated_table_expansion(self):
        table_row = {
            "key": "SEQDOC",
            "formulation_id": "SEQDOC_Table1_Row1",
            "raw_formulation_label": "Table 1 row",
            "candidate_source": "table_row_expansion_v1",
        }
        sequential_row = {
            "key": "SEQDOC",
            "formulation_id": "SEQDOC_Table2_Row1",
            "raw_formulation_label": "Table 2 row",
            "candidate_source": "sequential_optimization_interpreter_v1",
        }
        with patch(
            "src.stage2_sampling_labels.build_stage2_compatibility_projection_v1.run_doe_row_expansion_function_unit",
            return_value=([], [], [], {"enabled": False, "candidate_count": 0}),
        ), patch(
            "src.stage2_sampling_labels.build_stage2_compatibility_projection_v1.run_table_row_expansion",
            return_value=([table_row], [], [], {"function_unit": "table_row_expansion_v1"}),
        ), patch(
            "src.stage2_sampling_labels.build_stage2_compatibility_projection_v1.run_sequential_optimization_interpreter",
            return_value=([sequential_row], [], [], {"function_unit": "sequential_optimization_interpreter_v1", "called": True, "emitted_row_count": 1}),
        ) as sequential_mock:
            rows, _traces, _jsonl_rows, _recovery_summary, _guard_row = project_document(
                {"document_key": "SEQDOC", "formulation_candidates": []}
            )

        self.assertTrue(sequential_mock.called)
        self.assertEqual(
            {row["formulation_id"] for row in rows},
            {"SEQDOC_Table1_Row1", "SEQDOC_Table2_Row1"},
        )

    def test_column_anchor_expansion_drops_after_storage_measurement_column(self):
        matrix = [
            ["Parameters", "Nanoprecipitation method", "", "", "After storage at 4 °C for 3 months"],
            ["Drug:Polymer ratio", "1:20", "1:10", "1:6.66", "1:10"],
            ["Size (nm)", "90.21 ± 2.2", "88.05 ± 2.7", "95.45 ± 2.4", "100 ± 4.2"],
            ["PDI", "0.212 ± 0.07", "0.170 ± 0.05", "0.237 ± 0.09", "0.295 ± 0.08"],
            ["Encapsulation efficiency (EE, %)", "53.43 ± 2.8", "88.32 ± 3.3", "60.61 ± 3.5", ""],
        ]

        rows, reason = extract_column_anchor_rows_from_authority(
            authority_payload={"normalized_matrix": matrix},
            row_entries=[],
        )

        self.assertEqual(reason, "")
        self.assertEqual(
            [row["label"] for row in rows],
            [
                "Nanoprecipitation method / 1:20",
                "Nanoprecipitation method / 1:10",
                "Nanoprecipitation method / 1:6.66",
            ],
        )
        self.assertFalse(any("After storage" in row["label"] for row in rows))

    def test_auxiliary_measurement_column_detection_is_storage_specific(self):
        self.assertTrue(is_auxiliary_measurement_column_header(["After storage at 4 °C for 3 months", "1:10"]))
        self.assertTrue(is_auxiliary_measurement_column_header(["Stability", "1 month", "F1"]))
        self.assertFalse(is_auxiliary_measurement_column_header(["Nanoprecipitation method", "1:10"]))
        self.assertFalse(is_auxiliary_measurement_column_header(["Drug loaded", "PLGA 50:50"]))

    def test_repaired_summary_coordinate_grid_can_recover_split_column_sweep(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_csv = Path(tmpdir) / "TEST__table_05__pdf_table.csv"
            source_csv.write_text(
                "XAN nanospheres 3-MeOXAN nanospheres Theoretical,header\n"
                "50 13.0 26.1 19.0 38.1,noise\n"
                "60 20.0 33.0 24.9 41.5,noise\n"
                "70 Crystals of ND Crystals of ND,noise\n"
                "80 Crystals of ND Crystals of ND,noise\n",
                encoding="utf-8",
            )
            rows, reason = extract_split_column_concentration_sweep_rows_from_source_csv(
                authority_payload={
                    "representation_status": "repaired_summary",
                    "normalization_actions": ["preserve_coordinate_grid"],
                    "source_csv_path": str(source_csv),
                },
                document={"semantic_signals": {"has_variable_sweep": True}},
                scope={"table_id": "Table 5", "is_formulation_table": True},
            )

        self.assertEqual(reason, "")
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[0]["label"], "XAN nanospheres (Theoretical concentration 50 mg/mL)")
        self.assertEqual(rows[1]["label"], "3-MeOXAN nanospheres (Theoretical concentration 50 mg/mL)")

    def test_split_column_sweep_does_not_replicate_left_only_tail_to_right_family(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_csv = Path(tmpdir) / "TEST__table_03__pdf_table.csv"
            source_csv.write_text(
                "Table 3,\n"
                "XAN nanocapsules,3-MeOXAN nanocapsules\n"
                "200 0.4 178 89,1000 2.0 887 89\n"
                "700 1.4 Crystals of XAN ND,1600 3.2 Crystals of 3-MeOXAN ND\n"
                "800 1.6 Crystals of XAN ND,\n",
                encoding="utf-8",
            )
            rows, reason = extract_split_column_concentration_sweep_rows_from_source_csv(
                authority_payload={"source_csv_path": str(source_csv)},
                document={
                    "source_mode": "saved_raw_live_v2_replay_to_stage2_v2",
                    "semantic_signals": {"has_variable_sweep": True},
                },
                scope={"table_id": "Table 3", "is_formulation_table": True},
            )

        labels = [row["label"] for row in rows]
        self.assertEqual(reason, "")
        self.assertNotIn("XAN nanocapsules (Theoretical concentration 800 mg/mL)", labels)
        self.assertNotIn("3-MeOXAN nanocapsules (Theoretical concentration 800 mg/mL)", labels)

    def test_llm_variable_sweep_can_infer_source_table_scopes_after_family_collapse(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            table1 = Path(tmpdir) / "TEST__table_01__pdf_table.csv"
            table1.write_text(
                "Table 1,\n"
                "Encapsulation parameters of XAN and 3-MeOXAN in PLGA nanospheres,\n"
                "XAN nanospheres 3-MeOXAN nanospheres Theoretical,header\n"
                "50 13.0 26.1 19.0 38.1,noise\n"
                "60 20.0 33.0 24.9 41.5,noise\n",
                encoding="utf-8",
            )
            table2 = Path(tmpdir) / "TEST__table_02__pdf_table.csv"
            table2.write_text(
                "Table 2,\n"
                "Mean diameter of PLGA nanospheres,\n"
                "Empty nanospheres,XAN nanospheres,3-MeOXAN nanospheres\n"
                "Diameter (nm),154,164,164\n",
                encoding="utf-8",
            )
            document = {
                "document_key": "TEST",
                "source_mode": "saved_raw_live_v2_replay_to_stage2_v2",
                "stage2_semantic_source_mode": "llm_first_composite",
                "semantic_signals": {"has_variable_sweep": True},
                "formulation_candidates": [{"candidate_id": "F1"}],
                "source_table_files": [str(table1), str(table2)],
            }

            scopes = infer_table_scopes_from_source_sweep_tables(document)

        self.assertEqual([scope["table_id"] for scope in scopes], ["Table 1"])
        self.assertEqual(scopes[0]["marker_provenance"], "llm_parsed")

    def test_llm_variable_sweep_recovers_source_text_caption_tables_with_measured_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            table1 = Path(tmpdir) / "TEST__table_01__pdf_table.csv"
            table1.write_text(
                "Table 1,\n"
                "Encapsulation parameters of XAN and 3-MeOXAN in PLGA nanospheres,\n"
                "XAN nanospheres 3-MeOXAN nanospheres Theoretical,header\n"
                "50 13.0 26.1 19.0 38.1,noise\n"
                "60 20.0 33.0 24.9 41.5,noise\n",
                encoding="utf-8",
            )
            source_text = Path(tmpdir) / "TEST.txt"
            source_text.write_text(
                "Table 5 Mean diameter, polydispersity index (PI), zeta potential (z) and "
                "incorporation parameters of various nanocapsule formulations: empty nanocapsules "
                "(0.6 mL Myritol 318 and without xanthones), XAN-loaded nanocapsules "
                "(0.6 mL Myritol 318, XAN theoretical concentration of 1440 mg/mL) and "
                "3-MeOXAN-loaded nanocapsules (0.6 mL Myritol 318, 3-MeOXAN theoretical "
                "concentration of 3360 mg/mL) Sample Theoretical concentration (mg/mL) "
                "Final concentration (mg/mL) Encapsulation efficiency (%) Diameter (nm) PI z (mV) "
                "Empty nanocapsules – – – 261G17 0.48G0.06 K36.3G4.3 "
                "XAN nanocapsules 1440 1173G100 82G7 273G18 0.48G0.05 K36.4G9.3 "
                "3-MeOXAN nanocapsules 3360 2780G238 83G7 271G16 0.43G0.03 K41.8G5.4",
                encoding="utf-8",
            )
            document = {
                "document_key": "TEST",
                "source_mode": "saved_raw_live_v2_replay_to_stage2_v2",
                "stage2_semantic_source_mode": "llm_first_composite",
                "semantic_signals": {"has_variable_sweep": True},
                "formulation_candidates": [{"candidate_id": "F1"}],
                "source_table_files": [str(table1)],
                "source_text_path": str(source_text),
            }

            scopes = infer_table_scopes_from_source_sweep_tables(document)
            labels, reason = extract_caption_sample_rows_from_source_text(document=document, scope={"table_id": "Table 5"})

        self.assertIn("Table 5", [scope["table_id"] for scope in scopes])
        self.assertTrue(any(scope.get("source_text_caption_only") for scope in scopes if scope["table_id"] == "Table 5"))
        self.assertEqual(reason, "")
        self.assertEqual(
            [row["label"] for row in labels],
            [
                "empty nanocapsules (0.6 mL Myritol 318 and without xanthones)",
                "XAN-loaded nanocapsules (0.6 mL Myritol 318, XAN theoretical concentration of 1440 mg/mL)",
                "3-MeOXAN-loaded nanocapsules (0.6 mL Myritol 318, 3-MeOXAN theoretical concentration of 3360 mg/mL)",
            ],
        )

    def test_source_text_caption_scope_normalization_does_not_reattach_same_number_csv_payload(self):
        payload = {
            "table_id": "Table 5",
            "source_table_id": "Table 5",
            "source_table_asset_id": "TEST__table_05__pdf_table",
            "source_table_reference": "data/cleaned/goren_2025/tables/TEST/TEST__table_05__pdf_table.csv",
            "source_csv_path": "data/cleaned/goren_2025/tables/TEST/TEST__table_05__pdf_table.csv",
            "normalized_csv_path": "data/results/example/normalized_table_payloads/TEST__table_05__pdf_table__normalized.csv",
        }
        with patch(
            "src.stage2_sampling_labels.table_row_expansion_v1._load_normalized_table_payloads",
            return_value=([payload], ""),
        ):
            normalized = normalize_table_scope(
                {
                    "scope_id": "TEST__source_text_table_scope__05",
                    "table_id": "Table 5",
                    "source_table_reference": "source_text",
                    "source_text_caption_only": True,
                    "table_scope_locators": {"table_id": "Table 5", "source_table_reference": "source_text"},
                    "is_formulation_table": True,
                    "table_type": "full_formulation",
                    "marker_provenance": "llm_parsed",
                },
                document={"document_key": "TEST"},
            )

        self.assertTrue(normalized["source_text_caption_only"])
        self.assertEqual(normalized["source_table_reference"], "source_text")
        self.assertEqual(normalized["table_path"], "")
        self.assertEqual(normalized["source_table_asset_id"], "")

    def test_left_only_nd_crystal_tail_in_split_sweep_does_not_invent_extra_formulation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            table1 = Path(tmpdir) / "TEST__table_01__pdf_table.csv"
            table1.write_text(
                "Table 1,\n"
                "Encapsulation parameters of XAN and 3-MeOXAN in PLGA nanocapsules,\n"
                "XAN nanocapsules 3-MeOXAN nanocapsules Theoretical,header\n"
                "200 0.4 196G2 98G1,1000 2.0 870G30 87G3\n"
                "400 0.8 356G10 89G2,1200 2.4 1010G50 84G4\n"
                "700 1.4 Crystals of XAN ND,1600 3.2 Crystals of ND\n"
                "800 1.6 Crystals of XAN ND,\n"
                "Values express the mean resultsGSD values of three different batches. ND, not determined.\n",
                encoding="utf-8",
            )
            payload = {"source_csv_path": str(table1), "representation_status": "repair_insufficient"}
            document = {"semantic_signals": {"has_variable_sweep": True}}

            rows, reason = extract_split_column_concentration_sweep_rows_from_source_csv(
                authority_payload=payload,
                document=document,
                scope={"table_id": "Table 1"},
            )

        self.assertEqual(reason, "")
        self.assertEqual(
            [row["label"] for row in rows],
            [
                "XAN nanocapsules (Theoretical concentration 200 mg/mL)",
                "3-MeOXAN nanocapsules (Theoretical concentration 1000 mg/mL)",
                "XAN nanocapsules (Theoretical concentration 400 mg/mL)",
                "3-MeOXAN nanocapsules (Theoretical concentration 1200 mg/mL)",
                "XAN nanocapsules (Theoretical concentration 700 mg/mL)",
                "3-MeOXAN nanocapsules (Theoretical concentration 1600 mg/mL)",
            ],
        )

    def test_empty_loaded_characterization_table_recovers_only_measured_empty_control(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_text = Path(tmpdir) / "TEST.txt"
            source_text.write_text(
                "Table 2 Mean diameter, polydispersity index (PI) and zeta potential (z) of PLGA "
                "empty and loaded nanospheres Empty nanospheres XAN nanospheres 3-MeOXAN nanospheres Diameter (nm) "
                "154G6 164G8 164G9 PI 0.06G0.03 0.06G0.03 0.06G0.01 z (mV) "
                "K36.2G5.2 K38.9G1.3 K36.0G3.0 Values express mean results.",
                encoding="utf-8",
            )
            document = {"document_key": "TEST", "source_text_path": str(source_text)}

            rows, reason = extract_empty_control_characterization_row_from_source_text(document=document, scope={"table_id": "Table 2"})

        self.assertEqual(reason, "")
        self.assertEqual([row["label"] for row in rows], ["Empty nanospheres"])
        self.assertEqual(rows[0]["instance_role"], "control")

    def test_scope_locator_prefers_source_asset_over_duplicate_table_caption(self):
        payloads = [
            {
                "table_id": "Table 3",
                "source_csv_path": "/tmp/TEST__table_07__pdf_table.csv",
                "source_table_reference": "/tmp/TEST__table_07__pdf_table.csv",
                "source_table_asset_id": "TEST__table_07__pdf_table",
                "authority_rank": "2",
            },
            {
                "table_id": "Table 3",
                "source_csv_path": "/tmp/TEST__table_08__pdf_table.csv",
                "source_table_reference": "/tmp/TEST__table_08__pdf_table.csv",
                "source_table_asset_id": "TEST__table_08__pdf_table",
                "authority_rank": "1",
            },
        ]
        scope = {
            "table_id": "Table 3",
            "table_scope_locators": {
                "table_id": "Table 3",
                "source_table_asset_id": "TEST__table_07__pdf_table",
                "source_table_reference": "/tmp/TEST__table_07__pdf_table.csv",
            },
        }

        payload, reason = resolve_table_authority_payload_for_scope(scope, normalized_payloads=payloads)

        self.assertEqual(reason, "")
        self.assertEqual(payload["source_table_asset_id"], "TEST__table_07__pdf_table")

    def test_non_formulation_existing_table_scope_does_not_block_source_sweep_inference(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            table1 = Path(tmpdir) / "TEST__table_01__pdf_table.csv"
            table1.write_text(
                "Table 1,\n"
                "Encapsulation parameters of XAN and 3-MeOXAN in PLGA nanospheres,\n"
                "XAN nanospheres 3-MeOXAN nanospheres Theoretical,header\n"
                "50 13.0 26.1 19.0 38.1,noise\n"
                "60 20.0 33.0 24.9 41.5,noise\n",
                encoding="utf-8",
            )
            table8 = Path(tmpdir) / "TEST__table_08__pdf_table.csv"
            table8.write_text("Table 8,\nNon-formulation cytotoxicity table,\n", encoding="utf-8")
            document = {
                "document_key": "TEST",
                "source_mode": "saved_raw_live_v2_replay_to_stage2_v2",
                "stage2_semantic_source_mode": "llm_first_composite",
                "semantic_signals": {"has_variable_sweep": True},
                "source_table_files": [str(table1), str(table8)],
                "formulation_identity_candidates": [
                    {"formulation_candidate_id": "F1", "raw_formulation_label": "Nanospheres with variable amount", "instance_kind": "formulation_family"}
                ],
                "table_formulation_scopes": [
                    {
                        "table_id": "Table 8",
                        "is_formulation_table": False,
                        "table_type": "non_formulation",
                        "marker_provenance": "llm_parsed",
                    }
                ],
            }

            augmented = augment_document_with_table_markers(document)

        table_ids = [scope["table_id"] for scope in augmented["table_formulation_scopes"]]
        self.assertIn("Table 8", table_ids)
        self.assertIn("Table 1", table_ids)
        self.assertTrue(
            any(scope["table_id"] == "Table 1" and scope["is_formulation_table"] for scope in augmented["table_formulation_scopes"])
        )


class UniversalTableCellGridTests(unittest.TestCase):
    def test_build_table_cell_grid_from_payload_preserves_every_body_cell_without_semantic_roles(self):
        payload = {
            "table_id": "Table 1",
            "source_csv_path": "data/cleaned/goren_2025/tables/TEST/TEST__table_01__html_table.csv",
            "source_table_asset_id": "TEST__table_01__html_table",
            "source_caption_or_title": "Example table",
            "header_structure": {
                "header_row_count": 2,
                "header_rows": [
                    ["Composition", "Composition", "Response", "Response"],
                    ["Drug (mg)", "PLGA (mg)", "Size (nm)", "EE (%)"],
                ],
                "flattened_headers": [
                    "Composition Drug (mg)",
                    "Composition PLGA (mg)",
                    "Response Size (nm)",
                    "Response EE (%)",
                ],
            },
            "normalized_rows": [
                {"row_index": 3, "row_number": "F1", "cells": ["5", "90", "234.1 ± 0.5", "93.4"]},
                {"row_index": 4, "row_number": "F2", "cells": ["10", "80", "146.0 ± 0.6", "87.5"]},
            ],
        }

        grid = build_table_cell_grid_from_payload("TEST", payload)

        self.assertEqual(len(grid), 8)
        first = grid[0]
        self.assertEqual(first["paper_key"], "TEST")
        self.assertEqual(first["table_id"], "Table 1")
        self.assertEqual(first["row_index"], "3")
        self.assertEqual(first["column_index"], "0")
        self.assertEqual(first["raw_header_path_json"], '["Composition", "Drug (mg)"]')
        self.assertEqual(first["raw_header_text"], "Composition Drug (mg)")
        self.assertEqual(first["raw_cell_value"], "5")
        self.assertEqual(first["row_label_candidate"], "F1")
        self.assertEqual(first["cell_kind"], "body")
        self.assertEqual(first["structure_status"], "aligned")
        self.assertNotIn("canonical_field", first)
        self.assertNotIn("factor_or_measure_role", first)

    def test_build_table_cell_grid_keeps_unlabeled_cells_as_structure_not_semantic_skip(self):
        payload = {
            "table_id": "Table 2",
            "source_csv_path": "data/cleaned/goren_2025/tables/TEST/TEST__table_02__html_table.csv",
            "header_structure": {"header_row_count": 1, "header_rows": [["Run"]], "flattened_headers": ["Run"]},
            "normalized_rows": [
                {"row_index": 2, "cells": ["1", "extra value"]},
            ],
        }

        grid = build_table_cell_grid_from_payload("TEST", payload)

        self.assertEqual(len(grid), 2)
        self.assertEqual(grid[1]["raw_cell_value"], "extra value")
        self.assertEqual(grid[1]["raw_header_text"], "")
        self.assertEqual(grid[1]["structure_status"], "unlabeled_column")
        self.assertEqual(grid[1]["cell_kind"], "body")

    def test_grid_consumer_projects_unique_row_local_metric_bindings(self):
        payload = {
            "table_id": "Table 1",
            "source_csv_path": "data/cleaned/goren_2025/tables/TEST/TEST__table_01__html_table.csv",
            "header_structure": {
                "header_row_count": 1,
                "header_rows": [["Formulation", "Particle Size (nm)", "E.E. (%)"]],
                "flattened_headers": ["Formulation", "Particle Size (nm)", "E.E. (%)"],
            },
            "normalized_rows": [
                {"row_index": 2, "row_number": "F1", "cells": ["F1", "190.2 ± 18.0", "78.5 ± 1.8"]},
                {"row_index": 3, "row_number": "F2", "cells": ["F2", "201.0 ± 10.0", "80.0 ± 2.0"]},
            ],
        }
        grid = build_table_cell_grid_from_payload("TEST", payload)

        bindings, status = build_grid_cell_bindings_for_row(
            grid,
            paper_key="TEST",
            table_id="Table 1",
            row_label="F1",
        )

        self.assertEqual(status, "unique_grid_row_binding")
        by_field = {item["canonical_field"]: item for item in bindings}
        self.assertEqual(by_field["particle_size_nm"]["raw_cell_value"], "190.2 ± 18.0")
        self.assertEqual(by_field["particle_size_nm"]["raw_header"], "Particle Size (nm)")
        self.assertEqual(by_field["particle_size_nm"]["binding_rule"], "table_cell_grid_v1_row_local_header_binding")
        self.assertEqual(by_field["ee_percent"]["raw_cell_value"], "78.5 ± 1.8")
        self.assertEqual(by_field["ee_percent"]["source_row_index"], "2")

    def test_grid_consumer_skips_ambiguous_duplicate_row_labels(self):
        payload = {
            "table_id": "Table 1",
            "header_structure": {
                "header_row_count": 1,
                "header_rows": [["Formulation", "Size (nm)"]],
                "flattened_headers": ["Formulation", "Size (nm)"],
            },
            "normalized_rows": [
                {"row_index": 2, "row_number": "F1", "cells": ["F1", "190.2"]},
                {"row_index": 3, "row_number": "F1", "cells": ["F1", "201.0"]},
            ],
        }
        grid = build_table_cell_grid_from_payload("TEST", payload)

        bindings, status = build_grid_cell_bindings_for_row(
            grid,
            paper_key="TEST",
            table_id="Table 1",
            row_label="F1",
        )

        self.assertEqual(bindings, [])
        self.assertEqual(status, "ambiguous_grid_row_candidates")

    def test_grid_consumer_binds_transposed_metric_rows_to_formulation_column(self):
        payload = {
            "table_id": "Table 4",
            "header_structure": {
                "header_row_count": 1,
                "header_rows": [["", "Drug loaded 91.8 ± 2.74"]],
                "flattened_headers": ["", "Drug loaded 91.8 ± 2.74"],
            },
            "normalized_rows": [
                {"row_index": 2, "cells": ["Diameter (nm)", "91.8 ± 2.74"]},
                {"row_index": 3, "cells": ["PIa", "0.13 ± 0.01"]},
                {"row_index": 4, "cells": ["ZPb (mV)", "−21.23 ± 1.04"]},
            ],
        }
        grid = build_table_cell_grid_from_payload("TEST", payload)

        bindings, status = build_grid_cell_bindings_for_row(
            grid,
            paper_key="TEST",
            table_id="Table 4",
            row_label="plga_50/50_/_drug_loaded",
        )

        self.assertEqual(status, "unique_grid_transposed_metric_binding")
        by_field = {item["canonical_field"]: item for item in bindings}
        self.assertEqual(by_field["particle_size_nm"]["raw_cell_value"], "91.8 ± 2.74")
        self.assertEqual(by_field["pdi"]["raw_cell_value"], "0.13 ± 0.01")
        self.assertEqual(by_field["zeta_mV"]["raw_cell_value"], "−21.23 ± 1.04")
        self.assertEqual(by_field["zeta_mV"]["binding_rule"], "table_cell_grid_v1_transposed_row_metric_binding")

    def test_grid_bindings_project_into_blank_stage2_compatibility_fields_only(self):
        rows = [
            {
                "key": "TEST",
                "local_instance_id": "TEST__table_4__plga_50/50_/_drug_loaded",
                "formulation_id": "TEST__table_4__plga_50/50_/_drug_loaded",
                "raw_formulation_label": "PLGA 50/50 / Drug loaded",
                "table_id": "Table 4",
                "table_row_id": "Table 4::plga_50/50_/_drug_loaded",
                "size_nm_value": "",
                "pdi_value": "0.99",
                "zeta_mV_value": "",
            }
        ]
        jsonl_rows = [{"local_instance_id": "TEST__table_4__plga_50/50_/_drug_loaded"}]
        payload = {
            "table_id": "Table 4",
            "header_structure": {
                "header_row_count": 1,
                "header_rows": [["", "Drug loaded 91.8 ± 2.74"]],
                "flattened_headers": ["", "Drug loaded 91.8 ± 2.74"],
            },
            "normalized_rows": [
                {"row_index": 2, "cells": ["Diameter (nm)", "91.8 ± 2.74"]},
                {"row_index": 3, "cells": ["PIa", "0.13 ± 0.01"]},
                {"row_index": 4, "cells": ["ZPb (mV)", "−21.23 ± 1.04"]},
            ],
        }
        grid = build_table_cell_grid_from_payload("TEST", payload)

        stats = apply_table_cell_grid_bindings_to_rows(rows, jsonl_rows, document_key="TEST", grid_rows=grid)

        self.assertEqual(stats["rows_with_grid_bindings"], 1)
        self.assertEqual(rows[0]["size_nm_value"], "91.8")
        self.assertEqual(rows[0]["size_nm_value_text"], "91.8 ± 2.74")
        self.assertEqual(rows[0]["size_nm_membership_confidence"], "projected_direct")
        self.assertEqual(rows[0]["size_nm_evidence_region_type"], "row_local_table_cell_grid_binding")
        self.assertEqual(rows[0]["pdi_value"], "0.99")
        self.assertEqual(rows[0]["zeta_mV_value"], "-21.23")
        self.assertIn("table_cell_bindings_json", jsonl_rows[0])

    def test_generic_drug_mg_headers_project_to_drug_mass_binding(self):
        payload = {
            "table_id": "Table 2",
            "header_structure": {
                "header_row_count": 1,
                "header_rows": [["Formulation", "Drug (mg)", "Payload dye (mg)", "PLGA (mg)"]],
                "flattened_headers": ["Formulation", "Drug (mg)", "Payload dye (mg)", "PLGA (mg)"],
            },
            "normalized_rows": [
                {"row_index": 2, "row_number": "F1", "cells": ["F1", "5", "", "90"]},
            ],
        }
        grid = build_table_cell_grid_from_payload("TEST", payload)

        bindings, status = build_grid_cell_bindings_for_row(grid, paper_key="TEST", table_id="Table 2", row_label="F1")

        self.assertEqual(status, "unique_grid_row_binding")
        by_field = {item["canonical_field"]: item for item in bindings}
        self.assertEqual(by_field["drug_mass_mg"]["raw_cell_value"], "5")
        self.assertEqual(by_field["drug_mass_mg"]["raw_header"], "Drug (mg)")
        self.assertEqual(by_field["polymer_mass_mg"]["raw_cell_value"], "90")

    def test_grid_row_ordinal_label_disambiguates_repeated_formulation_labels_for_preparation_bindings(self):
        payload = {
            "table_id": "Table 1",
            "header_structure": {
                "header_row_count": 1,
                "header_rows": [["Formulation", "Acetone (mL)", "Aqueous phase (mL)", "Drug (mg)", "PLGA (mg)"]],
                "flattened_headers": ["Formulation", "Acetone (mL)", "Aqueous phase (mL)", "Drug (mg)", "PLGA (mg)"],
            },
            "normalized_rows": [
                {"row_index": 2, "row_number": "5", "cells": ["5", "5", "15", "5", "75"]},
                {"row_index": 3, "row_number": "5", "cells": ["5", "6", "16", "6", "76"]},
            ],
        }
        grid = build_table_cell_grid_from_payload("TEST", payload)

        bindings, status = build_grid_cell_bindings_for_row(grid, paper_key="TEST", table_id="Table 1", row_label="row_02__5")

        self.assertEqual(status, "unique_grid_row_binding")
        by_field = {item["canonical_field"]: item for item in bindings}
        self.assertEqual(by_field["O_volume_mL"]["raw_cell_value"], "6")
        self.assertEqual(by_field["external_aqueous_phase_volume_mL"]["raw_cell_value"], "16")
        self.assertEqual(by_field["drug_mass_mg"]["raw_cell_value"], "6")
        self.assertEqual(by_field["polymer_mass_mg"]["raw_cell_value"], "76")

    def test_row_local_binding_is_highest_authority_and_formats_direct_mass(self):
        row = {
            "key": "TEST",
            "drug_feed_amount_text_value_text": "acetylpuerarin",
            "table_cell_bindings_json": json.dumps([
                {
                    "canonical_field": "drug_mass_mg",
                    "raw_header": "Drug (mg)",
                    "raw_cell_value": "5",
                    "ambiguity_status": "unique_grid_header_cell",
                }
            ]),
        }

        value, source, evidence = get_system_value("drug_mass_mg", row, paper_key="TEST")

        self.assertEqual(value, "5 mg")
        self.assertEqual(source, "stage2_table_cell_binding_authority")
        self.assertEqual(evidence, "typed_direct_mass_value")

    def test_row_local_assignment_mass_header_is_authority_below_cell_binding(self):
        row = {
            "key": "TEST",
            "drug_feed_amount_text_value_text": "acetylpuerarin",
            "table_row_variable_assignments_json": json.dumps([
                {"('Composition', 'Drug (mg)')": "5", "('Composition', 'PLGA (mg)')": "75"}
            ]),
        }

        drug_value, drug_source, drug_evidence = get_system_value("drug_mass_mg", row, paper_key="TEST")
        polymer_value, polymer_source, polymer_evidence = get_system_value("polymer_mass_mg", row, paper_key="TEST")

        self.assertEqual((drug_value, drug_source, drug_evidence), ("5 mg", "row_local_table_assignment_authority", "typed_direct_mass_value"))
        self.assertEqual((polymer_value, polymer_source, polymer_evidence), ("75 mg", "row_local_table_assignment_authority", "typed_direct_mass_value"))

    def test_row_identity_formulation_label_can_supply_drug_name_when_direct_field_blank(self):
        row = {
            "key": "TEST",
            "drug_name_value_text": "",
            "table_row_id": "Table 7::3-MeOXAN nanocapsules (Theoretical concentration 1000 mg/mL)",
            "table_row_variable_assignments_json": json.dumps([
                {"formulation_identity_label": "3-MeOXAN nanocapsules (Theoretical concentration 1000 mg/mL)"}
            ]),
        }

        value, source, evidence = get_system_value("drug_name", row, paper_key="TEST")

        self.assertEqual((value, source, evidence), ("3-MeOXAN", "row_identity_drug_name", "supported_direct_row_identity_drug_name"))

    def test_row_identity_drug_name_rebinding_skips_blank_or_ambiguous_labels(self):
        blank_row = {
            "key": "TEST",
            "table_row_variable_assignments_json": json.dumps([
                {"formulation_identity_label": "Empty PLGA nanocapsules"}
            ]),
        }
        ambiguous_row = {
            "key": "TEST",
            "table_row_variable_assignments_json": json.dumps([
                {"formulation_identity_label": "XAN nanocapsules"},
                {"formulation_identity_label": "3-MeOXAN nanocapsules"},
            ]),
        }

        self.assertEqual(get_system_value("drug_name", blank_row, paper_key="TEST")[0], "")
        self.assertEqual(get_system_value("drug_name", ambiguous_row, paper_key="TEST")[0], "")

    def test_drug_feed_amount_rejects_identity_only_text(self):
        mass_validator = getattr(final_output, "is_valid_direct_mass_text", None)
        self.assertIsNotNone(
            mass_validator,
            "S5-2 should expose a generic direct mass validator before material-value carrythrough",
        )
        self.assertFalse(mass_validator("acetylpuerarin"))
        self.assertFalse(mass_validator("Dexibuprofen"))
        self.assertFalse(mass_validator("20 mg | PLGA"))
        self.assertTrue(mass_validator("5 mg acetylpuerarin"))

    def test_stage5_final_mass_boundary_blanks_invalid_identity_and_allows_lawful_carrythrough(self):
        source_text = "PLGA nanoparticles were prepared from 20 mg PLGA and 0.5 mg curcumin in acetone."
        row = {
            "raw_formulation_label": "F1 loaded nanoparticles",
            "drug_name_value": "curcumin",
            "drug_feed_amount_text_value": "curcumin",
            "drug_feed_amount_text_missing_reason": "",
            "plga_mass_mg_value": "PLGA",
            "plga_mass_mg_missing_reason": "",
            "polymer_name_raw": "PLGA",
        }

        materialized, applied = apply_global_preparation_material_carrythrough(
            final_row=row,
            source_text=source_text,
        )

        self.assertIn("drug_feed_amount_text", applied)
        self.assertIn("plga_mass_mg", applied)
        self.assertEqual(materialized["drug_feed_amount_text_value"], "0.5 mg")
        self.assertEqual(materialized["plga_mass_mg_value"], "20 mg")
        self.assertEqual(materialized["drug_feed_amount_text_missing_reason"], "")
        self.assertEqual(materialized["plga_mass_mg_missing_reason"], "")

    def test_stage5_final_mass_boundary_rejects_invalid_row_local_binding_before_carrythrough(self):
        source_text = "PLGA nanoparticles were prepared from 20 mg PLGA and 0.5 mg curcumin in acetone."
        row = {
            "raw_formulation_label": "F1 loaded nanoparticles",
            "drug_name_value": "curcumin",
            "drug_feed_amount_text_value": "",
            "drug_feed_amount_text_missing_reason": "",
            "plga_mass_mg_value": "",
            "plga_mass_mg_missing_reason": "",
            "polymer_name_raw": "PLGA",
            "table_cell_bindings_json": json.dumps([
                {
                    "binding_rule": "table_cell_grid_v1_row_local_header_binding",
                    "canonical_field": "drug_mass_mg",
                    "raw_header": "Drug (mg)",
                    "raw_cell_value": "curcumin",
                },
                {
                    "binding_rule": "table_cell_grid_v1_row_local_header_binding",
                    "canonical_field": "polymer_mass_mg",
                    "raw_header": "PLGA (mg)",
                    "raw_cell_value": "75",
                },
            ]),
        }

        materialized, applied = apply_global_preparation_material_carrythrough(
            final_row=row,
            source_text=source_text,
        )

        self.assertIn("drug_feed_amount_text", applied)
        self.assertEqual(materialized["drug_feed_amount_text_value"], "0.5 mg")
        self.assertEqual(materialized["plga_mass_mg_value"], "75")
        self.assertNotEqual(materialized["drug_feed_amount_text_value"], "curcumin")

    def test_typed_validator_rejects_ratio_identity_and_concentration_as_mass(self):
        for bad_value in ["75:25", "PLGA", "mg/mL", "1 mg/mL", "10 mL"]:
            ok, detail = validate_value_for_field("polymer_mass_mg", bad_value, raw_header="PLGA (mg)")
            self.assertFalse(ok, bad_value)
            self.assertTrue(detail.startswith("invalid_mass") or detail == "empty_value")
        ok, detail = validate_value_for_field("polymer_mass_mg", "90 mg", raw_header="PLGA (mg)")
        self.assertTrue(ok)
        self.assertEqual(detail, "typed_direct_mass_value")

    def test_extract_unique_shared_preparation_masses_requires_unique_direct_pair(self):
        source_text = (
            "PLGA nanoparticles were prepared by dissolving 20 mg PLGA and 0.5 mg curcumin "
            "in 2 mL acetone before emulsification. Characterization was then performed."
        )
        masses = extract_unique_shared_preparation_masses(source_text, drug_name="curcumin")
        self.assertEqual(masses["polymer_mass_mg"], "20 mg")
        self.assertEqual(masses["drug_mass_mg"], "0.5 mg")

    def test_shared_preparation_mass_carrythrough_skips_blank_helper_rows(self):
        source_text = "Nanoparticles were prepared from 20 mg PLGA and 0.5 mg curcumin in acetone."
        row = {
            "raw_formulation_label": "Blank NP control",
            "drug_name_value": "",
            "drug_feed_amount_text_value": "",
            "plga_mass_mg_value": "",
            "formulation_role": "control",
        }
        materialized, applied = apply_global_preparation_material_carrythrough(
            final_row=row,
            source_text=source_text,
        )
        self.assertNotIn("drug_feed_amount_text", applied)
        self.assertNotIn("plga_mass_mg", applied)
        self.assertEqual(materialized.get("drug_feed_amount_text_value", ""), "")
        self.assertEqual(materialized.get("plga_mass_mg_value", ""), "")

    def test_shared_preparation_mass_carrythrough_skips_doe_factor_rows(self):
        source_text = "Nanoparticles were prepared from 100 mg PLGA and 15 mg drug in acetone."
        row = {
            "raw_formulation_label": "F1 DOE row",
            "drug_name_value": "drug",
            "drug_feed_amount_text_value": "",
            "plga_mass_mg_value": "",
            "polymer_name_raw": "PLGA",
            "instance_context_tags": '["doe", "numbered_table_row"]',
            "candidate_source": "doe_numbered_table_row_recovery",
        }
        materialized, applied = apply_global_preparation_material_carrythrough(
            final_row=row,
            source_text=source_text,
        )
        self.assertNotIn("drug_feed_amount_text", applied)
        self.assertNotIn("plga_mass_mg", applied)
        self.assertEqual(materialized.get("drug_feed_amount_text_value", ""), "")
        self.assertEqual(materialized.get("plga_mass_mg_value", ""), "")

    def test_generic_doe_factor_abbreviation_materializes_drug_polymer_and_surfactant_concentrations(self):
        source_text = (
            "The independent variables were: X1 – Drug concentration in organic phase (%w/v); "
            "X2 – Polymer concentration in organic phase (%w/v); X3 – Surfactant concentration (%). "
            "Lopinavir-loaded PLGA nanoparticles were prepared using PVA as surfactant."
        )
        row = {
            "raw_formulation_label": "Run 1",
            "identity_variables_json": json.dumps([
                {"name": "X1", "value": "1.0"},
                {"name": "X2", "value": "2.5"},
                {"name": "X3", "value": "0.5"},
            ]),
            "drug_name_value": "",
            "polymer_name_value": "",
            "polymer_name_raw": "",
            "surfactant_name_value": "",
            "drug_concentration_value_value": "",
            "drug_concentration_unit_value": "",
            "polymer_concentration_value_value": "",
            "polymer_concentration_unit_value": "",
            "surfactant_concentration_text_value": "",
        }
        materialized, applied = apply_global_preparation_material_carrythrough(final_row=row, source_text=source_text)
        self.assertEqual(materialized["drug_name_value"], "Lopinavir")
        self.assertEqual(materialized["drug_concentration_value_value"], "1.0")
        self.assertEqual(materialized["drug_concentration_unit_value"], "%w/v")
        self.assertEqual(materialized["polymer_name_value"], "PLGA")
        self.assertEqual(materialized["polymer_concentration_value_value"], "2.5")
        self.assertEqual(materialized["polymer_concentration_unit_value"], "%w/v")
        self.assertEqual(materialized["surfactant_name_value"], "PVA")
        self.assertEqual(materialized["surfactant_concentration_text_value"], "0.5%")
        self.assertIn("drug_concentration_value", applied)
        self.assertIn("polymer_concentration_value", applied)
        self.assertIn("surfactant_concentration_text", applied)

    def test_generic_coded_material_abbreviation_maps_concentration_of_named_material(self):
        source_text = (
            "For the preparation of PLGA nanospheres, cFB, concentration of flurbiprofen (mg/mL), "
            "and cP188, concentration of poloxamer 188 (mg/mL), were varied."
        )
        row = {
            "raw_formulation_label": "Nanosphere 1",
            "identity_variables_json": json.dumps([
                {"name": "cFB", "value": "4"},
                {"name": "cP188", "value": "10"},
            ]),
            "drug_name_value": "",
            "surfactant_name_value": "",
            "drug_concentration_value_value": "",
            "drug_concentration_unit_value": "",
            "surfactant_concentration_text_value": "",
        }
        materialized, applied = apply_global_preparation_material_carrythrough(final_row=row, source_text=source_text)
        self.assertEqual(materialized["drug_name_value"], "flurbiprofen")
        self.assertEqual(materialized["drug_concentration_value_value"], "4")
        self.assertEqual(materialized["drug_concentration_unit_value"], "mg/mL")
        self.assertEqual(materialized["surfactant_name_value"], "poloxamer 188")
        self.assertEqual(materialized["surfactant_concentration_text_value"], "10 mg/mL")
        self.assertIn("drug_concentration_value", applied)
        self.assertIn("surfactant_concentration_text", applied)

    def test_table_row_expansion_blocks_aggregate_variant_list_in_multirow_surface(self):
        direct_rows = [
            {
                "label": "Garcinol loaded PLGA nanoparticles (GAR-NPs) | Blank PLGA nanoparticles (Blank NPs)",
                "assignments": [
                    {
                        "name": "formulation_identity_label",
                        "value": "Garcinol loaded PLGA nanoparticles (GAR-NPs) | Blank PLGA nanoparticles (Blank NPs)",
                    }
                ],
            },
            {
                "label": "FITC loaded PLGA nanoparticles (FITC-NPs) | 99mTc-labeled Garcinol loaded PLGA nanoparticles (99mTc-GAR-NPs)",
                "assignments": [
                    {
                        "name": "formulation_identity_label",
                        "value": "FITC loaded PLGA nanoparticles (FITC-NPs) | 99mTc-labeled Garcinol loaded PLGA nanoparticles (99mTc-GAR-NPs)",
                    }
                ],
            },
        ]

        self.assertTrue(direct_rows_look_like_aggregate_variant_list(direct_rows))

    def test_shared_preparation_drug_mass_uses_global_drug_identity_when_row_is_loaded_and_eligible(self):
        source_text = "Nanospheres were prepared from 100 mg PLGA and 15 mg flurbiprofen in acetone."
        row = {
            "raw_formulation_label": "Nanospheres produced with high viscosity PLGA",
            "drug_name_value": "flurbiprofen",
            "drug_name_scope": "global_shared",
            "drug_feed_amount_text_value": "",
            "plga_mass_mg_value": "",
            "instance_context_tags": '["table_row_expansion", "explicit_table_anchor"]',
            "evidence_span_text": "Nanospheres produced with high viscosity PLGA | loaded batch | 232.80 ± 1.93 | −25.79 ± 1.17 | 94.60 ± 0.42",
        }
        materialized, applied = apply_global_preparation_material_carrythrough(
            final_row=row,
            source_text=source_text,
        )
        self.assertIn("drug_feed_amount_text", applied)
        self.assertEqual(materialized.get("drug_feed_amount_text_value", ""), "15 mg")

    def test_shared_preparation_drug_mass_still_skips_global_drug_identity_on_blank_rows(self):
        source_text = "Flurbiprofen-loaded PLGA nanospheres were prepared from 100 mg PLGA and 15 mg flurbiprofen in acetone."
        row = {
            "raw_formulation_label": "Blank PLGA nanospheres",
            "drug_name_value": "flurbiprofen",
            "drug_name_scope": "global_shared",
            "drug_feed_amount_text_value": "",
            "plga_mass_mg_value": "",
            "evidence_span_text": "Blank PLGA nanospheres without drug",
        }
        materialized, applied = apply_global_preparation_material_carrythrough(
            final_row=row,
            source_text=source_text,
        )
        self.assertNotIn("drug_feed_amount_text", applied)
        self.assertEqual(materialized.get("drug_feed_amount_text_value", ""), "")

    def test_shared_preparation_mass_carrythrough_applies_to_loaded_rows_only_when_blank(self):
        source_text = "Nanoparticles were prepared from 20 mg PLGA and 0.5 mg curcumin in acetone."
        row = {
            "raw_formulation_label": "F1 loaded nanoparticles",
            "drug_name_value": "curcumin",
            "drug_feed_amount_text_value": "",
            "plga_mass_mg_value": "",
            "polymer_name_raw": "PLGA",
        }
        materialized, applied = apply_global_preparation_material_carrythrough(
            final_row=row,
            source_text=source_text,
        )
        self.assertIn("drug_feed_amount_text", applied)
        self.assertIn("plga_mass_mg", applied)
        self.assertEqual(materialized["drug_feed_amount_text_value"], "0.5 mg")
        self.assertEqual(materialized["plga_mass_mg_value"], "20 mg")
        self.assertEqual(materialized["drug_feed_amount_text_evidence_region_type"], "global_preparation_direct_mass_evidence")

    def test_shared_preparation_mass_extracts_parenthetical_and_of_polymer_forms(self):
        parenthetical = (
            "MB loaded-PLGA NPs were prepared by the nanoprecipitation method. "
            "Briefly, MB (0.5 mg), PLGA (20 mg) and different amounts of excipient were dissolved in acetone."
        )
        of_form = (
            "MF-loaded nanoparticles were prepared using the nanoprecipitation method. "
            "The organic phase, consisting of 2 mg of MF and 6 mg of PLGA in 1 mL of acetone, was rapidly poured into aqueous solution."
        )
        polymer_then_drug = (
            "The optimized formulation was prepared using nanoprecipitation method as follows: "
            "polymer (50 mg) and etoposide (5 mg) were dissolved in acetone."
        )
        self.assertEqual(extract_unique_shared_preparation_masses(parenthetical, drug_name="MB")["polymer_mass_mg"], "20 mg")
        self.assertEqual(extract_unique_shared_preparation_masses(of_form, drug_name="MF")["polymer_mass_mg"], "6 mg")
        self.assertEqual(extract_unique_shared_preparation_masses(polymer_then_drug, drug_name="etoposide")["polymer_mass_mg"], "50 mg")
        self.assertEqual(extract_unique_shared_preparation_masses(polymer_then_drug, drug_name="etoposide")["drug_mass_mg"], "5 mg")

    def test_shared_preparation_mass_can_be_scoped_by_nanocarrier_subtype(self):
        source_text = (
            "PLGA nanospheres were prepared by the solvent displacement technique. "
            "Briefly, an organic solution of PLGA (63 mg) and XAN in acetone was poured into aqueous solution. "
            "PLGA nanocapsules were prepared by interfacial polymer deposition. "
            "Briefly, about 50 mg of polymer and 100 mg of soybean lecithin were dissolved in acetone."
        )
        sphere_row = {
            "raw_formulation_label": "XAN nanospheres (Theoretical concentration 60 mg/mL)",
            "drug_name_value": "XAN",
            "plga_mass_mg_value": "",
            "drug_feed_amount_text_value": "",
        }
        capsule_row = {
            "raw_formulation_label": "XAN nanocapsules (Theoretical concentration 600 mg/mL)",
            "drug_name_value": "XAN",
            "plga_mass_mg_value": "",
            "drug_feed_amount_text_value": "",
        }
        sphere, sphere_applied = apply_global_preparation_material_carrythrough(final_row=sphere_row, source_text=source_text)
        capsule, capsule_applied = apply_global_preparation_material_carrythrough(final_row=capsule_row, source_text=source_text)
        self.assertIn("plga_mass_mg", sphere_applied)
        self.assertIn("plga_mass_mg", capsule_applied)
        self.assertEqual(sphere["plga_mass_mg_value"], "63 mg")
        self.assertEqual(capsule["plga_mass_mg_value"], "50 mg")

    def test_shared_preparation_polymer_mass_can_apply_to_loaded_plga_family_label_without_row_local_polymer_field(self):
        source_text = (
            "Rhodamine-loaded PLGA NPs (NPR1; Table 1) were prepared by nanoprecipitation. "
            "Briefly, 50 mg of PLGA and 2.5 mg Rh were dissolved in 4 mL acetone."
        )
        row = {
            "raw_formulation_label": "NPR1",
            "drug_name_value": "Rhodamine",
            "drug_feed_amount_text_value": "2.5 mg",
            "plga_mass_mg_value": "",
            "evidence_span_text": "NPR1 2.5 – 1 –",
        }
        materialized, applied = apply_global_preparation_material_carrythrough(
            final_row=row,
            source_text=source_text,
        )
        self.assertIn("plga_mass_mg", applied)
        self.assertEqual(materialized["plga_mass_mg_value"], "50 mg")

    def test_shared_preparation_organic_phase_volume_extracts_name_bound_solvent_volume(self):
        source_text = (
            "PLGA nanoparticles were prepared by solvent displacement. "
            "In summary, 90 mg of PLGA and 5 mg dexibuprofen were dissolved in 5 mL of acetone. "
            "The organic solution was added dropwise into aqueous surfactant solution."
        )
        self.assertEqual(
            extract_unique_global_preparation_organic_phase_volume(source_text, solvent_name="acetone"),
            "5 mL",
        )

    def test_shared_preparation_organic_phase_volume_requires_solvent_binding_and_unambiguous_prep_context(self):
        hplc_text = (
            "PLGA nanoparticles were prepared by nanoprecipitation. "
            "HPLC analysis used a mobile phase of acetonitrile and water. "
            "The calibration standard was dissolved in 5 mL acetone."
        )
        ambiguous_text = (
            "PLGA nanoparticles were prepared by dissolving polymer in 5 mL acetone. "
            "A second PLGA recipe dissolved polymer in 2 mL acetone."
        )
        self.assertEqual(extract_unique_global_preparation_organic_phase_volume(hplc_text, solvent_name="acetone"), "")
        self.assertEqual(extract_unique_global_preparation_organic_phase_volume(ambiguous_text, solvent_name="acetone"), "")
        self.assertEqual(extract_unique_global_preparation_organic_phase_volume("5 mL acetone was used.", solvent_name=""), "")

    def test_shared_preparation_organic_phase_volume_carries_to_row_with_global_solvent_without_overwriting(self):
        source_text = (
            "PLGA nanoparticles containing DXI were prepared by solvent displacement. "
            "90 mg of PLGA and 5 mg dexibuprofen were dissolved in 5 mL of acetone. "
            "The organic solution was added dropwise into 10 mL aqueous surfactant solution."
        )
        row = {
            "raw_formulation_label": "Formulation 1 loaded PLGA nanoparticles",
            "organic_solvent_value": "acetone",
            "organic_solvent_scope": "global_shared",
            "organic_phase_volume_mL_value": "",
            "organic_phase_volume_mL_value_text": "",
            "organic_phase_volume_mL_scope": "",
            "organic_phase_volume_mL_evidence_region_type": "",
        }
        materialized, applied = apply_global_preparation_material_carrythrough(
            final_row=row,
            source_text=source_text,
        )
        self.assertIn("organic_phase_volume_mL", applied)
        self.assertEqual(materialized["organic_phase_volume_mL_value"], "5 mL")
        self.assertEqual(materialized["organic_phase_volume_mL_scope"], "global_shared")
        self.assertEqual(materialized["organic_phase_volume_mL_evidence_region_type"], "global_preparation_organic_phase_volume_evidence")

        existing = dict(row)
        existing["organic_phase_volume_mL_value"] = "4 mL"
        existing["organic_phase_volume_mL_value_text"] = "4 mL"
        materialized_existing, applied_existing = apply_global_preparation_material_carrythrough(
            final_row=existing,
            source_text=source_text,
        )
        self.assertNotIn("organic_phase_volume_mL", applied_existing)
        self.assertEqual(materialized_existing["organic_phase_volume_mL_value"], "4 mL")

    def test_shared_preparation_external_aqueous_phase_volume_carries_from_preparation_text_without_overwriting(self):
        source_text = (
            "PLGA nanoparticles containing DXI were prepared by solvent displacement. "
            "90 mg of PLGA and 5 mg dexibuprofen were dissolved in 5 mL of acetone. "
            "The organic solution was added dropwise into 10 mL of an aqueous surfactant solution at pH 3.5."
        )
        row = {
            "raw_formulation_label": "Formulation 1 loaded PLGA nanoparticles",
            "external_aqueous_phase_volume_mL_value": "",
            "external_aqueous_phase_volume_mL_value_text": "",
            "external_aqueous_phase_volume_mL_scope": "",
            "external_aqueous_phase_volume_mL_evidence_region_type": "",
        }
        materialized, applied = apply_global_preparation_material_carrythrough(
            final_row=row,
            source_text=source_text,
        )
        self.assertIn("external_aqueous_phase_volume_mL", applied)
        self.assertEqual(materialized["external_aqueous_phase_volume_mL_value"], "10 mL")
        self.assertEqual(materialized["external_aqueous_phase_volume_mL_scope"], "global_shared")
        self.assertEqual(materialized["external_aqueous_phase_volume_mL_evidence_region_type"], "global_preparation_external_aqueous_phase_volume_evidence")

        existing = dict(row)
        existing["external_aqueous_phase_volume_mL_value"] = "15 mL"
        existing["external_aqueous_phase_volume_mL_value_text"] = "15 mL"
        materialized_existing, applied_existing = apply_global_preparation_material_carrythrough(
            final_row=existing,
            source_text=source_text,
        )
        self.assertNotIn("external_aqueous_phase_volume_mL", applied_existing)
        self.assertEqual(materialized_existing["external_aqueous_phase_volume_mL_value"], "15 mL")

    def test_shared_preparation_external_aqueous_phase_volume_rejects_non_prep_and_ambiguous_contexts(self):
        hplc_text = (
            "PLGA nanoparticles were prepared by nanoprecipitation. "
            "HPLC analysis used an aqueous mobile phase. "
            "The calibration standard was diluted into 10 mL of aqueous solution."
        )
        ambiguous_text = (
            "PLGA nanoparticles were prepared by adding the organic phase into 10 mL aqueous phase. "
            "A second preparation recipe used 20 mL aqueous phase."
        )
        for source_text in (hplc_text, ambiguous_text):
            materialized, applied = apply_global_preparation_material_carrythrough(
                final_row={
                    "raw_formulation_label": "loaded PLGA nanoparticles",
                    "external_aqueous_phase_volume_mL_value": "",
                    "external_aqueous_phase_volume_mL_value_text": "",
                },
                source_text=source_text,
            )
            self.assertNotIn("external_aqueous_phase_volume_mL", applied)
            self.assertEqual(materialized.get("external_aqueous_phase_volume_mL_value", ""), "")

    def test_generic_emulsifier_factor_assignment_materializes_name_and_concentration(self):
        row = {
            "change_descriptions": json.dumps(["cP188 (mg/mL)=15.0", "polymer=PLGA"]),
            "identity_variables_json": "",
            "raw_formulation_label": "DOE Row F1",
            "surfactant_name_value": "",
            "surfactant_name_value_text": "",
            "surfactant_concentration_text_value": "",
            "surfactant_concentration_text_value_text": "",
            "surfactant_concentration_text_scope": "",
        }
        source_text = "The factors were cP188, concentration of poloxamer 188 (mg/mL); cPLGA, concentration of PLGA."

        materialized, applied = apply_global_preparation_material_carrythrough(
            final_row=row,
            source_text=source_text,
        )

        self.assertEqual(materialized["surfactant_name_value_text"], "poloxamer 188")
        self.assertEqual(materialized["surfactant_concentration_text_value_text"], "15.0 mg/mL")
        self.assertIn("surfactant_name", applied)
        self.assertIn("surfactant_concentration_text", applied)

    def test_generic_surfactant_factor_concentration_without_material_name_uses_unit_definition_only(self):
        row = {
            "change_descriptions": json.dumps(["X3=0.75"]),
            "identity_variables_json": "",
            "raw_formulation_label": "DOE Row 8",
            "surfactant_name_value": "",
            "surfactant_name_value_text": "",
            "surfactant_concentration_text_value": "",
            "surfactant_concentration_text_value_text": "",
            "surfactant_concentration_text_scope": "",
        }
        source_text = "Independent variables were X1 drug amount, X2 polymer amount, and X3 surfactant concentration (%w/v)."

        materialized, applied = apply_global_preparation_material_carrythrough(
            final_row=row,
            source_text=source_text,
        )

        self.assertEqual(materialized["surfactant_concentration_text_value_text"], "0.75 %w/v")
        self.assertNotIn("surfactant_name", applied)
        self.assertIn("surfactant_concentration_text", applied)

    def test_row_local_surfactant_assignment_materializes_name_and_concentration(self):
        row = {
            "change_descriptions": json.dumps(["Gatifloxacin (mg)=5", "Polysorbate 80 (%)=1"]),
            "identity_variables_json": json.dumps([
                {"name_raw": "Gatifloxacin (mg)", "value_raw": "5"},
                {"name_raw": "Polysorbate 80 (%)", "value_raw": "1"},
            ]),
            "raw_formulation_label": "NPG2",
            "surfactant_name_value": "",
            "surfactant_name_value_text": "",
            "surfactant_concentration_text_value": "",
            "surfactant_concentration_text_value_text": "",
        }

        materialized, applied = apply_global_preparation_material_carrythrough(
            final_row=row,
            source_text="",
        )

        self.assertEqual(materialized["surfactant_name_value_text"], "Polysorbate 80")
        self.assertEqual(materialized.get("surfactant_concentration_text_value_text", ""), "")
        self.assertIn("surfactant_name", applied)
        self.assertNotIn("surfactant_concentration_text", applied)

    def test_row_local_polymer_and_drug_registry_uses_formulation_label_and_source_abbreviation(self):
        row = {
            "representative_source_raw_formulation_label": "MF-loaded PLGA Nanoparticles",
            "raw_formulation_label": "MF-loaded PLGA Nanoparticles",
            "change_descriptions": "",
            "polymer_identity_final": "",
            "polymer_name_raw": "",
            "drug_name_value": "",
            "drug_name_value_text": "",
            "loaded_state_final": "drug_loaded",
        }
        source_text = "Mometasone furoate (MF) was purchased from Acros Organics. PLGA-Purasorb PDLG 5010 (50:50) with an inherent viscosity midpoint of 1 dL/g was kindly donated."

        materialized, applied = apply_global_preparation_material_carrythrough(
            final_row=row,
            source_text=source_text,
        )

        self.assertEqual(materialized["polymer_name_raw"], "PLGA")
        self.assertEqual(materialized["drug_name_value_text"], "Mometasone furoate")
        self.assertIn("polymer_name", applied)
        self.assertIn("drug_name", applied)

    def test_material_identity_property_registry_materializes_row_bound_polymer_and_mw(self):
        row = {
            "representative_source_raw_formulation_label": "PCL / Drug loaded",
            "change_descriptions": json.dumps(["polymer_identity=PCL", "stabilizer concentration=1.0%w/v"]),
            "polymer_identity_final": "PCL",
            "polymer_name_raw": "",
            "polymer_mw_kDa_value": "",
            "polymer_mw_kDa_value_text": "",
            "drug_name_value": "",
            "drug_name_value_text": "",
            "loaded_state_final": "loaded",
        }
        source_text = "Etoposide-loaded PCL nanoparticles were prepared. PCL (Mw 4 kDa) and PLGA 50/50 (Mw 10 kDa) were used."

        materialized, applied = apply_global_preparation_material_carrythrough(
            final_row=row,
            source_text=source_text,
        )

        self.assertEqual(materialized["polymer_name_raw"], "PCL")
        self.assertEqual(materialized["polymer_mw_kDa_value_text"], "4 kDa")
        self.assertEqual(materialized["drug_name_value_text"], "Etoposide")
        self.assertIn("polymer_name", applied)
        self.assertIn("polymer_mw_kDa", applied)

    def test_material_registry_does_not_apply_ambiguous_polymer_mw(self):
        row = {
            "representative_source_raw_formulation_label": "PLGA / Drug loaded",
            "change_descriptions": json.dumps(["polymer_identity=PLGA"]),
            "polymer_identity_final": "PLGA",
            "polymer_name_raw": "",
            "polymer_mw_kDa_value": "",
            "polymer_mw_kDa_value_text": "",
        }
        source_text = "PLGA 50/50 (Mw 10 kDa) and PLGA 75/25 (Mw 20 kDa) were used."

        materialized, applied = apply_global_preparation_material_carrythrough(
            final_row=row,
            source_text=source_text,
        )

        self.assertEqual(materialized.get("polymer_mw_kDa_value_text", ""), "")
        self.assertNotIn("polymer_mw_kDa", applied)

    def test_shared_external_aqueous_volume_does_not_create_w2_alias_extra_surface(self):
        row = {
            "external_aqueous_phase_volume_mL_value_text": "10 mL",
            "change_descriptions": "",
            "supporting_evidence_refs": "",
            "evidence_span_text": "",
        }
        external_value, external_source, _ = get_system_value("external_aqueous_phase_volume_mL", row)
        w2_value, w2_source, w2_evidence = get_system_value("W2_volume_mL", row)
        self.assertEqual(external_value, "10 mL")
        self.assertEqual(external_source, "direct_extracted")
        self.assertEqual(w2_value, "")
        self.assertEqual(w2_source, "missing_system_field_surface")
        self.assertEqual(w2_evidence, "missing_system_field_surface")

    def test_shared_external_aqueous_volume_can_satisfy_nonempty_w2_gt_without_w2_extra_alias(self):
        gt_rows = [
            {
                "paper_key": "TEST",
                "doi": "",
                "gt_formulation_id": "TEST_G001",
                "formulation_label": "F1",
                "benchmark_default_include": "yes",
                "W2_volume_mL": "10 mL",
                "external_aqueous_phase_volume_mL": "",
            }
        ]
        system_rows = [
            {
                "key": "TEST",
                "formulation_id": "TEST_G001",
                "final_formulation_id": "TEST_F001",
                "external_aqueous_phase_volume_mL_value_text": "10 mL",
            }
        ]
        cells, _ = build_cells(gt_rows, system_rows)
        w2_cell = next(row for row in cells if row["field_name"] == "W2_volume_mL")
        external_cell = next(row for row in cells if row["field_name"] == "external_aqueous_phase_volume_mL")
        self.assertEqual(w2_cell["compare_status"], "present_and_match")
        self.assertEqual(w2_cell["system_value_source_type"], "external_aqueous_phase_alias_for_w2_gt")
        self.assertEqual(external_cell["compare_status"], "not_reported_in_gt")
        self.assertEqual(external_cell["system_value_source_type"], "suppressed_external_aqueous_duplicate_of_w2_gt")

    def test_derived_mass_provenance_does_not_write_direct_mass_fields(self):
        row = {
            "drug_feed_amount_text_value": "",
            "plga_mass_mg_value": "20 mg",
            "drug_to_polymer_ratio_raw_value": "1:10",
        }
        provenance = build_derived_mass_provenance_for_row(row, source_text="")
        self.assertEqual(row["drug_feed_amount_text_value"], "")
        self.assertEqual(provenance[0]["derived_field"], "drug_mass_mg")
        self.assertEqual(provenance[0]["derivation_rule"], "ratio_times_known_polymer_mass")
        self.assertEqual(provenance[0]["derived_value"], "2 mg")
        self.assertEqual(provenance[0]["direct_or_derived"], "derived")
        self.assertEqual(provenance[0]["direct_field_write_allowed"], "no")
        self.assertEqual(provenance[0]["evidence_binding_status"], "derived_without_direct_text")

    def test_concentration_times_volume_stays_in_derived_sidecar_not_direct_mass(self):
        row = {
            "drug_feed_amount_text_value": "",
            "drug_concentration_value_value": "2",
            "drug_concentration_unit_value": "mg/mL",
            "evidence_span_text": "The drug solution concentration was 2 mg/mL in 5 mL before nanoparticle preparation.",
        }

        provenance = build_derived_mass_provenance_for_row(row, source_text="")

        self.assertEqual(row["drug_feed_amount_text_value"], "")
        self.assertEqual(len(provenance), 1)
        self.assertEqual(provenance[0]["derived_field"], "drug_mass_mg")
        self.assertEqual(provenance[0]["derived_value"], "10 mg")
        self.assertEqual(provenance[0]["derivation_rule"], "mg_per_ml_times_volume_ml")
        self.assertEqual(provenance[0]["direct_or_derived"], "derived")
        self.assertEqual(provenance[0]["direct_field_write_allowed"], "no")
        self.assertEqual(provenance[0]["evidence_binding_status"], "derived_without_direct_text")

    def test_direct_formulation_rows_preserve_measurement_tail_size_values(self):
        row_entries = [
            {"row_index": 1, "cells": ["Formulation", "Polymer", "", "Average", "Polydispersity", "Zeta Potential", ""]},
            {"row_index": 2, "cells": ["", "", "Surfactant", "", "", "", "EE (%)"]},
            {"row_index": 3, "cells": ["Number", "Used", "", "Size (nm)", "Index (PI)", "(ZP, mV)", ""]},
            {
                "row_index": 4,
                "cells": ["1", "", "PVA", "234.1 ± 0.5", "0.081 ± 0.009", "−12.2 ± 1.3", "93.4"],
                "row_text": "1 | PVA | 234.1 ± 0.5 | 0.081 ± 0.009 | −12.2 ± 1.3 | 93.4",
            },
            {
                "row_index": 5,
                "cells": ["2", "", "Tween80", "146.0 ± 0.6", "0.054 ± 0.008", "−25.2 ± 0.6", "87.5"],
                "row_text": "2 | Tween80 | 146.0 ± 0.6 | 0.054 ± 0.008 | −25.2 ± 0.6 | 87.5",
            },
        ]
        rows, reason = extract_direct_formulation_rows_from_authority(
            authority_payload={},
            row_entries=row_entries,
        )
        self.assertEqual(reason, "")
        self.assertEqual(len(rows), 2)
        first_assignments = {(item["name"], item["value"]) for item in rows[0]["assignments"]}
        self.assertIn(("Average Size (nm)", "234.1 ± 0.5"), first_assignments)

    def test_direct_formulation_rows_do_not_shift_numbered_row_label_headers_into_measurements(self):
        row_entries = [
            {"row_index": 1, "cells": ["Surfactant", "EE (%)"]},
            {"row_index": 2, "cells": ["Number", "Used", "Size (nm)", "Index (PI)", "(ZP, mV)"]},
            {"row_index": 3, "cells": ["1", "PVA", "234.1 ± 0.5", "0.081 ± 0.009", "−12.2 ± 1.3", "93.4"]},
            {"row_index": 4, "cells": ["2", "Tween80", "146.0 ± 0.6", "0.054 ± 0.008", "−25.2 ± 0.6", "87.5"]},
        ]

        rows, reason = extract_direct_formulation_rows_from_authority(
            authority_payload={},
            row_entries=row_entries,
        )

        self.assertEqual(reason, "")
        first_assignments = {(item["name"], item["value"]) for item in rows[0]["assignments"]}
        self.assertIn(("Size (nm)", "234.1 ± 0.5"), first_assignments)
        self.assertNotIn(("Used", "234.1 ± 0.5"), first_assignments)

    def test_direct_formulation_rows_emit_structured_cell_bindings_for_measurement_headers(self):
        row_entries = [
            {"row_index": 1, "cells": ["Formulation", "Polymer", "", "Average", "Polydispersity", "Zeta Potential", ""]},
            {"row_index": 2, "cells": ["", "", "Surfactant", "", "", "", "EE (%)"]},
            {"row_index": 3, "cells": ["Number", "Used", "", "Size (nm)", "Index (PI)", "(ZP, mV)", ""]},
            {
                "row_index": 4,
                "cells": ["1", "", "PVA", "234.1 ± 0.5", "0.081 ± 0.009", "−12.2 ± 1.3", "93.4"],
            },
            {
                "row_index": 5,
                "cells": ["2", "", "Tween80", "146.0 ± 0.6", "0.054 ± 0.008", "−25.2 ± 0.6", "87.5"],
            },
        ]

        rows, reason = extract_direct_formulation_rows_from_authority(
            authority_payload={"source_csv_path": "data/cleaned/goren_2025/tables/INMUTV7L/INMUTV7L__table_01__html_table.csv"},
            row_entries=row_entries,
        )

        self.assertEqual(reason, "")
        bindings = rows[0].get("table_cell_bindings")
        self.assertIsInstance(bindings, list)
        by_field = {item["canonical_field"]: item for item in bindings if item.get("canonical_field")}
        self.assertEqual(by_field["particle_size_nm"]["raw_cell_value"], "234.1 ± 0.5")
        self.assertEqual(by_field["particle_size_nm"]["raw_header"], "Average Size (nm)")
        self.assertEqual(by_field["particle_size_nm"]["source_row_index"], "4")
        self.assertEqual(by_field["particle_size_nm"]["source_column_index"], "3")
        self.assertEqual(by_field["ee_percent"]["raw_cell_value"], "93.4")
        self.assertEqual(by_field["ee_percent"]["raw_header"], "EE (%)")
        self.assertEqual(by_field["zeta_mV"]["raw_cell_value"], "−12.2 ± 1.3")

    def test_field_header_alias_lexicon_maps_common_measurement_headers(self):
        self.assertEqual(canonical_field_for_header("%EE"), "ee_percent")
        self.assertEqual(canonical_field_for_header("E.E.% ± S.D."), "ee_percent")
        self.assertEqual(canonical_field_for_header("P.I."), "pdi")
        self.assertEqual(canonical_field_for_header("P. I. ± S.D."), "pdi")
        self.assertEqual(canonical_field_for_header("Polidispersity Index ± SD"), "pdi")
        self.assertEqual(canonical_field_for_header("ZP (mV)"), "zeta_mV")
        self.assertEqual(canonical_field_for_header("ZPb (mV)"), "zeta_mV")
        self.assertEqual(canonical_field_for_header("ζ-potential (mV)"), "zeta_mV")
        self.assertEqual(canonical_field_for_header("EEc (%)"), "ee_percent")
        self.assertEqual(canonical_field_for_header("Drug content (%)"), "lc_percent")
        self.assertEqual(canonical_field_for_header("D.L. (%)"), "dl_percent")
        self.assertEqual(compatibility_field_for_assignment("D.L. (%)"), "dl_percent")
        self.assertEqual(canonical_field_for_header("Y2 (PS, nm)"), "particle_size_nm")
        self.assertEqual(canonical_field_for_header("Z-average (nm)"), "particle_size_nm")
        self.assertEqual(canonical_field_for_header("Z-Ave (nm)"), "particle_size_nm")
        self.assertEqual(canonical_field_for_header("Z ave ± SD (nm)"), "particle_size_nm")
        self.assertEqual(canonical_field_for_header("Major axis (nm)"), "particle_size_nm")
        self.assertEqual(canonical_field_for_header("Minor axis (nm)"), "")
        self.assertEqual(canonical_field_for_header("Feret’s diameter (nm)"), "")
        self.assertEqual(canonical_field_for_header("L.C. (%)"), "lc_percent")
        self.assertEqual(canonical_field_for_header("LC (%)"), "lc_percent")
        self.assertEqual(canonical_field_for_header("PLGA mg/mL"), "polymer_concentration_value")
        self.assertEqual(canonical_field_for_header("Drug conc. Mg/mL"), "drug_concentration_value")
        self.assertEqual(canonical_field_for_header("PLGA (mg)"), "polymer_mass_mg")

    def test_field_header_alias_lexicon_guards_contextual_non_measurement_headers(self):
        self.assertEqual(canonical_field_for_header("HPLC method"), "")
        self.assertEqual(canonical_field_for_header("LC-MS assay"), "")
        self.assertEqual(canonical_field_for_header("Recovery (%)"), "")

    def test_table_cell_grid_rejects_mg_per_ml_polymer_header_as_direct_mass(self):
        bindings = [
            {
                "binding_rule": "table_cell_grid_v1_row_local_header_binding",
                "canonical_field": "polymer_mass_mg",
                "raw_header": "PLGA mg/mL",
                "raw_cell_value": "60",
            },
            {
                "binding_rule": "table_cell_grid_v1_row_local_header_binding",
                "canonical_field": "drug_mass_mg",
                "raw_header": "Drug conc. Mg/mL",
                "raw_cell_value": "3",
            },
        ]
        direct = final_output.direct_values_from_table_cell_grid_bindings(bindings)
        self.assertNotIn("plga_mass_mg", direct)
        self.assertEqual(direct["drug_feed_amount_text"], "3")
        self.assertNotIn("polymer_concentration_value", direct)
        self.assertNotIn("drug_concentration_value", direct)

    def test_table_cell_grid_maps_lc_and_dl_percent_headers_to_row_local_metrics(self):
        bindings = [
            {
                "binding_rule": "table_cell_grid_v1_row_local_header_binding",
                "canonical_field": "lc_percent",
                "raw_header": "LC (%)",
                "raw_cell_value": "7.5 ± 0.2",
            },
            {
                "binding_rule": "table_cell_grid_v1_row_local_header_binding",
                "canonical_field": "dl_percent",
                "raw_header": "D.L. (%)",
                "raw_cell_value": "3.1",
            },
        ]
        direct = final_output.direct_values_from_table_cell_grid_bindings(bindings)
        self.assertEqual(direct["loading_content_percent"], "7.5")
        self.assertEqual(direct["loading_content_percent_text"], "7.5 ± 0.2")
        self.assertEqual(direct["dl_percent"], "3.1")
        materialized = {"table_cell_bindings_json": json.dumps(bindings)}
        applied_fields = set()
        self.assertTrue(final_output.apply_row_local_table_cell_binding_values(materialized, applied_fields))
        self.assertEqual(materialized["loading_content_percent_value"], "7.5")
        self.assertEqual(materialized["loading_content_percent_value_text"], "7.5 ± 0.2")
        self.assertEqual(materialized["dl_percent_value"], "3.1")

    def test_source_csv_rebinding_maps_percent_dl_header_to_dl_percent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "characterization.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["('Particle size (nm)', 'Particle size (nm)')", "('% DL', '% DL')"])
                writer.writerow(["190.2 ± 18.0", "4.9 ± 0.1"])
            row_map = final_output.load_row_local_characterization_table_map(str(csv_path))
        self.assertEqual(row_map[2]["size_nm"], "190.2")
        self.assertEqual(row_map[2]["dl_percent"], "4.9")
        self.assertNotIn(1, row_map)
        self.assertEqual(row_map[2]["dl_percent_text"], "4.9 ± 0.1")

    def test_measurement_assignment_maps_abbreviated_ee_header_to_encapsulation_efficiency(self):
        self.assertEqual(
            compatibility_field_for_assignment("E.E.% ± S.D."),
            "encapsulation_efficiency_percent",
        )
        self.assertEqual(
            compatibility_field_for_assignment("Encapsulation efficiency (EE, %)"),
            "encapsulation_efficiency_percent",
        )
        self.assertEqual(compatibility_field_for_assignment("P.I."), "pdi")
        self.assertEqual(compatibility_field_for_assignment("ZP (mV)"), "zeta_mV")
        self.assertEqual(compatibility_field_for_assignment("D.L. (%)"), "dl_percent")


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

    def test_get_system_value_uses_final_polymer_identity_when_name_surface_blank(self):
        value, source, evidence = get_system_value(
            "polymer_name",
            {
                "polymer_name_raw": "",
                "polymer_identity_final": "PLGA",
                "polymer_identity": "",
                "final_formulation_id": "GENERIC__table_1__row_01",
            },
        )
        self.assertEqual(value, "PLGA")
        self.assertEqual(source, "final_polymer_identity")
        self.assertEqual(evidence, "supported")

    def test_get_system_value_does_not_use_unknown_final_polymer_identity_as_name(self):
        value, source, evidence = get_system_value(
            "polymer_name",
            {
                "polymer_name_raw": "",
                "polymer_identity_final": "unknown",
                "polymer_identity": "",
                "final_formulation_id": "GENERIC__table_1__row_01",
            },
        )
        self.assertEqual(value, "")
        self.assertEqual(source, "direct_extracted")
        self.assertEqual(evidence, "supported")

    def test_get_system_value_extracts_concentration_unit_from_combined_text_surface(self):
        value, source, evidence = get_system_value(
            "surfactant_concentration_unit",
            {"surfactant_concentration_text_value_text": "15.0 mg/mL"},
        )
        self.assertEqual(value, "mg/mL")
        self.assertEqual(source, "direct_extracted")
        self.assertEqual(evidence, "typed_direct_concentration_unit")

    def test_get_system_value_leaves_bare_percent_concentration_unit_embedded_in_value(self):
        value, source, evidence = get_system_value(
            "surfactant_concentration_unit",
            {"surfactant_concentration_text_value_text": "0.5%"},
        )
        self.assertEqual(value, "")
        self.assertEqual(source, "direct_extracted")
        self.assertEqual(evidence, "empty_value")

    def test_get_system_value_leaves_numeric_only_concentration_unit_blank(self):
        value, source, evidence = get_system_value(
            "surfactant_concentration_unit",
            {"surfactant_concentration_text_value_text": "15.0"},
        )
        self.assertEqual(value, "")
        self.assertEqual(source, "direct_extracted")
        self.assertEqual(evidence, "empty_value")

    def test_get_system_value_recovers_concentration_unit_from_row_local_factor_assignment_header(self):
        value, source, evidence = get_system_value(
            "surfactant_concentration_unit",
            {
                "surfactant_name_value_text": "PVA",
                "surfactant_concentration_text_value_text": "10.0",
                "change_descriptions": '["(\'Coded Levels of Factors\', \'cPVA (mg/mL)\')=−1", "cPVA (mg/mL)=10.0"]',
            },
        )
        self.assertEqual(value, "mg/mL")
        self.assertEqual(source, "row_local_assignment_header")
        self.assertEqual(evidence, "typed_direct_concentration_unit")

    def test_get_system_value_keeps_numeric_only_unit_blank_without_matching_assignment_header(self):
        value, source, evidence = get_system_value(
            "surfactant_concentration_unit",
            {
                "surfactant_name_value_text": "PVA",
                "surfactant_concentration_text_value_text": "10.0",
                "change_descriptions": '["polymer concentration (mg/mL)=8.5"]',
            },
        )
        self.assertEqual(value, "")
        self.assertEqual(source, "direct_extracted")
        self.assertEqual(evidence, "empty_value")

    def test_get_system_value_reverses_table_declared_drug_polymer_ratio_for_polymer_to_drug_field(self):
        row = {
            "key": "TEST",
            "raw_formulation_label": "1:10 / 50:50",
            "table_variable_roles_json": json.dumps({"varying_variables": ["Drug:Polymer ratio"]}),
            "identity_variables_json": json.dumps([
                {"name": "formulation_header_part_1", "value": "1:10"},
                {"name": "formulation_header_part_2", "value": "50:50"},
            ]),
        }
        drug_to_polymer, drug_source, _ = get_system_value("drug_to_polymer_ratio_raw", row)
        polymer_to_drug, polymer_source, _ = get_system_value("polymer_to_drug_ratio_raw", row)
        self.assertEqual(drug_to_polymer, "1:10")
        self.assertEqual(polymer_to_drug, "10:1")
        self.assertEqual(drug_source, "ratio_label_token_rebinding")
        self.assertEqual(polymer_source, "ratio_label_token_rebinding")

    def test_get_system_value_routes_multiple_table_declared_ratio_columns_generically(self):
        row = {
            "key": "TEST",
            "raw_formulation_label": "1:10 / 50:50",
            "table_variable_roles_json": json.dumps({"columns": ["Drug:Polymer ratio", "LA:GA ratio"]}),
        }
        polymer_to_drug, source, _ = get_system_value("polymer_to_drug_ratio_raw", row)
        la_ga, la_ga_source, _ = get_system_value("la_ga_ratio_raw", row)
        self.assertEqual(polymer_to_drug, "10:1")
        self.assertEqual(la_ga, "50:50")
        self.assertEqual(source, "ratio_label_token_rebinding")
        self.assertEqual(la_ga_source, "ratio_label_token_rebinding")

    def test_get_system_value_reverses_polymer_solvent_ratio_from_generic_header(self):
        row = {
            "key": "TEST",
            "raw_formulation_label": "1:8",
            "table_variable_roles_json": json.dumps({"columns": ["Solvent:Polymer ratio"]}),
        }
        value, source, _ = get_system_value("polymer_to_solvent_ratio_raw", row)
        self.assertEqual(value, "8:1")
        self.assertEqual(source, "ratio_label_token_rebinding")

    def test_get_system_value_does_not_invent_pluronic_stabilizer_from_polymer_family_label(self):
        value, source, evidence = get_system_value(
            "stabilizer_name",
            {"key": "TEST", "raw_formulation_label": "PLGA 50/50 / Drug loaded"},
        )
        self.assertEqual(value, "")
        self.assertEqual(source, "missing_system_field_surface")
        self.assertEqual(evidence, "missing_system_field_surface")

    def test_get_system_value_recovers_direct_drug_concentration_from_row_identity_label(self):
        value, source, evidence = get_system_value(
            "drug_concentration_value",
            {"formulation_id": "L3__table_5__xan_nanospheres_theoretical_concentration_60_mg/ml"},
        )
        self.assertEqual(value, "60")
        self.assertEqual(source, "row_identity_drug_concentration")
        self.assertEqual(evidence, "supported_direct_row_identity_concentration")
        unit, unit_source, unit_evidence = get_system_value(
            "drug_concentration_unit",
            {"formulation_id": "L3__table_5__xan_nanospheres_theoretical_concentration_60_mg/ml"},
        )
        self.assertEqual(unit, "mg/mL")
        self.assertEqual(unit_source, "row_identity_drug_concentration")
        self.assertEqual(unit_evidence, "supported_direct_row_identity_concentration_unit")

    def test_compare_values_normalizes_concentration_unit_spacing(self):
        strict, relaxed, canonicalized = compare_values(
            "emulsifier_stabilizer_concentration_unit",
            "%w/v",
            "% w/v",
        )
        self.assertFalse(strict)
        self.assertTrue(relaxed)
        self.assertTrue(canonicalized)

    def test_get_system_value_recovers_direct_surfactant_name_concentration_binding_from_row_identity_label(self):
        row = {"formulation_id": "Q__single_variable__poloxamer_188_concentration__2.5_mg/ml"}
        name, name_source, name_evidence = get_system_value("surfactant_name", row)
        self.assertEqual(name, "Poloxamer 188")
        self.assertEqual(name_source, "row_identity_surfactant_concentration")
        self.assertEqual(name_evidence, "supported_direct_row_identity_concentration_binding")
        value, value_source, value_evidence = get_system_value("surfactant_concentration_value", row)
        self.assertEqual(value, "2.5")
        self.assertEqual(value_source, "row_identity_surfactant_concentration")
        self.assertEqual(value_evidence, "supported_direct_row_identity_concentration_binding")
        unit, unit_source, unit_evidence = get_system_value("surfactant_concentration_unit", row)
        self.assertEqual(unit, "mg/mL")
        self.assertEqual(unit_source, "row_identity_surfactant_concentration")
        self.assertEqual(unit_evidence, "supported_direct_row_identity_concentration_unit")

    def test_compare_suppresses_duplicate_unit_when_gt_value_is_combined_concentration(self):
        self.assertTrue(
            should_suppress_duplicate_concentration_unit_cell(
                "emulsifier_stabilizer_concentration_unit",
                {"emulsifier_stabilizer_concentration_value": "0.25% (w/v)"},
                "% w/v",
            )
        )
        self.assertFalse(
            should_suppress_duplicate_concentration_unit_cell(
                "emulsifier_stabilizer_concentration_unit",
                {"emulsifier_stabilizer_concentration_value": "0.25% (w/v)"},
                "mg/mL",
            )
        )

    def test_compare_values_matches_decimal_fraction_to_percent_for_emulsifier_concentration(self):
        strict, relaxed, canonicalized = compare_values(
            "emulsifier_stabilizer_concentration_value",
            "0.005",
            "0.5%",
        )
        self.assertFalse(strict)
        self.assertTrue(relaxed)
        self.assertTrue(canonicalized)

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
            {"field_family": "drug_name", "surface_form": "KGN", "canonical_form": "Kartogenin", "scope": "global", "paper_key": "", "normalization_rule": "exact"},
            {"field_family": "surfactant_name", "surface_form": "Poloxamer", "canonical_form": "Poloxamer", "scope": "global", "paper_key": "", "normalization_rule": "exact"},
            {"field_family": "surfactant_name", "surface_form": "Poloxamer", "canonical_form": "Poloxamer 407", "scope": "paper_local", "paper_key": "UFXX9WXE", "normalization_rule": "exact"},
            {"field_family": "method_type", "surface_form": "solvent displacement method", "canonical_form": "single_emulsion_o_w", "scope": "global", "paper_key": "", "normalization_rule": "casefold_contains"},
        ])
        self.assertEqual(normalize_value_with_lexicon("drug_name", "KGN", lexicon=lexicon), "Kartogenin")
        self.assertEqual(normalize_value_with_lexicon("surfactant_name", "Poloxamer", paper_key="UFXX9WXE", lexicon=lexicon), "Poloxamer 407")
        self.assertEqual(normalize_value_with_lexicon("surfactant_name", "Poloxamer", paper_key="OTHER", lexicon=lexicon), "Poloxamer")
        self.assertEqual(
            normalize_value_with_lexicon("method_type", "prepared by solvent displacement method under stirring", lexicon=lexicon),
            "single_emulsion_o_w",
        )

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
        self.assertEqual(evidence, "typed_contract_not_restrictive")

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

    def test_get_system_value_recovers_solvent_from_row_local_solvent_volume_header(self):
        value, source, evidence = get_system_value(
            "solvent_name",
            {
                "organic_solvent_value_text": "",
                "decision_key_fields_used": "",
                "preparation_method": "nanoprecipitation",
                "change_descriptions": "[\"('Composition', 'Artemether (mg)')=5\", \"('Composition', 'PLGA (mg)')=75\", \"('Composition', 'Acetone (mL)')=5\", \"('Composition', 'Aqueous phase (mL)')=15\"]",
                "supporting_evidence_refs": "",
            },
        )
        self.assertEqual(value, "acetone")
        self.assertEqual(source, "row_local_solvent_volume_header")
        self.assertEqual(evidence, "supported")

    def test_get_system_value_recovers_o_volume_from_row_local_solvent_volume_header(self):
        value, source, evidence = get_system_value(
            "O_volume_mL",
            {
                "organic_phase_volume_mL_value_text": "",
                "decision_key_fields_used": "",
                "preparation_method": "nanoprecipitation",
                "change_descriptions": "[\"('Composition', 'Artemether (mg)')=5\", \"('Composition', 'PLGA (mg)')=75\", \"('Composition', 'Acetone (mL)')=5\", \"('Composition', 'Aqueous phase (mL)')=15\"]",
                "supporting_evidence_refs": "",
            },
        )
        self.assertEqual(value, "5 mL")
        self.assertEqual(source, "row_local_solvent_volume_header")
        self.assertEqual(evidence, "typed_direct_volume_value")

    def test_get_system_value_recovers_external_aqueous_volume_from_row_local_aqueous_phase_header(self):
        value, source, evidence = get_system_value(
            "external_aqueous_phase_volume_mL",
            {
                "external_aqueous_phase_volume_mL_value_text": "",
                "decision_key_fields_used": "",
                "preparation_method": "nanoprecipitation",
                "change_descriptions": "[\"('Composition', 'Artemether (mg)')=5\", \"('Composition', 'PLGA (mg)')=75\", \"('Composition', 'Acetone (mL)')=5\", \"('Composition', 'Aqueous phase (mL)')=15\"]",
                "supporting_evidence_refs": "",
            },
        )
        self.assertEqual(value, "15 mL")
        self.assertEqual(source, "row_local_solvent_volume_header")
        self.assertEqual(evidence, "typed_direct_volume_value")

    def test_get_system_value_does_not_treat_aqueous_phase_volume_as_solvent(self):
        value, source, evidence = get_system_value(
            "solvent_name",
            {
                "organic_solvent_value_text": "",
                "decision_key_fields_used": "",
                "preparation_method": "nanoprecipitation",
                "change_descriptions": "[\"('Composition', 'Aqueous phase (mL)')=15\"]",
                "supporting_evidence_refs": "",
            },
        )
        self.assertEqual(value, "")
        self.assertEqual(source, "direct_extracted")
        self.assertEqual(evidence, "supported")

    def test_get_system_value_does_not_rebind_ambiguous_multiple_solvent_volume_headers(self):
        value, source, evidence = get_system_value(
            "solvent_name",
            {
                "organic_solvent_value_text": "",
                "decision_key_fields_used": "",
                "preparation_method": "nanoprecipitation",
                "change_descriptions": "[\"('Composition', 'Acetone (mL)')=5\", \"('Composition', 'Dichloromethane (mL)')=5\"]",
                "supporting_evidence_refs": "",
            },
        )
        self.assertEqual(value, "")
        self.assertEqual(source, "direct_extracted")
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

    def test_get_system_value_recovers_ee_percent_from_labeled_evidence_span(self):
        value, source, evidence = get_system_value(
            "ee_percent",
            {
                "encapsulation_efficiency_percent_value_text": "",
                "evidence_span_text": "Size (nm)=88.05 ± 2.7 | PDI=0.170 ± 0.05 | Drug loading (DL, %)=9.25 ± 2.8 | Encapsulation efficiency (EE, %)=88.32 ± 3.3",
                "supporting_evidence_refs": "",
            },
        )
        self.assertEqual(value, "88.32 %")
        self.assertEqual(source, "evidence_span_metric_rebinding")
        self.assertEqual(evidence, "supported")

    def test_get_system_value_recovers_ee_percent_from_superscripted_ee_label(self):
        value, source, evidence = get_system_value(
            "ee_percent",
            {
                "encapsulation_efficiency_percent_value_text": "",
                "evidence_span_text": "Diameter (nm)=91.8 ± 2.74 | PIa=0.13 ± 0.01 | ZPb (mV)=−21.23 ± 1.04 | Recovery (%)=91.14 ± 0.28 | Drug content (%)=1.04 ± 0.06 | EEc (%)=57.64 ± 0.97",
                "supporting_evidence_refs": "",
            },
        )
        self.assertEqual(value, "57.64 %")
        self.assertEqual(source, "evidence_span_metric_rebinding")
        self.assertEqual(evidence, "supported")

    def test_get_system_value_does_not_rebind_ee_from_article_level_source_text(self):
        value, source, evidence = get_system_value(
            "ee_percent",
            {
                "encapsulation_efficiency_percent_value_text": "",
                "evidence_span_text": "Article front matter " + ("background text " * 200) + "entrapment efficiency: 1.0 ± 0.1 was reported elsewhere",
                "supporting_evidence_refs": "",
            },
        )
        self.assertEqual(value, "")
        self.assertEqual(source, "direct_extracted")
        self.assertEqual(evidence, "supported")

    def test_get_system_value_recovers_particle_size_from_labeled_evidence_span(self):
        value, source, evidence = get_system_value(
            "particle_size_nm",
            {
                "size_nm_value_text": "",
                "evidence_span_text": "Size (nm)=88.05 ± 2.7 | PDI=0.170 ± 0.05 | Drug loading (DL, %)=9.25 ± 2.8 | Encapsulation efficiency (EE, %)=88.32 ± 3.3",
                "supporting_evidence_refs": "",
            },
        )
        self.assertEqual(value, "88.05")
        self.assertEqual(source, "evidence_span_metric_rebinding")
        self.assertEqual(evidence, "supported")

    def test_get_system_value_recovers_particle_diameter_from_labeled_evidence_span(self):
        value, source, evidence = get_system_value(
            "particle_size_nm",
            {
                "size_nm_value_text": "",
                "evidence_span_text": "Diameter (nm)=91.8 ± 2.74 | PIa=0.13 ± 0.01 | ZPb (mV)=−21.23 ± 1.04 | Recovery (%)=91.14 ± 0.28 | Drug content (%)=1.04 ± 0.06 | EEc (%)=57.64 ± 0.97",
                "supporting_evidence_refs": "",
            },
        )
        self.assertEqual(value, "91.8")
        self.assertEqual(source, "evidence_span_metric_rebinding")
        self.assertEqual(evidence, "supported")

    def test_get_system_value_does_not_rebind_particle_size_from_article_level_source_text(self):
        value, source, evidence = get_system_value(
            "particle_size_nm",
            {
                "size_nm_value_text": "",
                "evidence_span_text": "Article front matter " + ("background text " * 200) + "particle size: 88.05 ± 2.7 was reported elsewhere",
                "supporting_evidence_refs": "",
            },
        )
        self.assertEqual(value, "")
        self.assertEqual(source, "direct_extracted")
        self.assertEqual(evidence, "supported")

    def test_compare_values_matches_decimal_fraction_to_percent_for_ee_percent(self):
        strict, canonicalized, numeric = compare_values("ee_percent", "0.8832", "88.32 %")
        self.assertFalse(strict)
        self.assertTrue(canonicalized)
        self.assertTrue(numeric)

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

    def test_get_system_value_recovers_particle_size_from_source_csv_header_aligned_table_row(self):
        value, source, evidence = get_system_value(
            "particle_size_nm",
            {
                "key": "BB3JUVW7",
                "size_nm_value_text": "",
                "evidence_span_text": "5 | 75 | 75 | 5 | 15 | 190.2 ± 18.0 | 0.06 ± 0.01 | −8.0 ± 0.58 | 78.5 ± 1.8 | 4.9 ± 0.1",
                "supporting_evidence_refs": '[{"source_locator_text":"Table 1::row_01__5","source_region_type":"table_row","supporting_snippet":"5 | 75 | 75 | 5 | 15 | 190.2 ± 18.0 | 0.06 ± 0.01 | −8.0 ± 0.58 | 78.5 ± 1.8 | 4.9 ± 0.1","target_field_name":"(\'Composition\', \'Artemether (mg)\')|(\'Composition\', \'PLGA (mg)\')|(\'Composition\', \'PVA (mg)\')|(\'Composition\', \'Acetone (mL)\')|(\'Composition\', \'Aqueous phase (mL)\')"}]',
            },
            paper_key="BB3JUVW7",
        )
        self.assertEqual(value, "190.2")
        self.assertEqual(source, "source_csv_header_metric_rebinding")
        self.assertEqual(evidence, "supported")

    def test_get_system_value_recovers_ee_from_source_csv_header_aligned_table_row(self):
        value, source, evidence = get_system_value(
            "ee_percent",
            {
                "key": "BB3JUVW7",
                "encapsulation_efficiency_percent_value_text": "",
                "evidence_span_text": "5 | 75 | 150 | 5 | 15 | 214.3 ± 6.2 | 0.238 | −1.8 ± 0.22 | 80.0 ± 0.1 | 5.0 ± 0.0",
                "supporting_evidence_refs": '[{"source_locator_text":"Table 1::row_02__5","source_region_type":"table_row","supporting_snippet":"5 | 75 | 150 | 5 | 15 | 214.3 ± 6.2 | 0.238 | −1.8 ± 0.22 | 80.0 ± 0.1 | 5.0 ± 0.0","target_field_name":"(\'Composition\', \'Artemether (mg)\')|(\'Composition\', \'PLGA (mg)\')|(\'Composition\', \'PVA (mg)\')|(\'Composition\', \'Acetone (mL)\')|(\'Composition\', \'Aqueous phase (mL)\')"}]',
            },
            paper_key="BB3JUVW7",
        )
        self.assertEqual(value, "80.0 %")
        self.assertEqual(source, "source_csv_header_metric_rebinding")
        self.assertEqual(evidence, "supported")

    def test_get_system_value_prefers_stage2_table_cell_binding_over_source_csv_rebinding(self):
        value, source, evidence = get_system_value(
            "ee_percent",
            {
                "key": "BB3JUVW7",
                "encapsulation_efficiency_percent_value_text": "",
                "evidence_span_text": "5 | 75 | 150 | 5 | 15 | 214.3 ± 6.2 | 0.238 | −1.8 ± 0.22 | 80.0 ± 0.1 | 5.0 ± 0.0",
                "supporting_evidence_refs": '[{"source_locator_text":"Table 1::row_02__5","source_region_type":"table_row","supporting_snippet":"5 | 75 | 150 | 5 | 15 | 214.3 ± 6.2 | 0.238 | −1.8 ± 0.22 | 80.0 ± 0.1 | 5.0 ± 0.0","target_field_name":"(\'Composition\', \'Artemether (mg)\')|(\'Composition\', \'PLGA (mg)\')|(\'Composition\', \'PVA (mg)\')|(\'Composition\', \'Acetone (mL)\')|(\'Composition\', \'Aqueous phase (mL)\')"}]',
                "table_cell_bindings_json": json.dumps(
                    [
                        {
                            "canonical_field": "ee_percent",
                            "raw_header": "%EE",
                            "raw_cell_value": "79.5 ± 1.8",
                            "binding_rule": "stage2_header_alias_cell_binding",
                            "ambiguity_status": "unique_header_cell",
                            "source_csv_path": "data/cleaned/goren_2025/tables/BB3JUVW7/BB3JUVW7__table_01__html_table.csv",
                            "source_row_index": "2",
                            "source_column_index": "8",
                        }
                    ]
                ),
            },
            paper_key="BB3JUVW7",
        )
        self.assertEqual(value, "79.5 %")
        self.assertEqual(source, "stage2_table_cell_binding_authority")
        self.assertEqual(evidence, "typed_direct_percent_value")

    def test_get_system_value_does_not_rebind_particle_size_when_source_csv_has_multiple_size_headers(self):
        value, source, evidence = get_system_value(
            "particle_size_nm",
            {
                "key": "BB3JUVW7",
                "size_nm_value_text": "",
                "evidence_span_text": "100 | 75:25 | 4x | Acetone | 15 | 234.1 ± 61.7 | 61.3 ± 8.7 | 3.8 ± 0.8 | 237.2 ± 61.9 | 77.8 ± 13.3 | 1.8",
                "supporting_evidence_refs": '[{"source_locator_text":"Table 2::row_01__100","source_region_type":"table_row","supporting_snippet":"100 | 75:25 | 4x | Acetone | 15 | 234.1 ± 61.7 | 61.3 ± 8.7 | 3.8 ± 0.8 | 237.2 ± 61.9 | 77.8 ± 13.3 | 1.8"}]',
            },
            paper_key="BB3JUVW7",
        )
        self.assertEqual(value, "")
        self.assertEqual(source, "direct_extracted")
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
        self.assertEqual(stabilizer_value, "")
        self.assertEqual(stabilizer_source, "missing_system_field_surface")

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

    def test_get_system_value_reverses_named_ratio_direction_for_opposite_target_field(self):
        value, source, evidence = get_system_value(
            "polymer_to_drug_ratio_raw",
            {
                "key": "GENERIC_RATIO",
                "raw_formulation_label": "ITZ:PLGA ratio=10:1",
            },
            paper_key="GENERIC_RATIO",
        )
        self.assertEqual(value, "1:10")
        self.assertEqual(source, "ratio_label_token_rebinding")
        self.assertEqual(evidence, "supported")

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

    def test_compare_values_reverses_named_la_ga_ratio_generically(self):
        strict, relaxed, canonicalized = compare_values(
            "la_ga_ratio_raw",
            "LA:GA ratio=75:25",
            "GA:LA ratio=25:75",
        )
        self.assertFalse(strict)
        self.assertTrue(relaxed)
        self.assertTrue(canonicalized)

    def test_compare_values_reverses_direction_neutral_named_phase_ratio(self):
        strict, relaxed, canonicalized = compare_values(
            "phase_ratio_raw",
            "water:oil ratio=1:2",
            "oil:water ratio=2:1",
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

    def test_ordinal_grid_semantics_is_scoped_to_its_declared_table_authority(self):
        polymer_mass, mass_source, evidence = get_system_value(
            "polymer_mass_mg",
            {
                "key": "PA3SPZ28",
                "raw_formulation_label": "1:10 / 50:50",
                "representative_source_formulation_id": "PA3SPZ28__table_1__1:10_/_50:50",
                "evidence_span_text": "Size (nm)=88.05 ± 2.7 | PDI=0.170 ± 0.05",
            },
            paper_key="PA3SPZ28",
        )
        self.assertEqual(polymer_mass, "")
        self.assertNotEqual(mass_source, "ordinal_grid_semantics")
        self.assertEqual(evidence, "missing_system_field_surface")

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
        self.assertEqual(evidence, "typed_contract_not_restrictive")

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

    def test_get_system_value_uses_decoded_structured_table_override_for_f_prefixed_doe_rows(self):
        row = {
            "key": "WIVUCMYG",
            "formulation_id": "WIVUCMYG_DOE_Row_F1",
            "raw_formulation_label": "F1",
            "preparation_method": "emulsion solvent evaporation",
            "polymer_name_raw": "PLGA",
            "surfactant_concentration_text_value_text": "10.0",
            "drug_feed_amount_text_value_text": "0.5",
            "evidence_span_text": "F1 | −1 | −1 | −1 | −1 | 566.50 ± 8.05 | 0.286 ± 0.02 | −8.28 ± 0.71 | 87.69 ± 0.13",
        }
        drug_conc_value, drug_conc_source, _ = get_system_value("drug_concentration_value", row, paper_key="WIVUCMYG")
        drug_unit_value, drug_unit_source, _ = get_system_value("drug_concentration_unit", row, paper_key="WIVUCMYG")
        drug_mass_value, drug_mass_source, _ = get_system_value("drug_mass_mg", row, paper_key="WIVUCMYG")
        polymer_mass_value, polymer_mass_source, _ = get_system_value("polymer_mass_mg", row, paper_key="WIVUCMYG")
        self.assertEqual(drug_conc_value, "0.5")
        self.assertEqual(drug_conc_source, "structured_table_rebinding")
        self.assertEqual(drug_unit_value, "mg/mL")
        self.assertEqual(drug_unit_source, "structured_table_rebinding")
        self.assertEqual(drug_mass_value, "")
        self.assertEqual(drug_mass_source, "structured_table_rebinding")
        self.assertEqual(polymer_mass_value, "")
        self.assertEqual(polymer_mass_source, "structured_table_rebinding")

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

    def test_choose_system_row_uses_sr_no_seed_ordinal_bridge_for_bare_numeric_final_label(self):
        gt_row = {
            "paper_key": "P4B",
            "gt_formulation_id": "P4B_G010",
            "seed_pred_representative_source_formulation_id": "F_SrNo10",
            "formulation_label": "Sr. No. 10",
            "polymer_name": "",
            "drug_name": "",
            "drug_mass_mg": "",
            "la_ga_ratio_normalized": "",
        }
        system_rows = [
            {
                "key": "P4B",
                "final_formulation_id": "P4B__fo__10",
                "formulation_id": "P4B_DOE_Row_10",
                "representative_source_formulation_id": "",
                "parent_core_row_id": "P4B_DOE_Row_10",
                "raw_formulation_label": "10",
                "polymer_identity_final": "",
                "polymer_name_raw": "",
                "drug_name_value_text": "",
                "drug_feed_amount_text_value_text": "",
                "la_ga_ratio_value_text": "",
            }
        ]
        row, rule, ok = choose_system_row(gt_row, system_rows)
        self.assertTrue(ok)
        self.assertEqual(rule, "scaffold_parent_ordinal_bridge")
        self.assertEqual(row["formulation_id"], "P4B_DOE_Row_10")

    def test_choose_system_row_keeps_duplicate_ordinal_candidates_blocked(self):
        gt_row = {
            "paper_key": "P4C",
            "gt_formulation_id": "P4C_G010",
            "seed_pred_representative_source_formulation_id": "F_SrNo10",
            "formulation_label": "Sr. No. 10",
            "polymer_name": "",
            "drug_name": "",
            "drug_mass_mg": "",
            "la_ga_ratio_normalized": "",
        }
        system_rows = [
            {
                "key": "P4C",
                "final_formulation_id": "P4C__fo__10a",
                "formulation_id": "P4C_DOE_Row_10",
                "representative_source_formulation_id": "",
                "parent_core_row_id": "P4C_DOE_Row_10",
                "raw_formulation_label": "10",
                "polymer_identity_final": "",
                "polymer_name_raw": "",
                "drug_name_value_text": "",
                "drug_feed_amount_text_value_text": "",
                "la_ga_ratio_value_text": "",
            },
            {
                "key": "P4C",
                "final_formulation_id": "P4C__fo__10b",
                "formulation_id": "P4C__table_13__10",
                "representative_source_formulation_id": "",
                "parent_core_row_id": "P4C__table_13__10",
                "raw_formulation_label": "10",
                "polymer_identity_final": "",
                "polymer_name_raw": "",
                "drug_name_value_text": "",
                "drug_feed_amount_text_value_text": "",
                "la_ga_ratio_value_text": "",
            },
        ]
        row, rule, ok = choose_system_row(gt_row, system_rows)
        self.assertFalse(ok)
        self.assertEqual(rule, "no_unique_alignment")
        self.assertIsNone(row)

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

    def test_choose_system_row_uses_row_local_design_signature_bridge(self):
        gt_row = {
            "paper_key": "P7B",
            "gt_formulation_id": "P7B_G006",
            "seed_pred_representative_source_formulation_id": "",
            "formulation_label": "100 75:25 4x Acetone 15",
            "polymer_name": "",
            "drug_name": "",
            "drug_mass_mg": "",
            "la_ga_ratio_normalized": "",
        }
        system_rows = [
            {
                "key": "P7B",
                "final_formulation_id": "P7B__fo__1",
                "formulation_id": "P7B__table_2__row_01_100",
                "representative_source_formulation_id": "P7B__table_2__row_01_100",
                "parent_core_row_id": "P7B__table_2__row_01_100",
                "raw_formulation_label": "row_01__100",
                "evidence_span_text": "100 | 75:25 | 4x | Acetone | 15 | 234.1 ± 61.7",
                "polymer_identity_final": "",
                "polymer_name_raw": "",
                "drug_name_value_text": "",
                "drug_feed_amount_text_value_text": "",
                "la_ga_ratio_value_text": "",
            },
            {
                "key": "P7B",
                "final_formulation_id": "P7B__fo__2",
                "formulation_id": "P7B__table_2__row_02_150",
                "representative_source_formulation_id": "P7B__table_2__row_02_150",
                "parent_core_row_id": "P7B__table_2__row_02_150",
                "raw_formulation_label": "row_02__150",
                "evidence_span_text": "150 | 75:25 | 4x | Acetone | 15 | 295.1 ± 64.9",
                "polymer_identity_final": "",
                "polymer_name_raw": "",
                "drug_name_value_text": "",
                "drug_feed_amount_text_value_text": "",
                "la_ga_ratio_value_text": "",
            },
        ]
        row, rule, ok = choose_system_row(gt_row, system_rows)
        self.assertTrue(ok)
        self.assertEqual(rule, "row_local_design_signature_bridge")
        self.assertEqual(row["formulation_id"], "P7B__table_2__row_01_100")

    def test_choose_system_row_uses_uppercase_method_token_to_disambiguate_design_signature(self):
        gt_row = {
            "paper_key": "P7B2",
            "gt_formulation_id": "P7B2_G006",
            "seed_pred_representative_source_formulation_id": "",
            "formulation_label": "100 75:25 4x Acetone 15",
            "polymer_name": "",
            "drug_name": "",
            "drug_mass_mg": "",
            "la_ga_ratio_normalized": "",
        }
        system_rows = [
            {
                "key": "P7B2",
                "final_formulation_id": "P7B2__fo__1",
                "formulation_id": "P7B2__table_2__row_01_100",
                "raw_formulation_label": "row_01__100",
                "evidence_span_text": "100 | 75:25 | 4x | Acetone | 15 | 234.1 ± 61.7",
                "polymer_identity_final": "",
                "polymer_name_raw": "",
                "drug_name_value_text": "",
                "drug_feed_amount_text_value_text": "",
                "la_ga_ratio_value_text": "",
            },
            {
                "key": "P7B2",
                "final_formulation_id": "P7B2__fo__7",
                "formulation_id": "P7B2__table_2__row_07_100",
                "raw_formulation_label": "row_07__100",
                "evidence_span_text": "100 | 75:25 | 4x | Heat* | 15 | 510.7 ± 114.6",
                "polymer_identity_final": "",
                "polymer_name_raw": "",
                "drug_name_value_text": "",
                "drug_feed_amount_text_value_text": "",
                "la_ga_ratio_value_text": "",
            },
        ]
        row, rule, ok = choose_system_row(gt_row, system_rows)
        self.assertTrue(ok)
        self.assertEqual(rule, "row_local_design_signature_bridge")
        self.assertEqual(row["formulation_id"], "P7B2__table_2__row_01_100")

    def test_choose_system_row_does_not_use_incomplete_design_signature_bridge_when_ambiguous(self):
        gt_row = {
            "paper_key": "P7C",
            "gt_formulation_id": "P7C_G006",
            "seed_pred_representative_source_formulation_id": "",
            "formulation_label": "100 75:25 Acetone 15",
            "polymer_name": "",
            "drug_name": "",
            "drug_mass_mg": "",
            "la_ga_ratio_normalized": "",
        }
        system_rows = [
            {
                "key": "P7C",
                "final_formulation_id": "P7C__fo__1",
                "formulation_id": "P7C__table_2__row_01_100",
                "raw_formulation_label": "row_01__100",
                "evidence_span_text": "100 | 75:25 | 4x | Acetone | 15",
                "polymer_identity_final": "",
                "polymer_name_raw": "",
                "drug_name_value_text": "",
                "drug_feed_amount_text_value_text": "",
                "la_ga_ratio_value_text": "",
            },
            {
                "key": "P7C",
                "final_formulation_id": "P7C__fo__4",
                "formulation_id": "P7C__table_2__row_04_100",
                "raw_formulation_label": "row_04__100",
                "evidence_span_text": "100 | 75:25 | 2x | Acetone | 15",
                "polymer_identity_final": "",
                "polymer_name_raw": "",
                "drug_name_value_text": "",
                "drug_feed_amount_text_value_text": "",
                "la_ga_ratio_value_text": "",
            },
        ]
        row, rule, ok = choose_system_row(gt_row, system_rows)
        self.assertFalse(ok)
        self.assertEqual(rule, "no_unique_alignment")
        self.assertIsNone(row)

    def test_choose_system_row_uses_loaded_state_disambiguation_bridge(self):
        gt_row = {
            "paper_key": "P8",
            "gt_formulation_id": "P8_G001",
            "seed_pred_representative_source_formulation_id": "Formulation_1",
            "formulation_label": "AP-PLGA-NPs",
            "polymer_name": "PLGA",
            "drug_name": "Acetylpuerarin",
            "drug_mass_mg": "7 mg",
            "ee_percent": "90.51 %",
            "la_ga_ratio_normalized": "",
        }
        system_rows = [
            {
                "key": "P8",
                "final_formulation_id": "P8__fo__loaded",
                "formulation_id": "P8__table_14__ap-plga-nps_/_drug_loaded",
                "raw_formulation_label": "AP-PLGA-NPs / Drug loaded",
                "decision_key_fields_used": '{"loaded_state": "drug_loaded", "identity_variables": "drug_identity=acetylpuerarin|polymer_identity=plga"}',
                "polymer_identity_final": "PLGA",
                "polymer_name_raw": "",
                "drug_name_value_text": "",
                "drug_feed_amount_text_value_text": "",
                "la_ga_ratio_value_text": "",
            },
            {
                "key": "P8",
                "final_formulation_id": "P8__fo__empty",
                "formulation_id": "P8__table_14__ap-plga-nps_/_empty",
                "raw_formulation_label": "AP-PLGA-NPs / Empty",
                "decision_key_fields_used": '{"loaded_state": "empty", "identity_variables": "polymer_identity=plga"}',
                "polymer_identity_final": "PLGA",
                "polymer_name_raw": "",
                "drug_name_value_text": "",
                "drug_feed_amount_text_value_text": "",
                "la_ga_ratio_value_text": "",
            },
        ]
        row, rule, ok = choose_system_row(gt_row, system_rows)
        self.assertTrue(ok)
        self.assertEqual(rule, "loaded_state_disambiguation_bridge")
        self.assertEqual(row["formulation_id"], "P8__table_14__ap-plga-nps_/_drug_loaded")

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


class S53PromptContractTests(unittest.TestCase):
    def test_s5_3_prompt_targets_gap_fill_surface_by_default(self):
        self.assertEqual(s5_3a.DEFAULT_TARGET_MODE, "missing_system")
        self.assertEqual(s5_3a.TARGET_FIELDS, s5_3a.S5_3_RESPONSIBLE_FIELDS)
        self.assertTrue(set(s5_3a.TARGET_FIELDS).issubset(set(SYSTEM_FIELD_MAP)))

    def test_s5_3_default_excludes_s5_2_mechanical_table_cell_fields(self):
        s5_2_mechanical_fields = {
            "drug_name",
            "polymer_name",
            "polymer_mass_mg",
            "drug_mass_mg",
            "O_volume_mL",
            "external_aqueous_phase_volume_mL",
            "surfactant_name",
            "particle_size_nm",
            "pdi",
            "zeta_mV",
            "ee_percent",
        }
        self.assertFalse(s5_2_mechanical_fields & set(s5_3a.TARGET_FIELDS))
        self.assertIn("method_type", s5_3a.TARGET_FIELDS)
        self.assertIn("polymer_mw_raw", s5_3a.TARGET_FIELDS)
        self.assertIn("solvent_name", s5_3a.TARGET_FIELDS)

    def test_s5_3_default_missing_system_mode_omits_already_filled_values(self):
        rows = [{"final_formulation_id": "P__F1", "method_type_value_text": "nanoprecipitation", "polymer_mw_raw_value_text": "10 kDa", "loading_content_percent_value_text": "7.5", "organic_solvent_value_text": ""}]
        selected = s5_3a.select_target_fields_for_rows(rows, [], s5_3a.DEFAULT_TARGET_MODE)
        by_row = s5_3a.select_target_fields_by_formulation_id(rows, [], s5_3a.DEFAULT_TARGET_MODE)
        self.assertNotIn("method_type", selected)
        self.assertNotIn("polymer_mw_raw", selected)
        self.assertNotIn("lc_percent", selected)
        self.assertIn("solvent_name", selected)
        self.assertNotIn("method_type", by_row["P__F1"])
        self.assertNotIn("lc_percent", by_row["P__F1"])
        self.assertIn("solvent_name", by_row["P__F1"])

    def test_s5_3_missing_system_targets_are_row_specific_not_chunk_wide_duplicates(self):
        rows = [
            {"final_formulation_id": "P__F1", "method_type_value_text": "nanoprecipitation"},
            {"final_formulation_id": "P__F2", "method_type_value_text": ""},
        ]
        selected = s5_3a.select_target_fields_for_rows(rows, ["method_type"], s5_3a.DEFAULT_TARGET_MODE)
        by_row = s5_3a.select_target_fields_by_formulation_id(rows, ["method_type"], s5_3a.DEFAULT_TARGET_MODE)
        self.assertEqual(selected, ["method_type"])
        self.assertEqual(by_row["P__F1"], [])
        self.assertEqual(by_row["P__F2"], ["method_type"])

    def test_s5_3_missing_system_treats_resolved_not_reported_as_not_a_gap(self):
        rows = [{"final_formulation_id": "P__F1", "polymer_concentration_unit_missing_reason": "not_reported"}]
        selected = s5_3a.select_target_fields_for_rows(rows, ["polymer_concentration_unit"], s5_3a.DEFAULT_TARGET_MODE)
        by_row = s5_3a.select_target_fields_by_formulation_id(rows, ["polymer_concentration_unit"], s5_3a.DEFAULT_TARGET_MODE)
        self.assertEqual(selected, [])
        self.assertEqual(by_row["P__F1"], [])

    def test_s5_3_missing_system_does_not_target_fields_without_final_table_surface(self):
        rows = [{"final_formulation_id": "P__F1"}]
        selected = s5_3a.select_target_fields_for_rows(rows, ["sonication_time_s"], s5_3a.DEFAULT_TARGET_MODE)
        by_row = s5_3a.select_target_fields_by_formulation_id(rows, ["sonication_time_s"], s5_3a.DEFAULT_TARGET_MODE)
        self.assertEqual(selected, [])
        self.assertEqual(by_row["P__F1"], [])

    def test_s5_3_missing_system_targets_unresolved_not_projectable_reason(self):
        rows = [{"final_formulation_id": "P__F1", "polymer_concentration_unit_missing_reason": "not_projectable_from_current_replacement_objects"}]
        selected = s5_3a.select_target_fields_for_rows(rows, ["polymer_concentration_unit"], s5_3a.DEFAULT_TARGET_MODE)
        by_row = s5_3a.select_target_fields_by_formulation_id(rows, ["polymer_concentration_unit"], s5_3a.DEFAULT_TARGET_MODE)
        self.assertEqual(selected, ["polymer_concentration_unit"])
        self.assertEqual(by_row["P__F1"], ["polymer_concentration_unit"])

    def test_s5_3_all_mode_does_not_reinclude_s5_2_filled_mechanical_values(self):
        rows = [{"drug_name_value_text": "paclitaxel", "size_nm_value_text": "100"}]
        selected = s5_3a.select_target_fields_for_rows(rows, [], "all")
        self.assertNotIn("drug_name", selected)
        self.assertNotIn("particle_size_nm", selected)
        self.assertIn("method_type", selected)

    def test_s5_3_row_offset_chunks_fixed_rows_without_field_selection(self):
        rows = [
            {"paper_key": "P", "final_formulation_id": "P__F1"},
            {"paper_key": "P", "final_formulation_id": "P__F2"},
            {"paper_key": "P", "final_formulation_id": "P__F3"},
        ]
        selected = s5_3a.select_final_rows(rows, "P", max_rows=1, row_offset=1)
        self.assertEqual([row["final_formulation_id"] for row in selected], ["P__F2"])


    def test_s5_3_uses_stage2_clean_evidence_blocks_before_raw_fallback(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            evidence_dir = root / "evidence_blocks" / "P"
            evidence_dir.mkdir(parents=True)
            evidence_path = evidence_dir / "evidence_blocks_v1.json"
            evidence_path.write_text(
                json.dumps(
                    {
                        "contract_version": "s2_2_evidence_blocks_v1",
                        "segmentation_profile": "section_aware_candidate_segmentation_v1",
                        "selection_mode": "evidence_priority_v1",
                        "evidence_blocks": [
                            {
                                "block_id": "P__method__01",
                                "block_type": "method",
                                "evidence_kind": "method",
                                "source_type": "clean_text_paragraph",
                                "origin_locator": "clean#method",
                                "text_content": "Downloaded from https://academic.oup.com/jpp/article/67/12/1650/6127230 by University user\nNanoparticles were prepared by nanoprecipitation using acetone.",
                                "is_table_derived": False,
                                "noise_flags": [],
                                "quality_flags": [],
                            },
                            {
                                "block_id": "P__front__01",
                                "block_type": "front_matter",
                                "evidence_kind": "metadata",
                                "source_type": "clean_text_paragraph",
                                "origin_locator": "clean#front",
                                "text_content": "Department of Noise University",
                                "is_table_derived": False,
                                "noise_flags": [],
                                "quality_flags": ["front_matter"],
                            },
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            raw_text = root / "raw.txt"
            raw_text.write_text("Journals & Books\nDownload full issue\nRAW NAVIGATION NOISE", encoding="utf-8")
            evidence, audit_rows, metadata = s5_3a.build_source_evidence(
                "P",
                raw_text,
                None,
                max_text_chars=10000,
                max_table_chars=10000,
                evidence_blocks_path=evidence_path,
            )
        self.assertIn("SOURCE_TEXT_AUTHORITY: stage2_clean_evidence_blocks", evidence)
        self.assertIn("BEGIN_STAGE2_EVIDENCE_BLOCK: P__method__01", evidence)
        self.assertIn("Nanoparticles were prepared by nanoprecipitation", evidence)
        self.assertNotIn("RAW NAVIGATION NOISE", evidence)
        self.assertNotIn("Department of Noise University", evidence)
        self.assertNotIn("Downloaded from", evidence)
        self.assertNotIn("academic.oup.com", evidence)
        self.assertEqual(metadata["evidence_source_mode"], "stage2_clean_evidence_blocks")
        self.assertEqual(metadata["stage2_prompt_noise_lines_removed"], 1)
        self.assertEqual(len(audit_rows), 1)
        self.assertEqual(audit_rows[0]["removed_prompt_noise_lines"], 1)

    def test_s5_4_rejects_s5_3_excluded_mechanical_fields_even_when_quoted_direct(self):
        decision, reason, review = s5_4.evaluate_candidate(
            {
                "field_name": "surfactant_name",
                "direct_or_derived": "direct",
                "source_quote": "aqueous phase containing surfactant (poloxamer 407)",
                "evidence_scope": "paper_global_unique",
            }
        )
        self.assertEqual(decision, "rejected")
        self.assertEqual(reason, "s5_3_excluded_mechanical_field_not_allowed")
        self.assertEqual(review, "no")


class MinimalPlusSharedSemanticsTests(unittest.TestCase):
    def test_stage3_resolves_arbitrary_shared_parameter_without_field_whitelist(self):
        relation_rows = [
            {
                "relation_row_id": "r1",
                "relation_type": "method_group_shared_field",
                "formulation_candidate_id": "",
                "method_group_id": "method_1",
                "field_name": "shared_param__aqueous_phase_ph",
                "field_value_raw": "7.4",
                "field_value_norm": "7.4",
                "deterministic_confidence": "high",
            }
        ]
        resolved = build_resolved_relation_fields_for_paper(
            paper_key="P1",
            candidate_items=[{"formulation_candidate_id": "P1__row1", "method_group_id": "method_1"}],
            relation_rows=relation_rows,
        )
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]["field_name"], "shared_param__aqueous_phase_ph")
        self.assertEqual(resolved[0]["field_value"], "7.4")
        self.assertEqual(resolved[0]["resolution_rule"], "method_group_shared_field")

    def test_stage5_materializes_arbitrary_resolved_field_to_bundle_or_shared_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "resolved_relation_fields_v1.tsv"
            path.write_text(
                "formulation_candidate_id\tpaper_key\tmethod_group_id\tscope_type\tfield_name\tfield_value\tfield_value_norm\tresolution_rule\tsource_relation_row_ids\tdeterministic_confidence\n"
                "P1__row1\tP1\tmethod_1\tmethod_group\taqueous_phase_pH\t7.4\t7.4\tmethod_group_shared_field\t1\thigh\n"
                "P1__row1\tP1\tmethod_1\tmethod_group\tshared_param__stirring_speed_rpm\t1200 rpm\t1200_rpm\tmethod_group_shared_field\t2\thigh\n",
                encoding="utf-8",
            )
            resolved = load_resolved_relation_fields(path)

        materialized, applied = apply_resolved_relation_fields(
            final_row={
                "aqueous_phase_pH_value": "",
                "aqueous_phase_pH_value_text": "",
                "aqueous_phase_pH_scope": "",
                "aqueous_phase_pH_evidence_region_type": "",
                "shared_parameters_json": "",
            },
            representative={"formulation_id": "P1__row1"},
            resolved_field_map=resolved,
        )
        self.assertIn("aqueous_phase_pH", applied)
        self.assertIn("shared_param__stirring_speed_rpm", applied)
        self.assertEqual(materialized["aqueous_phase_pH_value_text"], "")
        shared_params = json.loads(materialized["shared_parameters_json"])
        shared_by_name = {item["field_name"]: item for item in shared_params}
        self.assertEqual(shared_by_name["aqueous_phase_pH"]["field_value"], "7.4")
        self.assertEqual(shared_by_name["shared_param__stirring_speed_rpm"]["field_value"], "1200 rpm")
        self.assertEqual(shared_by_name["shared_param__stirring_speed_rpm"]["scope_type"], "method_group")

    def test_stage3_shared_numeric_relation_fields_are_materialized_and_typed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "resolved_relation_fields_v1.tsv"
            path.write_text(
                "formulation_candidate_id\tpaper_key\tmethod_group_id\tscope_type\tfield_name\tfield_value\tfield_value_norm\tresolution_rule\tsource_relation_row_ids\tdeterministic_confidence\n"
                "P1__row1\tP1\tmethod_1\tmethod_group\tsurfactant_concentration_text\t0.5% (w/v)\t0.5% (w/v)\tmethod_group_shared_field\t1\thigh\n"
                "P1__row1\tP1\tmethod_1\tmethod_group\tplga_mass_mg\tPLGA\tPLGA\tmethod_group_shared_field\t2\thigh\n"
                "P1__row1\tP1\tmethod_1\tmethod_group\tdrug_feed_amount_text\tDexibuprofen\tDexibuprofen\tmethod_group_shared_field\t3\thigh\n"
                "P1__row1\tP1\tmethod_1\tmethod_group\tdrug_name\tDexibuprofen\tDexibuprofen\tmethod_group_shared_field\t4\thigh\n",
                encoding="utf-8",
            )
            resolved = load_resolved_relation_fields(path)

        final_row = {
            "surfactant_concentration_text_value": "",
            "surfactant_concentration_text_value_text": "",
            "surfactant_concentration_text_scope": "",
            "surfactant_concentration_text_evidence_region_type": "",
            "plga_mass_mg_value": "",
            "plga_mass_mg_value_text": "",
            "drug_feed_amount_text_value": "",
            "drug_feed_amount_text_value_text": "",
            "drug_name_value": "",
            "drug_name_value_text": "",
        }
        materialized, applied = apply_resolved_relation_fields(
            final_row=final_row,
            representative={"formulation_id": "P1__row1"},
            resolved_field_map=resolved,
        )
        self.assertIn("surfactant_concentration_text", applied)
        self.assertIn("drug_name", applied)
        self.assertNotIn("plga_mass_mg", applied)
        self.assertNotIn("drug_feed_amount_text", applied)
        self.assertEqual(materialized["surfactant_concentration_text_value_text"], "0.5% (w/v)")
        self.assertEqual(materialized["surfactant_concentration_text_scope"], "global_shared")
        self.assertEqual(materialized["surfactant_concentration_text_evidence_region_type"], "relation_resolved")
        self.assertEqual(materialized["drug_name_value_text"], "Dexibuprofen")
        self.assertEqual(materialized["plga_mass_mg_value_text"], "")
        self.assertEqual(materialized["drug_feed_amount_text_value_text"], "")

    def test_stage5_uses_material_alias_binding_for_method_shared_direct_masses(self):
        row = {
            "key": "MVBIND1",
            "formulation_id": "MVBIND1_Row_1",
            "raw_formulation_label": "F1",
            "loaded_state_final": "loaded",
            "drug_name_value": "curcumin",
            "drug_name_value_text": "curcumin",
            "plga_mass_mg_value": "",
            "plga_mass_mg_value_text": "",
            "plga_mass_mg_scope": "",
            "plga_mass_mg_membership_confidence": "",
            "plga_mass_mg_evidence_region_type": "",
            "plga_mass_mg_missing_reason": "not_reported",
            "drug_feed_amount_text_value": "",
            "drug_feed_amount_text_value_text": "",
            "drug_feed_amount_text_scope": "",
            "drug_feed_amount_text_membership_confidence": "",
            "drug_feed_amount_text_evidence_region_type": "",
            "drug_feed_amount_text_missing_reason": "not_reported",
        }
        source_text = (
            "The drug curcumin (CUR) was used for all formulations. "
            "Nanoparticles were prepared by dissolving 100 mg of PLGA and 10 mg of CUR "
            "in acetone before addition to the aqueous phase."
        )
        materialized, applied = apply_global_preparation_material_carrythrough(
            final_row=row,
            source_text=source_text,
        )
        self.assertIn("plga_mass_mg", applied)
        self.assertIn("drug_feed_amount_text", applied)
        self.assertEqual(materialized["plga_mass_mg_value_text"], "100 mg")
        self.assertEqual(materialized["drug_feed_amount_text_value_text"], "10 mg")
        self.assertEqual(materialized["plga_mass_mg_evidence_region_type"], "material_value_binding_direct_text")
        self.assertEqual(materialized["drug_feed_amount_text_evidence_region_type"], "material_value_binding_direct_text")

    def test_centrifugation_collection_fields_are_not_core_compare_fields(self):
        self.assertNotIn("centrifugation_time_min", CORE_FIXED_FIELDS)
        self.assertNotIn("centrifugation_g", CORE_FIXED_FIELDS)
        cells, _ = build_cells(
            gt_rows=[
                {
                    "paper_key": "P1",
                    "doi": "10.1/example",
                    "gt_formulation_id": "P1_G001",
                    "formulation_label": "F1",
                    "gt_row_decision": "include_gt",
                    "benchmark_default_include": "yes",
                    "centrifugation_time_min": "30",
                    "centrifugation_g": "15000",
                }
            ],
            system_rows=[{"key": "P1", "doi": "10.1/example", "formulation_id": "P1__F1", "raw_formulation_label": "F1"}],
            alignment_scaffold_index={"P1_G001": {"gt_formulation_id": "P1_G001", "pred_row_id": "P1__F1"}},
            value_normalization_lexicon={},
        )
        self.assertFalse({"centrifugation_time_min", "centrifugation_g"} & {row["field_name"] for row in cells})

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

    def test_shrunken_context_inheritance_marker_survives_projection(self):
        shrunken_document = {
            "paper_key": "QLYKLPKT",
            "document_key": "QLYKLPKT",
            "doi": "10.1/example",
            "title": "Example",
            "source_mode": "saved_raw_live_v2_replay_to_stage2_v2",
            "table_scopes": [{"table_id": "Table 2", "scope_kind": "sequential_child", "is_formulation_bearing": True, "is_doe": False}],
            "semantic_signals": {
                "has_variable_sweep": True,
                "has_sequential_optimization": True,
                "primary_preparation_method_hint": "emulsion solvent evaporation",
                "primary_variable_names": ["surfactant concentration"],
                "selected_condition_hints": ["10:1 ratio selected and held fixed"],
            },
            "formulation_candidates": [
                {
                    "candidate_id": "QLYKLPKT_F2",
                    "candidate_kind": "single_formulation",
                    "source_table_id": "Table 2",
                    "label_hint": "F2",
                    "instance_role": "synthesis_core",
                    "status": "reported",
                    "confidence": "high",
                }
            ],
            "shared_semantics": {},
            "context_inheritance_markers": [
                {
                    "source_context_label": "base PLGA-ITZ nanosphere preparation",
                    "source_candidate_label_hint": "F1",
                    "source_table_id": "Table 1",
                    "target_contexts": [{"target_group_label": "surfactant concentration optimization", "target_table_id": "Table 2", "variation_axis": "surfactant concentration"}],
                    "inherited_fields": [{"field_name": "surfactant_name", "field_value": "poloxamer 188", "inheritance_basis": "shared_preparation_context", "confidence": "high"}],
                    "held_fixed_conditions": [{"field_name": "polymer_to_drug_ratio_raw", "field_value": "10:1", "inheritance_basis": "selected_as_optimal_then_fixed", "confidence": "high"}],
                    "evidence_cue": "The optimized ratio was used for the following surfactant concentration optimization.",
                    "evidence_source_hint": "methods text near Table 1/Table 2",
                    "confidence": "high",
                }
            ],
        }
        normalized = normalize_stage2_document_for_projection(shrunken_document)
        rows, _, _, _, _ = project_document(normalized)
        self.assertEqual(len(rows), 1)
        markers = json.loads(rows[0]["context_inheritance_markers_json"])
        self.assertEqual(markers[0]["inherited_fields"][0]["field_value"], "poloxamer 188")
        self.assertEqual(markers[0]["marker_readiness"], "execution_ready")

    def test_stage3_materializes_context_inheritance_marker_into_resolved_fields(self):
        marker = {
            "source_context_label": "base PLGA-ITZ nanosphere preparation",
            "source_candidate_label_hint": "F1",
            "source_table_id": "Table 1",
            "target_contexts": [{"target_group_label": "surfactant concentration optimization", "target_table_id": "Table 2", "variation_axis": "surfactant concentration"}],
            "inherited_fields": [{"field_name": "surfactant_name", "field_value": "poloxamer 188", "inheritance_basis": "shared_preparation_context", "confidence": "high"}],
            "held_fixed_conditions": [],
            "evidence_cue": "The optimized ratio was used for the following surfactant concentration optimization.",
            "evidence_source_hint": "methods text near Table 1/Table 2",
            "confidence": "high",
            "marker_readiness": "execution_ready",
            "marker_provenance": "llm_explicit",
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            weak_tsv = tmp_path / "weak_labels.tsv"
            fieldnames = [
                "key",
                "doi",
                "paper_title",
                "local_instance_id",
                "formulation_id",
                "raw_formulation_label",
                "candidate_source",
                "instance_kind",
                "formulation_role",
                "method_group_signature_hint",
                "table_id",
                "context_inheritance_markers_json",
                "instance_evidence_region_type",
                "evidence_section",
                "evidence_span_text",
            ]
            with weak_tsv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
                writer.writeheader()
                writer.writerow(
                    {
                        "key": "QLYKLPKT",
                        "doi": "10.1/example",
                        "paper_title": "Example",
                        "local_instance_id": "QLYKLPKT_F2",
                        "formulation_id": "QLYKLPKT_F2",
                        "raw_formulation_label": "F2",
                        "candidate_source": "table_row_expansion_v1",
                        "instance_kind": "new_formulation",
                        "formulation_role": "reported",
                        "method_group_signature_hint": "surfactant concentration optimization",
                        "table_id": "Table 2",
                        "context_inheritance_markers_json": json.dumps([marker]),
                        "instance_evidence_region_type": "table_row",
                        "evidence_section": "Table 2",
                        "evidence_span_text": "F2 row",
                    }
            )
            stats = build_relation_artifacts(weak_tsv, tmp_path / "relation")
            resolved_path = Path(stats["resolved_relation_fields_path"])
            with resolved_path.open("r", encoding="utf-8", newline="") as handle:
                resolved_rows = list(csv.DictReader(handle, delimiter="\t"))
        self.assertTrue(
            any(
                row["formulation_candidate_id"] == "QLYKLPKT_F2"
                and row["field_name"] == "surfactant_name"
                and row["field_value"] == "poloxamer 188"
                and row["resolution_rule"] == "direct_candidate_field_membership"
                for row in resolved_rows
            )
        )

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

    def test_l3h2rs2h_result_bearing_candidate_non_formulation_helpers_not_filtered(self):
        for formulation_id, label in [
            ("FC1", "XAN and 3-MeOXAN nanospheres at various concentrations"),
            ("FC2", "XAN nanospheres at concentration 60 mg/mL"),
            ("FC3", "XAN and 3-MeOXAN nanocapsules at various concentrations"),
        ]:
            should_filter, rule, reason = should_filter_non_formulation(
                {
                    "key": "L3H2RS2H",
                    "formulation_id": formulation_id,
                    "raw_formulation_label": label,
                    "instance_kind": "candidate_non_formulation",
                    "candidate_source": "live_llm_stage2_v2",
                    "formulation_role": "unclear",
                    "change_role": "unclear",
                    "change_context_tags": '["table_summary_helper", "result_reported"]',
                    "evidence_section": "Table 2 characterization results",
                    "supporting_evidence_refs": '[{"source_region_type":"table_row","target_field":"particle_size_nm"}]',
                    "particle_size_nm_value": "220",
                },
                {"loaded_state": "drug_loaded", "drug_name": "XAN", "polymer_identity": "PLGA"},
                paper_rows=[],
            )
            self.assertFalse(should_filter, msg=f"unexpected filter for {formulation_id}: {rule} {reason}")

    def test_candidate_non_formulation_helper_without_measured_result_still_filtered(self):
        should_filter, rule, reason = should_filter_non_formulation(
            {
                "key": "GENERIC",
                "formulation_id": "FC1",
                "raw_formulation_label": "PLGA nanoparticles summary",
                "instance_kind": "candidate_non_formulation",
                "candidate_source": "live_llm_stage2_v2",
                "change_context_tags": '["table_summary_helper"]',
                "drug_name_value": "Drug A",
                "polymer_identity": "PLGA",
                "supporting_evidence_refs": "[]",
            },
            {"loaded_state": "drug_loaded", "drug_name": "Drug A", "polymer_identity": "PLGA"},
            paper_rows=[],
        )
        self.assertTrue(should_filter)
        self.assertEqual(rule, "explicit_candidate_non_formulation")

    def test_candidate_non_formulation_blank_control_without_measured_result_still_filtered(self):
        should_filter, rule, reason = should_filter_non_formulation(
            {
                "key": "GENERIC",
                "formulation_id": "BLANK1",
                "raw_formulation_label": "Blank nanoparticles",
                "instance_kind": "candidate_non_formulation",
                "candidate_source": "live_llm_stage2_v2",
                "formulation_role": "control",
                "change_role": "non_synthesis",
                "change_context_tags": '["table_summary_helper"]',
                "supporting_evidence_refs": "[]",
            },
            {"loaded_state": "blank_control", "drug_name": "", "polymer_identity": "PLGA"},
            paper_rows=[],
        )
        self.assertTrue(should_filter)
        self.assertEqual(rule, "explicit_candidate_non_formulation")

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

    def test_inmutv7l_unparented_context_summaries_filtered_when_complete_table_enumeration_exists(self):
        paper_rows = [
            {"instance_kind": "new_formulation", "candidate_source": "doe_numbered_table_row_recovery", "semantic_scope_ref": "table15"}
            for _ in range(12)
        ]
        for label, instance_kind, role, change_role, tags in [
            ("PLGA 503 H with Tween 80", "variant_formulation", "unclear", "non_synthesis", '["synthesis_core"]'),
            ("PVA formulation binding study", "single_formulation", "characterization_only", "unclear", '["characterization_only"]'),
            ("Other polymer variants (e.g., PLGA 502 H)", "unclear", "comparative", "unclear", '["comparative"]'),
        ]:
            with self.subTest(label=label):
                should_filter, rule, reason = should_filter_non_formulation(
                    {
                        "key": "INMUTV7L",
                        "raw_formulation_label": label,
                        "instance_kind": instance_kind,
                        "parent_instance_id": "",
                        "formulation_role": role,
                        "change_role": change_role,
                        "candidate_source": "saved_raw_live_v2_replay_to_stage2_v2",
                        "instance_context_tags": tags,
                        "change_context_tags": '[]',
                        "evidence_section": "",
                        "supporting_evidence_refs": "",
                    },
                    {"loaded_state": "drug_loaded", "drug_name": "Dexibuprofen", "polymer_identity": "PLGA"},
                    paper_rows=paper_rows,
                )
                self.assertTrue(should_filter)
                self.assertEqual(rule, "semantic_context_summary_superseded_by_complete_table_enumeration")

    def test_l3h2rs2h_partial_compact_sweep_does_not_erase_llm_result_surfaces(self):
        paper_rows = [
            {
                "instance_kind": "new_formulation",
                "candidate_source": "table_row_expansion_v1",
                "raw_formulation_label": "XAN nanospheres (Theoretical concentration 60 mg/mL)",
                "semantic_scope_ref": "table1",
            }
            for _ in range(8)
        ]
        for label, instance_kind, role, parent, tags in [
            (
                "PLGA nanospheres with XAN or 3-MeOXAN at varying concentrations",
                "formulation_family",
                "unclear",
                "",
                '["synthesis_core"]',
            ),
            (
                "Characterization of nanocapsule formulations (size, PI, zeta) at selected condition (600 μg/mL XAN)",
                "unclear",
                "characterization_only",
                "F2",
                '["characterization_only"]',
            ),
        ]:
            with self.subTest(label=label):
                should_filter, rule, reason = should_filter_non_formulation(
                    {
                        "key": "L3H2RS2H",
                        "raw_formulation_label": label,
                        "instance_kind": instance_kind,
                        "parent_instance_id": parent,
                        "formulation_role": role,
                        "change_role": "unclear",
                        "candidate_source": "saved_raw_live_v2_replay_to_stage2_v2",
                        "instance_context_tags": tags,
                        "change_context_tags": '[]',
                        "evidence_section": "",
                        "supporting_evidence_refs": "",
                    },
                    {"loaded_state": "drug_loaded", "drug_name": "XAN", "polymer_identity": "PLGA"},
                    paper_rows=paper_rows,
                )
                self.assertFalse(should_filter)

    def test_doe_optimum_summary_survives_numbered_row_enumeration(self):
        paper_rows = [
            {"instance_kind": "new_formulation", "candidate_source": "table_row_expansion_v1", "semantic_scope_ref": "doe"}
            for _ in range(26)
        ]
        row = {
            "key": "UFXX9WXE",
            "raw_formulation_label": "Box-Behnken design runs varying PLGA, poloxamer, and phase ratio",
            "instance_kind": "formulation_family",
            "parent_instance_id": "",
            "formulation_role": "unclear",
            "change_role": "unclear",
            "candidate_source": "saved_raw_live_v2_replay_to_stage2_v2",
            "instance_context_tags": '["synthesis_core"]',
            "change_context_tags": '[]',
            "identity_variables_json": json.dumps([
                {"name": "PLGA concentration", "value": "optimal formulation based on Box-Behnken design"}
            ]),
            "evidence_section": "",
            "supporting_evidence_refs": "",
        }
        should_filter, rule, reason = should_filter_non_formulation(
            row,
            {"loaded_state": "drug_loaded", "drug_name": "LZP", "polymer_identity": "PLGA"},
            paper_rows=paper_rows,
        )
        self.assertFalse(should_filter)

    def test_inmutv7l_parented_context_summaries_filtered_when_complete_table_enumeration_exists(self):
        paper_rows = [
            {"instance_kind": "new_formulation", "candidate_source": "doe_numbered_table_row_recovery", "semantic_scope_ref": "table15"}
            for _ in range(12)
        ]
        for label, instance_kind, role, change_role, tags in [
            ("PLGA 503 H with Tween 80", "variant_formulation", "unclear", "non_synthesis", '["synthesis_core"]'),
            ("PLGA 503 H with Lutrol F68", "variant_formulation", "unclear", "non_synthesis", '["synthesis_core"]'),
            ("PVA formulation binding study", "single_formulation", "characterization_only", "unclear", '["characterization_only"]'),
            ("Lutrol formulation permeation study", "unclear", "characterization_only", "unclear", '["characterization_only"]'),
        ]:
            with self.subTest(label=label):
                should_filter, rule, reason = should_filter_non_formulation(
                    {
                        "key": "INMUTV7L",
                        "raw_formulation_label": label,
                        "instance_kind": instance_kind,
                        "parent_instance_id": "F1",
                        "formulation_role": role,
                        "change_role": change_role,
                        "candidate_source": "saved_raw_live_v2_replay_to_stage2_v2",
                        "instance_context_tags": tags,
                        "change_context_tags": '[]',
                        "evidence_section": "",
                        "supporting_evidence_refs": "",
                    },
                    {"loaded_state": "drug_loaded", "drug_name": "Dexibuprofen", "polymer_identity": "PLGA"},
                    paper_rows=paper_rows,
                )
                self.assertTrue(should_filter)
                self.assertIn(rule, {
                    "semantic_context_summary_superseded_by_complete_table_enumeration",
                    "parent_linked_non_synthesis_descendant_variant",
                })

    def test_later_measurement_table_duplicate_collapses_to_primary_formulation_identity(self):
        def row(formulation_id, table_id, label, values, snippet):
            return {
                "key": "PA3SPZ28",
                "formulation_id": formulation_id,
                "raw_formulation_label": label,
                "candidate_source": "table_row_expansion_v1",
                "table_id": table_id,
                "evidence_section": table_id,
                "identity_variables_json": json.dumps([
                    {"name": f"formulation_header_part_{idx}", "value": value}
                    for idx, value in enumerate(values, start=1)
                ]),
                "supporting_evidence_refs": json.dumps([{"supporting_snippet": snippet}]),
            }

        rows = [
            row("T1_20", "Table 1", "Nanoprecipitation method / 1:20 / 50:50", ["Nanoprecipitation method", "1:20", "50:50"], "Size (nm)=90 | Drug loading (DL, %)=4"),
            row("T1_10", "Table 1", "Nanoprecipitation method / 1:10 / 50:50", ["Nanoprecipitation method", "1:10", "50:50"], "Size (nm)=88 | Encapsulation efficiency (EE, %)=88"),
            row("T8_20", "Table 8", "Nanoprecipitation method / 1:20 / 50:50", ["Nanoprecipitation method", "1:20", "50:50"], "Size (nm)=90 | Drug loading (DL, %)=4"),
            row("T8_storage", "Table 8", "for 3 months / 1:10 / 50:50", ["for 3 months", "1:10", "50:50"], "Size (nm)=100 | PDI=0.295 | Zeta potential (mV)=-27"),
        ]
        collapse_map = final_output.build_doe_measurement_duplicate_collapse_map(rows)
        self.assertEqual(collapse_map["PA3SPZ28::T8_20"]["target_source_key"], "PA3SPZ28::T1_20")
        self.assertEqual(collapse_map["PA3SPZ28::T8_storage"]["target_source_key"], "PA3SPZ28::T1_10")
        self.assertEqual(
            collapse_map["PA3SPZ28::T8_storage"]["decision_rule"],
            "later_measurement_table_duplicate_of_primary_formulation_identity",
        )

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
        for row in [
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
            {
                "key": "UFXX9WXE",
                "raw_formulation_label": "optimized formulation",
                "instance_kind": "single_formulation",
                "parent_instance_id": "F1",
                "formulation_role": "unclear",
                "change_role": "unclear",
                "candidate_source": "saved_raw_live_v2_replay_to_stage2_v2",
                "instance_context_tags": '["synthesis_core"]',
                "change_context_tags": '[]',
                "change_descriptions": '["Selected based on desirability"]',
                "drug_name_value_text": "lorazepam",
                "polymer_identity": "PLGA",
                "identity_variables_json": json.dumps([
                    {"name": "drug_concentration", "value": "optimized formulation via box-behnken design"},
                    {"name": "plga_concentration", "value": "optimized formulation via box-behnken design"},
                ]),
            },
        ]:
            with self.subTest(label=row["raw_formulation_label"]):
                should_filter, rule, reason = should_filter_non_formulation(
                    row,
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

    def test_stage5_carries_doe_factor_emulsifier_name_from_source_definition(self):
        row = {
            "key": "SURF1",
            "formulation_id": "SURF1_DOE_Row_F1",
            "raw_formulation_label": "F1",
            "change_descriptions": '["(\\\'Coded Levels of Factors\\\', \\\'cP188 (mg/mL)\\\')=15.0"]',
            "surfactant_name_value": "",
            "surfactant_name_value_text": "",
            "surfactant_name_scope": "",
            "surfactant_name_membership_confidence": "low",
            "surfactant_name_evidence_region_type": "unknown",
            "surfactant_name_missing_reason": "not_reported",
            "surfactant_concentration_text_value": "15.0",
            "surfactant_concentration_text_value_text": "15.0",
        }
        source_text = (
            "Table 1. Initial full factorial design. cFB, concentration of flurbiprofen "
            "(mg/mL); cP188, concentration of poloxamer 188 (mg/mL)."
        )
        materialized, applied = apply_global_preparation_material_carrythrough(
            final_row=row,
            source_text=source_text,
        )
        self.assertIn("surfactant_name", applied)
        self.assertEqual(materialized["surfactant_name_value_text"], "poloxamer 188")
        self.assertEqual(materialized["surfactant_name_scope"], "global_shared")
        self.assertEqual(materialized["surfactant_name_evidence_region_type"], "global_emulsifier_factor_evidence")
        self.assertEqual(materialized["surfactant_name_missing_reason"], "")
        self.assertEqual(materialized["surfactant_concentration_text_value_text"], "15.0")

    def test_stage5_carries_doe_factor_emulsifier_concentration_from_row_assignment(self):
        row = {
            "key": "SURF_CONC1",
            "formulation_id": "SURF_CONC1_DOE_Row_F1",
            "raw_formulation_label": "F1",
            "change_descriptions": '["(\\\'Coded Levels of Factors\\\', \\\'cP188 (mg/mL)\\\') Factorial points=0", "cP188 (mg/mL)=15.0"]',
            "surfactant_name_value": "",
            "surfactant_name_value_text": "",
            "surfactant_name_scope": "",
            "surfactant_name_membership_confidence": "low",
            "surfactant_name_evidence_region_type": "unknown",
            "surfactant_name_missing_reason": "not_reported",
            "surfactant_concentration_text_value": "",
            "surfactant_concentration_text_value_text": "",
            "surfactant_concentration_text_scope": "",
            "surfactant_concentration_text_membership_confidence": "low",
            "surfactant_concentration_text_evidence_region_type": "unknown",
            "surfactant_concentration_text_missing_reason": "not_reported",
        }
        source_text = (
            "Table 1. Initial full factorial design. cFB, concentration of flurbiprofen "
            "(mg/mL); cP188, concentration of poloxamer 188 (mg/mL)."
        )
        materialized, applied = apply_global_preparation_material_carrythrough(
            final_row=row,
            source_text=source_text,
        )
        self.assertIn("surfactant_name", applied)
        self.assertIn("surfactant_concentration_text", applied)
        self.assertEqual(materialized["surfactant_name_value_text"], "poloxamer 188")
        self.assertEqual(materialized["surfactant_concentration_text_value_text"], "15.0 mg/mL")
        self.assertEqual(materialized["surfactant_concentration_text_scope"], "row_local")
        self.assertEqual(
            materialized["surfactant_concentration_text_evidence_region_type"],
            "row_emulsifier_factor_assignment",
        )
        self.assertEqual(materialized["surfactant_concentration_text_missing_reason"], "")

    def test_stage5_doe_factor_emulsifier_concentration_requires_source_definition(self):
        row = {
            "key": "SURF_CONC2",
            "formulation_id": "SURF_CONC2_DOE_Row_F1",
            "raw_formulation_label": "F1",
            "change_descriptions": '["X3=0.75%"]',
            "surfactant_name_value": "",
            "surfactant_name_value_text": "",
            "surfactant_concentration_text_value": "",
            "surfactant_concentration_text_value_text": "",
        }
        source_text = "The formulation used Pluronic F68, but the table does not define X3."
        materialized, applied = apply_global_preparation_material_carrythrough(
            final_row=row,
            source_text=source_text,
        )
        self.assertNotIn("surfactant_name", applied)
        self.assertNotIn("surfactant_concentration_text", applied)
        self.assertEqual(materialized["surfactant_concentration_text_value_text"], "")

    def test_stage5_carries_preparation_surfactant_concentration_list_by_row_local_name(self):
        row = {
            "key": "SURF_LIST1",
            "formulation_id": "SURF_LIST1__table_1__1",
            "raw_formulation_label": "1",
            "change_descriptions": '["Used=PVA", "Size (nm)=234.1 ± 0.5"]',
            "surfactant_name_value": "PVA",
            "surfactant_name_value_text": "PVA",
            "surfactant_concentration_text_value": "",
            "surfactant_concentration_text_value_text": "",
            "surfactant_concentration_text_scope": "",
            "surfactant_concentration_text_membership_confidence": "low",
            "surfactant_concentration_text_evidence_region_type": "unknown",
            "surfactant_concentration_text_missing_reason": "not_reported",
        }
        source_text = (
            "PLGA nanoparticles (NPs) containing DXI were prepared by using the solvent displacement method. "
            "Various optimized concentrations of surfactants were used (PVA 0.5%, Tween 80® 0.3% and Lutrol F68 (1%)."
        )
        materialized, applied = apply_global_preparation_material_carrythrough(
            final_row=row,
            source_text=source_text,
        )
        self.assertIn("surfactant_concentration_text", applied)
        self.assertEqual(materialized["surfactant_concentration_text_value_text"], "0.5%")
        self.assertEqual(materialized["surfactant_concentration_text_scope"], "global_shared")
        self.assertEqual(
            materialized["surfactant_concentration_text_evidence_region_type"],
            "global_preparation_surfactant_concentration_evidence",
        )
        self.assertEqual(materialized["surfactant_concentration_text_missing_reason"], "")

    def test_stage5_carries_preparation_surfactant_concentration_list_by_compact_or_short_row_name(self):
        source_text = (
            "Various optimized concentrations of surfactants were used (PVA 0.5%, Tween 80® 0.3% and Lutrol F68 (1%)."
        )
        cases = [("Tween80", "0.3%"), ("Lutrol", "1%")]
        for row_name, expected in cases:
            with self.subTest(row_name=row_name):
                row = {
                    "key": "SURF_LIST_ALIAS",
                    "formulation_id": f"SURF_LIST_ALIAS__{row_name}",
                    "raw_formulation_label": "1",
                    "change_descriptions": f'["Used={row_name}"]',
                    "surfactant_name_value": "",
                    "surfactant_name_value_text": "",
                    "surfactant_concentration_text_value": "",
                    "surfactant_concentration_text_value_text": "",
                    "surfactant_concentration_text_scope": "",
                    "surfactant_concentration_text_membership_confidence": "low",
                    "surfactant_concentration_text_evidence_region_type": "unknown",
                    "surfactant_concentration_text_missing_reason": "not_reported",
                }
                materialized, applied = apply_global_preparation_material_carrythrough(
                    final_row=row,
                    source_text=source_text,
                )
                self.assertIn("surfactant_concentration_text", applied)
                self.assertEqual(materialized["surfactant_concentration_text_value_text"], expected)

    def test_stage5_carries_subtype_scoped_preparation_surfactant_concentration_without_row_name(self):
        source_text = (
            "2.2. Preparation of nanospheres Nanospheres containing XAN were prepared by solvent displacement. "
            "An organic solution of PLGA was poured into 10 mL of an aqueous solution of Pluronic F-68 0.25% (w/v). "
            "2.4. Preparation of nanocapsules XAN-containing PLGA nanocapsules were prepared by interfacial polymer deposition. "
            "The final solution was poured into 20 mL of an aqueous solution of Pluronic F-68 0.5% (w/v)."
        )
        row = {
            "key": "SURF_SUBTYPE",
            "formulation_id": "SURF_SUBTYPE__table_5__xan_nanospheres_theoretical_concentration_50_mg/ml",
            "raw_formulation_label": "XAN nanospheres (Theoretical concentration 50 mg/mL)",
            "change_descriptions": '["formulation_identity_label=XAN nanospheres (Theoretical concentration 50 mg/mL)"]',
            "surfactant_name_value": "",
            "surfactant_name_value_text": "",
            "surfactant_concentration_text_value": "",
            "surfactant_concentration_text_value_text": "",
            "surfactant_concentration_text_scope": "",
            "surfactant_concentration_text_membership_confidence": "low",
            "surfactant_concentration_text_evidence_region_type": "unknown",
            "surfactant_concentration_text_missing_reason": "not_reported",
        }
        materialized, applied = apply_global_preparation_material_carrythrough(
            final_row=row,
            source_text=source_text,
        )
        self.assertIn("surfactant_concentration_text", applied)
        self.assertEqual(materialized["surfactant_concentration_text_value_text"], "0.25% (w/v)")
        self.assertEqual(materialized["surfactant_concentration_text_scope"], "global_shared")
        self.assertEqual(
            materialized["surfactant_concentration_text_evidence_region_type"],
            "global_preparation_surfactant_concentration_evidence",
        )

    def test_stage5_preparation_surfactant_concentration_list_requires_explicit_unit(self):
        row = {
            "key": "SURF_LIST_NO_UNIT",
            "formulation_id": "SURF_LIST_NO_UNIT__table_1__1",
            "raw_formulation_label": "1",
            "change_descriptions": '["Used=PVA"]',
            "surfactant_name_value": "PVA",
            "surfactant_name_value_text": "PVA",
            "surfactant_concentration_text_value": "",
            "surfactant_concentration_text_value_text": "",
            "surfactant_concentration_text_scope": "",
            "surfactant_concentration_text_membership_confidence": "low",
            "surfactant_concentration_text_evidence_region_type": "unknown",
            "surfactant_concentration_text_missing_reason": "not_reported",
        }
        source_text = (
            "Nanoparticle formulations were prepared with PVA 12.93 and reported measured results."
        )
        materialized, applied = apply_global_preparation_material_carrythrough(
            final_row=row,
            source_text=source_text,
        )
        self.assertNotIn("surfactant_concentration_text", applied)
        self.assertEqual(materialized["surfactant_concentration_text_value_text"], "")

    def test_stage5_prefers_stage2_table_cell_grid_bindings_without_raw_csv_reopen(self):
        row = {
            "key": "INMUTV7L",
            "formulation_id": "INMUTV7L__table_15__1",
            "raw_formulation_label": "1",
            "table_cell_bindings_json": json.dumps([
                {
                    "binding_rule": "table_cell_grid_v1_row_local_header_binding",
                    "canonical_field": "particle_size_nm",
                    "raw_cell_value": "234.1 ± 0.5",
                    "source_csv_path": "data/cleaned/goren_2025/tables/INMUTV7L/DO_NOT_REOPEN.csv",
                    "source_row_index": 4,
                },
                {
                    "binding_rule": "table_cell_grid_v1_row_local_header_binding",
                    "canonical_field": "pdi",
                    "raw_cell_value": "0.081 ± 0.009",
                    "source_csv_path": "data/cleaned/goren_2025/tables/INMUTV7L/DO_NOT_REOPEN.csv",
                    "source_row_index": 4,
                },
                {
                    "binding_rule": "table_cell_grid_v1_row_local_header_binding",
                    "canonical_field": "zeta_mV",
                    "raw_cell_value": "−12.2 ± 1.3",
                    "source_csv_path": "data/cleaned/goren_2025/tables/INMUTV7L/DO_NOT_REOPEN.csv",
                    "source_row_index": 4,
                },
                {
                    "binding_rule": "table_cell_grid_v1_row_local_header_binding",
                    "canonical_field": "ee_percent",
                    "raw_cell_value": "93.4",
                    "source_csv_path": "data/cleaned/goren_2025/tables/INMUTV7L/DO_NOT_REOPEN.csv",
                    "source_row_index": 4,
                },
                {
                    "binding_rule": "table_cell_grid_v1_row_local_header_binding",
                    "canonical_field": "surfactant_name",
                    "raw_cell_value": "PVA",
                    "source_csv_path": "data/cleaned/goren_2025/tables/INMUTV7L/DO_NOT_REOPEN.csv",
                    "source_row_index": 4,
                },
                {
                    "binding_rule": "table_cell_grid_v1_row_local_header_binding",
                    "canonical_field": "drug_mass_mg",
                    "raw_cell_value": "5",
                    "source_csv_path": "data/cleaned/goren_2025/tables/INMUTV7L/DO_NOT_REOPEN.csv",
                    "source_row_index": 4,
                },
                {
                    "binding_rule": "table_cell_grid_v1_row_local_header_binding",
                    "canonical_field": "polymer_mass_mg",
                    "raw_cell_value": "75",
                    "source_csv_path": "data/cleaned/goren_2025/tables/INMUTV7L/DO_NOT_REOPEN.csv",
                    "source_row_index": 4,
                },
                {
                    "binding_rule": "table_cell_grid_v1_row_local_header_binding",
                    "canonical_field": "O_volume_mL",
                    "raw_cell_value": "5",
                    "source_csv_path": "data/cleaned/goren_2025/tables/INMUTV7L/DO_NOT_REOPEN.csv",
                    "source_row_index": 4,
                },
            ]),
            "surfactant_name_value": "",
            "surfactant_name_value_text": "",
            "size_nm_value": "",
            "size_nm_value_text": "",
            "pdi_value": "",
            "pdi_value_text": "",
            "zeta_mV_value": "",
            "zeta_mV_value_text": "",
            "zeta_mV_scope": "",
            "zeta_mV_membership_confidence": "",
            "zeta_mV_evidence_region_type": "",
            "zeta_mV_missing_reason": "",
            "encapsulation_efficiency_percent_value": "Lutrol %",
            "encapsulation_efficiency_percent_value_text": "Lutrol %",
            "encapsulation_efficiency_percent_scope": "row_local_table_cell",
            "encapsulation_efficiency_percent_membership_confidence": "medium",
            "encapsulation_efficiency_percent_evidence_region_type": "row_local_table_cell",
            "encapsulation_efficiency_percent_missing_reason": "",
            "drug_feed_amount_text_value": "",
            "drug_feed_amount_text_value_text": "",
            "drug_feed_amount_text_scope": "",
            "drug_feed_amount_text_evidence_region_type": "",
            "plga_mass_mg_value": "",
            "plga_mass_mg_value_text": "",
            "plga_mass_mg_scope": "",
            "plga_mass_mg_evidence_region_type": "",
            "organic_phase_volume_mL_value": "",
            "organic_phase_volume_mL_value_text": "",
            "organic_phase_volume_mL_scope": "",
            "organic_phase_volume_mL_evidence_region_type": "",
            "polymer_name_raw": "",
        }
        materialized, applied = apply_global_preparation_material_carrythrough(
            final_row=row,
            source_text="",
        )
        self.assertEqual(materialized["size_nm_value"], "234.1")
        self.assertEqual(materialized["size_nm_value_text"], "234.1 ± 0.5")
        self.assertEqual(materialized["pdi_value"], "0.081")
        self.assertEqual(materialized["zeta_mV_value"], "-12.2")
        self.assertEqual(materialized["encapsulation_efficiency_percent_value"], "93.4")
        self.assertEqual(materialized["surfactant_name_value"], "PVA")
        self.assertEqual(materialized["drug_feed_amount_text_value"], "5")
        self.assertEqual(materialized["plga_mass_mg_value"], "75")
        self.assertEqual(materialized["organic_phase_volume_mL_value"], "5")
        self.assertEqual(materialized["organic_phase_volume_mL_scope"], "row_local_table_cell_grid")
        self.assertEqual(materialized["zeta_mV_scope"], "row_local_table_cell_grid")
        self.assertEqual(materialized["encapsulation_efficiency_percent_scope"], "row_local_table_cell_grid")
        self.assertEqual(materialized["encapsulation_efficiency_percent_evidence_region_type"], "row_local_table_cell_grid_binding")
        self.assertIn("size_nm", applied)
        self.assertIn("encapsulation_efficiency_percent", applied)

    def test_stage5_row_local_grid_binding_creates_metadata_bundle_for_late_added_preparation_fields(self):
        row = {
            "key": "BB3JUVW7",
            "formulation_id": "BB3JUVW7__table_1__row_02_5",
            "raw_formulation_label": "row_02__5",
            "table_cell_bindings_json": json.dumps([
                {
                    "binding_rule": "table_cell_grid_v1_row_local_header_binding",
                    "canonical_field": "O_volume_mL",
                    "raw_cell_value": "5",
                    "source_csv_path": "DO_NOT_REOPEN.csv",
                    "source_row_index": 2,
                },
                {
                    "binding_rule": "table_cell_grid_v1_row_local_header_binding",
                    "canonical_field": "external_aqueous_phase_volume_mL",
                    "raw_cell_value": "15",
                    "source_csv_path": "DO_NOT_REOPEN.csv",
                    "source_row_index": 2,
                },
            ]),
            # Simulate legacy Stage2 rows where Stage5 adds these output columns later.
            "organic_phase_volume_mL_value": "",
            "organic_phase_volume_mL_value_text": "",
            "external_aqueous_phase_volume_mL_value": "",
            "external_aqueous_phase_volume_mL_value_text": "",
            "polymer_name_raw": "",
        }
        materialized, applied = apply_global_preparation_material_carrythrough(
            final_row=row,
            source_text="",
        )
        self.assertIn("organic_phase_volume_mL", applied)
        self.assertEqual(materialized["organic_phase_volume_mL_value"], "5")
        self.assertEqual(materialized["organic_phase_volume_mL_scope"], "row_local_table_cell_grid")
        self.assertEqual(materialized["organic_phase_volume_mL_membership_confidence"], "medium")
        self.assertEqual(materialized["organic_phase_volume_mL_evidence_region_type"], "row_local_table_cell_grid_binding")
        self.assertEqual(materialized["organic_phase_volume_mL_missing_reason"], "")
        self.assertIn("external_aqueous_phase_volume_mL", applied)
        self.assertEqual(materialized["external_aqueous_phase_volume_mL_value"], "15")
        self.assertEqual(materialized["external_aqueous_phase_volume_mL_scope"], "row_local_table_cell_grid")
        self.assertEqual(materialized["external_aqueous_phase_volume_mL_membership_confidence"], "medium")
        self.assertEqual(materialized["external_aqueous_phase_volume_mL_evidence_region_type"], "row_local_table_cell_grid_binding")
        self.assertEqual(materialized["external_aqueous_phase_volume_mL_missing_reason"], "")

    def test_stage5_raw_csv_fallback_does_not_override_numeric_grid_bound_ee(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "shifted.csv"
            with csv_path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["formulation", "%EE"])
                writer.writerow(["row_01", "78.5 ± 1.8"])
                writer.writerow(["row_02", "80.0 ± 0.1"])
            row = {
                "key": "BB3JUVW7",
                "formulation_id": "BB3JUVW7__table_1__row_01_5",
                "raw_formulation_label": "row_01__5",
                "table_cell_bindings_json": json.dumps([
                    {
                        "binding_rule": "table_cell_grid_v1_row_local_header_binding",
                        "canonical_field": "ee_percent",
                        "raw_cell_value": "78.5 ± 1.8",
                        "source_csv_path": str(csv_path),
                        # Simulate a sidecar/raw-CSV row-index disagreement: the
                        # Stage2 grid binding is already row-local authority, so
                        # the diagnostic raw-CSV fallback must not overwrite it.
                        "source_row_index": 3,
                    }
                ]),
                "encapsulation_efficiency_percent_value": "",
                "encapsulation_efficiency_percent_value_text": "",
                "encapsulation_efficiency_percent_scope": "",
                "encapsulation_efficiency_percent_membership_confidence": "",
                "encapsulation_efficiency_percent_evidence_region_type": "",
                "encapsulation_efficiency_percent_missing_reason": "",
                "polymer_name_raw": "",
            }

            materialized, applied = apply_global_preparation_material_carrythrough(final_row=row, source_text="")

        self.assertIn("encapsulation_efficiency_percent", applied)
        self.assertEqual(materialized["encapsulation_efficiency_percent_value"], "78.5")
        self.assertEqual(materialized["encapsulation_efficiency_percent_value_text"], "78.5 ± 1.8")
        self.assertEqual(materialized["encapsulation_efficiency_percent_scope"], "row_local_table_cell_grid")
        self.assertEqual(materialized["encapsulation_efficiency_percent_evidence_region_type"], "row_local_table_cell_grid_binding")

    def test_stage5_rebinds_row_local_source_csv_characterization_cells_after_split_header_shift(self):
        row = {
            "key": "INMUTV7L",
            "formulation_id": "INMUTV7L__table_15__3",
            "raw_formulation_label": "3",
            "table_cell_bindings_json": json.dumps([
                {
                    "source_csv_path": "data/cleaned/goren_2025/tables/INMUTV7L/INMUTV7L__table_15__pdf_table.csv",
                    "source_row_index": 9,
                }
            ]),
            "surfactant_name_value": "",
            "surfactant_name_value_text": "",
            "size_nm_value": "",
            "size_nm_value_text": "",
            "pdi_value": "",
            "pdi_value_text": "",
            "zeta_mV_value": "",
            "zeta_mV_value_text": "",
            "encapsulation_efficiency_percent_value": "Lutrol %",
            "encapsulation_efficiency_percent_value_text": "Lutrol %",
            "encapsulation_efficiency_percent_scope": "row_local_table_cell",
            "encapsulation_efficiency_percent_membership_confidence": "medium",
            "encapsulation_efficiency_percent_evidence_region_type": "row_local_table_cell",
            "encapsulation_efficiency_percent_missing_reason": "",
            "polymer_name_raw": "",
        }
        materialized, applied = apply_global_preparation_material_carrythrough(
            final_row=row,
            source_text="",
        )
        self.assertIn("size_nm", applied)
        self.assertIn("pdi", applied)
        self.assertIn("zeta_mV", applied)
        self.assertIn("encapsulation_efficiency_percent", applied)
        self.assertIn("surfactant_name", applied)
        self.assertIn("polymer_name", applied)
        self.assertEqual(materialized["size_nm_value"], "159.5")
        self.assertEqual(materialized["pdi_value"], "0.058")
        self.assertEqual(materialized["zeta_mV_value"], "-26.0")
        self.assertEqual(materialized["encapsulation_efficiency_percent_value"], "85.1")
        self.assertEqual(materialized["surfactant_name_value"], "Lutrol")
        self.assertEqual(materialized["polymer_name_raw"], "PLGA 503 H")

    def test_stage5_doe_factor_emulsifier_name_does_not_override_row_local_name(self):
        row = {
            "key": "SURF2",
            "formulation_id": "SURF2_DOE_Row_F1",
            "raw_formulation_label": "F1",
            "change_descriptions": '["(\\\'Coded Levels of Factors\\\', \\\'cPVA (%)\\\')=1.0"]',
            "surfactant_name_value": "Tween 80",
            "surfactant_name_value_text": "Tween 80",
        }
        source_text = "The factors were cPVA, concentration of PVA (% w/v)."
        materialized, applied = apply_global_preparation_material_carrythrough(
            final_row=row,
            source_text=source_text,
        )
        self.assertNotIn("surfactant_name", applied)
        self.assertEqual(materialized["surfactant_name_value_text"], "Tween 80")

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

    def test_stage2_projection_blanks_invalid_direct_mass_values_with_diagnostics(self):
        document = {
            "document_key": "TYPEGUARD1",
            "paper_key": "TYPEGUARD1",
            "doi": "10.1/typeguard1",
            "model_name": "test-model",
            "source_mode": "saved_raw_live_v2_replay_to_stage2_v2",
            "semantic_signals": {},
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
                    "amount_expression_raw": "PLGA",
                    "parsed_value_raw": "",
                },
                {
                    "component_id": "c_drug",
                    "formulation_candidate_id": "",
                    "component_role_raw": "drug",
                    "component_name_raw": "Curcumin",
                    "amount_expression_raw": "1 mg/mL",
                    "parsed_value_raw": "1",
                },
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
        rows, traces, _, _, _ = project_document(document)
        row = rows[0]
        self.assertEqual(row["plga_mass_mg_value"], "")
        self.assertEqual(row["plga_mass_mg_value_text"], "")
        self.assertEqual(row["plga_mass_mg_missing_reason"], "invalid_mass_no_numeric_value")
        self.assertEqual(row["drug_feed_amount_text_value"], "")
        self.assertEqual(row["drug_feed_amount_text_value_text"], "")
        self.assertEqual(row["drug_feed_amount_text_missing_reason"], "invalid_mass_concentration_only")
        invalid_traces = [trace for trace in traces if trace["mapping_status"] == "invalid_type_rejected"]
        self.assertTrue(any(trace["legacy_field"] == "plga_mass_mg" for trace in invalid_traces))
        self.assertTrue(any(trace["legacy_field"] == "drug_feed_amount_text" for trace in invalid_traces))

    def test_stage5_final_boundary_blanks_invalid_volume_before_lawful_carrythrough(self):
        source_text = (
            "PLGA nanoparticles were prepared by dissolving PLGA in acetone (2 mL) "
            "and added dropwise into 10 mL aqueous phase."
        )
        row = {
            "raw_formulation_label": "Loaded PLGA nanoparticles",
            "drug_name_value": "curcumin",
            "polymer_name_raw": "PLGA",
            "organic_solvent_value": "acetone",
            "organic_phase_volume_mL_value": "acetone",
            "organic_phase_volume_mL_value_text": "acetone",
            "organic_phase_volume_mL_missing_reason": "",
            "external_aqueous_phase_volume_mL_value": "water",
            "external_aqueous_phase_volume_mL_value_text": "water",
            "external_aqueous_phase_volume_mL_missing_reason": "",
        }

        materialized, applied = apply_global_preparation_material_carrythrough(final_row=row, source_text=source_text)

        self.assertIn("organic_phase_volume_mL", applied)
        self.assertIn("external_aqueous_phase_volume_mL", applied)
        self.assertEqual(materialized["organic_phase_volume_mL_value"], "2 mL")
        self.assertEqual(materialized["external_aqueous_phase_volume_mL_value"], "10 mL")
        self.assertEqual(materialized["organic_phase_volume_mL_missing_reason"], "")
        self.assertEqual(materialized["external_aqueous_phase_volume_mL_missing_reason"], "")

    def test_stage2_projection_rejects_compressed_direct_mass_with_invalid_segment(self):
        document = {
            "document_key": "TYPEGUARD2",
            "paper_key": "TYPEGUARD2",
            "doi": "10.1/typeguard2",
            "model_name": "test-model",
            "source_mode": "saved_raw_live_v2_replay_to_stage2_v2",
            "semantic_signals": {},
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
                    "component_id": "c_poly_1",
                    "formulation_candidate_id": "",
                    "component_role_raw": "polymer",
                    "component_name_raw": "PLGA",
                    "amount_expression_raw": "20 mg",
                    "parsed_value_raw": "20",
                },
                {
                    "component_id": "c_poly_2",
                    "formulation_candidate_id": "",
                    "component_role_raw": "polymer",
                    "component_name_raw": "PLGA",
                    "amount_expression_raw": "PLGA",
                    "parsed_value_raw": "",
                },
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
        row = rows[0]
        self.assertEqual(row["plga_mass_mg_value"], "")
        self.assertEqual(row["plga_mass_mg_value_text"], "")
        self.assertEqual(row["plga_mass_mg_missing_reason"], "invalid_mass_no_numeric_value")

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
