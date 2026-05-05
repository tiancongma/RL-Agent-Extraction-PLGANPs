from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.stage2_sampling_labels import build_dictionary_promotion_review_v1 as promotion_review
from src.stage2_sampling_labels import build_paper_local_abbreviation_registry_v1 as abbrev_registry
from src.stage2_sampling_labels import extract_semantic_stage2_objects_v2 as stage2_objects
from src.stage2_sampling_labels import table_row_expansion_v1 as row_expansion
from src.stage2_sampling_labels import table_structure_dictionary_v1 as structure_dictionary
from src.stage5_benchmark import build_minimal_final_output_v1 as s5_final


INMUTV7L_ROWS = [
    ["0", "1", "2", "3", "4", "5", "6"],
    ["triblocks were used.", "", "", "", "", "", ""],
    ["Table 1. Characterization of the different formulations developed.", "", "", "", "", "", ""],
    ["Formulation", "Polymer", "", "Average", "Polydispersity", "Zeta Potential", ""],
    ["", "", "Surfactant", "", "", "", "EE (%)"],
    ["Number", "Used", "", "Size (nm)", "Index (PI)", "(ZP, mV)", ""],
    ["1", "", "PVA", "234.1 ± 0.5", "0.12 ± 0.01", "-18.1 ± 0.4", "65.2 ± 1.3"],
    ["", "PLGA 503 H", "®", "", "", "", ""],
    ["2", "", "Tween80", "220.4 ± 0.7", "0.15 ± 0.02", "-17.8 ± 0.3", "63.0 ± 1.0"],
    ["3", "", "Lutrol", "228.3 ± 0.4", "0.14 ± 0.01", "-18.0 ± 0.2", "64.1 ± 0.9"],
    ["4", "", "PVA", "240.5 ± 0.8", "0.13 ± 0.02", "-17.1 ± 0.5", "66.4 ± 1.5"],
    ["", "PLGA-5%", "®", "", "", "", ""],
    ["5", "", "Tween80", "238.8 ± 0.3", "0.16 ± 0.01", "-16.8 ± 0.2", "61.5 ± 0.8"],
]


def make_row_entries() -> list[dict[str, object]]:
    return [
        {"row_index": idx, "cells": row, "row_text": " | ".join(cell for cell in row if cell)}
        for idx, row in enumerate(INMUTV7L_ROWS)
    ]


def assignment_value(row: dict[str, object], canonical_field: str) -> str:
    for item in row.get("assignments", []):
        if item.get("canonical_field") == canonical_field:
            return item.get("value", "")
    return ""


class TableStructureDictionaryV1Tests(unittest.TestCase):
    def with_dictionary_rows(self, rows: list[dict[str, str]]):
        class _DictionaryRowsContext:
            def __enter__(inner_self):
                inner_self.old_cache = structure_dictionary._LEXICON_CACHE
                structure_dictionary._LEXICON_CACHE = rows

            def __exit__(inner_self, exc_type, exc, tb):
                structure_dictionary._LEXICON_CACHE = inner_self.old_cache

        return _DictionaryRowsContext()

    def test_infer_header_structure_skips_index_and_metadata_rows(self) -> None:
        inferred = structure_dictionary.infer_header_structure(INMUTV7L_ROWS)
        self.assertEqual(inferred["header_row_count"], 3)
        self.assertEqual(
            inferred["header_rows"],
            [
                ["Formulation", "Polymer", "", "Average", "Polydispersity", "Zeta Potential", ""],
                ["", "", "Surfactant", "", "", "", "EE (%)"],
                ["Number", "Used", "", "Size (nm)", "Index (PI)", "(ZP, mV)", ""],
            ],
        )
        self.assertEqual(
            inferred["flattened_headers"],
            [
                "Formulation Number",
                "Polymer Used",
                "Surfactant",
                "Average Size (nm)",
                "Polydispersity Index (PI)",
                "Zeta Potential (ZP, mV)",
                "EE (%)",
            ],
        )

    def test_normalized_payload_full_table_preserves_coordinate_columns(self) -> None:
        normalized_rows, actions, metadata = stage2_objects.normalize_selected_table_rows(
            INMUTV7L_ROWS,
            table_role_hint="formulation",
        )
        self.assertIn("preserve_coordinate_grid", actions)
        self.assertNotIn("drop_sparse_placeholder_columns", actions)
        self.assertNotIn("left_align_shifted_numbered_matrix", actions)
        self.assertEqual(metadata["numbered_row_column_index"], "")
        self.assertEqual(len({len(row) for row in normalized_rows}), 1)
        inferred = structure_dictionary.infer_header_structure(normalized_rows)
        self.assertEqual(inferred["flattened_headers"][2], "Surfactant")
        self.assertEqual(inferred["flattened_headers"][6], "EE (%)")
        row_entries = stage2_objects.build_normalized_row_entries(
            normalized_rows,
            header_structure=inferred,
            numbered_row_column_index="",
        )
        first_data = next(row for row in row_entries if row["cells"][0] == "1")
        self.assertEqual(first_data["cell_map"].get("Surfactant"), "PVA")
        self.assertEqual(first_data["cell_map"].get("EE (%)"), "65.2 ± 1.3")
        self.assertNotIn("Polymer EE (%) Used", first_data["cell_map"])

    def test_normalized_payload_full_table_drops_leading_spillover_not_geometry(self) -> None:
        rows_with_spillover = [
            [str(i) for i in range(7)],
            ["triblocks were used.", "", "", "", "", "", ""],
            ["", "", "Table 1. Characterization of the different formulations developed.", "", "", "", ""],
        ] + INMUTV7L_ROWS
        normalized_rows, actions, metadata = stage2_objects.normalize_selected_table_rows(
            rows_with_spillover,
            table_role_hint="formulation",
        )
        self.assertIn("drop_enumerator_index_row", actions)
        self.assertIn("drop_leading_non_table_rows", actions)
        self.assertIn("preserve_coordinate_grid", actions)
        self.assertEqual(len({len(row) for row in normalized_rows}), 1)
        inferred = structure_dictionary.infer_header_structure(normalized_rows)
        self.assertEqual(inferred["flattened_headers"][2], "Surfactant")
        self.assertEqual(inferred["flattened_headers"][6], "EE (%)")
        row_entries = stage2_objects.build_normalized_row_entries(
            normalized_rows,
            header_structure=inferred,
            numbered_row_column_index=metadata["numbered_row_column_index"],
        )
        first_data = next(row for row in row_entries if row["cells"][0] == "1")
        self.assertEqual(first_data["cell_map"].get("Surfactant"), "PVA")
        self.assertEqual(first_data["cell_map"].get("EE (%)"), "65.2 ± 1.3")

    def test_extract_direct_rows_uses_recovered_headers_and_group_carrydown(self) -> None:
        extracted_rows, reason = row_expansion.extract_direct_formulation_rows_from_authority(
            authority_payload={"source_csv_path": "INMUTV7L__table_15__pdf_table.csv"},
            row_entries=make_row_entries(),
        )
        self.assertEqual(reason, "")
        self.assertGreaterEqual(len(extracted_rows), 5)
        self.assertEqual(assignment_value(extracted_rows[0], "surfactant_name"), "PVA")
        self.assertEqual(assignment_value(extracted_rows[0], "pdi"), "0.12 ± 0.01")
        self.assertEqual(assignment_value(extracted_rows[0], "ee_percent"), "65.2 ± 1.3")
        self.assertEqual(assignment_value(extracted_rows[0], "polymer_name"), "PLGA 503 H®")
        self.assertEqual(assignment_value(extracted_rows[1], "polymer_name"), "PLGA 503 H®")
        self.assertEqual(assignment_value(extracted_rows[2], "surfactant_name"), "Lutrol")
        self.assertEqual(assignment_value(extracted_rows[3], "polymer_name"), "PLGA-5%®")

    def test_dictionary_layer_normalizes_headers_and_entity_surfaces(self) -> None:
        self.assertEqual(structure_dictionary.canonical_field_for_header("Index (PI)"), "pdi")
        self.assertEqual(structure_dictionary.canonical_field_for_header("Polymer Used"), "polymer_name")
        self.assertEqual(structure_dictionary.normalize_dictionary_value("drug_name", "DXI"), "dexibuprofen")
        self.assertEqual(structure_dictionary.normalize_dictionary_value("surfactant_name", "Tween 80"), "Tween80")
        self.assertEqual(structure_dictionary.normalize_dictionary_value("polymer_name", "plga 503 h"), "PLGA 503 H")
        self.assertEqual(s5_final.normalize_global_drug_candidate("DXI"), "dexibuprofen")
        self.assertEqual(s5_final._canonical_preparation_solvent("DCM"), "dichloromethane")
        self.assertEqual(s5_final._canonical_preparation_solvent("methylene chloride"), "dichloromethane")

    def test_dictionary_scope_boundaries_and_paper_local_priority(self) -> None:
        rows = [
            {"field_family": "surfactant_name", "surface_form": "Poloxamer", "canonical_form": "Poloxamer", "scope": "global", "paper_key": "", "normalization_rule": "exact"},
            {"field_family": "surfactant_name", "surface_form": "Poloxamer", "canonical_form": "Poloxamer 407", "scope": "paper_local", "paper_key": "UFXX9WXE", "normalization_rule": "exact"},
            {"field_family": "surfactant_name", "surface_form": "LOCALONLY", "canonical_form": "Local Surfactant", "scope": "paper_local", "paper_key": "UFXX9WXE", "normalization_rule": "exact"},
            {"field_family": "method_type", "surface_form": "local prep", "canonical_form": "paper_local_method", "scope": "paper_local", "paper_key": "", "normalization_rule": "exact"},
        ]
        with self.with_dictionary_rows(rows):
            self.assertEqual(
                structure_dictionary.normalize_dictionary_value("surfactant_name", "Poloxamer", paper_key="UFXX9WXE"),
                "Poloxamer 407",
            )
            self.assertEqual(
                structure_dictionary.normalize_dictionary_value("surfactant_name", "Poloxamer", paper_key="OTHER"),
                "Poloxamer",
            )
            self.assertEqual(
                structure_dictionary.normalize_dictionary_value("surfactant_name", "LOCALONLY", paper_key="OTHER"),
                "LOCALONLY",
            )
            self.assertEqual(
                structure_dictionary.normalize_dictionary_value("method_type", "local prep", paper_key="ANY"),
                "local prep",
            )

    def test_stage5_material_normalization_uses_dictionary_scope(self) -> None:
        rows = [
            {"field_family": "surfactant_name", "surface_form": "CP188", "canonical_form": "poloxamer 188", "scope": "global", "paper_key": "", "normalization_rule": "exact"},
            {"field_family": "surfactant_name", "surface_form": "LocalF68", "canonical_form": "Lutrol F68", "scope": "paper_local", "paper_key": "INMUTV7L", "normalization_rule": "exact"},
        ]
        with self.with_dictionary_rows(rows):
            self.assertEqual(s5_final.normalize_emulsifier_factor_candidate("CP188"), "poloxamer 188")
            self.assertEqual(s5_final.normalize_emulsifier_factor_candidate("LocalF68", paper_key="INMUTV7L"), "Lutrol F68")
            self.assertEqual(s5_final.normalize_emulsifier_factor_candidate("LocalF68", paper_key="OTHER"), "")

    def test_paper_local_drug_abbreviation_overlay_enables_drug_mass_header_binding(self) -> None:
        rows = [
            {
                "field_family": "drug_name",
                "surface_form": "SND",
                "canonical_form": "some new drug",
                "alias_type": "abbreviation",
                "scope": "paper_local",
                "paper_key": "PAPER1",
                "normalization_rule": "exact",
                "status": "approved_paper_local",
            }
        ]
        with self.with_dictionary_rows(rows):
            self.assertEqual(structure_dictionary.canonical_field_for_header("SND (mg)", paper_key="PAPER1"), "drug_mass_mg")
            self.assertEqual(structure_dictionary.canonical_field_for_header("SND (mg)", paper_key="OTHER"), "")

    def test_overlay_tsv_is_loaded_as_paper_local_dictionary_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            overlay = Path(tmpdir) / "paper_local_overlay.tsv"
            overlay.write_text(
                "field_family\tsurface_form\tcanonical_form\talias_type\tscope\tpaper_key\tnormalization_rule\tstatus\n"
                "drug_name\tSND\tsome new drug\tabbreviation\tpaper_local\tPAPER1\texact\tapproved_paper_local\n",
                encoding="utf-8",
            )
            with structure_dictionary.dictionary_overlay_paths([overlay]):
                self.assertEqual(structure_dictionary.normalize_dictionary_value("drug_name", "SND", paper_key="PAPER1"), "some new drug")
                self.assertEqual(structure_dictionary.canonical_field_for_header("SND (mg)", paper_key="PAPER1"), "drug_mass_mg")
                self.assertEqual(structure_dictionary.normalize_dictionary_value("drug_name", "SND", paper_key="OTHER"), "SND")

    def test_abbreviation_registry_extracts_paper_local_drug_candidate_from_text(self) -> None:
        rows = abbrev_registry.extract_abbreviation_candidates_from_text(
            paper_key="PAPER1",
            text="The model drug Some New Drug (SND) was dissolved. SND loaded PLGA nanoparticles were prepared.",
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["field_family"], "drug_name")
        self.assertEqual(row["surface_form"], "SND")
        self.assertEqual(row["canonical_form"], "Some New Drug")
        self.assertEqual(row["scope"], "paper_local")
        self.assertEqual(row["status"], "candidate")

    def test_promotion_review_recommends_global_candidate_only_without_conflicts(self) -> None:
        registry_rows = [
            {"paper_key": "P1", "field_family": "drug_name", "surface_form": "SND", "canonical_form": "Some New Drug", "scope": "paper_local", "status": "approved_paper_local"},
            {"paper_key": "P2", "field_family": "drug_name", "surface_form": "SND", "canonical_form": "Some New Drug", "scope": "paper_local", "status": "approved_paper_local"},
            {"paper_key": "P3", "field_family": "drug_name", "surface_form": "ABC", "canonical_form": "Alpha Beta", "scope": "paper_local", "status": "approved_paper_local"},
            {"paper_key": "P4", "field_family": "drug_name", "surface_form": "ABC", "canonical_form": "Alternate Beta", "scope": "paper_local", "status": "approved_paper_local"},
        ]
        review_rows = promotion_review.build_promotion_review_rows(registry_rows, global_lexicon_rows=[], min_support_papers=2)
        by_surface = {row["surface_form"]: row for row in review_rows}
        self.assertEqual(by_surface["SND"]["recommended_action"], "promote_to_global_candidate")
        self.assertEqual(by_surface["ABC"]["recommended_action"], "keep_paper_local_conflict")


if __name__ == "__main__":
    unittest.main()
