#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


OBJECT_TYPES: list[dict[str, Any]] = [
    {
        "object_type": "formulation_identity_candidate",
        "purpose": "Represent one candidate formulation identity with parent or variant semantics preserved.",
        "llm_emitted": True,
        "deterministic_postprocessed": True,
        "maps_to_db_v2": ["formulation_identity"],
        "example": {
            "document_key": "5ZXYABSU",
            "formulation_candidate_id": "NPR1",
            "raw_formulation_label": "NPR1",
            "instance_kind": "new_formulation",
            "formulation_role": "baseline",
            "candidate_source": "llm_extracted",
        },
        "fields": [
            {"field_name": "document_key", "required": "yes", "description": "Paper key for object grouping.", "source_stage": "Stage2 LLM", "normalization_later": "no", "evidence_role": "grouping_anchor", "notes": "Maps to document.zotero_key."},
            {"field_name": "doi", "required": "no", "description": "Paper DOI when available in the extraction context.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "grouping_anchor", "notes": "Useful but not identity-defining inside one paper."},
            {"field_name": "formulation_candidate_id", "required": "yes", "description": "Stage2 candidate-local formulation identifier.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "object_id", "notes": "Maps to formulation_identity.source_stage2_row_id before later canonicalization."},
            {"field_name": "raw_formulation_label", "required": "no", "description": "Raw article-native formulation label.", "source_stage": "Stage2 LLM", "normalization_later": "no", "evidence_role": "semantic_label", "notes": "Preserve author wording exactly when possible."},
            {"field_name": "parent_candidate_id", "required": "no", "description": "Raw parent or inherited formulation reference.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "relation_hint", "notes": "Used later for parent-link resolution."},
            {"field_name": "instance_kind", "required": "yes", "description": "Primary identity routing label such as new_formulation or variant_formulation.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "semantic_identity", "notes": "Keep compact enum; reconcile deterministically later if needed."},
            {"field_name": "formulation_role", "required": "no", "description": "Role label such as baseline, variant, control, or characterization_only.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "semantic_identity", "notes": "Useful for later Stage5 identity closure."},
            {"field_name": "identity_confidence", "required": "no", "description": "Coarse confidence that the candidate is a distinct formulation identity.", "source_stage": "Stage2 LLM", "normalization_later": "no", "evidence_role": "uncertainty_hint", "notes": "Formulation-level only; avoid per-field confidence sprawl."},
            {"field_name": "candidate_source", "required": "no", "description": "Source mode such as llm_extracted or deterministic_table_addition.", "source_stage": "Stage2 LLM or deterministic", "normalization_later": "yes", "evidence_role": "provenance_hint", "notes": "Deterministic additive rows may override this."},
            {"field_name": "change_descriptions", "required": "no", "description": "Short raw change descriptions that explain what differs from related formulations.", "source_stage": "Stage2 LLM", "normalization_later": "no", "evidence_role": "semantic_relation_hint", "notes": "Keep concise; avoid arbitration narrative."},
            {"field_name": "instance_context_tags", "required": "no", "description": "Compact semantic tags for context such as doe, sweep, or control.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "semantic_relation_hint", "notes": "Tags stay compact and non-exhaustive."},
            {"field_name": "change_context_tags", "required": "no", "description": "Compact semantic tags for change interpretation.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "semantic_relation_hint", "notes": "Used later for conservative relation and closure logic."},
        ],
    },
    {
        "object_type": "component_candidate",
        "purpose": "Represent one formulation component without fixed slot limits.",
        "llm_emitted": True,
        "deterministic_postprocessed": True,
        "maps_to_db_v2": ["formulation_component"],
        "example": {
            "formulation_candidate_id": "NPR1",
            "component_id": "NPR1__component_01",
            "component_name_raw": "PLGA",
            "component_role_raw": "polymer",
            "amount_expression_raw": "50 mg",
        },
        "fields": [
            {"field_name": "formulation_candidate_id", "required": "yes", "description": "Owning formulation candidate.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "grouping_anchor", "notes": "Links the component to one formulation candidate."},
            {"field_name": "component_id", "required": "yes", "description": "Stable within-payload component identifier.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "object_id", "notes": "Can be deterministically regenerated later."},
            {"field_name": "component_name_raw", "required": "yes", "description": "Raw component wording.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "semantic_value", "notes": "Preserve exact wording for audit."},
            {"field_name": "component_role_raw", "required": "yes", "description": "Raw role such as polymer, drug, surfactant, solvent, lipid, or additive.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "semantic_value", "notes": "Avoid fixed polymer or surfactant slots."},
            {"field_name": "phase_hint_raw", "required": "no", "description": "Raw phase assignment hint such as W1, O, W2, organic phase, or aqueous phase.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "relation_hint", "notes": "Phase-aware but low commitment."},
            {"field_name": "amount_expression_raw", "required": "no", "description": "Raw author-side amount expression.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "semantic_value", "notes": "Preserve direct mass, concentration, percentage, ratio, or volume forms."},
            {"field_name": "amount_kind_hint", "required": "no", "description": "Coarse hint such as mass, concentration, percent, volume, ratio, or unknown.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "parse_hint", "notes": "Deterministic parsing remains authoritative."},
            {"field_name": "parsed_value_raw", "required": "no", "description": "Raw numeric token surface if trivially available.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "parse_hint", "notes": "Do not force canonicalization here."},
            {"field_name": "parsed_unit_raw", "required": "no", "description": "Raw unit token surface if trivially available.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "parse_hint", "notes": "Useful for later deterministic conversion."},
            {"field_name": "component_properties_raw", "required": "no", "description": "Optional raw property list such as polymer MW text or LA:GA ratio text.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "semantic_value", "notes": "Use for family-specific properties without universalizing them."},
        ],
    },
    {
        "object_type": "phase_candidate",
        "purpose": "Represent an explicit formulation phase when the paper describes phase structure.",
        "llm_emitted": True,
        "deterministic_postprocessed": True,
        "maps_to_db_v2": ["formulation_phase"],
        "example": {
            "formulation_candidate_id": "F2",
            "phase_id": "F2__phase_O",
            "phase_code_raw": "organic phase",
            "phase_role_hint": "O",
        },
        "fields": [
            {"field_name": "formulation_candidate_id", "required": "yes", "description": "Owning formulation candidate.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "grouping_anchor", "notes": "Links the phase to one formulation candidate."},
            {"field_name": "phase_id", "required": "yes", "description": "Stable within-payload phase identifier.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "object_id", "notes": "Can be regenerated deterministically."},
            {"field_name": "phase_code_raw", "required": "yes", "description": "Raw phase wording or code.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "semantic_value", "notes": "Examples: W1, O, W2, organic phase."},
            {"field_name": "phase_role_hint", "required": "no", "description": "Coarse normalized hint for the phase role.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "parse_hint", "notes": "Deterministic canonicalization remains later."},
            {"field_name": "phase_order_hint", "required": "no", "description": "Order hint when the sequence matters.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "relation_hint", "notes": "Needed for W1/O/W2 style forms."},
        ],
    },
    {
        "object_type": "process_step_candidate",
        "purpose": "Represent preparation route or process-step semantics without collapsing them into fixed scalar slots.",
        "llm_emitted": True,
        "deterministic_postprocessed": True,
        "maps_to_db_v2": ["formulation_process"],
        "example": {
            "formulation_candidate_id": "NPR1",
            "process_step_id": "NPR1__proc_01",
            "process_name_raw": "solvent displacement",
            "parameter_name_raw": "stirring speed",
            "parameter_expression_raw": "800 rpm",
        },
        "fields": [
            {"field_name": "formulation_candidate_id", "required": "yes", "description": "Owning formulation candidate.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "grouping_anchor", "notes": "Links process facts to one formulation candidate."},
            {"field_name": "process_step_id", "required": "yes", "description": "Stable within-payload process-step identifier.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "object_id", "notes": "Can be regenerated later."},
            {"field_name": "process_name_raw", "required": "yes", "description": "Raw preparation route or step wording.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "semantic_value", "notes": "Supports emulsion and non-emulsion routes."},
            {"field_name": "process_step_order_hint", "required": "no", "description": "Order hint when the paper clearly sequences steps.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "relation_hint", "notes": "Deterministic ordering can refine this later."},
            {"field_name": "parameter_name_raw", "required": "no", "description": "Raw process parameter name.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "semantic_value", "notes": "Examples: sonication time, stirring speed, evaporation time."},
            {"field_name": "parameter_expression_raw", "required": "no", "description": "Raw parameter value expression.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "semantic_value", "notes": "Keep raw expression instead of early conversion."},
        ],
    },
    {
        "object_type": "variable_or_factor_candidate",
        "purpose": "Represent manipulated variables and DOE factors even when they are outside the legacy fixed field list.",
        "llm_emitted": True,
        "deterministic_postprocessed": True,
        "maps_to_db_v2": ["formulation_process", "formulation_component", "formulation_measurement"],
        "example": {
            "formulation_candidate_id": "Run_11",
            "factor_id": "Run_11__factor_01",
            "factor_name_raw": "organic-to-aqueous phase ratio",
            "factor_expression_raw": "1:6",
            "identity_defining_signal": "yes",
        },
        "fields": [
            {"field_name": "formulation_candidate_id", "required": "yes", "description": "Owning formulation candidate.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "grouping_anchor", "notes": "Links the factor to the formulation identity it helps define."},
            {"field_name": "factor_id", "required": "yes", "description": "Stable within-payload factor identifier.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "object_id", "notes": "May later map to process or component records."},
            {"field_name": "factor_name_raw", "required": "yes", "description": "Raw manipulated variable or factor wording.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "semantic_value", "notes": "Not restricted to current CORE_FIELDS."},
            {"field_name": "factor_expression_raw", "required": "no", "description": "Raw reported expression for the factor value.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "semantic_value", "notes": "Supports arbitrary DOE expressions."},
            {"field_name": "factor_entity_hint", "required": "no", "description": "Hint about what the factor modifies such as component, phase, process, or measurement condition.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "relation_hint", "notes": "Deterministic consolidation resolves ownership later."},
            {"field_name": "identity_defining_signal", "required": "no", "description": "Whether the paper treats this factor as formulation-identity defining.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "semantic_relation_hint", "notes": "Low-commitment cue only."},
        ],
    },
    {
        "object_type": "measurement_candidate",
        "purpose": "Represent measured formulation outcomes or reported outputs separately from identity fields.",
        "llm_emitted": True,
        "deterministic_postprocessed": True,
        "maps_to_db_v2": ["formulation_measurement"],
        "example": {
            "formulation_candidate_id": "NPR1",
            "measurement_id": "NPR1__measurement_01",
            "measurement_name_raw": "encapsulation efficiency",
            "measurement_value_raw": "55.3",
            "measurement_unit_raw": "%",
        },
        "fields": [
            {"field_name": "formulation_candidate_id", "required": "yes", "description": "Owning formulation candidate.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "grouping_anchor", "notes": "Links the measurement to one candidate formulation."},
            {"field_name": "measurement_id", "required": "yes", "description": "Stable within-payload measurement identifier.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "object_id", "notes": "Can be regenerated later."},
            {"field_name": "measurement_name_raw", "required": "yes", "description": "Raw measurement name.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "semantic_value", "notes": "Examples: size, PDI, zeta potential, EE, loading content."},
            {"field_name": "measurement_value_raw", "required": "no", "description": "Raw measurement value text.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "semantic_value", "notes": "Keep raw before canonical numeric parsing."},
            {"field_name": "measurement_unit_raw", "required": "no", "description": "Raw measurement unit text.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "parse_hint", "notes": "Needed for later deterministic conversion."},
            {"field_name": "statistic_qualifier_raw", "required": "no", "description": "Raw statistic qualifier such as mean, mean+sd, or single value.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "parse_hint", "notes": "Useful for later measurement normalization."},
            {"field_name": "measurement_context_raw", "required": "no", "description": "Raw contextual qualifier such as release timepoint or medium.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "semantic_value", "notes": "Keeps measurement context separate from identity."},
        ],
    },
    {
        "object_type": "relation_cue",
        "purpose": "Represent low-commitment semantic links among discovered objects without forcing final deterministic arbitration into Stage2.",
        "llm_emitted": True,
        "deterministic_postprocessed": True,
        "maps_to_db_v2": ["formulation_identity", "formulation_process", "formulation_component"],
        "example": {
            "cue_id": "cue_01",
            "cue_type": "inherits_from",
            "source_object_ref": "F2",
            "target_object_ref": "F1",
            "cue_text": "prepared similarly to F1 except for PVA concentration",
        },
        "fields": [
            {"field_name": "cue_id", "required": "yes", "description": "Stable within-payload relation cue identifier.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "object_id", "notes": "Can be regenerated later."},
            {"field_name": "cue_type", "required": "yes", "description": "Semantic link type such as inherits_from, varies_on, shared_with, or measured_for.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "semantic_relation_hint", "notes": "Keep cue ontology compact."},
            {"field_name": "source_object_ref", "required": "yes", "description": "Local source object identifier or formulation candidate id.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "relation_hint", "notes": "Can point to an object in the same payload."},
            {"field_name": "target_object_ref", "required": "yes", "description": "Local target object identifier or formulation candidate id.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "relation_hint", "notes": "Used later by deterministic materialization."},
            {"field_name": "cue_text", "required": "no", "description": "Short raw explanation for the semantic link.", "source_stage": "Stage2 LLM", "normalization_later": "no", "evidence_role": "semantic_relation_hint", "notes": "Avoid long arbitration notes."},
            {"field_name": "cue_confidence", "required": "no", "description": "Coarse confidence in the link.", "source_stage": "Stage2 LLM", "normalization_later": "no", "evidence_role": "uncertainty_hint", "notes": "Optional and compact."},
        ],
    },
    {
        "object_type": "evidence_handoff",
        "purpose": "Represent low-commitment source hints that downstream deterministic binding can refine.",
        "llm_emitted": True,
        "deterministic_postprocessed": True,
        "maps_to_db_v2": ["evidence_binding"],
        "example": {
            "handoff_id": "handoff_01",
            "target_object_ref": "NPR1__component_01",
            "target_field_name": "amount_expression_raw",
            "source_region_type": "table_row",
            "source_locator_text": "Table 1, NPR1 row",
        },
        "fields": [
            {"field_name": "handoff_id", "required": "yes", "description": "Stable within-payload evidence handoff identifier.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "object_id", "notes": "Can be regenerated later."},
            {"field_name": "target_object_ref", "required": "yes", "description": "Local object identifier receiving the evidence hint.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "binding_hint", "notes": "Not final ownership binding."},
            {"field_name": "target_field_name", "required": "yes", "description": "Field on the local object that the source hint refers to.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "binding_hint", "notes": "Useful for later deterministic evidence assembly."},
            {"field_name": "source_region_type", "required": "yes", "description": "Coarse region type such as table_row, caption, methods_sentence, or results_sentence.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "coarse_handoff", "notes": "Coarse only; not audit-grade binding."},
            {"field_name": "source_locator_text", "required": "no", "description": "Short human-readable locator text.", "source_stage": "Stage2 LLM", "normalization_later": "no", "evidence_role": "coarse_handoff", "notes": "Examples: Table 2 row F3, methods paragraph on nanoprecipitation."},
            {"field_name": "supporting_snippet", "required": "no", "description": "Short excerpt for deterministic routing only.", "source_stage": "Stage2 LLM", "normalization_later": "no", "evidence_role": "coarse_handoff", "notes": "Do not treat this as final evidence ownership."},
            {"field_name": "support_role", "required": "no", "description": "Hint such as direct_support or contextual_support.", "source_stage": "Stage2 LLM", "normalization_later": "yes", "evidence_role": "coarse_handoff", "notes": "Downstream evidence binding remains authoritative."},
        ],
    },
]

LEGACY_MAPPING_ROWS: list[dict[str, str]] = [
    {"legacy_field": "formulation_id", "replacement_object_type": "formulation_identity_candidate", "replacement_field": "formulation_candidate_id", "mapping_kind": "rename", "preserve_raw": "yes", "deprecated": "no", "notes": "Candidate-local id remains needed during transition."},
    {"legacy_field": "raw_formulation_label", "replacement_object_type": "formulation_identity_candidate", "replacement_field": "raw_formulation_label", "mapping_kind": "direct", "preserve_raw": "yes", "deprecated": "no", "notes": "Preserve author wording."},
    {"legacy_field": "parent_instance_id", "replacement_object_type": "formulation_identity_candidate", "replacement_field": "parent_candidate_id", "mapping_kind": "rename", "preserve_raw": "yes", "deprecated": "no", "notes": "Resolved parent ids happen later."},
    {"legacy_field": "instance_kind", "replacement_object_type": "formulation_identity_candidate", "replacement_field": "instance_kind", "mapping_kind": "direct", "preserve_raw": "yes", "deprecated": "no", "notes": "Keep compact routing enum."},
    {"legacy_field": "formulation_role", "replacement_object_type": "formulation_identity_candidate", "replacement_field": "formulation_role", "mapping_kind": "direct", "preserve_raw": "yes", "deprecated": "no", "notes": "Supports Stage5 later."},
    {"legacy_field": "change_descriptions", "replacement_object_type": "formulation_identity_candidate", "replacement_field": "change_descriptions", "mapping_kind": "direct", "preserve_raw": "yes", "deprecated": "no", "notes": "Stay concise."},
    {"legacy_field": "instance_context_tags", "replacement_object_type": "formulation_identity_candidate", "replacement_field": "instance_context_tags", "mapping_kind": "direct", "preserve_raw": "yes", "deprecated": "no", "notes": "Compact semantic tags."},
    {"legacy_field": "change_context_tags", "replacement_object_type": "formulation_identity_candidate", "replacement_field": "change_context_tags", "mapping_kind": "direct", "preserve_raw": "yes", "deprecated": "no", "notes": "Compact semantic tags."},
    {"legacy_field": "polymer_identity", "replacement_object_type": "component_candidate", "replacement_field": "component_role_raw=polymer + component_name_raw", "mapping_kind": "split", "preserve_raw": "yes", "deprecated": "yes", "notes": "Do not keep as a single fixed-slot field in the long term."},
    {"legacy_field": "polymer_name_raw", "replacement_object_type": "component_candidate", "replacement_field": "component_name_raw", "mapping_kind": "direct", "preserve_raw": "yes", "deprecated": "yes", "notes": "Belongs on polymer component records."},
    {"legacy_field": "la_ga_ratio_value_text", "replacement_object_type": "component_candidate", "replacement_field": "component_properties_raw.la_ga_ratio", "mapping_kind": "move_to_component_property", "preserve_raw": "yes", "deprecated": "yes", "notes": "Family-specific polymer property, not universal formulation field."},
    {"legacy_field": "plga_mw_kDa_value_text", "replacement_object_type": "component_candidate", "replacement_field": "component_properties_raw.molecular_weight", "mapping_kind": "legacy_rename_and_move", "preserve_raw": "yes", "deprecated": "yes", "notes": "Retire PLGA-specific naming; use generic polymer component property."},
    {"legacy_field": "polymer_mw_kDa_value_text", "replacement_object_type": "component_candidate", "replacement_field": "component_properties_raw.molecular_weight", "mapping_kind": "move_to_component_property", "preserve_raw": "yes", "deprecated": "yes", "notes": "Generic polymer MW belongs on the polymer component."},
    {"legacy_field": "plga_mass_mg_value_text", "replacement_object_type": "component_candidate", "replacement_field": "amount_expression_raw", "mapping_kind": "move_to_component", "preserve_raw": "yes", "deprecated": "yes", "notes": "Use polymer component amount expression, not a formulation-wide slot."},
    {"legacy_field": "surfactant_name_value_text", "replacement_object_type": "component_candidate", "replacement_field": "component_name_raw", "mapping_kind": "move_to_component", "preserve_raw": "yes", "deprecated": "yes", "notes": "Use one record per surfactant or stabilizer."},
    {"legacy_field": "surfactant_concentration_text_value_text", "replacement_object_type": "component_candidate", "replacement_field": "amount_expression_raw", "mapping_kind": "move_to_component", "preserve_raw": "yes", "deprecated": "yes", "notes": "Supports multiple stabilizers cleanly."},
    {"legacy_field": "pva_conc_percent_value_text", "replacement_object_type": "component_candidate", "replacement_field": "amount_expression_raw", "mapping_kind": "demote_special_case", "preserve_raw": "yes", "deprecated": "yes", "notes": "Retire dedicated PVA slot; represent as a component amount."},
    {"legacy_field": "organic_solvent_value_text", "replacement_object_type": "component_candidate", "replacement_field": "component_name_raw", "mapping_kind": "move_to_component", "preserve_raw": "yes", "deprecated": "yes", "notes": "Supports multiple solvents and co-solvents."},
    {"legacy_field": "drug_name_value_text", "replacement_object_type": "component_candidate", "replacement_field": "component_name_raw", "mapping_kind": "move_to_component", "preserve_raw": "yes", "deprecated": "yes", "notes": "Drug becomes a component record."},
    {"legacy_field": "drug_feed_amount_text_value_text", "replacement_object_type": "component_candidate", "replacement_field": "amount_expression_raw", "mapping_kind": "move_to_component", "preserve_raw": "yes", "deprecated": "yes", "notes": "Keep raw amount before deterministic conversion."},
    {"legacy_field": "emul_method_value_text", "replacement_object_type": "process_step_candidate", "replacement_field": "process_name_raw", "mapping_kind": "move_to_process", "preserve_raw": "yes", "deprecated": "yes", "notes": "Generalize beyond emulsion-specific naming."},
    {"legacy_field": "emul_type_value_text", "replacement_object_type": "process_step_candidate", "replacement_field": "process_name_raw or phase relation", "mapping_kind": "split", "preserve_raw": "yes", "deprecated": "yes", "notes": "Some values belong to process type, some to phase structure."},
    {"legacy_field": "preparation_method", "replacement_object_type": "process_step_candidate", "replacement_field": "process_name_raw", "mapping_kind": "move_to_process", "preserve_raw": "yes", "deprecated": "no", "notes": "Useful generalized process field."},
    {"legacy_field": "emulsion_structure", "replacement_object_type": "phase_candidate", "replacement_field": "phase_code_raw", "mapping_kind": "move_to_phase", "preserve_raw": "yes", "deprecated": "no", "notes": "Useful phase-aware field."},
    {"legacy_field": "size_nm_value_text", "replacement_object_type": "measurement_candidate", "replacement_field": "measurement_value_raw", "mapping_kind": "move_to_measurement", "preserve_raw": "yes", "deprecated": "yes", "notes": "Use measurement object with name=size_nm."},
    {"legacy_field": "pdi_value_text", "replacement_object_type": "measurement_candidate", "replacement_field": "measurement_value_raw", "mapping_kind": "move_to_measurement", "preserve_raw": "yes", "deprecated": "yes", "notes": "Use measurement object with name=pdi."},
    {"legacy_field": "zeta_mV_value_text", "replacement_object_type": "measurement_candidate", "replacement_field": "measurement_value_raw", "mapping_kind": "move_to_measurement", "preserve_raw": "yes", "deprecated": "yes", "notes": "Use measurement object with name=zeta_mV."},
    {"legacy_field": "encapsulation_efficiency_percent_value_text", "replacement_object_type": "measurement_candidate", "replacement_field": "measurement_value_raw", "mapping_kind": "move_to_measurement", "preserve_raw": "yes", "deprecated": "yes", "notes": "Primary outcome measurement."},
    {"legacy_field": "loading_content_percent_value_text", "replacement_object_type": "measurement_candidate", "replacement_field": "measurement_value_raw", "mapping_kind": "move_to_measurement", "preserve_raw": "yes", "deprecated": "yes", "notes": "Primary outcome measurement when present."},
    {"legacy_field": "*_scope", "replacement_object_type": "relation_cue", "replacement_field": "cue_type/shared_or_instance_specific", "mapping_kind": "demote_to_relation_hint", "preserve_raw": "optional", "deprecated": "yes", "notes": "Do not keep per-field ownership as the main Stage2 surface."},
    {"legacy_field": "*_membership_confidence", "replacement_object_type": "formulation_identity_candidate", "replacement_field": "identity_confidence", "mapping_kind": "collapse_and_demote", "preserve_raw": "optional", "deprecated": "yes", "notes": "Per-field confidence burden moves out of Stage2."},
    {"legacy_field": "*_evidence_region_type", "replacement_object_type": "evidence_handoff", "replacement_field": "source_region_type", "mapping_kind": "collapse_and_demote", "preserve_raw": "yes", "deprecated": "yes", "notes": "Keep only coarse evidence hints."},
    {"legacy_field": "*_missing_reason", "replacement_object_type": "measurement_candidate or component_candidate", "replacement_field": "notes_raw", "mapping_kind": "optional_annotation", "preserve_raw": "optional", "deprecated": "yes", "notes": "Useful for audit, not core contract."},
    {"legacy_field": "supporting_evidence_refs", "replacement_object_type": "evidence_handoff", "replacement_field": "source_locator_text/supporting_snippet", "mapping_kind": "demote_to_coarse_handoff", "preserve_raw": "yes", "deprecated": "yes", "notes": "Avoid mixed-bag evidence ownership logic in Stage2."},
    {"legacy_field": "instance_evidence_region_type", "replacement_object_type": "evidence_handoff", "replacement_field": "source_region_type", "mapping_kind": "demote_to_coarse_handoff", "preserve_raw": "yes", "deprecated": "yes", "notes": "Coarse handoff only."},
    {"legacy_field": "evidence_section", "replacement_object_type": "evidence_handoff", "replacement_field": "source_locator_text", "mapping_kind": "demote_to_coarse_handoff", "preserve_raw": "yes", "deprecated": "yes", "notes": "Keep only as a routing hint."},
    {"legacy_field": "evidence_span_text", "replacement_object_type": "evidence_handoff", "replacement_field": "supporting_snippet", "mapping_kind": "demote_to_coarse_handoff", "preserve_raw": "yes", "deprecated": "yes", "notes": "Not final binding authority."},
    {"legacy_field": "evidence_span_start/evidence_span_end", "replacement_object_type": "evidence_handoff", "replacement_field": "none", "mapping_kind": "remove_from_llm_contract", "preserve_raw": "no", "deprecated": "yes", "notes": "Exact anchoring should be deterministic."},
    {"legacy_field": "paper_notes", "replacement_object_type": "none", "replacement_field": "none", "mapping_kind": "retire_required_output", "preserve_raw": "optional", "deprecated": "yes", "notes": "Do not require arbitration narrative from the LLM."},
]

RESPONSIBILITY_BOUNDARY = {
    "stage2_llm_should_do": [
        "semantic object discovery",
        "formulation identity discovery",
        "component discovery",
        "phase discovery",
        "process-step discovery",
        "variable and factor discovery",
        "measurement discovery",
        "raw expression capture",
        "relation cue extraction",
        "coarse evidence handoff",
    ],
    "stage2_llm_should_not_do": [
        "final fixed-slot normalization",
        "audit-grade evidence ownership binding",
        "definitive unit conversion",
        "premature derived-value generation",
        "long conflict-arbitration narration",
        "final modeling-table filling",
    ],
    "deterministic_postprocessing_should_do": [
        "object consolidation",
        "normalization and canonicalization",
        "derived value generation",
        "exact evidence refinement and binding",
        "compatibility projection for Stage3 and Stage5",
        "database assembly",
    ],
}

COMPATIBILITY_PROJECTION = {
    "status": "transitional_non_authoritative_until_adopted",
    "purpose": "Provide a deterministic bridge from semantic-object Stage2 outputs back to the legacy wide-row surface expected by current Stage3 and Stage5 consumers.",
    "current_downstream_consumers": [
        "src/stage3_relation/build_formulation_relation_artifacts_v1.py",
        "src/stage5_benchmark/build_minimal_final_output_v1.py",
    ],
    "projection_groups": [
        {"group_name": "identity_projection", "legacy_fields": ["formulation_id", "raw_formulation_label", "instance_kind", "parent_instance_id", "formulation_role", "change_descriptions", "instance_context_tags", "change_context_tags"], "source_objects": ["formulation_identity_candidate", "relation_cue"]},
        {"group_name": "fixed_slot_component_compatibility", "legacy_fields": ["polymer_identity", "polymer_name_raw", "plga_mass_mg_*", "surfactant_name_*", "surfactant_concentration_text_*", "organic_solvent_*", "drug_name_*", "drug_feed_amount_text_*"], "source_objects": ["component_candidate"]},
        {"group_name": "measurement_projection", "legacy_fields": ["size_nm_*", "pdi_*", "zeta_mV_*", "encapsulation_efficiency_percent_*", "loading_content_percent_*"], "source_objects": ["measurement_candidate"]},
        {"group_name": "coarse_evidence_projection", "legacy_fields": ["supporting_evidence_refs", "instance_evidence_region_type", "evidence_section", "evidence_span_text"], "source_objects": ["evidence_handoff"]},
    ],
}


def build_manifest() -> dict[str, Any]:
    return {
        "schema_name": "stage2_replacement_semantic_contract_v1",
        "schema_date": "2026-03-25",
        "status": "transitional_design_scaffold",
        "active_runtime_status": "not_active_mainline",
        "source_note": "Initial Stage2 replacement scaffold. Active benchmark runtime remains the current wide-row Stage2 extractor until a deterministic compatibility projection is adopted.",
        "responsibility_boundary": RESPONSIBILITY_BOUNDARY,
        "object_types": OBJECT_TYPES,
        "compatibility_projection": COMPATIBILITY_PROJECTION,
    }


def flatten_contract_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for object_type in OBJECT_TYPES:
        for field in object_type["fields"]:
            rows.append(
                {
                    "object_type": object_type["object_type"],
                    "field_name": field["field_name"],
                    "required": field["required"],
                    "description": field["description"],
                    "source_stage": field["source_stage"],
                    "normalization_later": field["normalization_later"],
                    "evidence_role": field["evidence_role"],
                    "notes": field["notes"],
                }
            )
    return rows


def write_tsv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_outputs(
    *,
    manifest_path: Path,
    contract_tsv_path: Path,
    mapping_tsv_path: Path,
) -> dict[str, str]:
    manifest = build_manifest()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    contract_rows = flatten_contract_rows()
    write_tsv(
        contract_tsv_path,
        contract_rows,
        [
            "object_type",
            "field_name",
            "required",
            "description",
            "source_stage",
            "normalization_later",
            "evidence_role",
            "notes",
        ],
    )
    write_tsv(
        mapping_tsv_path,
        LEGACY_MAPPING_ROWS,
        [
            "legacy_field",
            "replacement_object_type",
            "replacement_field",
            "mapping_kind",
            "preserve_raw",
            "deprecated",
            "notes",
        ],
    )
    return {
        "manifest_path": str(manifest_path),
        "contract_tsv_path": str(contract_tsv_path),
        "mapping_tsv_path": str(mapping_tsv_path),
        "object_type_count": str(len(OBJECT_TYPES)),
        "contract_field_count": str(len(contract_rows)),
        "legacy_mapping_count": str(len(LEGACY_MAPPING_ROWS)),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write the Stage2 replacement semantic contract scaffold artifacts."
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data/db/db_v2/schema_manifest_v2_replacement.json"),
    )
    parser.add_argument(
        "--contract-tsv-path",
        type=Path,
        default=Path("data/db/db_v2/stage2_replacement_output_contract.tsv"),
    )
    parser.add_argument(
        "--mapping-tsv-path",
        type=Path,
        default=Path("data/db/db_v2/stage2_legacy_to_replacement_mapping.tsv"),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = write_outputs(
        manifest_path=args.manifest_path,
        contract_tsv_path=args.contract_tsv_path,
        mapping_tsv_path=args.mapping_tsv_path,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
