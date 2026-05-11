# Stage2 Table Authority Contract Alignment (2026-04-14)

## Purpose
Record the bounded contract hardening that separates the semantic-facing table
summary surface from the execution-facing full-table authority surface inside
maintained Stage2.

## Updated Paths
- `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`
- `src/stage2_sampling_labels/table_row_expansion_v1.py`
- `src/stage2_sampling_labels/function_units/doe_row_expansion_function_unit_v1.py`
- `project/2_ARCHITECTURE.md`
- `project/ACTIVE_PIPELINE_FLOW.md`
- `project/ACTIVE_PIPELINE_RUNBOOK.md`
- `project/PIPELINE_SCRIPT_MAP.md`
- `project/4_DECISIONS_LOG.md`
- `docs/maintained_script_surface.tsv`
- `docs/src_script_registry.tsv`
- `data/mem/v1/err.tsv`
- `data/mem/v1/idx.tsv`
- `data/mem/v1/prm.tsv`
- `data/mem/v1/run.tsv`

## Final Contract Wording
- `S2-2a` owns the execution-facing full-table authority surface.
- The current maintained full-table authority artifact is:
  `semantic_stage2_objects/normalized_table_payloads/<paper_key>/normalized_table_payloads_v1.json`
- The colocated `payloads/*.csv` files are additive execution payload members
  referenced by that artifact.
- Each preserved authority record must carry:
  - stable `table_id`
  - `source_table_reference`
  - deterministic `table_type`
  - `row_count`
  - `has_row_numbering`
  - `header_structure`
  - `raw_cells`
  - execution-facing `normalized_rows`
  - `row_identity_signals`
  - `reconstruction_confidence`
- S2-2 also writes `analysis/table_authority_validation_v1.tsv` as the
  maintained validation and observability surface for preserved table
  authority.
- `S2-2b` owns the semantic-facing summary or evidence packaging surface:
  `semantic_stage2_objects/evidence_blocks/<paper_key>/evidence_blocks_v1.json`
- Table-derived semantic-facing summary blocks must carry stable `table_id`
  and explicit `summary_is_lossy=true`.
- `S2-3` prompt assembly may consume only the semantic-facing summary or
  evidence surface.
- When deterministic row materialization is semantically authorized,
  downstream Stage2 execution must resolve back to the preserved S2-2
  full-table authority surface by stable table identity.
- DOE and non-DOE deterministic row materialization now share that same
  execution contract.
- The summary or evidence surface must never become the sole execution source
  of truth when a full-table authority surface exists.
- Stage1 table assets may remain a deterministic reconstruction fallback
  inside S2-2a only; they are no longer the downstream execution source of
  truth once authority exists.
- Engineering principle:
  the LLM sees a semantic-facing summary of a table, while deterministic
  execution operates on the preserved table entity.

## Code Surface Alignment
- No new owner scripts were introduced.
- The contract is aligned to the existing maintained owner surfaces:
  - `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`
  - `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`
  - `src/stage2_sampling_labels/table_row_expansion_v1.py`
  - `src/stage2_sampling_labels/function_units/doe_row_expansion_function_unit_v1.py`
- `extract_semantic_stage2_objects_v2.py` now emits an execution-grade
  authority record rather than a thin path manifest only.
- `table_row_expansion_v1.py` now resolves execution input through preserved
  S2-2 authority payloads and no longer treats Stage1 tables as the non-DOE
  execution source of truth.
- `doe_row_expansion_function_unit_v1.py` now treats authorized S2-2
  authority payload binding as required rather than silently falling back to
  heuristic execution input selection.
- The governed memory index was rebuilt with `python src/utils/build_mem_v1.py`
  so the new contract is discoverable through the repository memory layer.

## Open Questions Or Mismatches
- The maintained artifact name remains `normalized_table_payloads_v1.json`
  rather than a more explicit `full_table_authority_v1.json`.
  This note keeps the current name to avoid a disruptive artifact rename.
- The semantic-facing summary view is implemented through
  `evidence_blocks_v1.json` rather than a separately named
  `table_summary_view_v1.json`.
  The contract treats `evidence_blocks_v1.json` as the maintained summary or
  evidence surface.
- Older historical `normalized_table_payloads_v1.json` artifacts may still
  expose only `normalized_csv_path` rather than embedded `normalized_rows`.
  The maintained non-DOE execution path now accepts that authority-carried
  fallback, but long-term validation should prefer the richer embedded
  execution-grade record form.
- Some existing scoped S2-5 validation runs still reference authority-poor or
  marker-poor historical replay payloads.
  That affects bounded runtime validation coverage, but it is no longer a
  justification for direct Stage1 execution fallback.
