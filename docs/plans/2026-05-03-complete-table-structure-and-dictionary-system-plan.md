# Complete Table Structure and Dictionary System Plan

Goal: implement two bounded, reusable improvements for simple/complete formulation tables after semantic authorization:

1. complete table-structure recovery
2. a systematic dictionary layer shared by Stage2 and Stage5 consumers

This plan is implementation-facing. It is not a governance document and must stay outside `project/`.

---

## Problem statement

Current simple-table recovery is still split across narrow heuristics:
- S2 header inference can preserve CSV index rows (`0,1,2,...`) as headers.
- prelude/header flattening can mix caption/prose fragments into assignment headers.
- continuation/group rows that carry polymer or family identity are not treated as structured carry-down context.
- header aliases, material aliases, drug aliases, and paper-local abbreviations are not exposed through one shared dictionary API.

This leaves Stage5 with residual gaps that should have been resolved deterministically from the preserved table surface.

---

## Scope

In scope:
- improve Stage2 structure recovery for bounded complete formulation tables
- remove top numeric index rows from header surfaces
- ignore caption/prose noise rows when inferring header hierarchy
- recover multi-row headers into stable flattened headers
- recover continuation/group rows as deterministic carry-down context for explicit formulation rows
- add one reusable dictionary module for:
  - `header -> canonical field`
  - `surface value -> canonical material/drug/surfactant/polymer identity`
  - paper-local abbreviation lookup from the same lexicon surface
- wire Stage2 and Stage5 consumers to the same dictionary API
- add bounded unit tests covering the INMUTV7L-style pattern

Out of scope:
- new semantic discovery
- generalized arbitrary table mining
- creating new formulation rows without semantic authorization
- turning Stage5 into a new semantic authority layer

---

## Target files

New files:
- `src/stage2_sampling_labels/table_structure_dictionary_v1.py`
- `tests/test_table_structure_dictionary_v1.py`

Modify:
- `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`
- `src/stage2_sampling_labels/table_row_expansion_v1.py`
- `src/stage5_benchmark/build_minimal_final_output_v1.py`
- `data/cleaned/reference/value_normalization_lexicon_v1.tsv`

Do not modify:
- `project/`
- GT authority assets
- benchmark reporting contracts

---

## Design

### A. Shared structure + dictionary module

Create `table_structure_dictionary_v1.py` as the single reusable bounded surface for:
- lexicon loading/caching
- canonical header lookup
- canonical entity normalization by `field_family`
- numeric index row detection
- caption/prose metadata row filtering
- multi-row header recovery
- continuation/group row carry-down extraction

The module must remain deterministic and structure-only. It may interpret table geometry and lexicon aliases, but it must not create row identity beyond already explicit formulation rows.

### B. Header recovery rules

For simple/complete formulation tables:
- drop leading numeric index rows like `0,1,2,...`
- skip caption/prose metadata rows during header inference
- preserve the trailing contiguous header block immediately before the first explicit formulation row
- flatten multi-row headers column-wise with deduped text joins
- keep measurement tails such as `Average Size (nm)`, `Polydispersity Index (PI)`, `Zeta Potential (ZP, mV)`, `EE (%)`
- recover assignment headers such as `Polymer Used` and `Surfactant` without caption pollution

### C. Continuation/group row carry-down

For rows after the first explicit formulation row:
- if a non-explicit row contains sparse text only in variable/identity columns and no measurement payload, treat it as continuation/group context
- apply that context to:
  - the nearest previous explicit formulation row
  - subsequent explicit rows until the next continuation/group row
- allow a newer continuation/group row to replace an older carried context for the immediately previous explicit row

This is the bounded mechanism needed for tables where polymer/family identity is emitted as a separate structural row.

### D. Dictionary system

Use the existing lexicon TSV as the source of truth, but expose one shared API with:
- `canonical_field_for_header(...)`
- `normalize_dictionary_value(field_family=..., value=..., paper_key=...)`
- paper-local lookup support when `scope=paper_local`

Extend lexicon entries for the immediate bounded class:
- `Polymer Used -> polymer_name`
- `Surfactant -> surfactant_name`
- common surfactant aliases such as `Tween80`, `Tween 80`, `Lutrol`
- common polymer surfaces such as `PLGA`, `PLGA 503 H`, `PLGA-5%`, `PLGA 10%`, `PLGA 15%`

Stage5 should use the same normalization surface for drug/material identity carrythrough instead of ad hoc partial normalization only.

---

## Implementation tasks

### Task 1. Add the shared module
- implement lexicon cache/lookup helpers
- implement bounded row classifiers:
  - numeric index row
  - caption/prose metadata row
  - header-like row
  - explicit formulation row
  - continuation/group row
- implement header recovery helpers
- implement continuation/group carry-down helpers

### Task 2. Rewire Stage2 header inference
- update `extract_semantic_stage2_objects_v2.py::infer_header_structure(...)`
- consume the shared structure helpers
- ensure normalized row entries start after recovered header rows, not after numeric CSV index rows

### Task 3. Rewire Stage2 direct table-row expansion
- replace local `canonical_field_for_header` ownership with shared module delegation
- replace old `combined_prelude_headers(...)` based logic with shared recovered-structure helpers
- apply continuation/group carry-down so extracted direct rows contain row-local polymer/surfactant identity where structurally authorized

### Task 4. Rewire Stage5 identity normalization
- keep Stage5 materialization-only
- use shared dictionary normalization for drug/material/surfactant/polymer canonicalization where Stage5 already has lawful source-backed carrythrough paths
- do not add new semantic discovery

### Task 5. Expand the lexicon
- append bounded field and identity aliases needed by the simple complete-table class
- prefer generic reusable entries over paper-specific branches
- use paper-local entries only where the surface is truly paper-local

### Task 6. Validate
- add unit tests for:
  - header recovery on an INMUTV7L-style matrix
  - canonical header mapping
  - dictionary value normalization
  - continuation/group carry-down
  - direct extraction producing `surfactant_name`, `pdi`, `ee_percent`, and carried `polymer_name`
- run targeted unittest coverage

---

## Acceptance criteria

The implementation is successful when all are true:
- a leading `0,1,2,...` row is no longer treated as header structure
- caption/prose rows are not flattened into assignment headers
- `Polymer Used` and `Surfactant` can be recovered as variable headers for complete formulation tables
- continuation/group rows can carry polymer/family identity onto explicit formulation rows
- one shared dictionary API is used by both Stage2 and Stage5 consumers
- tests pass for the bounded INMUTV7L-style class

---

## Validation commands

```bash
python3 -m unittest tests/test_table_structure_dictionary_v1.py
python3 -m unittest tests/test_s5_value_layer_contract_v1.py
```

If a bounded paper replay is needed after code tests, keep it single-paper and diagnostic-only.

---

## 2026-05-04 DEV15 boundary-fix diagnostic addendum

After wiring the shared structure/dictionary layer into maintained mainline replay, DEV15 exposed a boundary bug: stronger table parsing increased direct-row recoverability on non-primary helper/sequential-child surfaces, and those rows could enter the benchmark-facing formulation universe through the existing Stage2 direct-row path.

### Diagnosed failure mode
- problem was not the shared table repair itself
- problem was hidden dependence on parse failure as accidental noise suppression
- once non-primary tables parsed cleanly, existing direct-row emission logic could over-admit them

### Generic boundary fixes applied in `table_row_expansion_v1.py`
- non-primary direct rows now pass a measurement/helper screening step before emission
- helper-style non-primary tables may remain available as anchor evidence for downstream bounded recovery, without being emitted directly into the main formulation universe
- non-primary surfaces that duplicate an already-emitted primary label set are blocked as duplicate label surfaces
- primary-looking formulation identity labels remain eligible so legitimate non-full-formulation benchmark rows are not blindly removed

### DEV15 replay outcomes so far
- first mainline replay regressed badly due to over-admission, especially `INMUTV7L` doubling from 12 to 24 rows
- boundary fixes restored `INMUTV7L` to 12 and later restored `QLYKLPKT` to 7
- current remaining count regressions are concentrated in:
  - `5GIF3D8W` under by 8
  - `PA3SPZ28` over by 4
- latest diagnostic replay reduced the broad regression substantially, but has not yet fully matched the accepted baseline

### Current rule-level conclusion
The correct architecture is:
- shared table repair may improve parsing for all tables
- benchmark-facing formulation admission must be governed independently
- non-primary/helper/sequential-child tables must never rely on parse failure for exclusion

### Boundaryfix8 result
A later bounded fix in `extract_column_anchor_rows_from_authority(...)` repaired the transposed optimized-formulation table failure mode directly. When shared header recovery consumes too many leading rows on a horizontal authority table, column-anchor extraction now retries against the raw normalized matrix / CSV rows instead of only the header-trimmed row-entry view.

That restored `5GIF3D8W` Table 4 optimized formulation rows without reopening the helper/sequential-child over-admission regression.

Latest DEV15 maintained diagnostic replay:
- Stage2: `229_stage2_full_replay_tabledict_boundaryfix8_diagnostic`
- Stage3: `230_stage3_tabledict_boundaryfix8_diagnostic/relation_artifacts`
- Stage5: `231_stage5_tabledict_boundaryfix8_no_s53_diagnostic`
- final compare: `233_compare_final_tabledict_boundaryfix8_diagnostic`
- layer3 compare: `234_layer3_compare_tabledict_boundaryfix8_diagnostic`

Observed outcomes:
- `5GIF3D8W` restored from `18 -> 26` final rows
- `INMUTV7L` remained restored at `12 -> 12`
- `QLYKLPKT` remained restored at `7 -> 7`
- `UFXX9WXE` remained at `27 -> 27`
- `PA3SPZ28` remains the only final-count mismatch in this diagnostic lineage (`7 vs 3` in the current GT-count file)

Accepted-baseline comparison at Layer3:
- `core_fixed_fields system_nonempty_on_gt_cells = 2074` (matches accepted baseline exactly)
- `core_fixed_fields value_recall = 0.664957` (matches accepted baseline exactly)
- `core_fixed_fields conditional_accuracy_strict = 0.612343` vs baseline `0.595468`
- `core_fixed_fields conditional_accuracy_relaxed = 0.933462` vs baseline `0.922372`
- `core_fixed_fields extra_in_system_cells = 27` vs baseline `30`
- `risk_review_queue_rows = 1608` vs baseline `1634`

Interpretation:
- the main regression introduced by stronger shared table parsing is repaired
- baseline Layer3 coverage has been recovered
- accuracy improved and risk rows decreased relative to the accepted baseline
- remaining follow-up is narrow: reconcile `PA3SPZ28` over-retention against the current GT-count authority without regressing the recovered mainline totals

### Boundaryfix14 result

`PA3SPZ28` residual was traced to table-row admission rather than table parsing:
- Table 1 is a ratio-coordinate DOE-style surface and remains the lawful 3-row count source for the current GT authority.
- Table 8 is a sequential-child/helper surface with the same ratio-coordinate labels and must not duplicate Table 1 into the benchmark-facing formulation universe.
- Table 7 is a single aggregate variant-list row listing helper/control particle types and must remain evidence/anchor context, not a main formulation row.

Generic bounded repair in `table_row_expansion_v1.py`:
- block weak ratio-coordinate row-universe surfaces only when they are auxiliary/non-primary table roles such as sequential-child/helper surfaces, while preserving true DOE/full primary surfaces;
- block aggregate variant-list rows before they can become main formulation rows;
- apply the same admission guard to direct-row, column-anchor, and assignment-row extraction paths;
- keep blocked helper rows available as anchor/evidence context instead of deleting table evidence.

Maintained DEV15 replay without S5-3:
- Stage2: `273_stage2_full_replay_tabledict_boundaryfix14_diagnostic`
- Stage3: `274_stage3_tabledict_boundaryfix14_diagnostic/relation_artifacts`
- Stage5: `275_stage5_tabledict_boundaryfix14_no_s53_diagnostic`
- identity freeze: `276_identity_freeze_tabledict_boundaryfix14_diagnostic`
- final compare: `277_compare_final_tabledict_boundaryfix14_diagnostic`
- Layer3 compare: `278_layer3_compare_tabledict_boundaryfix14_diagnostic`

Final-count outcome:
- all DEV15 papers match current GT counts;
- `PA3SPZ28` closed from boundaryfix8 `7 vs 3` to `3 vs 3`;
- recovered totals remain intact: `5GIF3D8W = 26/26`, `INMUTV7L = 12/12`, `QLYKLPKT = 7/7`, `WFDTQ4VX = 30/30`, `WIVUCMYG = 26/26`, `YGA8VQKU = 17/17`.

Layer3 comparison against accepted baseline:
- baseline: `system_nonempty_on_gt_cells = 2074`, `value_recall = 0.664957`, `strict = 0.595468`, `relaxed = 0.922372`, `extra_in_system_cells = 30`;
- boundaryfix14: `system_nonempty_on_gt_cells = 2086`, `value_recall = 0.668804`, `strict = 0.610259`, `relaxed = 0.933845`, `extra_in_system_cells = 30`, `risk_review_queue_rows = 1599`.

Interpretation:
- shared table-structure repair is now a net improvement rather than a noise source;
- the remaining PA3 over-retention was closed by a generic row-universe admission contract, not by paper-key hardcoding;
- S5-3 should remain a post-S5-2 gap-filler and should not rerun mechanical table materialization already handled by deterministic Stage2/Stage5 paths.
