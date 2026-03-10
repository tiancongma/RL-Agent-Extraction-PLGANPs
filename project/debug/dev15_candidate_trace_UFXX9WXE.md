# DEV15 Candidate Trace Audit: UFXX9WXE

Date: 2026-03-09  
Target paper:
- `paper_key`: `UFXX9WXE`
- `doi`: `10.1155/2014/156010`
- `title`: *Formulation and Optimization of Polymeric Nanoparticles for Intranasal Delivery of Lorazepam Using Box-Behnken Design: In Vitro and In Vivo Evaluation*

## 1) Workbook vs candidate scaffold row counts

### Current review workbook
- File: `data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_formulation_skeleton_review_v1_fixed.xlsx`
- Sheet: `review_formulations`
- Rows for `UFXX9WXE`: **26**

### Candidate scaffold used by skeleton tool
- File: `data/cleaned/labels/manual/dev15_formulation_skeleton/candidates/dev15_formulation_candidates.tsv`
- Rows for `UFXX9WXE`: **8**
- Header columns:
  - `paper_key, doi, candidate_formulation_label, candidate_formulation_id, source_type_candidate, evidence_pointer_candidate, notes_candidate`

- File: `data/cleaned/labels/manual/dev15_formulation_skeleton/candidates/UFXX9WXE__10_1155_2014_156010__candidates.jsonl`
- Rows: **8**

Interpretation:
- `candidate_rows = 8` in scaffold/source summary is real scaffold count.
- `GT_rows = 26` in the workbook reflects additional review rows currently present in the editable review sheet.

## 2) Workbook generation logic trace

Primary code path:
- `src/stage3_gt/build_dev15_formulation_skeleton_review_v1.py`
  - Source discovery: lines `293-297`
  - Candidate normalization/padding call: lines `299-303`
  - Default minimum row setting: lines `270-273` (`default=8`)

- `src/stage3_gt/formulation_skeleton_common.py`
  - Candidate source search patterns: lines `170-175`
  - Source-row to candidate-row mapping: function at line `225`
  - Uses upstream `formulation_id` when building raw candidates: line `260`
  - Dedup by `(paper_key, candidate_formulation_id)`: lines `292-301`
  - Min-row padding logic: line `318` (`target_n = max(default_rows_per_paper, len(existing))`)
  - Deterministic skeleton ID assignment: line `327` (`{paper_key}_F{idx:02d}`)

For `UFXX9WXE` in this run:
- Raw source-mapped candidates before padding: **8** (unique IDs `1..8` from source)
- Default minimum rows: **8**
- Padding applied: **No extra padding** (`max(8, 8) = 8`)

## 3) Candidate scaffold provenance for UFXX9WXE

Selected source for this workbook:
- `data/results/run_20260227_1016_a8d884b_goren2025_step1dev_v1/step1_dev/weak_labels__gemini_flashlite.tsv`

Observed counts in selected source:
- Total rows in source file: **363**
- Rows matched by key `UFXX9WXE`: **16**
- Unique `formulation_id` among those rows: **8** (`1..8`)
- DOE-like columns in header (`doe|box|behnken|coded|factor|level|response`): **none found**

This indicates scaffold candidates for this paper came from actual extraction output rows (then deduped), not from empty placeholder-only generation.

### Other candidate-like sources found for same paper (comparison)
Among files scanned by current candidate-source search patterns, this paper appears in multiple weak-label outputs:
- `.../run_20260219_1623_780eb83_goren18_weaklabels_v1/weak_labels__gemini.tsv` -> `26` rows, `26` unique `formulation_id`
- `.../run_20260226_1519_5576d8b.../weak_labels__gemini__with_gate_anchor.tsv` -> `26` rows, `26` unique `formulation_id`
- `.../run_20260227_1016.../weak_labels__flashlite.tsv` -> `8` rows, `8` unique IDs
- `.../run_20260227_1016.../weak_labels__gemini.tsv` -> `8` rows, `8` unique IDs
- `.../run_20260227_1016.../weak_labels__gemini_flashlite.tsv` -> `16` rows, `8` unique IDs (selected source)
- `.../run_20260227_1059.../weak_labels__flashlite.tsv` -> `13` rows, `13` unique IDs
- `.../run_20260227_1059.../weak_labels__gemini.tsv` -> `8` rows, `8` unique IDs
- `.../run_20260227_1059.../weak_labels__gemini_flashlite.tsv` -> `22` rows, `13` unique IDs
- `.../run_20260228_1634.../weak_labels__gemini.tsv` -> `8` rows, `8` unique IDs

There are also formulation-level benchmark tables with DOI matches, but those have no `formulation_id` and are not the selected source in this run.

## 4) DOE-aware logic inventory

### DOE-specific logic exists in repository
Examples:
- `src/stage5_benchmark/derive_doe_coded_factors_v1.py`
  - CLI description explicitly derives DOE coded/decoded factors from source tables.
- `src/stage5_benchmark/run_derivation_v1.py`
  - Derivation pipeline that includes DOE decoding.
- `src/stage5_benchmark/build_two_table_schema_v3.py`
  - Uses DOE signatures and decoded-rate gating.
- `src/stage5_benchmark/build_audit_pack_human_evidence_v1.py`
  - References `derived_doe_decode`.
- `src/stage5_benchmark/export_full_database_v1.py`
  - Appends DOE coded/decoded factors to exported database when present.
- `src/stage5_benchmark/audit_evidence_resolver_v1.py`
  - Has DOE-like table detection heuristics (`box-behnken`, `coded`, `level`, etc.).

### Is DOE logic in current skeleton workbook path?
- `src/stage3_gt/*` has no DOE-aware extraction/decode logic in the skeleton flow.
- Current skeleton path:
  - detect candidate source
  - map weak-label-style rows to lightweight candidate scaffold
  - pad to minimum rows
  - export workbook for manual dropdown review
- No DOE decode outputs are consumed in this path.

Conclusion on DOE wiring:
- DOE-aware logic **exists** in the repo, but it is **not wired into** the current stage3 skeleton workbook generation workflow.

## 5) Final diagnosis for UFXX9WXE

- **B**: `candidate_rows = 8` comes from true scaffold candidates (from selected extraction output, deduped to 8 IDs).
- **C**: DOE-aware logic exists, but is not wired into the current skeleton workflow.

Why:
- Selected candidate source file contributes actual `UFXX9WXE` rows with `formulation_id` values.
- Current scaffold builder does not call any stage5 DOE decode/signature pipeline artifacts.

## 6) Recommended next step (no code changes in this task)

Pin and audit candidate-source selection for skeleton generation:
1. Add a reproducible source-selection report for each paper (selected file + alternative files + candidate ID counts).
2. Decide whether skeleton candidates should come from:
   - step1 weak-label source (current), or
   - a DOE-aware post-derivation source when available.
3. If DOE-aware candidates are desired, wire a deterministic optional input from stage5 DOE outputs into stage3 scaffold build as an explicit mode switch.

