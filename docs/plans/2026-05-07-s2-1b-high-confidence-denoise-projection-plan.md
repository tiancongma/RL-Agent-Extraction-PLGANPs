# S2-1b High-Confidence Denoise Projection Plan

> **For Hermes:** Use the main/default agent as orchestrator in this repo. Use read-only specialists for audit/regression review when useful, keep writes serial, and verify outputs before treating them as final.

## Goal

Add a small deterministic Stage2 substep, `S2-1b High-confidence source denoise projection`, before evidence construction. S2-1b removes only high-confidence source boilerplate/noise and emits the text surface consumed by later Stage2 evidence selection. It must reduce selector pollution before full-source extraction expansion without changing semantic authority.

## Why this is needed now

The 2026-05-07 DEV15 diagnostic lineage showed that the new full-cleantext/Marker integration was useful as an integration diagnostic but not an improvement baseline. One concrete blocker before scaling all original-source extraction is that Stage2 selector/gate logic still sees obvious non-evidence source noise: publisher chrome, downloaded markers, page/header/footer lines, reference tails, and journal navigation. The existing Stage2 cleanup is too late and too weak: it often records residual-noise flags or penalties after candidate construction instead of preventing obvious hard noise from entering evidence selection.

## Governance boundaries

- S2-1b is an internal Stage2 projection between `S2-1 Scope resolution` and `S2-2 Evidence construction`.
- S2-1b does **not** modify Stage1 raw/clean source authority.
- S2-1b does **not** perform formulation semantic discovery, table-importance judgment, row-universe construction, or GT/source-anchor completion.
- LLM semantic discovery remains the Stage2 semantic authority.
- The raw source surface before S2-1b remains audit authority; the denoised projection after S2-1b is the default text consumed by S2-2/S2-3/S2-4.
- User-provided original-source excerpts in `docs/methods/layer3_field_gt_protocol_v1.md:1098-1864` remain audit anchors only, not runtime extraction inputs.
- No live LLM expansion should run until S2-1b no-live acceptance passes and the user gives explicit batch-level approval.

## Conceptual flow

```text
S2-1 Scope resolution
  input: manifest row, raw/current clean text, optional Stage1 structure/table sidecars
  output: resolved raw source text path and source metadata

S2-1b High-confidence source denoise projection
  input: resolved raw/current clean text plus optional non-authoritative structure hints
  output: denoised_for_stage2 text + audit JSON/TSV

S2-2 Evidence construction
  input: denoised_for_stage2 text plus preserved table/structure authority
  output: candidate blocks, normalized table payloads, evidence blocks

S2-3/S2-4 Prompt assembly/call
  input: canonical S2-2 evidence blocks only
```

## S2-1b contract

### Inputs

- `paper_key`
- raw/current clean text path resolved by S2-1
- optional Stage1 structure sidecar for block/source locator hints
- optional table-cell/normalized-table authority sidecars for preservation checks

### Outputs

Run-scoped outputs under the Stage2 run child:

```text
semantic_stage2_objects/s2_1b_denoised_text/<paper_key>.txt
semantic_stage2_objects/s2_1b_denoise_audit/<paper_key>_s2_1b_denoise_audit_v1.json
analysis/s2_1b_denoise_summary_v1.tsv
```

Required summary/audit fields:

```text
paper_key
input_text_path
output_text_path
raw_char_count
denoised_char_count
removed_char_count
raw_line_count
denoised_line_count
removed_line_count
rule_id
rule_class
rule_confidence
source_locator
removed_text_preview
preservation_exception
```

### Required manifest / downstream provenance fields

Later S2 artifacts must record:

```text
source_text_projection = s2_1b_denoised
source_raw_clean_text_path
source_s2_1b_denoised_text_path
s2_1b_denoise_audit_path
s2_1b_denoise_summary_path
```

## Hard-delete classes

S2-1b may hard-delete only high-confidence boilerplate/noise that cannot lawfully define PLGA formulation identity, preparation, composition, table authority, or result authority.

Allowed high-confidence hard-delete classes:

- `publisher_chrome`
  - examples: `Journals & Books`, `Help`, `Search`, `View PDF`, `Download full issue`, `Outline`
- `download_marker`
  - examples: `Downloaded from ...`, `This article was downloaded by ...`
- `page_header_footer`
  - repeated journal titles, page numbers, DOI-only page-footers, `Page x of y`
- `author_page_running_line`
- `copyright_or_license_boilerplate`
- `reference_tail`
- `isolated_reference_line`
- `article_recommendation_or_related_articles`
- obvious publisher navigation/crossref/pubmed metadata lines outside article body

## Suppress/tag but do not hard-delete by default

These may be prompt-ineligible or low-priority for formulation identity, but S2-1b should not hard-delete them unless a separate rule proves they are pure boilerplate:

- figure captions
- downstream assay paragraphs/captions such as cellular uptake, PK, biodistribution, release-only, stability-only
- abstract/highlights/graphical abstract
- broad characterization-only paragraphs

## Must preserve

S2-1b must never hard-delete:

- materials paragraphs
- preparation / methods paragraphs
- formulation/design/result table captions
- table bodies or coordinate-preserving table payloads
- DOE/design matrices
- composition/result rows
- row labels / first-column identity rows
- carrythrough sentences such as `prepared similarly`, `same procedure`, `varying X`, `according to Table`, or direct formulation/result statements

## Implementation work packages

### Task 1 — Contract-first implementation scaffold

Create a deterministic S2-1b helper in `src/stage2_sampling_labels/` with tests before integration.

Planned file:

```text
src/stage2_sampling_labels/denoise_stage2_source_text_s2_1b_v1.py
```

Planned tests:

```text
tests/test_denoise_stage2_source_text_s2_1b_v1.py
```

Acceptance:

```bash
PYTHONPATH=. python3 -m unittest tests.test_denoise_stage2_source_text_s2_1b_v1
python3 -m py_compile src/stage2_sampling_labels/denoise_stage2_source_text_s2_1b_v1.py
```

### Task 2 — High-confidence rule set

Implement rule IDs for publisher chrome, download marker, page/header/footer, author/page running line, reference tail, isolated reference line, copyright/license boilerplate, and related-article metadata.

Rules must be explainable and auditable; no rule may use GT status or formulation-value expectations.

### Task 3 — Denoise projection artifacts

Write denoised text plus per-paper audit JSON and run summary TSV. Preserve raw source path and character/line deltas.

### Task 4 — Integrate S2-2 consumption

Modify maintained Stage2 composite / internal extractor so S2-2 consumes the S2-1b projection by default when available, while retaining raw text path metadata for audit.

Likely files to inspect/modify:

```text
src/stage2_sampling_labels/run_stage2_composite_v1.py
src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py
tests/test_stage2_stage1_sidecar_consumption_v1.py
```

### Task 5 — Acceptance gate and no-live replay

Run a no-live S2-1b -> S2-2 acceptance pass on DEV15 or the next declared scope. Check that obvious noise no longer appears in selected evidence while method/materials/table authority remains visible.

Required no-live checks:

- `canonical_evidence_noise_present` decreases for publisher/reference/header/footer classes.
- method/materials/table evidence is not removed for papers with source-anchor expectations.
- table authority preservation remains `CONFIRMED_NOISE` vs `PRESERVE`; S2-1b does not become an importance filter.
- all outputs are diagnostic-only until explicitly promoted.

### Task 6 — Memory and governance refresh

After implementation and verification, update:

- `project/4_DECISIONS_LOG.md` with the final implemented decision if behavior changes.
- `docs/maintained_script_surface.tsv` and `docs/src_script_registry.tsv` only after the script exists and tests pass.
- `data/mem/v1/` via governed memory update/rebuild, not by creating an alternate memory tree.

## Cron execution intent

A cron job should execute this plan in a fresh session with no live LLM calls, starting with Task 1. The cron prompt must be self-contained, read governance files first, inspect `git status --short`, protect existing dirty files, and write only planned S2-1b implementation/test/governance updates. It must not update `data/results/ACTIVE_RUN.json` and must not attempt WeChat/Weixin/live LLM.

## Progress ledger

Use:

```text
docs/plans/2026-05-07-s2-1b-high-confidence-denoise-projection-progress.tsv
```

Columns:

```text
task_id	status	test_status	notes	updated_at
```
