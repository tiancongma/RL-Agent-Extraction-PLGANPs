# Formulation-Centered Review UI Implementation Plan

Date: 2026-05-20

## Purpose

Build a lightweight human review app for formulation-centered GT and audit work.
The review unit is a formulation object, not a spreadsheet cell. The app should
let a reviewer start from one predicted formulation, decide the formulation
boundary, inspect field values, and record evidence/value review decisions.

This is a reviewer-facing support surface. It must not redefine Stage2 semantic
discovery, Stage5 final-output semantics, GT counting rules, or benchmark
validity.

## Governance Boundary

Inputs must be explicit frozen artifacts:

- Stage5 `final_formulation_table_v1.tsv`, or
- Stage5 audit-ready export `final_formulation_table_audit_ready_v1.tsv`, and
- optional Layer3 field seed rows from `build_field_gt_review_workbook_v1.py`.
- optional explicit source index TSV with `paper_key`, `doi`, `pdf_path`, and
  `html_path` columns.

Outputs are review-only:

- append-only reviewer decisions JSONL
- review session metadata JSON

The app must not:

- update `data/results/ACTIVE_RUN.json`
- mutate authoritative Stage5 or GT TSVs
- use reviewer decisions as a lawful Stage3 or Stage5 resume boundary
- report benchmark performance

## Implementation Shape

Add a small standard-library Python web app under `src/stage5_benchmark/` so it
can run without adding new UI dependencies. The app reads explicit TSV inputs,
builds formulation cards, serves a local browser UI, and appends reviewer
decisions to an output directory.

The UI is organized as three working areas:

1. Boundary Review
   - one card per `(paper_key, formulation_id)`
   - decisions include accept, merge, split, missing, control/comparator, and
     unclear states

2. Value Review
   - field rows are attached to the accepted formulation card when seed rows are
     available
   - priority fields include EE, size, PDI, zeta, polymer MW, LA:GA,
     surfactant, solvent, and drug/polymer ratio when present in source rows

3. Evidence Review
   - evidence text, source type, support status, warnings, and normalization
     status remain visible at field level
   - reviewer decisions can mark supported, unsupported text, unresolved table,
   normalization pending, or unclear evidence

4. Source Access
   - DOI is displayed as text only
   - PDF/HTML source files open through local allowlist routes backed by the
     explicit source index
   - raw local paths are not exposed to the browser payload

## Execution Steps

1. Register the new support script in `docs/src_script_registry.tsv`.
2. Implement `src/stage5_benchmark/formulation_review_app_v1.py`.
3. Add minimal fixtures and tests for card construction and append-only
   decision writing.
4. Run focused unit tests.
5. Launch the local UI against fixtures and perform a browser smoke test that
   verifies the screen renders and a reviewer decision is recorded.

## Acceptance Criteria

- The app requires explicit input TSV paths.
- The first browser screen is the working review interface, not a landing page.
- The UI groups rows by formulation and exposes field/evidence detail.
- PDF/HTML source buttons are available only for files registered in the
  explicit source index.
- Reviewer decisions are append-only and include source-file provenance.
- Unit tests pass.
- A local browser smoke test passes.
