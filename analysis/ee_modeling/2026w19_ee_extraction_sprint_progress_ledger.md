# 2026W19 EE Extraction Sprint Progress Ledger

Last updated: 2026-05-05 21:50:44 EDT

## Decisions

- Run all locally downloaded/clean-text-available papers first.
- Audit clean text before relying on live LLM.
- Allow live LLM this week only with explicit batch-level user approval immediately before each live run, and only after pre-live evidence/noise gate.
- Target locked to EE.
- Tiered modeling dataset accepted.
- New non-governance plan/progress artifacts allowed.

## Completed

### Scope / clean-text audit

Created:

- `analysis/ee_modeling/2026w19_downloaded_fulltext_cleantext_scope_audit_v1.tsv`
- `analysis/ee_modeling/2026w19_ee_extraction_scope_summary_v1.md`
- `analysis/ee_modeling/2026w19_ee_extraction_scope_manifest_v1.tsv`
- `analysis/ee_modeling/2026w19_ee_extraction_scope_batch001_25_manifest_v1.tsv`
- `analysis/ee_modeling/2026w19_ee_extraction_scope_batch002_50_manifest_v1.tsv`
- `analysis/ee_modeling/2026w19_ee_extraction_scope_batch003_100_manifest_v1.tsv`

Results:

- keyed clean text audited: 432 rows before manifest deduplication
- deduplicated Stage2 sprint scope: 382 papers
- initial included/review-first rows before dedupe: 426

### Batch 001 dry run

Command class: Stage2 composite, live mode, Gemini backend, `--stop-before-live-call`.

Run directory:

- `data/results/20260505_c1ad6ca/01_ee_sprint_batch001_25_dryrun/`

Generated:

- `analysis/stage2_prompt_preview_v1.tsv`
- `analysis/table_selection_debug_v1.json`
- `analysis/candidate_segmentation_debug_v1.tsv`
- `semantic_stage2_objects/candidate_blocks/`
- `semantic_stage2_objects/evidence_blocks/`
- `RUN_CONTEXT.md`

Note: `success_count=0 failure_count=25` is expected because the run stopped before live calls. The prompt/evidence artifacts are the deliverable.

### Pre-live preparation/evidence audit

Command:

```bash
python3 src/stage2_sampling_labels/audit_preparation_evidence_sufficiency_v1.py \
  --stage2-run-dir data/results/20260505_c1ad6ca/01_ee_sprint_batch001_25_dryrun \
  --out-dir data/results/20260505_c1ad6ca/01_ee_sprint_batch001_25_dryrun/analysis/pre_llm_evidence_sufficiency_audit_v1
```

Summary:

- cleaned_text_missing_method_body: 20
- evidence_selection_missing_preparation_core: 2
- materialization_or_carrythrough_boundary: 3

Interpretation: the existing preparation audit is conservative and flags many method-heading/body issues; it is useful as a risk signal, not by itself a live-call blocker for EE result discovery.

### Pre-live EE noise gate

Created:

- `data/results/20260505_c1ad6ca/01_ee_sprint_batch001_25_dryrun/analysis/pre_llm_ee_noise_gate_v1/pre_llm_ee_noise_gate_v1.tsv`
- `data/results/20260505_c1ad6ca/01_ee_sprint_batch001_25_dryrun/analysis/pre_llm_ee_noise_gate_v1/pre_llm_ee_noise_gate_summary_v1.json`

Summary:

- row_count: 25
- pass_for_live_llm: 14
- hold_for_selector_or_cleantext_review: 11

Reasons:

- oversized_prompt: 2
- tail_noise_with_weak_ee_signal: 4
- selected_evidence_missing_ee_or_loading_signal: 6
- missing_preparation_core: 2

Created live/hold manifests:

- `analysis/ee_modeling/2026w19_ee_extraction_scope_batch001_live_gatepass_14_manifest_v1.tsv`
- `analysis/ee_modeling/2026w19_ee_extraction_scope_batch001_hold_for_review_11_manifest_v1.tsv`

Gate-pass paper keys:

- `49T9E5EQ`
- `6AT9RFVD`
- `6P3Q8G9F`
- `DDWII6ZS`
- `HTUHFC6S`
- `LIU25HSK`
- `M923IXC6`
- `MWUP53IB`
- `PNKAM3D7`
- `RM4BRF9X`
- `S7SSV9SN`
- `TIDBBF25`
- `XDIRIJ74`
- `ZB76MB3J`

Hold paper keys:

- `UFXX9WXE`
- `YZYKTTFE`
- `IC5L6Z3X`
- `KTNLRQZU`
- `VHSU9G7W`
- `Z7B3LLMJ`
- `TFT6JTT6`
- `XVWN2PDB`
- `EHABVGGA`
- `V99GKZEI`
- `7IG3TLJ6`

## Next actions

1. Run live LLM for `batch001_live_gatepass_14` only.
2. Run maintained downstream Stage2 compatibility, Stage3, and Stage5 for live Batch 001 if Stage2 completes.
3. Build a Batch 001 EE extraction/readiness summary.
4. Review held 11 papers for selector noise/clean-text problems; do not spend live calls on them until repaired or manually approved.
5. Repeat dryrun/gate for Batch 002, then live only pass rows.

## Open risks

- Some clean text bodies may be lossy even when keyword signals exist.
- Evidence selector can still include non-EE result noise; the gate catches obvious cases but not all subtle misalignment.
- Current run remains diagnostic/modeling-sprint, not benchmark-valid.
- Stage2 live completion must be checked before scaling beyond Batch 001.
