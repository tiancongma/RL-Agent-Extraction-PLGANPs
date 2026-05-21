# 274 Formulation Universe To Final Table Evidence Audit Plan

## Goal

Build a new diagnostic lineage where
`274_formulation_universe_ee331_deepseek_stream_combined` is the fixed row
creation authority for downstream final-table, Evidence Binding, Gemma audit,
and human adjudication work.

This plan does not call DeepSeek. It may call Gemini/Gemma only for evidence
audit over system-built evidence packs.

## Authority Boundary

Input row authority:

- `data/results/20260511_b069802/274_formulation_universe_ee331_deepseek_stream_combined/formulation_universe_frozen_v1.tsv`

Rows admitted by that frozen universe become the only formulation rows in this
lineage. Downstream steps must not create additional formulation rows. If a
later value or evidence pass detects a possible missing formulation, it must
write a review queue row, not mutate the frozen row universe.

The prior Stage5 table
`115_stage5_deepseek_multi_anchor_complete_final_table/final_formulation_table_v1.tsv`
is advisory only. It may donate field values and evidence only when a
deterministic, auditable identity/evidence match is unique.

## Execution Steps

1. Promote 274 into a completed Stage2-compatible surface.
   - Assign stable row IDs from `paper_key`, `canonical_formulation_id`, and
     `formulation_label`.
   - Preserve 274 identity fields, evidence quotes, source locators, confidence,
     and human-review markers.
   - Emit a compatibility TSV plus a promotion ledger.

2. Build a 2853-row final-table candidate.
   - Use the promoted 274 rows as the fixed row universe.
   - Reuse old 115 values only for unique paper-local identity matches.
   - Leave unmatched values blank.
   - Write value-reuse and gap ledgers.

3. Rebuild Evidence Binding over the new 2853-row candidate.
   - Evidence Binding remains a sidecar.
   - It explains frozen rows and frozen values.
   - It must not create rows, create values, or mutate the final table.

4. Rebuild risk flags.
   - Risk assessment consumes only frozen Evidence Binding packs.
   - It prioritizes rows/fields for review.

5. Run Gemma 4 31B audit.
   - Audit input is the system-built evidence pack, not full paper text.
   - Reuse previous Gemma field reviews only when paper key, field name,
     frozen value, value evidence, row identity evidence, and source cell text
     are identical.
   - Otherwise call Gemma through the Gemini backend.

6. Adjudicate Gemma results.
   - Gemma is an audit signal, not final-table authority.
   - If Gemma finds a pure evidence defect, repair the evidence sidecar only.
   - If Gemma indicates a possible value error, clean text/table context must
     bind the proposed value to the same row identity and field before any
     candidate final-table correction is written.
   - Historical source artifacts must not be overwritten.

## Outputs

The expected campaign-local lineage is:

- `289_formulation_universe_274_promoted_stage2_compat`
- `290_formulation_universe_274_final_table_candidate`
- `291_evidence_binding_authority_manifest_universe274`
- `292_evidence_binding_packs_universe274`
- `293_evidence_binding_risk_universe274`
- `294_evidence_binding_gemma_review_universe274`
- `295_gemma_audit_cleantext_adjudication_universe274`

All outputs are diagnostic-only unless a later governance decision promotes
them through the maintained mainline contract.

## Non-Goals

- Do not call DeepSeek.
- Do not update `data/results/ACTIVE_RUN.json`.
- Do not overwrite run 115 or run 274.
- Do not report benchmark performance.
- Do not fill blank values by assumption or donor-fill.
