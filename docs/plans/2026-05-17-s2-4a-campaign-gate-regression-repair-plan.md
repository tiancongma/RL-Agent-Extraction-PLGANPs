# S2-4a Campaign Gate Regression Repair Plan

## Scope

This plan covers the no-live repair pass for campaign
`data/results/20260511_b069802`, focused on the current pre-live chain:

- Stage1 unified current/Marker surface
- S2-2 evidence construction
- S2-4a prompt construction
- S2-4a hard-gate audit

The task is diagnostic and pre-live only. It must not call S2-4b live LLM, must
not advance `data/results/ACTIVE_RUN.json`, and must not report benchmark
performance.

## Current Trigger

The latest no-live full-campaign replay produced:

- S2-2 child `78`: `380/380` deterministic artifacts completed
- S2-4a child `79`: `380/380` prompts built
- prompt audit: `372/380` pass
- hard gate: `361/380` pass and `19/380` fail

Compared with child `66`, child `79` recovered several old failures but also
introduced ten old-pass to new-fail regressions. Therefore the current surface
is not yet cleared for full live LLM execution.

## Governed Inputs

- Repair index:
  `docs/repair_index/success_pattern_index_v1.tsv`
- Campaign progress:
  `data/results/20260511_b069802/CAMPAIGN_PROGRESS.md`
- Current full replay evidence:
  `data/results/20260511_b069802/78_stage2_s2_2_campaign_generic_repair_no_live/semantic_stage2_objects/evidence_blocks`
- Current prompt audit:
  `data/results/20260511_b069802/79_stage2_s2_4a_campaign_generic_repair_no_live/analysis/s2_4a_prompt_audit_v1.tsv`
- Current hard gate:
  `data/results/20260511_b069802/79_stage2_s2_4a_campaign_generic_repair_no_live/analysis/s2_4a_hard_gate_campaign_generic_repair_v1.tsv`
- Previous comparison gate:
  `data/results/20260511_b069802/66_stage2_s2_4a_campaign_current_no_live_recheck/analysis/s2_4a_hard_gate_ee380_recheck_v1.tsv`

## Candidate Repair Patterns To Check

- `PAT_STAGE1_MARKER_TABLE_BLOCK_AUTHORITY_PROMOTION_V1`
- `PAT_S2_2_HOLLOW_LABEL_ONLY_PAYLOAD_GUARD_V1`
- `PAT_S2_4A_RECOVERED_TABLE_SUMMARY_SURFACE_V1`
- `PAT_S2_2_METHOD_RESULT_UNDERSELECTION_FLOOR_V1`
- `PAT_S2_7_TABLE_IDENTITY_ALIAS_LOCATOR_PRIORITY_V1`

Patterns with anything weaker than explicit governed activation must be treated
as historical or partial until the current lineage proves them.

## Execution Checklist

1. Build a failure ledger for child `79`.
   - Separate `upstream_evidence_nonconformant` from prompt-audit-pass
     `evidence_underselected`.
   - Mark old-pass to new-fail regressions against child `66`.

2. Inspect evidence by failure bucket, not paper-specific patching.
   - Compare Stage1 unified text availability, candidate blocks, selected
     evidence blocks, normalized table payloads, table summaries, and hard-gate
     satisfied paths.
   - Use remapped source-resolution artifacts when source files must be checked.

3. Locate the responsible generic function.
   - Acceptable repair units include selector floor, table summary surface,
     table authority payload visibility, method/preparation path matching,
     prompt audit conformance, and Stage1 source completeness.
   - Do not add paper-key special cases.
   - Do not let deterministic code define formulation semantics or row
     membership before the LLM.

4. Patch only the minimal deterministic unit.
   - Add or update focused regression tests for the generic pattern.
   - Keep table summaries structural and prompt-only.
   - Keep execution-grade table authority separate from prompt summaries.

5. Run bounded replay first.
   - Use a bounded key set covering the failing buckets and the old-pass
     regressions.
   - Replay no-live Stage1 if needed, then S2-2, S2-4a, and hard gate.
   - Validate that the repaired bucket improves and that known pass examples do
     not regress.

6. Run the full campaign no-live gate only after bounded validation passes.
   - Rebuild full S2-2 and S2-4a no-live artifacts from explicit source paths.
   - Re-run hard gate.
   - Report exact pass/fail counts and remaining failure keys.

## Stop Conditions

Stop only when one of the following is true:

- Full campaign no-live S2-2 -> S2-4a gate passes at least the intended
  current surface without old-pass regressions.
- A remaining failure is proven to require live LLM semantic judgment or a
  broader governed design change.
- A boundary-legality violation is detected; in that case mark the affected
  result invalid and stop interpretation.

## Execution Result

Status: completed for deterministic no-live gate repair.

Implemented generic repairs:

- Stage1 same-key HTML -> canonical PDF clean-text supplement fallback for
  cases where remapped HTML current text is too short but the canonical PDF
  clean text is available.
- S2-2 preparation/evidence floor expansion for source-backed formulation
  method context, including `typical procedure`, `w/o/w`, solvent evaporation,
  spray-dried formulation methods, and `sPLGA` family text.
- S2-4a hard-gate expansion for structural table-summary surfaces and
  source-backed preparation-core text, while keeping table summaries
  semantic-facing and execution-grade table authority separate.

Validation:

- Focused unit tests:
  `python3 -m unittest tests.test_stage1_unified_marker_table_promotion_v1 tests.test_stage2_preparation_core_selector_floor_v1 -q`
  passed, `60` tests.
- Bounded replay after Stage1 -> S2-2 -> S2-4a:
  `84_stage2_s2_4a_gate_repair_bounded_no_live`
  produced `22/28` pass. The remaining `6` were all metadata-only
  source-completeness failures.
- Full no-live replay:
  - Stage1:
    `85_stage1_unified_campaign_gate_repair_no_live`
  - S2-2:
    `86_stage2_s2_2_campaign_gate_repair_no_live`
  - S2-4a:
    `87_stage2_s2_4a_campaign_gate_repair_no_live`

Full campaign gate outcome:

- Prompt construction: `380/380` success.
- Prompt audit: `374/380` pass, `6/380` fail.
- Hard gate: `374/380` pass, `6/380` fail.
- Hard-gate satisfied paths among passing papers:
  `path1=139`, `path2=9`, `path3=203`, `path4=23`.

Remaining blocked keys:

- `GNQWKY3J`
- `HA8A6XKC`
- `HZY5DREJ`
- `M75R5N92`
- `ULCW6JTQ`
- `YVDWQU9Y`

All remaining failures are `upstream_evidence_nonconformant` at prompt audit
and `evidence_underselected` at hard gate, with `ordered_block_order=metadata`.
They are source-completeness/clean-text availability blockers, not the repaired
table summary, evidence selector, table authority, or payload locator
regression buckets.

Regression check:

- Child `66`: `362/380` pass, `18/380` fail.
- Child `79`: `361/380` pass, `19/380` fail.
- Child `87`: `374/380` pass, `6/380` fail.
- Recovered versus child `66`:
  `4UPTKEWP 5RKMV25Z 67FVVTMK 782JNNJL B2X4XLLJ DQW9XTTW FJQQ7H9M H8F4TQLS KJXMGXWE VN86KN6P XXLXM9FM YRPA2Z9U`
- Recovered versus child `79`:
  `2T32C4VP 67FVVTMK 782JNNJL 8PT5W352 DCSRFP8X FJQQ7H9M KW8UTVFV KXMBCZSZ NXWGJR8B RBUY36NL TP6VRSML TT2JDLQK YGA8VQKU`
- New regressions versus child `66`: none.
- New regressions versus child `79`: none.

Live-LLM readiness:

The deterministic no-live gate now clears `374/380` papers. The remaining
`6/380` require source-completeness repair or a governed decision to exclude or
separately handle metadata-only inputs. This result should not be treated as
`380/380` live-call clearance.
