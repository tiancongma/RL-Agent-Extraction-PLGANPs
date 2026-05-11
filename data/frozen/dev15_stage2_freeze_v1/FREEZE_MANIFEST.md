# DEV15 Stage2 Freeze v1

## Freeze Scope

This portable freeze promotes the minimal governed Stage2 frozen chain needed
for cross-machine reuse without syncing full `data/results/` run directories.

Included layers:

- `S2-2` frozen candidate and evidence bundles
- `S2-3` frozen prompt-preview observability surface
- `S2-4a` frozen prompt-assembly readiness artifacts
- `S2-4b` frozen live-call output set

Paper scope:

- DEV15 scope from:
  - `data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/dev15_scope.tsv`
- Covered paper keys:
  - `5ZXYABSU`
  - `L3H2RS2H`
  - `WIVUCMYG`
  - `5GIF3D8W`
  - `7ZS858NS`
  - `BB3JUVW7`
  - `BXCV5XWB`
  - `INMUTV7L`
  - `PA3SPZ28`
  - `QLYKLPKT`
  - `RHMJWZX8`
  - `UFXX9WXE`
  - `V99GKZEI`
  - `WFDTQ4VX`
  - `YGA8VQKU`

## Source Run Directories

S2-2 and S2-3 portable source:

- `data/results/20260411_312d44b/12_FINAL_segmentation_closure`

Why this source is authoritative for S2-2 and S2-3:

- Its run context records the governed composite Stage2 contract with the
  formal S2-2 boundary and the canonical candidate/evidence artifacts.
- It is the final segmentation-closure run used for the frozen DEV15 S2-2
  closure state.
- Its feature activation report shows active evidence-artifact, design-status,
  prompt-preview, role-aware selector, DOE overlay, and duplicate-table
  suppression evidence.
- Governance records freeze `candidate_blocks_v1.json`, freeze selector logic,
  and define `evidence_blocks_v1.json` as the canonical S2-3 input.

S2-4a portable source:

- `data/results/20260410_a165cd1/14_s2_prompt_assembly_readiness_v2`

Why this source is authoritative for S2-4a:

- It contains the only located complete frozen prompt-assembly readiness triplet:
  - `s2_4a_prompt_template_v1.txt`
  - `s2_4a_prompts_v1.jsonl`
  - `s2_4a_prompt_audit_v1.tsv`
- The run uses the same DEV15 scope and the same replay-backed governed Stage2
  prompt-assembly configuration as the frozen chain.
- The prompt artifacts explicitly resolve back to frozen
  `evidence_blocks_v1.json` inputs and preserve the governed ordered-block
  prompt surface.

S2-4b portable source:

- `data/results/20260412_8517d36/04_s2_4b_live_llm_call_dev15_v1`
- `data/results/20260412_8517d36/10_E_s2_4b_parallel2_t180_r1_v1`
- `data/results/20260412_8517d36/13_H_s2_4b_parallel2_t180_r1_sleep5_v1`
- `data/results/20260412_8517d36/14_s2_4b_completion_remaining5_v1`

Why this source is authoritative for S2-4b:

- The portable `s2_4b/` freeze is now a same-lineage consolidation of
  successful raw-response payloads whose request metadata resolves to the same
  frozen `S2-4a` prompt SHA surface.
- `04_s2_4b_live_llm_call_dev15_v1` remains the original full-DEV15 dedicated
  maintained `S2-4b` run and still supplies the baseline successful papers
  from the frozen prompt lineage.
- `10_E_s2_4b_parallel2_t180_r1_v1` and
  `13_H_s2_4b_parallel2_t180_r1_sleep5_v1` supply additional same-lineage
  successful papers that were previously failed or missing in the baseline
  full run.
- `14_s2_4b_completion_remaining5_v1` is the bounded completion child run that
  reran only papers with no successful persisted raw response for this frozen
  prompt lineage and completed the remaining coverage under:
  - model:
    `gemini-2.5-flash`
  - request mode:
    `stream_collect`
  - request timeout seconds:
    `180`
  - request retries:
    `1`
  - retry sleep seconds:
    `5.0`
  - max parallel requests:
    `2`
- The resulting portable `s2_4b/` freeze preserves one successful raw payload
  plus aligned request metadata for all `15/15` DEV15 papers without drifting
  into `S2-5`, `S2-6`, `S2-7`, or any later stage.

## Why These Artifacts Are Authoritative

The governed Stage2 architecture defines:

- `candidate_blocks_v1.json` as the explicit pre-selector candidate-segmentation
  surface inside `S2-2`
- `evidence_blocks_v1.json` as the canonical pre-LLM evidence artifact and the
  canonical `S2-3` prompt-assembly input
- `stage2_prompt_preview_v1.tsv` as derived observability that must resolve
  back to the canonical evidence artifact

This freeze promotes those governed artifacts into a stable portable path so
cross-machine work does not depend on copying the original run roots.

## Portable Directory Contract

Portable frozen root:

- `data/frozen/dev15_stage2_freeze_v1/`

Portable subpaths:

- `s2_2/semantic_stage2_objects/candidate_blocks/<paper_key>/candidate_blocks_v1.json`
- `s2_2/semantic_stage2_objects/evidence_blocks/<paper_key>/evidence_blocks_v1.json`
- `s2_3/analysis/stage2_prompt_preview_v1.tsv`
- `s2_4a/analysis/s2_4a_prompt_template_v1.txt`
- `s2_4a/analysis/s2_4a_prompts_v1.jsonl`
- `s2_4a/analysis/s2_4a_prompt_audit_v1.tsv`
- `s2_4b/RUN_CONTEXT.md`
- `s2_4b/stage2_s2_4b_run_metadata_v1.json`
- `s2_4b/analysis/s2_4b_request_summary_v1.tsv`
- `s2_4b/raw_responses/<paper_key>__stage2_v2_raw_response.json`
- `s2_4b/request_metadata/<paper_key>__stage2_v2_request_metadata.json`

Filenames and relative paths are intentionally stable and machine-portable.

## Downstream Consumption Rule

Downstream stages or audits that need this frozen chain must consume the
portable artifacts from `data/frozen/dev15_stage2_freeze_v1/` instead of
treating the original run-local `data/results/...` paths as the portable
authority.

Expected downstream consumers:

- `S2-2` stage-local audit, regression, and closure verification on frozen
  candidate/evidence bundles
- `S2-3` prompt-assembly audits that need the canonical evidence input and
  prompt-preview observability surface
- `S2-4a` prompt-assembly readiness and prompt-contract review workflows that
  need the frozen template, assembled prompts, and prompt audit
- future `S2-5` replay or rehydration work that needs the frozen `S2-4b`
  raw-response set plus request-level metadata without syncing the source run
  directory

Non-goal:

- This freeze does not replace the lawful downstream `Stage3` resume boundary,
  which remains the completed Stage2 compatibility projection artifact.
- This freeze does not itself perform `S2-5` semantic parsing or create the
  completed Stage2 authority surface.

## Portable Authority Statement

For this frozen chain:

- the original run-local `data/results/...` artifact paths remain provenance
  sources only
- they are no longer the portable authority for cross-machine sync of the
  DEV15 frozen Stage2 chain
- the portable authority is this directory:
  - `data/frozen/dev15_stage2_freeze_v1/`

## Promotion Update (`2026-04-13`)

Promotion action:

- extend the frozen Stage2 authority tree by promoting already-executed,
  already-validated `S2-5`, `S2-6`, and `S2-7` artifacts from explicit
  maintained result children under:
  - `data/results/20260413_8517d36/01_s2_5_semantic_parsing`
  - `data/results/20260413_8517d36/02_s2_6_contract_validation`
  - `data/results/20260413_8517d36/03_s2_7_compatibility_projection`
- promotion is organizational only:
  - no rerun
  - no regeneration
  - no semantic changes
  - no extraction-logic changes

Frozen chain structure now present in this tree:

- `S2-4a -> S2-4b -> S2-5 -> S2-6 -> S2-7`

Portable promoted subpaths:

- `s2_5/semantic_stage2_objects/`
- `s2_5/RUN_CONTEXT.md`
- `s2_5/stage2_s2_5_run_metadata_v1.json`
- `s2_6/analysis/stage2_semantic_authority_contract_report_v1.json`
- `s2_6/RUN_CONTEXT.md`
- `s2_6/stage2_s2_6_run_metadata_v1.json`
- `s2_7/semantic_to_widerow_adapter/`
- `s2_7/RUN_CONTEXT.md`
- `s2_7/stage2_s2_7_run_metadata_v1.json`

Stage chain contract:

- `S2-4a`
  - input:
    canonical `S2-2` evidence artifacts
  - output:
    frozen prompt artifacts under `s2_4a/analysis/`
  - boundary type:
    `internal_intermediate`
  - next lawful step:
    `S2-4b`
- `S2-4b`
  - input:
    frozen `s2_4a/analysis/s2_4a_prompts_v1.jsonl`
  - output:
    replayable raw responses under `s2_4b/raw_responses/`
  - boundary type:
    `diagnostic_boundary`
  - next lawful step:
    `S2-5`
- `S2-5`
  - input:
    frozen `S2-4b` raw responses only
  - output:
    `s2_5/semantic_stage2_objects/semantic_stage2_v2_objects.jsonl`
    and `s2_5/semantic_stage2_objects/semantic_stage2_v2_summary.tsv`
  - boundary type:
    `internal_intermediate`
  - next lawful step:
    `S2-6`
- `S2-6`
  - input:
    frozen `s2_5/semantic_stage2_objects/semantic_stage2_v2_objects.jsonl`
  - output:
    `s2_6/analysis/stage2_semantic_authority_contract_report_v1.json`
  - boundary type:
    `internal_intermediate`
  - next lawful step:
    `S2-7`
- `S2-7`
  - input:
    passing `s2_6/analysis/stage2_semantic_authority_contract_report_v1.json`
    plus the referenced `S2-5` semantic-intermediate artifacts
  - output:
    completed Stage2 compatibility-projected artifacts under
    `s2_7/semantic_to_widerow_adapter/`
  - boundary type:
    `mainline_resume_boundary`
  - next lawful step:
    `Stage3 relation materialization`

Boundary statements:

- `S2-5`: `internal_intermediate`
- `S2-6`: `internal_intermediate` (validation gate)
- `S2-7`: `mainline_resume_boundary`

Authoritative downstream statement:

- `S2-7` output is the authoritative completed Stage2 artifact and lawful
  Stage3 resume boundary.

Lineage notes:

- `S2-5` source run:
  - `data/results/20260413_8517d36/01_s2_5_semantic_parsing`
- `S2-6` source run:
  - `data/results/20260413_8517d36/02_s2_6_contract_validation`
- `S2-7` source run:
  - `data/results/20260413_8517d36/03_s2_7_compatibility_projection`
- promotion method:
  - copy only minimal authoritative artifacts into the frozen tree
- semantic-change statement:
  - promoted files are byte-preserving copies of the source artifacts

Coverage note:

- The promoted `S2-5` -> `S2-7` lineage is valid and fully reproducible for
  the explicit source runs above.
- Those source runs currently cover the bounded validated paper subset:
  - `UFXX9WXE`
  - `WIVUCMYG`
- Therefore this freeze tree now contains the full Stage2 step structure
  through `S2-7`, but full-paper DEV15 coverage for the promoted `S2-5` ->
  `S2-7` layers is not established by these source runs alone.
