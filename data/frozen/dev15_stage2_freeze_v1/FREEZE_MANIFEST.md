# DEV15 Stage2 Freeze v1

## Freeze Scope

This portable freeze promotes the minimal governed Stage2 frozen chain needed
for cross-machine reuse without syncing full `data/results/` run directories.

Included layers:

- `S2-2` frozen candidate and evidence bundles
- `S2-3` frozen prompt-preview observability surface
- `S2-4a` frozen prompt-assembly readiness artifacts

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

Non-goal:

- This freeze does not replace the lawful downstream `Stage3` resume boundary,
  which remains the completed Stage2 compatibility projection artifact.

## Portable Authority Statement

For this frozen chain:

- the original run-local `data/results/...` artifact paths remain provenance
  sources only
- they are no longer the portable authority for cross-machine sync of the
  DEV15 frozen Stage2 chain
- the portable authority is this directory:
  - `data/frozen/dev15_stage2_freeze_v1/`
