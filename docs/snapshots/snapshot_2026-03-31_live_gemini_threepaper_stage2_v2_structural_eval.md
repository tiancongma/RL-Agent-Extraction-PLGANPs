# Snapshot: 2026-03-31 Live Gemini Three-Paper Stage2 v2 Structural Evaluation

## Purpose

This snapshot records a live Gemini execution of the governed three-paper
Stage2 v2 comparison slice for:

- `WIVUCMYG`
- `UFXX9WXE`
- `5GIF3D8W`

The purpose of this run was architecture validation, not benchmark reporting.
The question was whether live Gemini Stage2 v2 behavior is structurally better
aligned with the frozen contract than prior extraction behavior.

Guardrail:

- this is `diagnostic-only, not benchmark-valid final output`
- this does not replace `ACTIVE_RUN`
- deterministic Stage2 semantic surfaces remain comparator surfaces only

## Run record

- run id:
  - `run_20260331_1156_03e5d25_threepaper_stage2_v2_live_gemini_structural_eval_v2`
- run root:
  - `data/results/run_20260331_1156_03e5d25_threepaper_stage2_v2_live_gemini_structural_eval_v2/`
- execution mode:
  - `live_llm`
- backend:
  - `gemini`
- model:
  - `gemini-2.5-flash`
- max text chars:
  - `18000`

Execution command:

```powershell
python src/utils/run_threepaper_stage2_v2_comparison.py --run-id run_20260331_1156_03e5d25_threepaper_stage2_v2_live_gemini_structural_eval_v2 --source-mode live_llm --llm-backend gemini --model gemini-2.5-flash --max-text-chars 18000
```

## Resolved governed sources

- active source run:
  - `data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/`
- active scope manifest:
  - `data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/dev15_scope.tsv`
- active deterministic semantic JSONL:
  - `data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/semantic_stage2_objects/semantic_stage2_objects_v1.jsonl`
- active deterministic compatibility TSV:
  - `data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`
- GT boundary scaffold:
  - `data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_formulation_skeleton_gt_v2_variantaware.tsv`
- replay Stage2 v2 comparator:
  - `data/results/run_20260331_1015_03e5d25_threepaper_stage2_v2_comparison_v1/semantic_stage2_v2/semantic_stage2_v2_objects.jsonl`
- historical legacy comparator:
  - `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/weak_labels_v7pilot_r3_fixparse/weak_labels__v7pilot_r3_fixparse.tsv`

## Minimal live-path adapter changes

Two minimal execution-safety adjustments were required:

- `src/stage2_sampling_labels/extract_semantic_stage2_v2_threepaper.py`
  - request Gemini JSON output explicitly with `response_mime_type=application/json`
  - replace raw tab characters before JSON parsing to avoid invalid-control-char
    failures from copied table text
- `src/utils/run_threepaper_stage2_v2_comparison.py`
  - pass through `--max-text-chars`
  - pass explicit `gt_skeleton_tsv` into the comparison builder

These changes were execution adapters only. They do not promote Stage2 v2 and
do not change `ACTIVE_RUN.json`.

## Output surfaces

- `semantic_stage2_v2/semantic_stage2_v2_objects.jsonl`
- `semantic_stage2_v2/semantic_stage2_v2_summary.tsv`
- `analysis/stage2_v2_threepaper_comparison/paper_level_counts.tsv`
- `analysis/stage2_v2_threepaper_comparison/boundary_review.tsv`
- `analysis/stage2_v2_threepaper_comparison/component_completeness_review.tsv`
- `analysis/stage2_v2_threepaper_comparison/expression_richness_review.tsv`
- `analysis/stage2_v2_threepaper_comparison/variable_detection_review.tsv`
- `analysis/stage2_v2_threepaper_comparison/ambiguity_handling_review.tsv`
- `analysis/stage2_v2_threepaper_comparison/structural_comparison_summary.tsv`
- `analysis/stage2_v2_threepaper_comparison/structural_comparison_report.md`

## Structural findings

### Facts

- Live Gemini succeeded and wrote object-first outputs for all three papers.
- Boundary counts were unstable relative to the GT boundary scaffold:
  - `WIVUCMYG`: GT `26`, live `35`, deterministic `26`, replay `29`
  - `UFXX9WXE`: GT `26`, live `3`, deterministic `26`, replay `5`
  - `5GIF3D8W`: GT `26`, live `8`, deterministic `24`, replay `8`
- Expression and ambiguity handling were stronger than a strict slotting
  surface:
  - raw variables and process conditions were preserved
  - unassigned or ambiguous observations were emitted instead of silently
    resolved
- Boundary stability remained the dominant failure mode:
  - `WIVUCMYG` over-split by adding broad umbrella formulations on top of row
    formulations
  - `UFXX9WXE` collapsed DOE row structure into a few coarse formulation objects
  - `5GIF3D8W` retained only a small subset of formulation identities, similar
    to the earlier replay behavior

### Inference

- Live Gemini is better aligned with the frozen architecture principle on
  ambiguity honesty and raw-expression preservation than a wide-table-first
  extractor.
- Live Gemini is not yet better aligned on the most important contract axis:
  formulation boundary stability.
- Because boundary stability matters more than count convenience or variable
  richness, this run is not sufficient evidence to continue current Stage2 v2
  prompting unchanged.

### Uncertainty

- This evaluation covers only three papers.
- The current Stage2 v2 schema still limits some richer multi-expression
  representation, so boundary and schema effects are not fully separated.
- The recommendation below is about this current live prompt/schema slice, not
  about all possible LLM-owned Stage2 designs.

## Recommendation

- judgment:
  - `no-go`
- engineering recommendation:
  - `stop and redesign`

Reason:

- the live slice preserved semantic richness and ambiguity more honestly than
  earlier slotting-oriented behavior
- but it failed the formulation-boundary stability requirement on all three
  target papers
- that means it is not yet structurally better aligned with the frozen Stage2
  contract than the prior extraction behavior
