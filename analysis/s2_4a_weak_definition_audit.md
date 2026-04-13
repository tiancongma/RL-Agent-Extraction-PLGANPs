# S2-4a Weak Definition Audit

## Scope

This audit is read-only and limited to:

- weak-signal definition in the current S2-4a prompt-noise analysis
- alignment of that diagnostic with human reference evidence

## Authoritative source resolution

Per `project/ACTIVE_DATA_SOURCE_CONTRACT.md`, the current repository authority pointer resolves to:

- `data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1`

That active benchmark lineage does not contain the prompt-noise artifacts named in this audit. The requested prompt-noise files were found only at the explicit frozen path:

- `data/frozen/dev15_stage2_freeze_v1/s2_4a/analysis/s2_4a_prompt_noise_metrics.tsv`
- `data/frozen/dev15_stage2_freeze_v1/s2_4a/analysis/s2_4a_prompt_noise_summary.md`
- `data/frozen/dev15_stage2_freeze_v1/s2_4a/analysis/s2_4a_prompt_noise_examples.md`

## Phase A: Weak-signal definition

### Exact maintained upstream producer

The maintained code that produced the frozen S2-4a prompt payloads is:

- `src/stage2_sampling_labels/run_stage2_s2_4a_prompt_construction_v1.py`
- `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::build_live_prompt`

Relevant maintained behavior:

- the S2-4a runner imports `build_live_prompt` and writes `s2_4a_prompts_v1.jsonl` plus `s2_4a_prompt_audit_v1.tsv`
- `build_evidence_blocks_artifact` constructs ordered prompt blocks from selector output
- `build_live_prompt` concatenates those blocks into a governed evidence pack for the live prompt

Concrete maintained references:

- `src/stage2_sampling_labels/run_stage2_s2_4a_prompt_construction_v1.py:15-29`
- `src/stage2_sampling_labels/run_stage2_s2_4a_prompt_construction_v1.py:125-141`
- `src/stage2_sampling_labels/run_stage2_s2_4a_prompt_construction_v1.py:235-260`
- `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py:3671-3791`
- `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py:4144-4394`

### Important negative finding

I did not find a maintained source file, function, notebook, or checked-in script that defines the three prompt-noise labels:

- `VALID_SIGNAL`
- `WEAK_SIGNAL`
- `NOISE`

Repository searches over `src/`, `docs/`, `project/`, and the frozen S2-4a surface found the output artifacts and the frozen prompts, but not a checked-in classifier implementation for those labels.

That means the exact prompt-noise labeler is not presently recoverable as governed maintained code. The best available reconstruction is from the frozen outputs themselves.

### Recovered label logic

From the frozen artifacts, the label set behaves like a surface heuristic audit over already assembled S2-4a prompt spans:

- `VALID_SIGNAL`
  - appears to mean a clearly formulation-relevant span in an explicit prompt block such as preparation, formulation/result table content, or explicit optimization closure
- `WEAK_SIGNAL`
  - appears to be a broad residual bucket for spans that are not obvious noise but are not clean formulation-defining evidence either
  - in practice this includes mixed-purpose materials, context-heavy narrative, partially relevant result prose, and contaminated blocks that still contain some relevant evidence
- `NOISE`
  - appears to be triggered by surface artifacts such as author strings, affiliations, title or abstract bleed, reference-tail content, parser corruption, duplicate payloads, and suspicious tail sections

Observed rule properties:

- block-level type:
  - yes, because the examples are reported as prompt blocks such as `FORMULATION_RESULT_BLOCK`, `OPTIMIZATION_RESULT_BLOCK`, and `PARAGRAPH_BLOCK`
- span-level heuristics:
  - yes, because the examples are shorter representative spans extracted from larger blocks
- regex or pattern rules:
  - strongly implied for `NOISE`
  - evidence: the examples are dominated by affiliations, article metadata, abstract-like bleed, and corrupted fragments rather than semantic misclassification
- duplication detection:
  - yes, at least as a metric input
  - evidence: `duplicate_block_count` is an explicit metric column
- position and tail handling:
  - strongly implied
  - evidence: `reference_section_detected` is an explicit metric column and the examples repeatedly describe noisy tail content
- semantic understanding:
  - no direct evidence of semantic modeling inside the prompt-noise audit
  - the recovered behavior looks surface-rule driven, not meaning-aware

### Governance status

`WEAK_SIGNAL` is not defined in:

- `project/ACTIVE_PIPELINE_FLOW.md`
- `project/ACTIVE_PIPELINE_RUNBOOK.md`
- `project/2_ARCHITECTURE.md`

The same is true for the S2-4a prompt-noise label taxonomy as a whole. The governed documents do mention:

- conservative S2-2a candidate noise filtering
- frozen human reference passages for S2-2b selector auditing

But they do not define `VALID_SIGNAL`, `WEAK_SIGNAL`, or `NOISE` as canonical S2-4a prompt labels.

Conclusion:

- `WEAK_SIGNAL` is an audit-local construct, not a governed pipeline concept
- `VALID_SIGNAL` and `NOISE`, as used in this frozen prompt-noise analysis, are also audit-local labels
- the only nearby governed concept is S2-2a candidate noise filtering, which is different from this prompt-noise metric

## Phase B: Human reference evidence alignment

### What the human reference files are for

The human reference passages in `docs/selector_calibration/` are explicitly governed for S2-2b selector auditing, not for S2-4a prompt-noise scoring.

Governance evidence:

- `project/2_ARCHITECTURE.md:266-269`
- `project/ACTIVE_PIPELINE_FLOW.md:562-569`
- `project/ACTIVE_PIPELINE_RUNBOOK.md:479-491`

Those docs describe the selector calibration set as:

- frozen
- authoritative for S2-2b selector auditing
- stage-local only

They do not extend that role to S2-4a prompt-noise labels.

### Does the prompt-noise audit use human references?

I found no explicit linkage between the frozen prompt-noise artifacts and `docs/selector_calibration/`.

I also found no stored mapping between:

- a human reference passage
- a matching prompt span
- a `VALID_SIGNAL`, `WEAK_SIGNAL`, or `NOISE` label

Therefore the safest factual statement is:

- the current prompt-noise audit is independent of human reference evidence
- any apparent agreement with human reference passages is incidental, not encoded

### Small alignment check

I checked representative passages from:

- `docs/selector_calibration/5ZXYABSU.json`
- `docs/selector_calibration/L3H2RS2H.json`
- `docs/selector_calibration/QLYKLPKT.json`

Key observations:

- clean, role-explicit passages such as the L3H2RS2H materials inventory and optimization saturation passage align well with the recovered `VALID_SIGNAL` behavior
- explicit optimization closure in QLYKLPKT also aligns well with `VALID_SIGNAL`
- mixed-purpose materials blocks create mismatches
  - in 5ZXYABSU and QLYKLPKT, the prompt block that contains the human-reference materials evidence is fused with unrelated assay, biodistribution, or background content
  - under the recovered audit behavior, those mixed blocks are more likely to land in `WEAK_SIGNAL` than `VALID_SIGNAL`

That is a meaning-level mismatch:

- the human reference treats the supplier and material inventory text as formulation-relevant supporting evidence
- the prompt-noise audit appears to down-weight it because the block is structurally messy

## Observed issues

### 1. `WEAK_SIGNAL` is under-defined

The frozen outputs make `WEAK_SIGNAL` behave like a residual bucket:

- not obviously good
- not obviously noise
- therefore weak

That makes the category broad and internally mixed.

### 2. The classifier is not preserved as maintained code

Because the actual labeler script is not recoverable from maintained source:

- the metric is not fully auditable
- the definition is not reproducible from governance alone
- exact boundary cases cannot be verified against code

### 3. The audit is surface-oriented, not evidence-aligned

The examples and metric fields indicate that the audit mainly reacts to:

- block type
- local lexical cues
- tail/reference contamination
- duplication
- corrupted extraction fragments

It does not appear to ask whether a span is formulation-relevant according to the human reference anchor.

### 4. Human-reference material can be downgraded to `WEAK_SIGNAL`

The spot check shows a concrete failure mode:

- a human-calibrated materials passage can be present in the prompt
- but if it is merged with assay or context-heavy prose, the prompt-noise audit likely downgrades it to `WEAK_SIGNAL`

That means `weak_ratio` is not a safe proxy for "human-reference misalignment" or "bad prompt evidence."

## Confidence assessment

High confidence:

- the S2-4a prompt producer is maintained and traceable
- `WEAK_SIGNAL` is not a governed concept in the listed governance docs
- the human reference passage set is governed for S2-2b selector auditing, not for S2-4a prompt-noise scoring
- no explicit linkage between selector calibration passages and prompt-noise labels was found

Medium confidence:

- the recovered definitions of `VALID_SIGNAL`, `WEAK_SIGNAL`, and `NOISE`

Why only medium:

- the checked-in repository preserves the outputs of the prompt-noise audit, but not the maintained implementation that generated the span labels

## Recommendation

`weak_ratio` should be treated as a rough heuristic only.

It is useful for:

- spotting visibly messy prompt packs
- identifying reference-tail or parser-artifact contamination
- comparing coarse prompt cleanliness across papers

It should not be used as a strong decision metric for:

- formulation-evidence quality
- alignment with human reference evidence
- whether a prompt is semantically adequate for formulation extraction

Bottom line:

- `WEAK_SIGNAL` is ad hoc, not governed
- the prompt-noise audit is not explicitly tied to human reference evidence
- `weak_ratio` is not reliable enough to support high-confidence decision-making beyond rough diagnostics
