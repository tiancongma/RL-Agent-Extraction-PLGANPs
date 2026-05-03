# Architecture

This document defines the conceptual and data architecture of the project.
It specifies the stable structure of the pipeline, the responsibilities of each
stage, and the data contracts between stages.

This file is intentionally conservative and should change only when the project
scope changes, not when individual prompts, models, or one-off experiments
change.

---

## Design Philosophy

This project is organized around data semantics, not script order.

- Scripts may change.
- Prompts may change.
- Models may change.

However, the data boundaries, directory semantics, and single sources of truth
must remain stable.

The goal is to ensure that:

- results are reproducible,
- experimental variation is explicit,
- "latest" is never inferred from filenames or memory.

---

## Pipeline Overview

The active implementation is restricted to Stage 0 through Stage 5.
Stages are conceptual contracts first and script namespaces second.

The canonical production path is:

1. Stage 0: raw metadata and relevance filtering
2. Stage 1: cleaned content and manifest construction
3. Stage 2: composite semantic extraction and deterministic post-LLM completion
4. Stage 3: deterministic formulation relation materialization
5. Stage 4: evaluation and diagnostics
6. Stage 5: final formulation closure and benchmark comparison

Manual GT assets are reference inputs to evaluation and comparison. They are
not a production transformation stage.

Architecture note:

- Frozen corrective contract after the 2026-03-30 Stage2 authority-transition
  audit:
  - Stage2 authority belongs to LLM semantic discovery, not deterministic
    semantic reconstruction.
  - Deterministic Stage2 semantic emitters or semantic lifts are fallback,
    comparator, migration-support, or diagnostic infrastructure only.
  - Future drift that re-promotes deterministic semantic Stage2 authority
    should be treated as a contract violation.
- The deterministic post-LLM completion step remains inside Stage2. It does not
  own semantic discovery authority and it is not a separate numbered stage.
- Stage2.5 is retired from the active mainline and remains archived only as a
  historical exploratory path.

## Fine-Grained Internal Hierarchy

The active runtime remains Stage 0 through Stage 5 plus the separate benchmark
comparison node. The following fine-grained hierarchy is a governance mapping
inside those coarse stages; it does not introduce new runtime namespaces.

### Stage0 / Stage1

- `S1-1 Raw ingestion`
  - Zotero-derived raw corpus intake and attachment availability.
- `S1-2 Multi-source manifest assembly`
  - deterministically assemble the canonical manifest key universe and
    bibliographic identity from one or more explicitly declared raw Zotero
    sources.
- `S1-3 Manifest hydration`
  - deterministically hydrate the assembled manifest into the fully usable
    canonical manifest consumed by downstream extraction.
- `S1-3a Asset hydration`
  - bind manifest rows to already-governed cleaned text and table asset
    surfaces.
- `S1-3b Scope overlays`
  - bind deterministic dataset and benchmark scope overlays onto the hydrated
    manifest.

### Stage2

- `S2-1 Scope resolution`
  - resolve the declared manifest scope, cleaned assets, and table assets for
    the current Stage2 execution.
- `S2-2 Evidence construction`
  - the first engineering freeze point in Stage2.
  - outputs:
    `semantic_stage2_objects/normalized_table_payloads/<paper_key>/normalized_table_payloads_v1.json`
    and
    `semantic_stage2_objects/evidence_blocks/<paper_key>/evidence_blocks_v1.json`
  - this boundary may include candidate segmentation, full-table authority
    preservation, and selector work inside S2-2, but prompt assembly must
    consume the persisted semantic-facing evidence artifact rather than
    rereading clean text.
- `S2-2a Candidate segmentation`
  - candidate discovery, structure recovery, and execution-grade full-table
    authority preservation only.
- `S2-2b Selector evidence prioritization`
  - deterministic evidence-driven semantic-facing evidence selection over frozen
    candidate blocks and preserved table authority.
  - selector authority is limited to conservative denoising, minimum evidence
    coverage, and bounded packing.
  - irreversible table preservation is governed by a strict two-class rule:
    `CONFIRMED_NOISE` or `PRESERVE`.
  - rules must not decide whether a table is important and must not downrank,
    suppress, or remove a table based on guessed semantic value.
  - any non-noise table must remain preserved in the pre-LLM authority
    surface, even if additional bounded summary-view labels are still carried
    for observability.
- `S2-3 Prompt assembly`
  - assemble prompt inputs from `evidence_blocks_v1.json` only.
  - must not reread clean text, rescore candidates, or perform new selection or
    ranking.
  - all LLM-facing table evidence at `S2-3` / `S2-4a` is summary-only; full
    tables must not be placed into the prompt surface.
  - the maintained summary path is neutral across preserved tables; the main
    residual risk is lossy compression, not primary-table reranking.
  - header / column schema and first-column row-identity surfaces are the
    primary summary contract; sample rows are optional aids only.
  - the LLM may see a lossy or compact summary of a table here, but this
    surface must never become the sole execution source of truth.
- `S2-4a Prompt construction freeze boundary`
  - optional frozen local boundary that materializes prompt artifacts from the
    canonical S2-3 evidence handoff and stops before the live LLM call.
  - this boundary preserves the summary-only table contract and explicitly
    tells the LLM to determine semantic table scope itself when multiple
    candidate table summaries are present.
- `S2-4b Live LLM call freeze boundary`
  - the only nondeterministic Stage2 substep.
  - output: raw LLM response payloads under the run-scoped raw-response
    surface.
- `S2-5 Semantic parsing`
  - parse raw LLM responses into semantic-intermediate object artifacts.
- `S2-6 Contract validation`
  - validate Stage2 authority and provenance contracts.
  - this is a guardrail and legality check, not selector logic.
- `S2-7 Compatibility projection`
  - deterministic Stage2 handoff into the downstream-ready compatibility
    surface consumed by Stage3.
  - this is compatibility projection and Stage3 handoff, not evidence
    construction.

### Stage3

- `S3-1 Relation materialization`
  - construct explicit paper-level relation artifacts from the completed Stage2
    candidate surface.
- `S3-2 Relation resolution`
  - resolve inherited or shared relation-backed fields for downstream
    materialization.

### Stage5

- `S5-1 Fixed-row candidate intake`
  - materialize the fixed formulation-row universe from the completed Stage2
    candidate surface plus required Stage3 relation artifacts.
  - this substep may consume Stage3 resolved fields, but downstream S5 value
    layers must not create or remove formulation rows.
- `S5-2 Deterministic direct materialization`
  - apply source-faithful deterministic rules over the fixed row universe:
    DOE/table row materialization already authorized upstream, row-local direct
    table-cell binding, unique source-backed shared carry-through, value/unit
    split from direct source cells, filtering, normalization, and identity
    guardrails.
  - it must not perform donor-fill, assumption-based inference, modeling-target
    projection, or derived arithmetic.
- `S5-3 LLM-assisted direct value candidate extraction`
  - optional Stage5-local LLM value layer after formulation boundaries are fixed.
  - the LLM may propose direct value candidates only for existing Stage5 rows,
    with exact source evidence, scope, direct/derived classification, prompt
    hash, model identity, and input artifact hashes.
  - this substep must not change formulation membership, rediscover row
    boundaries, calculate values, or treat system final-table values as source
    authority.
- `S5-4 Value authority validation and merge`
  - validate and merge S5-2 deterministic values and S5-3 LLM candidates under
    the same authority ladder: row-local direct evidence, typed row-local
    assignments, unique table-scoped direct evidence, unique paper/global direct
    constants, and only then accepted LLM direct candidates satisfying the same
    evidence and scope rules.
  - derived, ambiguous, conflict-bearing, or quote-less candidates are rejected
    from the direct layer or moved to a review queue.
- `S5-5 Derived reasoning / calculated value materialization`
  - compute separately-provenanced derived values such as `%w/v × mL -> mg`,
    `mg/mL × mL -> mg`, concentration × volume, ratio-derived mass, and unit
    conversions only from accepted direct inputs.
  - derived outputs must live in sidecars with formula IDs, input provenance,
    and `eligible_for_direct_compare=no`; they must not contaminate current
    direct-evidence GT comparison.
- `S5-6 Final table closure and audit export`
  - emit the primary benchmark-facing final formulation table, linked lower-level
    descendant records, decision trace, and value-layer sidecars.
  - when Stage5 excludes or collapses downstream/post-processing descendants
    that do not define independent benchmark-facing formulation identity,
    preserve them in a governed linked lower-level record surface rather than
    silently dropping them.
  - the final table remains the Stage5 benchmark-facing object. Direct-value and
    derived-value sidecars must remain distinguishable in downstream compare,
    audit, and modeling-ready outputs.

### Benchmark

- `B-1 GT compare`
  - compare only the Stage5 final table to the frozen GT reference inputs.

### Cross-cutting Layers

- Feature governance layer
  - run-scoped feature activation, execution-ledger, and governance observability.
- Memory layer
  - the governed supporting memory surface under `data/mem/v1/`.
  - it is not a numbered pipeline stage.

---

## Stage 0 - Raw Metadata and Relevance Filtering

### Purpose
Identify candidate papers that are potentially relevant to PLGA nanoparticle
formulation and build the checked raw Zotero-derived input surface.

### Typical Operations
- regex-based pre-filtering
- LLM-based relevance classification
- Zotero tagging and attachment fetching

### Characteristics
- outputs are rerunnable
- outputs are non-binding
- downstream stages should depend on the checked raw JSONL artifact, not on
  transient helper state

### Location
`data/raw/zotero/`

---

## Stage 1 - Cleaned Content and Manifest

### Purpose
Convert HTML/PDF documents into cleaned text and stable corpus assets, then
establish a manifest linking papers to those assets.

### Key Outputs
- cleaned full text
- key-to-text mappings
- dataset-local table assets
- assembled manifest rows linking key, DOI, title, and declared upstream source
  lineage
- hydrated manifest rows linking key, DOI, title, content paths, table paths,
  and deterministic scope overlays

### Single Source of Truth
`data/cleaned/index/manifest_current.tsv`

### Location
- `data/cleaned/content/`
- `data/cleaned/index/`
- dataset-local cleaned roots such as `data/cleaned/goren_2025/`

### Invariants
- Any change to files in `data/cleaned/index/` requires re-running all
  downstream stages.
- Only one `manifest_current.tsv` may exist as the active authoritative manifest.
- `manifest_current.tsv` may be assembled from one or more explicitly declared
  raw Zotero-source manifests, but the assembly contract must preserve row-level
  source provenance rather than assuming a single upstream corpus.
- `manifest_current.tsv` is contract-complete only after explicit manifest
  hydration populates asset-binding and scope-overlay fields from governed
  Stage1 source surfaces.

---

## Stage 2 - Composite Semantic Extraction Layer

### Purpose
Run one composite Stage2 contract over cleaned paper content and structured
assets:

1. LLM semantic discovery
2. deterministic post-LLM completion for downstream readiness

Stage2 is the authoritative extraction stage. Raw LLM semantic objects are an
internal Stage2 intermediate. The completed Stage2 artifact is the only valid
Stage3 input and the only authoritative Stage2 evaluation target.

The maintained Stage2 path now also includes one formal internal pre-LLM
boundary:

- S2-2: clean text -> governed evidence package
- explicit internal sub-boundary inside S2-2:
  - clean text / extracted tables -> candidate segmentation -> evidence-driven
    selector -> governed evidence package
- candidate-segmentation artifact:
  `data/results/run_<run_id>/semantic_stage2_objects/candidate_blocks/<paper_key>/candidate_blocks_v1.json`
- execution-grade full-table authority artifact:
  `data/results/run_<run_id>/semantic_stage2_objects/normalized_table_payloads/<paper_key>/normalized_table_payloads_v1.json`
- execution payload members:
  `data/results/run_<run_id>/semantic_stage2_objects/normalized_table_payloads/<paper_key>/payloads/*.csv`
- authority validation artifact:
  `data/results/run_<run_id>/analysis/table_authority_validation_v1.tsv`
- canonical artifact:
  `data/results/run_<run_id>/semantic_stage2_objects/evidence_blocks/<paper_key>/evidence_blocks_v1.json`
- producer:
  `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`
- consumer:
  the same maintained extractor's evidence-driven selector consumes the persisted
  candidate surface and then the prompt-assembly path consumes the canonical
  evidence artifact before live LLM calls or replay normalization; downstream
  deterministic execution may resolve back to the preserved S2-2 full-table
  authority surface by stable table identity
- candidate responsibility rule:
  candidate segmentation performs structure recovery, candidate generation,
  conservative table isolation, and conservative high-confidence noise
  filtering only
- full-table authority rule:
  when a formulation-relevant table is detected, S2-2 must preserve an
  execution-grade table surface that is lossless or maximally
  structure-preserving relative to the best available Stage1 table asset
- full-table preservation rule:
  the preserved execution-facing table surface must retain row numbering, row
  order, column structure, header hierarchy when available, and table-local
  identifiers when available
- authority storage rule:
  the execution-facing table surface must be stored in S2-2 and must remain
  bound to the selected table identity rather than being rebuilt ad hoc
  downstream
- authority schema rule:
  each preserved table authority record must carry stable `table_id`,
  `source_table_reference`, deterministic `table_type`, `row_count`,
  `has_row_numbering`, `header_structure`, `raw_cells`, execution-facing
  `normalized_rows`, `row_identity_signals`, and `reconstruction_confidence`
- table-authority ranking rule:
  S2-2a may rank recovered table payloads into primary versus secondary
  preserved authority using conservative artifact-level signals such as repair
  quality, row-anchor stability, formulation-structure density, and obvious
  downstream-result demotions; it must not emit semantic roles, semantic
  signals, or pre-LLM paper interpretation
- structure-first primary rule:
  coarse labels such as `table_type=non_formulation_table` or
  `table_role_hint=characterization/results` are noisy priors only
- primary-eligibility rule:
  those coarse labels may demote a recovered table inside S2-2a ranking, but
  they must not by themselves make a structurally strong table ineligible for
  primary authority
- hard-guardrail rule:
  only structural failure such as repair-insufficient payloads, narrative /
  figure-caption domination, or obvious non-tabular spillover may block
  primary authority
- selector responsibility rule:
  selector prioritizes and retains semantic-facing evidence from candidate
  blocks; it does not own lossless table preservation anymore
- semantic-facing summary rule:
  `evidence_blocks_v1.json` is the maintained semantic-facing summary or
  evidence surface for selector behavior, prompt assembly, and LLM semantic
  authorization
- summary observability rule:
  table-derived summary blocks must carry stable `table_id` and explicit
  `summary_is_lossy=true`
- execution-facing authority rule:
  `normalized_table_payloads_v1.json` is the maintained execution-facing
  full-table authority surface for downstream deterministic table execution
- execution-input rule:
  DOE and non-DOE deterministic table row materialization must resolve
  semantic target -> stable `table_id` -> preserved S2-2 full-table authority
  surface; Stage1 table assets may remain a reconstruction fallback inside
  S2-2a only and must not remain the downstream execution source of truth
- authority-metadata boundary rule:
  deterministic handles such as `authority_run_dir`,
  `authority_payload_root`, and table-scope locators are execution-side
  metadata, not LLM semantic content; replay compatibility should reattach
  them through governed sidecars or reattachment surfaces
- table-surface principle:
  the LLM sees a semantic-facing summary of a table, while deterministic
  execution operates on the preserved table entity bound to the same stable
  table identity
- observability rule:
  `analysis/stage2_prompt_preview_v1.tsv` is derived from the canonical S2-2
  artifact and is not the primary truth surface
- candidate observability rule:
  `analysis/candidate_segmentation_debug_v1.tsv` is the maintained run-level
  surface for inspecting candidates before selector prioritization
- selector rule:
  the maintained S2-2 path uses deterministic evidence-driven evidence
  selection with conservative denoising, but irreversible table removal is
  governed by a confirmed-noise-only rule
- selection policy rule:
  the selector must not decide whether a table is important; if a table is not
  confirmed pure noise, preserve it in the pre-LLM authority surface
- summary-neutrality rule:
  the maintained `S2-3` / `S2-4a` summary path is neutral across preserved
  tables; the main residual risk is lossy compression rather than
  cross-table importance bias
- summary-structure rule:
  header / column schema and first-column row-identity surfaces are the
  primary summary contract; sample rows are optional aids only
- success rule:
  the S2-2 artifact must distinguish `technical_status` from `design_status`
  so artifact emission is not mistaken for input-contract conformance
- segmentation closure freeze rule:
  after a governed segmentation-closure decision is recorded for the current
  cycle, S2-2a candidate segmentation is frozen by default
- segmentation closure non-regression rule:
  selector and evidence-prioritization work must not modify segmentation logic
  unless a concrete regression is demonstrated against the frozen closure state
- selector-phase focus rule:
  after segmentation closure, remaining S2-2 design failures are treated as
  selector-evidence or table-extraction-quality investigations first, not as
  justification for segmentation redesign
- S2-2b stage-local debugging rule:
  S2-2b is strict stage-local selector debugging on frozen S2-2a inputs only
- S2-2b auditing uses a frozen human reference passage set sourced from docs/selector_calibration/.
- selector non-discovery rule:
  selector must not introduce new candidate discovery behavior and operates
  strictly on existing `candidate_blocks_v1.json`
- S2-2b forbidden closure inputs:
  no use of S2-3 through S2-7 behavior, no use of downstream Stage3, Stage4,
  or Stage5 outputs, and no use of GT comparison for closure
- S2-2b non-benchmark rule:
  this is a stage-local freeze only, not downstream system validation
- frozen discoverability rule:
  once a fine-grained Stage2 substep is frozen in the current cycle, the repo
  must expose discoverable ownership, inputs, outputs, stop boundary, and next
  lawful step in maintained governance and execution-facing surfaces
- current-cycle explicit frozen discoverability mapping:
  - `S2-2a`
    - owner:
      `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::build_candidate_segmentation_artifact`
    - outputs:
      `semantic_stage2_objects/candidate_blocks/<paper_key>/candidate_blocks_v1.json`
      `semantic_stage2_objects/normalized_table_payloads/<paper_key>/normalized_table_payloads_v1.json`
      `semantic_stage2_objects/normalized_table_payloads/<paper_key>/payloads/*.csv`
      `analysis/candidate_segmentation_debug_v1.tsv`
      and `analysis/table_authority_validation_v1.tsv`
    - stop boundary:
      candidate segmentation, conservative table-authority ranking, and
      execution-grade table preservation only; no semantic role packaging and
      no row materialization
    - next lawful step:
      `S2-2b`
  - `S2-2b`
    - owner:
      `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::build_evidence_blocks_artifact`
      plus `build_evidence_priority_selection`
    - outputs:
      `semantic_stage2_objects/evidence_blocks/<paper_key>/evidence_blocks_v1.json`
      and `analysis/table_selection_debug_v1.json`
    - selection contract:
      deterministic evidence selection remains evidence-level only and now enforces a minimal evidence sufficiency floor after ranking
      the floor may add one best method block, one best materials block, or one bounded supporting block when clearly available
      the floor must not assign semantic roles, infer semantic signals, or perform pre-LLM semantic extraction
    - stop boundary:
      semantic-facing evidence handoff written; execution-facing full-table
      authority remains preserved from S2-2a
    - next lawful step:
      `S2-3`
  - `S2-3`
    - owner:
      `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::build_live_prompt`
      plus `build_prompt_preview_row`
    - outputs:
      in-memory semantic-only prompt payload and maintained observability
      `analysis/stage2_prompt_preview_v1.tsv`
    - prompt contract:
      LLM-facing prompt text contains semantic task instructions, schema, paper identity, and the governed evidence pack only; runtime packing metadata remains in preview/audit surfaces rather than the default live prompt header
    - next lawful step:
      `S2-4b live LLM call`, or explicit `S2-4a` prompt materialization when prompt freezing is active
  - `S2-4a`
    - owner:
      `src/stage2_sampling_labels/run_stage2_s2_4a_prompt_construction_v1.py`
    - outputs:
      `analysis/s2_4a_prompt_template_v1.txt`
      `analysis/s2_4a_prompts_v1.jsonl`
      `analysis/s2_4a_prompt_audit_v1.tsv`
      and stage-local `RUN_CONTEXT.md`
    - next lawful step:
      `S2-4b live LLM call`
    - governance note:
      `S2-4a` audit is a governance layer with three separated questions:
      Hard Gate legality/readiness, Feature Activation Audit, and
      Calibration Review only
  - `S2-4b`
    - owner:
      `src/stage2_sampling_labels/run_stage2_s2_4b_live_llm_call_v1.py`
    - inputs:
      frozen `analysis/s2_4a_prompts_v1.jsonl`
    - outputs:
      replayable `raw_responses/<paper_key>__stage2_v2_raw_response.json`
      request metadata sidecars under `request_metadata/`
      `analysis/s2_4b_request_summary_v1.tsv`
      and stage-local `RUN_CONTEXT.md`
    - next lawful step:
      `S2-5 semantic parsing` only through the maintained composite Stage2 path

### Key Artifacts
- Stage2 internal semantic-intermediate artifacts:
  - canonical S2-2 evidence-block JSON artifacts
  - run-scoped semantic-object JSONL payloads
  - run-scoped semantic-object summary TSVs
  - run-scoped raw response copies when replay or live LLM execution is used
  - paper-local evidence handoff references carried inside the semantic objects
  - object families:
  - `formulation_identity_candidate`
  - `component_candidate`
  - `phase_candidate`
  - `process_step_candidate`
  - `variable_or_factor_candidate`
  - `measurement_candidate`
  - `relation_cue`
  - `evidence_handoff`
- Stage2 completed downstream-ready artifacts:
  - compatibility-projected legacy wide-row TSV
  - compatibility-projected legacy wide-row JSONL
  - projection trace TSV
  - projection summary JSON

### Characteristics
- Stage2 is one composite stage, not multiple numbered stages
- open semantic discovery and formulation-boundary discovery are owned by the
  LLM substep
- deterministic post-LLM completion remains inside Stage2 and exists only to
  make Stage2 outputs reconstructable and relation-ready for unchanged
  downstream consumers
- Stage2 may emit governed table-authorization markers that declare a table as
  formulation-bearing without enumerating its rows in the LLM substep
- deterministic row expansion remains inside the Stage2 completion step and may
  enumerate explicit row-level candidates only from:
  - LLM-declared DOE scope through the existing DOE enumerator
  - LLM-declared non-DOE formulation-table scope through the table-row
    expansion contract
- no formulation candidate may enter authoritative Stage2 output unless it is
  traceable to `llm_semantic_discovery` or an explicitly declared governed
  fallback semantic source
- formulation identity discovery, component discovery, factor discovery, and
  raw expression capture are owned by the LLM substep
- raw semantic objects may remain incomplete where the paper support is
  incomplete
- deterministic post-LLM completion must not be mistaken for Stage3 or Stage5
- Stage2 final output must not be treated as final benchmark materialization
- deterministic Stage2 semantic reconstruction paths are non-authoritative and
  must not replace the LLM Stage2 boundary as active mainline authority
- deterministic DOE row enumeration is allowed only as row-level expansion
  within LLM-declared DOE scope unless an explicitly declared governed fallback
  mode is active
- deterministic non-DOE table row enumeration is allowed only when the LLM
  declares the table formulation-bearing and non-DOE through the table
  authorization contract
- a bounded simple-table deterministic enumeration path is allowed inside the
  non-DOE table-row executor when:
  - the table is already LLM-authorized as formulation-bearing
  - the table is not on the DOE path
  - preserved `S2-2` normalized payload authority is available
  - the table is a low-ambiguity `full_formulation` surface with stable
    first-column row identity
  - base formulation rows can be instantiated from the preserved table alone
- this simple-table path does not require LLM row-level candidates and does
  not cover DOE matrices, non-DOE sweep recovery, or cross-table decoding
- direct comparison of raw semantic objects to formulation-level GT is
  diagnostic only when the deterministic completion substep has not been
  applied
- deterministic execution ownership after Stage2 decomposition must remain
  provable from run artifacts rather than inferred from code presence or
  registry presence alone
- silent non-activation of governed deterministic execution units is an
  auditable failure state
- `S2-4a` audit architecture is intentionally split:
  - Hard Gate evaluates whether the frozen pre-LLM input is legal and ready
    for `S2-4b`
  - Feature Activation Audit evaluates whether maintained repaired capabilities
    are actually active in run artifacts
  - Calibration Review evaluates semantic correctness on known-answer papers
    only
- Hard Gate must not decide the true semantic primary table
- semantic table truth remains LLM-owned in the active pipeline and may be
  reviewed post hoc through calibration, but it is not hard-gated by selector
  rules
  architecture failure when the semantic authorization signal is present

### Current Functional-Unit Execution Status

- The intended active contract remains:
  - LLM semantic discovery and authorization
  - deterministic function units for governed execution
- The DEV15 decomposition-era audit established a real architecture failure:
  semantic signals could exist while governed deterministic function units were
  not reliably taking control of execution on the mainline.
- In the failed DEV15 lineage, sequential optimization behavior remained
  active, while DOE and non-DOE table-row execution were not yet provably
  reliable on-path across the same governed lineage.
- The DOE execution path is now restored on the mainline for the confirmed
  `UFXX9WXE` repair case:
  - `data/results/20260406_ced19d6/07_doe_fu_ufxx_scopefix`
  - governed DOE function-unit activation emitted `26` deterministic rows
- Non-DOE table-row execution has partial downstream repair only:
  - rule and unit-level execution issues improved observability and restored
    already-authorized cases
  - broader DEV15 coverage remains upstream-blocked when
    `table_formulation_scopes` are missing from the Stage2 evidence handoff
- a bounded simple formulation-table deterministic enumeration rule is now
  validated for low-ambiguity `full_formulation` tables:
  - `INMUTV7L` is the anchor case
  - deterministic replay emits `12` preserved table rows after semantic table
    authorization without requiring LLM row-level output
  - `WIVUCMYG` remains on the DOE path
  - `5GIF3D8W` remains on the non-DOE single-variable recovery path
  - `UFXX9WXE` remains stable with no regression
- This means the dominant remaining limitation is now upstream Stage2
  extraction, selector, or evidence-handoff completeness rather than a claim
  that deterministic execution no longer matters.

### Current Clarification On Stage2 Contract Pressure

Observed facts from the maintained Stage2 contract audit:

- the recent authority correction restored the intended semantic-authority
  split:
  - deterministic semantic overreach is no longer the active mainline
    architecture
  - marker provenance is preserved
  - deterministic execution is marker-authorized only
  - the maintained validator is enforcing the governed contract
- however, the current live Stage2 semantic contract still places substantial
  burden on the LLM before deterministic completion begins
- the remaining bottleneck is therefore not only model quality
- the remaining bottleneck also includes:
  - semantic contract rigidity
  - LLM role overload
  - suppression of governed markers under execution-level uncertainty

Locked interpretation:

- this is not a rollback of the restored LLM semantic-authority boundary
- this is not permission for deterministic semantic inference
- this is not permission for vague uncontrolled free-text outputs
- it is a clarification that semantic understanding is not the same thing as
  executable structure

Future-facing design direction:

- the preferred Stage2 contract direction is for the LLM to emit reusable
  semantic cues and governed intermediate markers
- the preferred Stage2 contract direction is not for the LLM to emit
  execution-ready formulation structures earlier than needed
- when governance permits, partial semantic markers may be preferable to hard
  suppression if the paper-level semantic signal is present but some
  execution-level grounding remains incomplete
- stricter execution completion, row expansion, decomposition, relation
  binding, normalization, and validation should remain downstream deterministic
  responsibilities

Current implemented clarification:

- the maintained Stage2 contract now permits governed partial semantic markers
  for the suppression-prone `selection_marker` and `inheritance_marker`
  families
- these partial markers are allowed only for non-execution-critical grounding
  gaps:
  - `selection_marker.source_table_id`
  - `selection_marker.selected_variable`
  - `selection_marker.selected_value`
  - `inheritance_marker.from_table`
  - `inheritance_marker.to_table`
- execution-critical fields remain strict in the active runtime:
  - `inheritance_marker.inherit_type`
  - `inheritance_marker.variable`
  - `inheritance_marker.value`
- the maintained contract now distinguishes:
  - `execution_ready`
  - `partial_semantic`
- only `execution_ready` markers continue into the current deterministic
  Stage2-to-Stage3 handshake
- `partial_semantic` markers are preserved in the Stage2 semantic-intermediate
  artifact for auditability and future governed completion work, but they do
  not authorize current row expansion or current Stage3 inheritance
  materialization

Non-change statement:

- the active runtime remains the current composite Stage2 contract
- no new pipeline stage is introduced by this clarification
- no deterministic fallback path is promoted by this clarification

### Stage2 Internal Intermediate Artifact
`data/results/run_<run_id>/semantic_stage2_objects/semantic_stage2_v2_objects.jsonl`

### Stage2 Internal Pre-LLM Evidence Artifact
`data/results/run_<run_id>/semantic_stage2_objects/evidence_blocks/<paper_key>/evidence_blocks_v1.json`

### Stage2 Internal Full-Table Authority Artifact
`data/results/run_<run_id>/semantic_stage2_objects/normalized_table_payloads/<paper_key>/normalized_table_payloads_v1.json`

Contract note:
- this is the current maintained implementation of the Stage2 full-table
  authority surface
- it is execution-facing, not prompt-facing
- it must preserve table identity and execution-grade structure for downstream
  authorized deterministic row materialization
- it now records stable `table_id`, `source_table_reference`, deterministic
  `table_type`, `row_count`, `has_row_numbering`, `header_structure`,
  `raw_cells`, execution-facing `normalized_rows`, `row_identity_signals`,
  and `reconstruction_confidence`
- the colocated `payloads/*.csv` files are additive execution payload members
  referenced by the JSON artifact and are not an ad hoc downstream rebuild
- DOE and non-DOE deterministic table execution must resolve through this
  preserved authority surface rather than using the semantic-facing summary
  view as the execution input

### Stage2 Authoritative Completion Artifact
`data/results/run_<run_id>/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`

---

## Stage 3 - Deterministic Formulation Relation Materialization

### Purpose
Convert the completed Stage2 candidate formulation-instance rows into explicit
paper-level relation artifacts without any LLM usage.

### Why this stage exists
The pipeline needs an auditable intermediate layer that makes relation
structure explicit before final flattening. This layer separates relation
reasoning from final benchmark-facing row closure.

### Key Artifacts
- relation-record TSVs
- per-paper logic-graph JSON artifacts
- per-paper relation summary TSVs

### Characteristics
- deterministic and reproducible
- no LLM or external API calls
- explicit method-group, shared-field, variation-axis, and parent-link structure
- intermediate production artifact, not benchmark-valid output

### Location
- `src/stage3_relation/`
- `data/results/run_<run_id>/...`

---

## Stage 4 - Evaluation and Diagnostics

### Purpose
Quantitatively evaluate extraction quality and support targeted debugging and
review.

### Typical Operations
- rule-based grading
- metric calculation
- reviewer-facing mismatch and alignment surfaces

### Characteristics
- downstream of the compatibility-projected legacy wide-row surface
- diagnostic only
- not the benchmark-valid system endpoint

### Location
`src/stage4_eval/`

---

## Evaluation Reference Assets

### Purpose
Provide partial, human-curated labels for comparison and review.

### Characteristics
- ground truth is intentionally incomplete
- manual labels do not overwrite weak labels
- disagreement between system output and GT is expected during development

### Location
`data/cleaned/labels/manual/`

---

## Stage 5 - Final Formulation Closure And Benchmark Comparison

### Current Phase: Diagnostic Development Mode

- The repository is currently operating in diagnostic development mode.
- In this phase, Stage5 outputs are diagnostic baselines for debugging work.
- Identity freeze is a diagnostic-only, non-blocking risk signal.
- Benchmark mode is disabled in the current phase.
- Legal recovery must remain LLM-first: the LLM owns semantic formulation understanding and declaration of the candidate universe, while deterministic rules may only validate, normalize, align, or refill values that are already semantically authorized by the LLM or by governed explicit evidence handoff.
- Deterministic value restoration must not become a substitute semantic engine and must not let rules silently redefine formulation meaning, row membership, or candidate identity.
- Any repository reference to a baseline in this phase means diagnostic baseline unless a governed contract explicitly states otherwise.
- DEV15 is a governed diagnostic-development set used to observe whether diagnosis baselines improve as pipeline repairs land; it must not be described as if a complete benchmark-certified endpoint already exists for that set.
- Outside explicit frozen GT scopes such as DEV15, runs may have no GT at all; those runs must be described as diagnosis, audit, or extraction-development runs rather than benchmark runs.

### Purpose
Convert candidate formulation-instance outputs into final one-row-per-
formulation records and compare only those final records to GT.

### Key Principle
Stage 5 is the only benchmark-valid reporting layer. Earlier stages may produce
diagnostic comparisons, but they are not the official system result.

In the current repo phase, this benchmark-validity statement is a reserved future-state rule, not a requirement that the current DEV15 work produce benchmark-certified outputs.

Benchmark-validity clarification:

- Stage5 final-table generation is necessary but not sufficient for
  benchmark-valid reporting.
- Benchmark legality additionally requires the separate GT compare node.
- In the current diagnostic-development phase, identity freeze remains visible
  as a risk signal but does not block execution.
- For current DEV15 work, the target artifact is a diagnosis baseline that can be compared repeatedly against the same frozen GT to measure directional improvement during development.
- If no explicit governed GT exists for a run scope, no benchmark-valid claim should be attempted and no benchmark language should be used beyond historical/governance reference.
- The full DEV15 run
  `data/results/20260401_5d9f4e6/09_dev15_count_validation`
  reached Stage5 final-table materialization but failed the mandatory
  identity-freeze gate and therefore did not produce legal benchmark-valid GT
  compare or modeling-ready continuation outputs.
- The governed repair lineage has localized the failure classes as:
  - row count drift
  - identity reassignment
  - unresolved scaffold binding
- Scaffold-binding and representation repair work are part of the governed
  follow-on repair lineage, but they do not by themselves prove that a lawful
  full-pipeline benchmark run now passes the hard identity-freeze gate.

### Internal Stage5 Families

- Benchmark-final family
  - canonical object:
    `final_formulation_table_v1.tsv`
  - linked lower-level preserved record surface:
    `downstream_variant_records_v1.tsv`
  - maintained entrypoints:
    `src/stage5_benchmark/build_minimal_final_output_v1.py`
    `src/stage5_benchmark/enforce_identity_freeze_v1.py`
    `src/stage5_benchmark/compare_final_table_to_gt_v1.py`
  - role:
    source-faithful final closure, identity-preserving filtering, explicit
    Stage3-resolved field carry-through, preservation of excluded
    downstream/post-processing descendant records in a linked lower-level
    surface, diagnostic identity-freeze risk reporting, and GT compare
  - legality rule:
    `final_formulation_table_v1.tsv` is currently managed as a diagnostic
    baseline surface; final-table materialization legalizes diagnostic compare
    in the current phase, while benchmark mode remains disabled
  - primary identity rule:
    benchmark-facing formulation identity is one row per independently
    reported formulation identity; downstream/post-processing descendants do
    not join the primary final table unless the paper explicitly reports them
    as independent formulation identities
  - lower-level preservation rule:
    excluded downstream/post-processing descendants must remain visible in the
    linked `downstream_variant_records_v1.tsv` surface with parent linkage,
    change-role semantics, downstream variable payloads, and exclusion
    provenance
  - benchmark-final must not:
    replace paper-reported values with convenience-normalized values, perform
    donor-fill, perform assumption-based inference, or change formulation
    membership after identity freeze

- Downstream modeling-ready family
  - role:
    consume the frozen benchmark-final object and build downstream
    normalization, harmonization, derivation, or curated projection surfaces
    for non-benchmark use
  - first maintained surface:
    `src/stage5_benchmark/build_modeling_ready_sidecar_v1.py` emits a
    row-linked sidecar from `final_formulation_table_v1.tsv` using explicit
    deterministic parse/math rules only
  - first row-wise surface:
    `src/stage5_benchmark/build_modeling_ready_table_v1.py` pivots selected
    sidecar values back into one row per frozen `final_formulation_id` while
    keeping raw benchmark-final carry-through fields distinct from transformed
    modeling columns
  - allowed operations:
    canonical label cleanup, unit harmonization, safe deterministic derivation,
    curated projection, and preservation of raw benchmark-final values plus
    provenance
  - modeling-ready outputs must not:
    replace `final_formulation_table_v1.tsv`, redefine benchmark-final
    semantics, or change formulation membership

- Downstream audit/review family
  - role:
    consume the frozen benchmark-final object for reviewer-facing audit exports
    and workbooks
  - these surfaces are downstream support artifacts, not benchmark-final
    builders and not modeling-ready projections

---

## Separation Of Concerns

- Stage2 owns semantic discovery from paper text and tables plus the
  deterministic post-LLM completion required for downstream readiness.
- Stage2 does not let the LLM enumerate table rows.
- Stage2 may let the LLM declare:
  - `table_formulation_scope`
  - `variable_roles`
  - `selection_marker`
  - `inheritance_marker`
  - `boundary_marker`
- those markers authorize deterministic row expansion but do not themselves
  create formulation rows.
- Stage3 owns relation resolution over the compatibility-projected rows.
- Stage3 consumes deterministic DOE rows and deterministic non-DOE table rows
  through one compatible candidate-row surface, then applies inheritance and
  shared-context binding without Cartesian reconstruction.
- Stage5 owns final materialization and benchmark-facing closure.
- Stage5 must not absorb semantic inference that belongs to Stage2.

### Current Phase-1 Boundary
- Stage 5 consumes the compatibility-projected legacy wide-row TSV produced by
  the deterministic adapter.
- It may also consume the Stage 3 relation-record TSV as deterministic
  provenance.
- The relation artifact is not yet the sole driver of final keep/drop/collapse
  decisions.
- Some branch or historical Stage5 helper scripts still project from legacy
  weak-label inputs rather than from the frozen benchmark-final object.
  Those helpers are not part of the active benchmark-final family and should be
  treated as legacy or branch-only modeling utilities until they are
  explicitly re-anchored downstream of `final_formulation_table_v1.tsv`.

### Location
- `src/stage5_benchmark/`
- `data/results/run_<run_id>/`

---

## Run Tracking and Reproducibility

Each pipeline execution is assigned a unique `run_id`:

`run_YYYYMMDD_HHMM_<git_commit>_<sample>`

Run outputs live under:

`data/results/run_<run_id>/`

Every run directory must contain a reproducibility-grade `RUN_CONTEXT.md`.
That run context must record both script lineage and feature activation lineage
for governed runs.

### Lineage containment policy

A top-level `data/results/run_*` directory now represents one benchmark or
experiment lineage, not every internal retry or repair step.

Child executions that belong to the same lineage must live under the parent
run directory, for example:

- `data/results/<parent_run_id>/lineage/children/<ordered_role>/<child_run_id>/`

This child-execution rule applies to:

- stage-local retries
- partial reruns
- recovery passes for failed papers
- deterministic refresh steps
- stage-only materialization runs
- merge or completion steps that serve the same declared lineage objective

The parent lineage directory remains the authoritative human-facing entrypoint.
It must expose:

- its own `RUN_CONTEXT.md`
- a lineage mapping artifact if child runs were moved or nested
- an explicit child-step index when the lineage includes retries or repair work

### Active data-source authority

For current `data/results` workflows, the repository-level active source must
be declared explicitly.

Authority order:

1. explicit CLI source such as `--run-dir`
2. `data/results/ACTIVE_RUN.json`
3. otherwise hard error

The architecture forbids resolving the active source by:

- lexical sort order
- modification time
- parent fallback
- glob-first matching
- unstated defaulting

Independent top-level runs are allowed only when the declared objective, scope,
or benchmark contract is materially separate from an existing lineage.

---

## Architectural Invariants

- Files in `data/cleaned/index/` are pipeline-critical.
- `manifest_current.tsv` and `key2txt.tsv` are unique active authorities.
- No script may silently depend on legacy artifacts.
- Code presence is not activation; feature activation must be proven by run
  artifacts, including the generated Feature Unit Activation section inside
  `RUN_CONTEXT.md`.
- Experimental variation must be expressed via:
  - `run_id`
  - explicit configuration or CLI arguments
  - git commit
- Official benchmark reporting may occur only from Stage 5 final-table
  comparison outputs.

---

## Relationship to Project Governance

- project scope and stage transitions are defined here
- architectural decisions are recorded in `project/4_DECISIONS_LOG.md`
- agent run procedure is defined in `project/ACTIVE_PIPELINE_RUNBOOK.md`
- active script registry and stage roles are defined in
  `project/PIPELINE_SCRIPT_MAP.md` and `docs/src_script_registry.tsv`

This architecture is intentionally minimal, explicit, and extensible.

---

## Debug And Human Review Contract

All debug artifacts intended for manual inspection must include DOI-level
metadata.

Any Excel or TSV generated for human review must contain:

- doc_key (Zotero key)
- reference_normalized_doi (DOI)
- doi_url (`https://doi.org/<DOI>`)
- paper_title when available
- publication_year when available

Rationale:

- manual verification requires immediate access to original publications
- DOI is the canonical external identifier and must be included

Scope:

- benchmark debug matrices
- patch-queue outputs
- per-document regression diagnostics
- modeling-ready merged instance summaries
- any artifact labeled debug, audit, review, or manual

No debug artifact is considered complete if DOI is missing.

---

## LLM Extraction Layer, Deterministic Arbitration Layer, And Audit Boundary

### Why the pipeline is layered
The project separates semantic extraction, deterministic arbitration, and audit
so that:

- semantic interpretation is handled where language context is strongest
  (LLM stage),
- reproducible rule behavior is handled where strict consistency is required
  (deterministic stages),
- human and machine audit can verify both without mixing responsibilities.

This prevents hidden logic drift where semantic decisions are silently embedded
into late-stage scripts.

### Layer 1: LLM extraction responsibilities
The LLM extraction layer is responsible for:

- identifying formulation instances and instance boundaries
- assigning field-role semantics
- distinguishing shared-vs-instance-specific meaning in prose and tables
- emitting structured candidate rows with explicit missingness rather than
  silent omission
- emitting flexible semantic objects rather than relying on deterministic
  semantic reconstruction as the active authority

The LLM extraction layer is not responsible for final arbitration of
conflicting evidence.

Current clarification:

- the LLM's primary responsibility is full-document semantic understanding,
  formulation scope detection, structural signal detection, and governed
  marker-level authorization
- structural signal detection includes governed motifs such as:
  - DOE structure
  - selection signals
  - inheritance signals
  - sequential optimization signals
  - other governed formulation-boundary patterns
- the preferred future contract is for the LLM to emit reusable semantic cues
  and intermediate markers even when execution-level completion is deferred
- the preferred future contract is not to force the LLM to resolve every cue
  into candidate-level or execution-like structure before deterministic
  function units can act
- in complex papers, semantic understanding may be present even when exact
  execution-level grounding is incomplete
- hard suppression of markers under such uncertainty can create downstream
  execution starvation:
  - no marker family
  - no governed expansion
  - no downstream formulation rows
- the maintained runtime now preserves selected partial semantic markers for:
  - `selection_marker`
  - `inheritance_marker`
  while keeping current downstream execution restricted to `execution_ready`
  markers only

Inference from the maintained audit:

- the remaining failure mode is better described as semantic-contract overload
  than as restored architectural impurity
- future contract work should reduce LLM output burden rather than simply
  tightening prompts for more execution-ready certainty

### Layer 2: Deterministic arbitration responsibilities
Deterministic scripts are responsible for:

- formulation relation materialization and grouping provenance
- numeric evidence binding and token-level support checks
- deterministic derivation and unit or ratio normalization
- schema assembly and export formatting
- stable filtering and gating for reproducible benchmarking and release outputs

These responsibilities must remain deterministic to preserve run-to-run
reproducibility and auditability.

Stage2 enforcement note:

- Deterministic semantic emitters, deterministic semantic lifts, and similar
  rule-heavy Stage2 reconstruction paths may exist for fallback, comparator,
  migration-support, or diagnostic work.
- They must not be described or selected as active Stage2 mainline authority.
- Promoting such paths as Stage2 authority is an architecture contract
  violation.

Clarification on deterministic scope:

- deterministic layers should perform execution-level completion only within
  governed semantic scope
- deterministic layers should not synthesize new semantics outside that
  governed scope
- the preferred future contract direction is to move execution strictness
  downstream into governed function units and validators rather than forcing
  the LLM to emit prematurely executable structures

### Layer 3: Audit boundary responsibilities
Audit occurs at the boundary between extracted candidates and publishable
database outputs. The audit boundary must expose:

- evidence pointers and span traceability
- field-level and formulation-level QC outcomes
- explicit conflict and uncertainty artifacts for targeted human review

Current governed interpretation:

- Layer 3 is not only an evaluation helper.
- It is also part of the production-grade audit and governance layer around the
  formulation database.
- The benchmark-valid endpoint remains the Stage 5 final formulation table.
- Reviewer-facing Layer 3 audit outputs remain downstream support surfaces and
  must not mutate benchmark-valid outputs.

Formulation-centered audit direction:

- the preferred reviewer entry object is one formulation row
- human review is split into two linked layers:
  - formulation existence and identity audit
  - value credibility audit
- these layers are not parallel:
  - value credibility depends on structure and identity correctness
  - many apparent value errors are projections of structure or identity errors
- current repo capability is partially present but not yet unified into one
  governed formulation-centered audit system contract

### Stable downstream deterministic rule families
The following rule families are considered stable deterministic core:

- formulation relation materialization
- numeric evidence realignment and token QC gating
- derivation and normalized field computation
- schema assembly and export formatting
- benchmark-facing final-table comparison
