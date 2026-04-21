# Minimal Authority Reopen Contract Design

## Executive Conclusion

The smallest governance-ready fix is a **minimal explicit authority reopen contract**:

- keep `S2-2 normalized_table_payloads` as the unique downstream row-bearing source of truth
- keep `S2-3` / `S2-4a` summary-only
- keep `S2-5` as semantic authorization plus downstream execution cues
- but add an **explicit stable authority handle** so `S2-7` can reopen the correct upstream authority root without deriving it from `source_raw_response_path`

Recommended design:

- **Option B: explicit authority reopen contract with stable handles to earlier surfaces**

This is the best fit with current docs and decisions. It is smaller than explicit payload handoff, more stable than path-derived reopen, and keeps semantic authority with the LLM while allowing downstream deterministic execution to reopen earlier row-bearing material when `S2-5` alone is insufficient.

## Facts Already Established

- The docs most naturally imply that:
  - `S2-2` preserves execution-facing full-table authority
  - `S2-4a` remains summary-only
  - downstream deterministic execution should operate on preserved table authority rather than prompt summaries
- The current implementation is `C_PARTIAL_REOPEN`:
  - `S2-7` consumes `S2-5 semantic_jsonl`, not explicit normalized payload input
  - downstream expansion tries to reopen row-bearing authority from `source_raw_response_path`
  - that mechanism is lineage-fragile and fails in `data/results/20260421_43ed145`
- The decision log already states the intended invariant:
  - semantic target -> stable `table_id` -> preserved S2-2 full-table authority
  - Stage1 table assets are not the downstream execution source of truth once S2-2 authority exists

Key governance anchors:

- [project/2_ARCHITECTURE.md](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/project/2_ARCHITECTURE.md:93)
- [project/ACTIVE_PIPELINE_RUNBOOK.md](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/project/ACTIVE_PIPELINE_RUNBOOK.md:430)
- [project/4_DECISIONS_LOG.md](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/project/4_DECISIONS_LOG.md:3626)

## Design Requirements

The contract must satisfy all of the following:

1. `S2-5` semantic output is **not** the sole information source.
2. Downstream may legally reopen earlier row-bearing authority surfaces after `S2-5` semantic authorization.
3. Reopen must be:
   - explicit
   - governed
   - auditable
   - reproducible
   - not path-guessed
4. The contract must stay minimal.
5. The summary-only prompt contract must remain unchanged.
6. The design must not re-promote deterministic semantic authority.
7. The design must preserve the current invariant:
   - LLM owns semantic discovery
   - deterministic downstream units execute only within LLM-authorized scope

## Candidate Earlier Authority Surfaces

### 1. `S2-2 normalized_table_payloads`

Status:
- **recommended primary reopen source**

Why:
- already defined in governance as the execution-facing full-table authority surface
- already self-describing
- already the intended downstream execution source of truth
- aligns with MDEC102

Use:
- primary row-bearing source for DOE and non-DOE downstream expansion

### 2. `S2-2 evidence_blocks`

Status:
- **supporting only, not row-bearing**

Why:
- semantic-facing, lossy, prompt-oriented
- useful for provenance and linkage checks
- not suitable as the row-bearing execution source

Use:
- optional audit/provenance cross-check
- not a reopen execution source

### 3. `S2-2 candidate_blocks`

Status:
- **not recommended as downstream reopen source**

Why:
- pre-selector candidate surface
- useful for diagnostics, not for authoritative row materialization
- would blur the clean S2-2a / S2-2b / S2-7 contract

Use:
- diagnostics only

### 4. Stage1 extracted table assets

Status:
- **not recommended as normal downstream reopen source**

Why:
- current governance already says Stage1 table assets are fallback inside `S2-2a`, not the downstream source of truth once S2-2 authority exists
- allowing direct downstream Stage1 reopening as a normal path would expand the authority surface and increase drift risk

Use:
- not part of the minimal normal reopen contract
- if retained at all, only as an explicitly coded exceptional fallback with explicit audit flags, not as the standard path

### 5. Other earlier surfaces

- `analysis/table_authority_validation_v1.tsv`
  - observability only
  - not an execution source

## Minimal Explicit Handle Proposal

The minimal contract should introduce **one document-level authority root handle** plus **one scope-level table locator bundle**.

### Document-level handle

Required:

- `authority_run_dir`
  - the exact upstream Stage2 run directory that owns the preserved S2-2 authority surface
- `authority_payload_family`
  - fixed value such as `normalized_table_payloads_v1`
- `authority_payload_root`
  - explicit root path:
    `.../semantic_stage2_objects/normalized_table_payloads`

Why this is enough:
- it removes lineage guessing
- it keeps the contract explicit and auditable
- it avoids passing whole payload files downstream

### Scope-level locator bundle

Required for each LLM-authorized table scope:

- `paper_key`
- `table_id`
- `source_table_asset_id`
- `source_table_reference`

Recommended representation:

- `authority_table_locator = {paper_key, table_id, source_table_asset_id, source_table_reference}`

Why:
- `table_id` alone may be insufficient when lineages or repaired tables drift
- `source_table_asset_id` and `source_table_reference` make resolution stable and falsifiable
- no separate opaque `payload_id` is required if these fields already identify the same authority record

### Unnecessary for the minimal contract

Not required as the reopen contract:

- raw `source_raw_response_path`
- prompt path
- `evidence_blocks` path
- `candidate_blocks` path
- direct full payload embedding in `S2-5`

These may remain for provenance, but they should not be the authority locator.

## Reopen Priority Ladder

Recommended governed reopen order:

1. **Primary**
   - `S2-2 normalized_table_payloads`
   - resolve via:
     - `authority_run_dir`
     - `authority_payload_root`
     - `authority_table_locator`

2. **Secondary disambiguation within the same S2-2 authority family**
   - if `table_id` alone matches multiple rows, disambiguate by:
     - `source_table_asset_id`
     - then `source_table_reference`

3. **Fail loudly**
   - if no authority row resolves inside the declared S2-2 authority root

### Not in the minimal normal ladder

- `evidence_blocks`
  - not row-bearing
- `candidate_blocks`
  - diagnostic only
- Stage1 table assets
  - not a normal downstream execution source under the minimal contract

This is intentional. The minimal contract should stabilize the intended S2-2 reopen path first instead of widening downstream execution to multiple row-bearing sources.

## Failure Contract

The reopen contract should expose explicit, typed failure modes.

Minimum required failure labels:

- `authorized_target_unresolved`
  - semantic scope exists, but no authority row could be resolved
- `authority_root_missing`
  - declared upstream authority root does not exist
- `payload_locator_missing`
  - required table locator fields are missing
- `multiple_candidate_payloads`
  - more than one authority record matches and cannot be disambiguated
- `semantic_authorized_but_row_bearing_source_unavailable`
  - semantic authorization succeeded, but no usable authority payload was available
- `lineage_root_mismatch`
  - declared authority root and currently derived/runtime lineage root disagree
- `stale_handle_into_wrong_lineage`
  - handle points to an older or unrelated lineage than the one declared for this completion chain

Recommended additional failure labels:

- `authority_record_present_but_payload_csv_missing`
- `authority_record_present_but_execution_fields_incomplete`
- `table_locator_conflict`

## Audit / Provenance Contract

Minimum required audit fields on every reopen attempt:

- `reopen_source_type`
  - `normalized_table_payloads`
- `reopen_authority_run_dir`
- `reopen_authority_root`
- `reopen_payload_locator`
  - serialized stable locator bundle
- `reopen_resolution_status`
  - `resolved`, `unresolved`, `ambiguous`, `failed`
- `reopen_failure_reason`
- `semantic_scope_ref`
- `table_scope_ref`
- `normalized_payload_used`
  - yes/no
- `stage1_table_asset_used`
  - yes/no

Recommended additional fields:

- `reopen_resolution_notes`
- `resolved_table_id`
- `resolved_source_table_asset_id`
- `resolved_source_table_reference`
- `resolved_normalized_payload_path`

This is enough to make reopen behavior falsifiable in run artifacts without turning the contract into a full data-duplication handoff.

## Option A vs B vs C Comparison

### Option A

Definition:
- explicit `normalized_table_payloads` handoff into `S2-7` input

Architectural fit:
- acceptable, but heavier than needed

Compatibility with summary-only prompt design:
- compatible

Reproducibility:
- very strong

Auditability:
- very strong

Risk of semantic/deterministic drift:
- low

Engineering complexity:
- medium to high

Suitability when `S2-5` is incomplete:
- strong

Downside:
- broadens the `S2-7` runner contract more than necessary
- pushes toward explicit data handoff everywhere instead of minimal reopen

### Option B

Definition:
- explicit authority reopen contract with stable handles to earlier surfaces

Architectural fit:
- **best fit**

Compatibility with summary-only prompt design:
- **best fit**

Reproducibility:
- strong if the authority root and locator are explicit

Auditability:
- strong

Risk of semantic/deterministic drift:
- low, because the execution source of truth remains S2-2

Engineering complexity:
- low to medium

Suitability when `S2-5` is incomplete:
- strong, because downstream can reopen the needed row-bearing source without embedding it in the semantic payload

Downside:
- requires disciplined explicit handle propagation

### Option C

Definition:
- hybrid design with semantic authorization plus explicit fallback locator set that may reopen normalized payloads or Stage1 tables

Architectural fit:
- mixed

Compatibility with summary-only prompt design:
- compatible

Reproducibility:
- moderate to strong

Auditability:
- moderate to strong, but more complicated

Risk of semantic/deterministic drift:
- higher than B because it widens the downstream row-bearing authority set

Engineering complexity:
- medium to high

Suitability when `S2-5` is incomplete:
- very strong

Downside:
- too wide for the minimal contract
- risks reintroducing Stage1 as a practical downstream authority source
- makes execution behavior harder to reason about

## Final Recommendation

Recommended design:

- **Option B**

### Exact recommendation

Adopt a minimal explicit authority reopen contract where:

1. `S2-2 normalized_table_payloads` remains the sole normal downstream row-bearing source of truth.
2. `S2-5` / `S2-7` must carry:
   - `authority_run_dir`
   - `authority_payload_root`
   - a stable `authority_table_locator`
3. downstream deterministic expansion may reopen earlier authority surfaces only through that explicit contract
4. downstream must fail explicitly when the declared authority handle cannot resolve
5. Stage1 table assets remain outside the minimal normal reopen path

### Why this is the right minimal contract

- It respects the already-established architecture.
- It preserves summary-only prompt design.
- It does not require stuffing full row-bearing material into `S2-5`.
- It avoids path-derived reopen.
- It keeps reopen explicit, governed, auditable, and reproducible.
- It is the smallest design that solves the established failure mode.

## FACTS

- `S2-2 normalized_table_payloads` is already the maintained execution-facing table authority surface.
- `S2-4a` table evidence is required to remain summary-only.
- Current downstream reopen is derived from `source_raw_response_path` and is lineage-fragile.
- Decision-log language already says deterministic execution should operate on preserved S2-2 authority by stable table identity.

## INFERENCES

- The repository does not need a broad pipeline redesign.
- The main missing piece is an explicit authority pointer contract, not more semantic content inside `S2-5`.
- Stabilizing reopen against S2-2 authority is enough to satisfy the user requirement that downstream may reopen earlier authority when `S2-5` alone is insufficient.

## UNCERTAINTIES

- Whether any limited Stage1 fallback should exist at all in the future remains a separate policy question.
- Some edge cases may require an ambiguity-resolution rule beyond `table_id`, `source_table_asset_id`, and `source_table_reference`.

## NOT RECOMMENDED

- Not recommended:
  - broad explicit full-payload handoff everywhere
  - continuing path-derived reopen from `source_raw_response_path`
  - using `evidence_blocks` or `candidate_blocks` as row-bearing downstream execution sources
  - treating Stage1 table assets as the normal downstream execution source once S2-2 authority exists

