# Evidence Alignment Hardening Plan

## Scope
This document captures the hardening implemented for evidence alignment quality and formulation-level confidence scoring on branch `feature/ee-coverage-rl`.

Target run used for output generation:
- `run_20260219_1623_780eb83_goren18_weaklabels_v1`

## Where The Logic Lives
Primary implementation files:
- `src/stage5_benchmark/run_evidence_realign_v1.py`
- `src/stage5_benchmark/run_evidence_token_qc_v1.py`

Relevant output artifacts consumed/produced:
- `data/results/<run_id>/evidence_realign_log.tsv`
- `data/results/<run_id>/qc_field_evidence_gate_summary__realigned.tsv`
- `data/results/<run_id>/benchmark_goren_2025/derivation_v1/evidence_token_qc_checks__realigned.tsv`
- `data/results/<run_id>/risk_flags__shared_spans.tsv`
- `data/results/<run_id>/confidence_tiers__formulation_level.tsv`

## Hardening Changes Implemented

### 1) Evidence Quality Filters
Implemented in `run_evidence_token_qc_v1.py`.

- Header/footer boilerplate rejection:
  - For each source document, repeated long lines are detected as boilerplate (`repeat >= 3`, minimum line length `20`).
  - If evidence span contains a boilerplate line, the field is failed (`fail_boilerplate=1`).

- Numeric anchoring (numeric fields):
  - Evidence span must support the extracted numeric value through exact/normalized token match or tight numeric tolerance (`rel_tol=1e-3`, `abs_tol=1e-6`).
  - Comma/decimal formatting variants are accepted.
  - Failure is recorded as `fail_numeric_token=1`.

- Section-based deprioritization:
  - `abstract/introduction/background` spans are treated as low-trust.
  - Such spans are invalid unless numeric anchoring passes.
  - Failure recorded as `fail_deprioritized_section=1`.

### 2) Field-Level Anchor IDs
Implemented in `run_evidence_token_qc_v1.py`.

- Anchor derivation:
  - Per field, uses realigned span start when available (`evidence_realign_log.tsv`), otherwise original span start.
  - Computes `anchor_id` as `doc_key + paragraph_id + paragraph_hash` (fallback to start/hash-based ID).

### 3) Shared-Span Risk Flags
Implemented in `run_evidence_token_qc_v1.py`.

- Detects reuse of the same `anchor_id` across multiple formulation rows for the same doc + field.
- Emits run-level TSV:
  - `data/results/<run_id>/risk_flags__shared_spans.tsv`
- Includes reuse cardinality and impacted `group_key` list for triage.

### 4) Formulation-Level Confidence Scoring
Implemented in `run_evidence_token_qc_v1.py`.

For each formulation record (`key::formulation_id`):
- `n_core_fields_supported`
- `n_core_fields_sharing_top_anchor`
- `anchor_cohesion_score`
- `shared_span_risk_fields_count`
- `confidence_tier` (`A/B/C`)

Tiering currently favors conservative filtering:
- `A`: strong support and high anchor cohesion with low shared-span risk
- `B`: moderate support/cohesion
- `C`: weak/fragmented support or elevated risk

## Outputs Generated
For `run_20260219_1623_780eb83_goren18_weaklabels_v1`:
- `data/results/run_20260219_1623_780eb83_goren18_weaklabels_v1/risk_flags__shared_spans.tsv`
- `data/results/run_20260219_1623_780eb83_goren18_weaklabels_v1/confidence_tiers__formulation_level.tsv`

Additional regenerated supporting outputs:
- `data/results/run_20260219_1623_780eb83_goren18_weaklabels_v1/qc_field_evidence_gate_summary__realigned.tsv`
- `data/results/run_20260219_1623_780eb83_goren18_weaklabels_v1/benchmark_goren_2025/derivation_v1/evidence_token_qc_checks__realigned.tsv`

## Why This Improves Modeling Readiness
- Reduces false confidence from repeated boilerplate and generic sections.
- Enforces deterministic numeric support for numeric fields.
- Surfaces cross-row span reuse risk that often indicates templated or misaligned evidence.
- Produces a reproducible formulation-level confidence layer for downstream model filtering without manual full-audit.
