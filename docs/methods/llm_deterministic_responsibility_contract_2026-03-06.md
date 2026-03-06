# LLM vs Deterministic Responsibility Contract (2026-03-06)

## Current Problem Statement
The extraction pipeline has matured into a layered system, but semantic and deterministic responsibilities were not explicitly documented as a formal contract. As a result, some downstream post-processing rules have expanded into semantic repair behavior that should eventually be moved upstream into extraction schema design.

This note defines a stable responsibility split for current engineering and future redesign.

## Current Pipeline Shape
The current implementation behaves as a three-layer architecture:

1. LLM extraction layer (Stage 2 and related extraction prompts)
- Produces structured candidate formulation rows.
- Emits row-level evidence context.

2. Deterministic arbitration layer (Stage 4/5 post-processing, derivation, schema/export)
- Performs reproducible normalization, alignment, derivation, and gating.
- Produces stable benchmark and database artifacts.

3. Audit boundary layer (evaluation packs, conflict queues, evidence/QC reports)
- Exposes traceability and risk for human review and release confidence.

## Responsibility Split

### LLM extraction is responsible for
- semantic membership and formulation instance boundaries,
- field-role interpretation from prose/table context,
- shared-vs-instance-specific meaning representation,
- explicit missingness in structured output.

### Deterministic layers are responsible for
- numeric evidence binding and token-level checks,
- deterministic normalization and derivation,
- schema assembly and database export,
- reproducible QC filters and confidence tiers.

### Audit boundary is responsible for
- explicit evidence trace and pointer stability,
- conflict visibility and triage artifacts,
- formulation-level and field-level risk signaling.

## Findings from Pipeline Responsibility Audit
The audit confirms that many deterministic scripts are correctly performing reproducible work, while a subset of downstream rule families currently performs semantic reconstruction that should be reduced over time.

Observed stable deterministic core:
- evidence realignment and evidence token QC,
- derivation and projection,
- schema table construction and export,
- deterministic multi-model consensus/conflict reporting.

Observed semantic-compensation hotspots:
- formulation regrouping/signature reconstruction,
- global baseline inheritance into instance rows,
- drug/surfactant normalization used to recover semantic membership,
- condition-instance key reconstruction and inherited-base donor logic,
- core signature assignment logic compensating for missing upstream structure.

## Top Downstream Rule Areas to Reduce via Future Upstream LLM Schema Redesign
1. Formulation grouping and regrouping rules.
2. Global baseline inheritance of shared conditions.
3. Drug/surfactant semantic normalization for alignment recovery.
4. Condition-instance boundary reconstruction and donor inference.
5. Core signature assignment logic that infers missing semantic structure.

## Immediate Engineering Implications
- Keep release-critical deterministic rules active for current runs.
- Track semantic-repair-heavy downstream rules as redesign backlog, not as unbounded rule growth.
- Update extraction schema/prompt contracts to carry stronger instance semantics and shared-vs-instance structure.
- Preserve audit artifacts so migration impact can be measured objectively.

## What Remains Intentionally Deterministic
- Numeric evidence anchoring, token checks, and evidence mismatch gating.
- Unit-safe derivation and canonical value computation.
- Stable schema builds (formulation_core/measurements/factors) and full database export.
- PLGA-only publication filter at database/export layer.
- Run-level reproducibility controls and explicit trace manifests.
