# Layer2 Identity Scaffold Contract v1

Status:
- proposed and minimally implemented as a diagnostic-only compatibility feature
- validated first on safe normalization papers only: `WIVUCMYG`, `5ZXYABSU`

Purpose:
- freeze a stable formulation-identity anchor upstream of Layer3 value work
- separate identity resolution from later attribute accumulation and final presentation
- prevent downstream compare and audit workflows from binding on unstable final-row ids

## Why Current Final-Row Binding Is Unstable

Current failure mode:
- the benchmark-facing final table is the correct Layer2-reviewed row object
- but downstream compare/audit work can still bind through unstable surfaces such as:
  - namespaced representative ids like `WIVUCMYG_F23`
  - regenerated final formulation ids
  - coarse full-row similarity fallback

This is unsafe because:
- final presentation ids are materialization outputs, not identity anchors
- namespaced ids can drift even when article-native identity is unchanged
- coarse fallback can explode DOE papers by attaching one GT row to many sibling rows

Observed safe-normalization examples:
- `WIVUCMYG_F23 -> F23`
- `5ZXYABSU_NPB1 -> NPB1`

## Contract Layers

### 1. Identity Scaffold Layer

Authoritative role:
- define the frozen formulation entity boundary for downstream audit and value comparison

Primary source:
- the reviewed/frozen Layer2-style boundary surface when available
- for the current DEV15 diagnostic validation, that source is the repaired-v4 manual workbook identity surface:
  - `seed_pred_representative_source_formulation_id`

Scaffold key v1:
- `l2_identity_scaffold_key_v1 = <paper_key>::native::<normalized_article_native_label>`

Fallback key only when native label is absent:
- `l2_identity_scaffold_key_v1 = <paper_key>::gt::<normalized_gt_formulation_id>`

Hard rule:
- the scaffold key is not the same thing as:
  - `final_formulation_id`
  - namespaced `representative_source_formulation_id`
  - a display label

### 2. Attribute Accumulation Layer

Allowed behavior:
- downstream stages may add fields, provenance, measurements, and enrichment to an existing scaffolded identity node
- downstream stages may preserve additional identity-bearing metadata such as `identity_variables_json`

Not allowed by default:
- silently redefining the scaffolded row boundary
- silently splitting one scaffold row into many rows

### 3. Final Presentation / Materialization Layer

Role:
- produce final benchmark-facing tables, display ids, reviewer exports, and materialized collapse decisions

Constraint:
- presentation/materialization surfaces may describe the current-system row
- they do not replace the scaffold identity authority for downstream comparison

## Identity Source Policy

When a reviewed/frozen Layer2 boundary surface exists:
- it is the identity authority for downstream compare and value-audit binding

Downstream stages may:
- enrich the node
- preserve additional identity-bearing variables
- carry presentation aliases

Downstream stages may not:
- silently split identity unless split is explicitly authorized

## Split Authorization Policy

Field classes:

### Identity-defining fields

Examples:
- explicit article-native formulation labels
- reviewed family/core/variant boundaries
- explicitly authorized identity-bearing variables

These fields may authorize a split only when:
- the split is paper-local
- the distinction is formulation-defining
- the distinction is traceable in the scaffold or an explicitly approved identity extension

### Non-identity fields

Examples:
- helper annotations
- presentation aliases
- audit notes

Default behavior:
- enrich existing identity
- do not split

### Measurement / outcome fields

Examples:
- size
- PDI
- zeta
- EE
- LC

Default behavior:
- enrich existing identity
- do not redefine formulation identity

Uncertainty rule:
- if uncertain, enrich the existing scaffolded identity instead of splitting

## Binding Ladder Policy

### Priority 1. Exact article-native formulation label match

Fields:
- GT / scaffold native label
- current-system `representative_source_formulation_id`

Why safe:
- closest to the paper-reported formulation identity surface

### Priority 2. Normalized namespaced-label match

Examples:
- `WIVUCMYG_F23 -> F23`
- `5ZXYABSU_NPB1 -> NPB1`

Why safe:
- this recovers benign namespacing drift without changing formulation meaning

### Priority 3. Strict identity-equivalent binding

Fields:
- scaffold native label
- normalized article-native label
- exact paper-local formulation-label equivalent surface

Why only conditionally safe:
- useful when the paper preserves a stable native label in a slightly different column
- should stay label-based, not value-similarity-based

### Priority 4. Coarse fallback

Examples:
- matching on shared polymer, drug, EE, or other broad field bundles

Policy:
- allowed for manual review only
- not allowed as benchmark-grade compare binding

## Minimal Insertion Point

The v1 implementation is additive and diagnostic-only:
- a report-only Stage5 support tool builds a scaffold-binding surface from:
  - repaired-v4 GT workbook
  - baseline final table
  - new final table

It does not:
- mutate Stage5 outputs
- replace compare semantics
- rewrite Layer2 authority

## What Stages Are Allowed To Do

Stage2:
- may discover identity-bearing variables
- may preserve raw structure
- may not by itself redefine frozen reviewed identity after the boundary is accepted

Stage3:
- may materialize relations and enrich a candidate row
- may not silently split frozen identity in downstream audit binding

Stage5:
- remains the benchmark-valid final output layer
- may materialize presentation ids and collapse decisions
- does not become the sole identity authority for downstream compare when a reviewed scaffold exists

## Why This Reduces False Row Explosion

Without the scaffold:
- namespaced id drift breaks direct binding
- compare workflows fall into coarse fallback
- DOE papers fan out incorrectly

With the scaffold:
- article-native identity is frozen first
- safe namespaced normalization recovers one-to-one binding where appropriate
- new parameters can enrich an existing row without automatically exploding the compare surface

## Scope And Non-Goals

In scope for v1:
- two-paper validation only: `WIVUCMYG`, `5ZXYABSU`
- additive diagnostic surface
- strict binding ladder

Out of scope for v1:
- broad repair of hard papers such as `UFXX9WXE`, `WFDTQ4VX`, `YGA8VQKU`, `BB3JUVW7`, `5GIF3D8W`, `INMUTV7L`, `L3H2RS2H`
- coarse value-level benchmark binding
- hidden semantic inference in Stage5 materialization
- pipeline-wide redesign
