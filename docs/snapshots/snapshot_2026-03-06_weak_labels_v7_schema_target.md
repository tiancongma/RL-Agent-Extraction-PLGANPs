# Snapshot: weak_labels_v7 Schema Target (2026-03-06)

## Why v6 Is Insufficient
v6 captures formulation rows and row-level evidence, but it does not explicitly encode shared-vs-instance scope, field membership confidence, or evidence region type. This leaves downstream stages to reconstruct semantic structure with growing rule logic.

## What v7 Changes Structurally
- adds explicit schema versioning and formulation-level semantic fields,
- upgrades each field from scalar to structured field object,
- introduces semantic typing keys:
  `scope`, `membership_confidence`, `evidence_region_type`, `formulation_role`, `instance_confidence`,
- preserves compatibility path via flattening to TSV for staged adoption.

## Downstream Semantic Repair Areas Targeted
- formulation regrouping/signature reconstruction,
- global baseline inheritance inference,
- condition-instance key reconstruction,
- drug/surfactant membership recovery rules.

## What Remains Deterministic
- numeric evidence binding and token QC,
- derivation and unit-safe normalization,
- schema assembly/export and release gating,
- PLGA-only database filter at database layer.
