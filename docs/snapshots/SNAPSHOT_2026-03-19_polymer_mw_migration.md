# Snapshot — Polymer MW Canonical Migration + Materials-Priority Packing

## Status

- Layer 2 complete
- Relation-first mainline active
- Stage 2 updated (packing + canonical field) in the pre-authority-migration transition-era legacy Stage2 path

## Authority Clarification

- This snapshot records a transition-era Stage2 field/canonical update and does not define the current active Stage2 authority.
- The field and packing changes described here occurred before the later semantic Stage2 authority migration.
- Current active Stage2 authority is the semantic emitter plus deterministic compatibility adapter, while `auto_extract_weak_labels_v7pilot_r3_fixparse.py` is legacy fallback/debug infrastructure only.

## Key Changes

- `polymer_mw_kDa` is now the canonical molecular-weight field
- `plga_mw_kDa` is retained as a legacy read alias only
- `materials_procurement` was introduced as an explicit Stage 2 evidence-pack block type

## What did NOT change

- Stage 5 semantics
- relation-first logic
- boundary decisions

## Implication

- LLM input changed -> fresh LLM regression required

## One-line summary

We corrected polymer MW field semantics and improved Materials visibility in LLM input, without altering the relation-first architecture.
