# Snapshot — Polymer MW Canonical Migration + Materials-Priority Packing

## Status

- Layer 2 complete
- Relation-first mainline active
- Stage 2 updated (packing + canonical field)

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
