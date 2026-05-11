# SYNTHESIS_METHOD_BLOCK Packing Update (2026-03-10)

## Rationale

For formulation-instance extraction, tables are useful for enumerating candidate rows, but synthesis-defining preparation paragraphs are more important for identity, grouping, and parent/variant reasoning. These paragraphs explain which preparation route is shared, which variables are intentionally varied, and which conditions remain fixed.

## New block types

The stage2 packer now promotes selected paragraph-sized body text blocks to:

- `SYNTHESIS_METHOD_BLOCK`
- `MATERIALS_PROCUREMENT_BLOCK`

This is a lightweight classifier over existing paragraph blocks. It looks for preparation-family language such as:

- `prepared`
- `preparation`
- `same procedure`
- `fixed amounts`
- `varying`
- `while maintaining`
- `this led to`
- polymer / surfactant / oil-volume / concentration change language

It avoids promoting obvious reference-heavy and conclusion-heavy paragraphs.

`MATERIALS_PROCUREMENT_BLOCK` is a narrower procurement-aware class. It targets
Materials-style blocks that explicitly carry polymer/material defaults such as:

- polymer identity
- molecular weight
- polymer grade or supplier-coded product names
- procurement language such as `purchased`, `procured`, `gifted`, or
  `analytical grade`

## Updated packing priority

The effective packing order is now:

1. `metadata`
2. `synthesis_method`
3. `materials_procurement`
4. `table`
5. `caption`
6. `paragraph`

This keeps the packer schema-first for formulation-instance extraction:
preparation-family logic is shown first, then shared/default procurement blocks,
before row-enumeration-heavy tables.

## L3H2RS2H audit result

For `L3H2RS2H`, the earliest explanatory/preparation block moved from packing rank `20` to packing rank `2`. The promoted synthesis-method blocks now appear before the table stack, and the packed evidence looks more suitable for preserving parent/variant reasoning than the previous table-dominant order.

## Fixed 3-paper pilot result

Relative to the previous block-pack run:

- `L3H2RS2H` improved from `28` predicted formulation rows to `20` and recovered parent-linked variant behavior
- `5ZXYABSU` stayed stable at `9`
- `WIVUCMYG` kept `26` formulation rows but lost the previously recovered `candidate_non_formulation` freeze-dry rows

## Readout

The new block type helped on the target PDF-derived paper by reducing table dominance and restoring inheritance-aware behavior. The remaining tradeoff is that non-formulation suppression on `WIVUCMYG` regressed, so the next refinement should target how post-processing paragraphs are retained after the synthesis-method blocks and tables, not a return to naive table-first packing.

## Materials Procurement Block (Post-Update)

`MATERIALS_PROCUREMENT_BLOCK` is now a distinct packing class from
`SYNTHESIS_METHOD_BLOCK`.

Definition

- `SYNTHESIS_METHOD_BLOCK` remains the highest-priority textual block because it
  carries preparation-route and variation logic.
- `MATERIALS_PROCUREMENT_BLOCK` is narrower and targets Materials-style blocks
  that explicitly describe shared/global material defaults rather than the
  preparation procedure itself.

Trigger pattern

- procurement-style language such as `purchased`, `procured`, `gifted`,
  `obtained from`, `supplied by`, `used as received`, or `analytical grade`
- plus polymer-identity or polymer-default signals
- plus molecular-weight / grade / supplier-coded product signals

Why it is separate

- These blocks often define shared defaults such as polymer identity,
  polymer molecular weight, polymer grade, and supplier-coded material names.
- They are important for extraction, but they are not themselves the
  preparation-logic block that defines formulation routing.

Why it is placed before tables

- Shared procurement/default parameters should reach the model before generic
  table or caption content.
- This improves early visibility of paper-level shared formulation parameters
  without changing the extraction schema or any Stage 5 behavior.
