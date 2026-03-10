# SYNTHESIS_METHOD_BLOCK Packing Update (2026-03-10)

## Rationale

For formulation-instance extraction, tables are useful for enumerating candidate rows, but synthesis-defining preparation paragraphs are more important for identity, grouping, and parent/variant reasoning. These paragraphs explain which preparation route is shared, which variables are intentionally varied, and which conditions remain fixed.

## New block type

The stage2 packer now promotes selected paragraph-sized body text blocks to:

- `SYNTHESIS_METHOD_BLOCK`

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

## Updated packing priority

The effective packing order is now:

1. `metadata`
2. `synthesis_method`
3. `table`
4. `caption`
5. `paragraph`

This makes the packer schema-first for formulation-instance extraction: preparation-family logic is shown before row-enumeration-heavy tables.

## L3H2RS2H audit result

For `L3H2RS2H`, the earliest explanatory/preparation block moved from packing rank `20` to packing rank `2`. The promoted synthesis-method blocks now appear before the table stack, and the packed evidence looks more suitable for preserving parent/variant reasoning than the previous table-dominant order.

## Fixed 3-paper pilot result

Relative to the previous block-pack run:

- `L3H2RS2H` improved from `28` predicted formulation rows to `20` and recovered parent-linked variant behavior
- `5ZXYABSU` stayed stable at `9`
- `WIVUCMYG` kept `26` formulation rows but lost the previously recovered `candidate_non_formulation` freeze-dry rows

## Readout

The new block type helped on the target PDF-derived paper by reducing table dominance and restoring inheritance-aware behavior. The remaining tradeoff is that non-formulation suppression on `WIVUCMYG` regressed, so the next refinement should target how post-processing paragraphs are retained after the synthesis-method blocks and tables, not a return to naive table-first packing.
