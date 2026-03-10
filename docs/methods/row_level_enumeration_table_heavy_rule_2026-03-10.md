# Row-Level Enumeration Rule for Table-Heavy Sweep Papers (2026-03-10)

## Rule

For table-heavy or sweep-style formulation studies, the extractor must enumerate formulation candidates row by row before any abstraction.

Operationally:

- table rows or experimental runs are treated as candidate formulation instances first
- parent/variant relationships are assigned only after row-level enumeration is complete
- grouped family summaries must not replace row-level formulation instances
- standalone global/shared pseudo-formulations must not be emitted

## Trigger

The stage2 pilot extractor now prepends a stricter enumeration block when the assembled paper text shows one or more table-heavy sweep signals, such as:

- multiple formulation tables
- repeated formulation labels like `F1`, `F2`, `Run1`, `Sample2`
- repeated parameter rows with similar structure
- design/sweep language indicating row-wise experimental runs

## Motivation

This rule was validated on `10.1016/j.ejpb.2004.09.002` (`L3H2RS2H`), where the regression was caused by premature abstraction of table rows into family-level summaries. The prompt-side row-enumeration constraint improved formulation-instance recall without changing the ontology or adding DOE decoding.

## Fixed 3-paper pilot check

Before vs after formulation-row counts on the fixed `v7pilot3` subset:

- `5ZXYABSU` (`10.2147/ijn.s130908`): `9 -> 9` against GT `9`
- `L3H2RS2H` (`10.1016/j.ejpb.2004.09.002`): `8 -> 24` against GT `22`
- `WIVUCMYG` (`10.1002/jps.24101`): `11 -> 26` against GT `26`

The rule improved under-enumeration on the two table-heavy papers. The remaining tradeoff is that `L3H2RS2H` now slightly over-enumerates in the 3-paper run, so the next refinement should focus on duplicate row suppression after row-level enumeration, not on reverting to family-level summarization.

## Principle

For formulation extraction:

`table rows -> formulation candidates -> parent/variant abstraction`

Never collapse table rows into family summaries before enumeration.
