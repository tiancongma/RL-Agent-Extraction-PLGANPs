Purpose:
- Provide long-term selector calibration cases for the S2-2 evidence-selection boundary
- Keep paper archetypes and evidence-priority expectations comparable across representative papers

Not:
- Not GT
- Not runtime input
- Not benchmark artifact
- Not runtime rule configuration
- Not Stage3 semantic closure guidance

Used for:
- selector debugging
- scoring calibration
- failure analysis
- conservative review of keep/drop expectations and minimal evidence sets

Multi-evidence-anchor boundary:
- The selector must not collapse any evidence kind to a single category slot.
- Multiple source-body anchors may be required for LLM semantic discovery,
  including base preparation, variant preparation, surface conjugation or
  functionalization, selected-formulation recap, materials identities,
  optimization rationale, table-adjacent result context, and scheme/table
  pointers.
- Deterministic selection may filter only structural/noise failures and exact
  duplicates. It must not decide whether multiple anchors are semantically the
  same formulation, independent rows, inherited variants, controls, downstream
  descendants, or redundant discussion.
- Calibration cases may specify a category count as a minimum evidence set, not
  as a maximum. Cases such as `V99GKZEI` explicitly require two distinct
  preparation-method blocks.
