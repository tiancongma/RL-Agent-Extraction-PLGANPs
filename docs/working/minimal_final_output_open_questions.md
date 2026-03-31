# Minimal Final-Output Layer Open Questions

Only unresolved design issues are listed here.

## 1. Exact core-parameter signature fields

The repository still needs a fixed minimal signature for collapse decisions. Open points include how to weigh polymer identity, LA:GA ratio, drug feed amount, polymer amount, stabilizer concentration, solvent, and phase-ratio-like fields when some are missing or inherited.

## 2. Baseline and optimized provenance handling

The repository has not yet fixed a single rule for when baseline/optimized labels remain distinct final formulations versus when they are provenance labels attached to the same final formulation identity.

## 3. Parent/variant collapse policy

`parent_instance_id`, `instance_kind`, and `change_role` already exist at the candidate layer. The final-output layer still needs an explicit rule for when a variant remains a distinct final formulation and when it should collapse into an existing retained formulation.

## 4. Benchmark table schema versus modeling table schema

The repository needs a final decision on whether the benchmark-valid final formulation table and the later modeling table are the same artifact or two closely related exports with different column surfaces.

## 5. Placement of the final benchmark comparison script

The repository has not yet decided whether the future benchmark-valid comparison step should live alongside current Stage4 evaluators, in a new downstream benchmark family, or as a very small adapter around the minimal final-output layer outputs.

## 6. Narrow DoE handling in the minimal layer

The repository has a documented DoE reconciliation rule for `WFDTQ4VX`, but it is still open whether the minimal final-output layer should initially support only generic collapse plus explicit paper-local exceptions, or whether a small generalized coordinate-aware rule belongs in phase 1.
