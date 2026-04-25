# Ordinal Grid Semantics v1 Notes

Goal: replace INMUTV7L-style row-number semantics recovery with a generalized ordinal-grid mechanism, not another paper-key branch.

Current remaining paper-local branches in compare:
- 5ZXYABSU, 5GIF3D8W, YGA8VQKU, INMUTV7L

## Observed INMUTV7L failure mode
- Current system rows preserve only numeric row identity surfaces (`raw_formulation_label`, `representative_source_formulation_id`, `formulation_id` like DOE_Row_7).
- GT row labels encode richer semantics (`7 PLGA 10% PVA`).
- Compare currently restores:
  - polymer_name
  - polymer_grade
  - surfactant_name
  - polymer_mass_mg
  - shared method_type
  - shared solvent_name
- The current implementation uses a direct `INMUTV7L_ROW_LABEL_MAP` and paper-local branching.

## Candidate generalized mechanism
Name: `ordinal_grid_semantics_v1`

### Problem class
A paper has a bounded row grid where row ordinal deterministically maps to one value from each of several small axes.

### Example axis structure validated by INMUTV7L
- axis A (polymer family / grade): changes every block of 3 rows
- axis B (surfactant): cycles every row within each block of 3
- shared constants: method_type, solvent_name, polymer_mass_mg

### Proposed generic schema shape
- eligibility_fn(row)
- ordinal_source_fields
- row_count_bound
- axis_definitions:
  - name
  - values
  - cycle_length or block_size
  - target_fields
- shared_constant_fields

### Guardrails required
- Only activate when ordinal source is explicit and unique.
- Only for small bounded grids.
- Only when the row semantics are fully determined by ordinal position inside the grid.
- No cross-row inference from GT labels.
- No firing on arbitrary numeric labels without an explicit schema match.

### Implementation caution
A paper-key-free implementation still needs a bounded schema registry. The key is:
- schema selection must be by table/grid class signals, not by hand-coded field returns in the override function.
- avoid immediate implementation until schema eligibility signals are explicit enough to avoid false positives.

## Next coding step
Refactor current INMUTV7L direct map into:
1. generic ordinal extraction helper
2. generic ordinal-grid schema container
3. one schema instance for the validated grid class
4. route through generalized engine before remaining paper-local override
5. delete direct INMUTV7L branch only after behavior is preserved
