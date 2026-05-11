**Executive Conclusion**
Implemented a bounded general rule for simple formulation-table deterministic enumeration after semantic authorization. The repaired path activates only for low-ambiguity non-DOE `full_formulation` tables with preserved normalized payload authority and stable first-column row identity. Validation succeeded on the anchor and guards, so the rule was landed into governance docs, the decision log, memory, and the repair index.

**Why This Is A Bounded General Rule, Not A Paper-Specific Hack**
- The rule does not branch on `paper_key`.
- It does not require LLM row-level output.
- It activates only when all of the following hold:
  - the table is already LLM-authorized as formulation-bearing
  - the table is not on the DOE path
  - the preserved authority payload is available from `S2-2 normalized_table_payloads`
  - the scope is a low-ambiguity `full_formulation` table
  - the first-column row identity surface is stable enough to instantiate rows directly
- It explicitly does not take over DOE matrices, non-DOE sweep-family recovery, or cross-table decode cases.

**Contract Definition**
- Rule family:
  `simple formulation-table deterministic enumeration after semantic authorization`
- Runtime contract:
  - semantic authorization remains LLM-owned
  - deterministic row enumeration remains execution-owned
  - the executor may instantiate row-level formulation instances directly from preserved normalized table payloads when the simple-table contract holds
- Current bounded activation signals:
  - `simple_table_enumeration_attempted`
  - `simple_table_enumeration_activated`
  - `simple_table_rows_emitted`
  - `simple_table_block_reason`
  - `row_identity_surface_used`

**Code Files Changed**
- [src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py)
  - extended sidecar locator merge to recognize table-number aliases from preserved asset references
  - this lets semantic scopes such as `Table 15` rebind to the preserved authority locator when the locator surface still carries the paper-native `Table 1`
- [src/stage2_sampling_labels/table_row_expansion_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/table_row_expansion_v1.py)
  - added the bounded simple-table contract and audit fields
  - allowed direct deterministic row enumeration before the old variable-role gate when the simple-table contract holds
  - preserved the prior direct-row path for non-simple cases so existing recovered papers do not regress

**Bounded Validation**
- Maintained runner used:
  - [run_stage2_s2_7_compatibility_projection_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/run_stage2_s2_7_compatibility_projection_v1.py)
- Source replay boundary:
  - `data/results/20260421_c8f4b61/02_s2_6`
- Validation run:
  - [20260421_f4b9d62/01_s2_7_simple_table_enumeration_validation_v2](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_f4b9d62/01_s2_7_simple_table_enumeration_validation_v2)
- Key validation artifact:
  - [compatibility_projection_summary_v1.json](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_f4b9d62/01_s2_7_simple_table_enumeration_validation_v2/semantic_to_widerow_adapter/compatibility_projection_summary_v1.json)

Anchor:
- `INMUTV7L`
  - before: `4` Stage2 rows, no deterministic base-row expansion
  - after: `16` Stage2 rows
  - new deterministic rows emitted: `12`
  - activation:
    - `simple_table_enumeration_attempted=yes`
    - `simple_table_enumeration_activated=yes`
    - `simple_table_rows_emitted=12`
    - `row_identity_surface_used=numeric_first_column`
  - result:
    - the preserved numbered `Table 1` now materializes row-level formulation instances without requiring LLM row objects

DOE guard:
- `WIVUCMYG`
  - before: `29` Stage2 rows
  - after: `29` Stage2 rows
  - DOE path remains primary
  - simple-table activation stays off for the relevant scopes

Non-DOE complex guard:
- `5GIF3D8W`
  - before: `12` Stage2 rows
  - after: `12` Stage2 rows
  - existing non-DOE single-variable recovery remains active
  - simple-table activation stays off with `simple_table_block_reason=doe_scope_table`

Stability guard:
- `UFXX9WXE`
  - before: `28` Stage2 rows
  - after: `28` Stage2 rows
  - existing deterministic recovery remains intact
  - no regression was introduced while adding the bounded simple-table path

**What Was Landed**
Landed into governance/docs:
- [project/2_ARCHITECTURE.md](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/project/2_ARCHITECTURE.md)
- [project/ACTIVE_PIPELINE_FLOW.md](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/project/ACTIVE_PIPELINE_FLOW.md)
- [project/ACTIVE_PIPELINE_RUNBOOK.md](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/project/ACTIVE_PIPELINE_RUNBOOK.md)
- [project/4_DECISIONS_LOG.md](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/project/4_DECISIONS_LOG.md)
- [README.md](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/README.md)

Landed into repair index:
- [docs/repair_index/success_pattern_index_v1.md](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/docs/repair_index/success_pattern_index_v1.md)
- [docs/repair_index/success_pattern_index_v1.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/docs/repair_index/success_pattern_index_v1.tsv)

Memory updates were also added through the governed updater:
- one decision row for the bounded simple-table deterministic enumeration rule
- one error/failure-family row for simple formulation-table family collapse anchored by `INMUTV7L`
- memory IDs:
  - `MDEC098`
  - `MERR1135`

**Residual Limitations**
- This rule only covers low-ambiguity simple tables with stable first-column row identity.
- It does not synthesize rich field assignment for every extracted row yet; the current win is row preservation and lawful deterministic instantiation.
- It does not solve DOE execution, non-DOE sweep recovery, or cross-table coded/decode cases.
- It does not change Stage3 or Stage5 behavior.

**What This Rule Does NOT Cover**
- not DOE row enumeration
- not non-DOE single-variable sweep recovery
- not prompt redesign
- not LLM row-level object generation
- not generic “enumerate every formulation table”
- not cross-table semantic linking or decode logic
