**Executive Conclusion**
`INMUTV7L` is not failing because the paper is complex. It is a simple paper with one strong preparation paragraph and one clear numbered formulation table. The dominant first blocker is `SEMANTIC_UNDERENUMERATION_AT_S2_5`: the replay semantic document preserves the right table scope, but collapses the 12 numbered Table 1 formulations into one family-level candidate instead of row-like formulation instances. A later `S2-7` table-emission stop (`missing_table_authority_payload` for `Table 15`) is real, but it happens after the semantic collapse and is not the first dominant blocker.

**Upstream Sufficiency Check**
- Upstream text contains one strong method paragraph at `data/cleaned/content/text/INMUTV7L.pdf.txt#paragraph:1#segment:0`:
  `PLGA nanoparticles (NPs) containing DXI were prepared by using the solvent displacement method...`
- Upstream table asset [INMUTV7L__table_15__pdf_table.csv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/cleaned/goren_2025/tables/INMUTV7L/INMUTV7L__table_15__pdf_table.csv) is a clear numbered formulation table with rows `1` through `12`, polymer blocks, surfactant choice, size, PDI, zeta potential, and EE.
- Upstream table asset `INMUTV7L__table_06__pdf_table.csv` is a downstream sterilization table, not the main formulation authority.
- Conclusion: yes, one sufficient method paragraph plus one sufficient formulation-bearing table exist upstream.

**S2-2a Check**
- Candidate artifact:
  [candidate_blocks_v1.json](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_3579206/09_selector_contract_dev15_prellm/semantic_stage2_objects/candidate_blocks/INMUTV7L/candidate_blocks_v1.json)
- Key method candidate survives as:
  - `INMUTV7L__candidate_paragraph__05`
  - `block_type=synthesis_method`
  - origin `data/cleaned/content/text/INMUTV7L.pdf.txt#paragraph:1#segment:0`
- Key table candidates survive as:
  - `INMUTV7L__candidate_table__02`
    - origin `.../INMUTV7L__table_15__pdf_table.csv`
    - `table_role_hint=formulation`
    - `selector_readiness_label=ready`
    - `representation_status=repaired_summary`
  - `INMUTV7L__candidate_table__01`
    - origin `.../INMUTV7L__table_06__pdf_table.csv`
    - also survives as a formulation-tagged table summary
- Conclusion: not lost at `S2-2a`.

**S2-2b / Evidence Preservation Check**
- Evidence artifact:
  [evidence_blocks_v1.json](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_3579206/09_selector_contract_dev15_prellm/semantic_stage2_objects/evidence_blocks/INMUTV7L/evidence_blocks_v1.json)
- Key preserved blocks:
  - `INMUTV7L__method__01`
    - `selection_reason=selected_method_evidence`
  - `INMUTV7L__table__01`
    - `table_id=Table 1`
    - candidate `INMUTV7L__candidate_table__02`
    - origin `.../INMUTV7L__table_15__pdf_table.csv`
  - `INMUTV7L__table__02`
    - `table_id=Table 2`
    - candidate `INMUTV7L__candidate_table__01`
    - origin `.../INMUTV7L__table_06__pdf_table.csv`
- Normalized payload artifact also preserves both:
  [normalized_table_payloads_v1.json](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_3579206/09_selector_contract_dev15_prellm/semantic_stage2_objects/normalized_table_payloads/INMUTV7L/normalized_table_payloads_v1.json)
  - `Table 1` -> `INMUTV7L__table_15__pdf_table__normalized.csv`
  - `Table 2` -> `INMUTV7L__table_06__pdf_table__normalized.csv`
- Conclusion: not lost at `S2-2b`.

**S2-4a Prompt Sufficiency Check**
- Prompt audit row:
  [s2_4a_prompt_audit_v1.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_43ed145/01_s2_4a/analysis/s2_4a_prompt_audit_v1.tsv)
  - `status=pass`
  - `ordered_block_order=metadata > method > table > table > table > table`
  - `all_selected_blocks_included=yes`
  - `truncation_detected=no`
- Frozen prompt:
  [s2_4a_prompts_v1.jsonl](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_43ed145/01_s2_4a/analysis/s2_4a_prompts_v1.jsonl)
- The prompt clearly contains:
  - the full preparation paragraph
  - a summary for `Table 1`
  - a summary for `Table 2`
- Important caveat:
  - the `Table 1` summary is lossy and structurally weak
  - it renders `table_id: Table 15`, `key_columns: 0 || 1`, `row_identifier_pattern: mixed identifiers`
  - sample rows are poor (`triblocks were used.`, `PLGA-5% | ®`, `12 | Lutrol`)
- Conclusion: the prompt is materially sufficient to recognize the paper as a formulation study with a formulation-bearing table, but the summary is lossy. The first major collapse is still later at semantics rather than prompt omission.

**S2-5 Semantic Check**
- Semantic artifact:
  [semantic_stage2_v2_objects.jsonl](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_c8f4b61/01_s2_5/semantic_stage2_objects/semantic_stage2_v2_objects.jsonl)
- Correct high-level recognition survives:
  - `Table 15` -> `scope_kind=formulation_table`, `is_formulation_bearing=true`, `confidence=high`
  - `Table 6` -> `scope_kind=sequential_child`, `is_formulation_bearing=false`
  - `primary_preparation_method_hint=solvent displacement method`
  - variable names include `PLGA concentration`, `Dexibuprofen concentration`, `Surfactant type`, `Surfactant concentration`, `pH`
- But row-bearing semantics do not survive:
  - only four `formulation_candidates` are emitted
  - only one table-derived synthesis candidate is emitted for the core table:
    - `Table1_Formulation_Family`
    - `candidate_kind=formulation_family`
    - label `Formulations from Table 1`
  - there are no 12 numbered row-level candidates for Table 1
- Stage2 completed artifact from `S2-7` still shows the same four upstream semantic instances, not a row-materialized set:
  [weak_labels__v7pilot_r3_fixparse.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_c8f4b61/03_s2_7/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv)
  - `Method_DXI_PLGA_NP`
  - `Table1_Formulation_Family`
  - `Table2_Sterilized_Variants`
  - `Table3_PK_Characterized`
- Conclusion: this is the first dominant collapse. The semantic layer recognizes the right table but under-enumerates it into one family-level object.

**S2-7 Emission Check**
- Execution ledger:
  [execution_ledger_v2.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_c8f4b61/03_s2_7/analysis/execution_ledger_v2.tsv)
- Relevant rows:
  - `INMUTV7L  table_row_expansion_v1  Table 15 ... was_unit_authorized=yes ... was_unit_called=no ... rows_emitted=0 ... skip_reason=missing_table_authority_payload`
  - `Table 6`, `Table 8`, `Table 9` are unauthorized with `skip_reason=not_formulation_table`
- Important nuance:
  - the replay sidecar does contain a resolved locator for `INMUTV7L` `Table 1`
  - the upstream normalized payload file also exists for `INMUTV7L__table_15__pdf_table__normalized.csv`
  - so `S2-7` shows a real projection-time reopen/binding failure even though the authority surface exists
- Conclusion: `S2-7` is additionally blocked, but this comes after the earlier `S2-5` family-level under-enumeration.

**Stage5 Check**
- Final table:
  [final_formulation_table_v1.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_c8f4b61/05_stage5/final_formulation_table_v1.tsv)
  - `INMUTV7L` final rows: `3`
  - retained formulations:
    - `Method_DXI_PLGA_NP`
    - `Table1_Formulation_Family`
    - `Table2_Sterilized_Variants`
- Decision trace:
  [final_output_decision_trace_v1.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_c8f4b61/05_stage5/final_output_decision_trace_v1.tsv)
  - three rows are `kept`
  - one row (`Table3_PK_Characterized`) is `filtered_non_formulation`
- Compare output:
  [final_table_vs_gt_counts.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_c8f4b61/06_compare/final_table_vs_gt_counts.tsv)
  - `final_table_count=3`
  - `gt_count=12`
  - `count_diff=-9`
  - `comparison_status=under`
- Conclusion: later filtering is minor. Stage5 mostly preserves the already-collapsed family-level representation inherited from Stage2.

**Final Classification**
`SEMANTIC_UNDERENUMERATION_AT_S2_5`

**One-Sentence Bottom Line**
`INMUTV7L` is still missing nine rows because the pipeline correctly preserves its key method and numbered formulation table upstream, but `S2-5` collapses that 12-row table into a single family-level semantic object, and later `S2-7` row emission never recovers the lost row structure.

**FACTS**
- The paper has one strong method paragraph and a clear 12-row numbered formulation table upstream.
- The key method and key table both survive `S2-2a`, `S2-2b`, and normalized payload preservation.
- The frozen `S2-4a` prompt includes both the method evidence and the table summaries.
- `S2-5` marks `Table 15` as a high-confidence formulation-bearing table.
- `S2-5` emits only one family-level candidate for `Table 15`, not 12 row-level candidates.
- `S2-7` records `skip_reason=missing_table_authority_payload` for `Table 15`.
- Stage5 retains three family/method rows and filters only one characterization row.

**INFERENCES**
- The first dominant loss is semantic under-enumeration, not upstream table loss.
- The weak summary structure for `Table 1` likely contributes to the LLM collapsing the table into a family-level object.
- The later `S2-7` stop is a secondary practical blocker that prevents recovery after the semantic collapse.

**UNCERTAINTIES**
- The exact internal cause of the `S2-7` `missing_table_authority_payload` condition for `Table 15` remains unresolved here, because the upstream normalized payload and replay sidecar locator both exist.
- This audit does not determine whether a repaired `S2-7` reopen path alone would fully recover all 12 rows without first improving `S2-5` row-level semantic binding.
