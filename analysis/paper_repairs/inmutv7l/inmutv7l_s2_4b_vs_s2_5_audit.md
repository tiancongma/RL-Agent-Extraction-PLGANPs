**Executive Conclusion**
For `INMUTV7L`, the row-level collapse happens at `S2-4b`, not at `S2-5`. The frozen raw LLM response already represents the main formulation table as one family-level candidate, `Table1_Formulation_Family`, rather than twelve row-level formulations. `S2-5` then preserves and normalizes that already-collapsed structure. Confidence: high.

**S2-4b Raw Response Inspection**
- Raw response artifact:
  [INMUTV7L__stage2_v2_raw_response.json](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_43ed145/02_s2_4b/raw_responses/INMUTV7L__stage2_v2_raw_response.json)
- The raw response is already in the minimal Stage2 semantic contract with only these top-level keys:
  - `paper_key`
  - `table_scopes`
  - `semantic_signals`
  - `formulation_candidates`
- Row-level presence classification: `NONE`
- Evidence:
  - `table_scopes` correctly recognizes the main table:
    - `{"table_id": "Table 15", "scope_kind": "formulation_table", "is_formulation_bearing": true, "confidence": "high"}`
  - But `formulation_candidates` contains only four high-level candidates:
    - `Method_DXI_PLGA_NP`
    - `Table1_Formulation_Family`
    - `Table2_Sterilized_Variants`
    - `Table3_PK_Characterized`
  - The key table-derived candidate is explicitly family-level:
    - `candidate_id: "Table1_Formulation_Family"`
    - `candidate_kind: "formulation_family"`
    - `source_table_id: "Table 15"`
    - `label_hint: "Formulations from Table 1"`
    - `core_change_hint: "Variations in drug, polymer, loading, and surfactant type/concentration"`
- What is missing from the raw response:
  - no numbered entries `1` through `12`
  - no per-row formulation objects
  - no array of row-level candidates
  - no partial enumeration of row IDs

Representative raw-response fragment:

```json
{
  "candidate_id": "Table1_Formulation_Family",
  "candidate_kind": "formulation_family",
  "source_table_id": "Table 15",
  "label_hint": "Formulations from Table 1",
  "instance_role": "synthesis_core",
  "core_change_hint": "Variations in drug, polymer, loading, and surfactant type/concentration",
  "status": "reported",
  "confidence": "high"
}
```

**S2-5 Semantic Object Inspection**
- Semantic artifact:
  [semantic_stage2_v2_objects.jsonl](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_c8f4b61/01_s2_5/semantic_stage2_objects/semantic_stage2_v2_objects.jsonl)
- `S2-5` row-level presence classification: `NONE`
- The replayed semantic document for `INMUTV7L` still contains:
  - `table_scopes` with `Table 15` as `formulation_table`
  - the same four `formulation_candidates`
  - the same family-level `Table1_Formulation_Family`
- There are no row-level candidate objects in the semantic document.
- Evidence of preservation rather than new collapse:
  - `source_raw_response_path` points back to the frozen raw response:
    - `data/results/20260421_43ed145/02_s2_4b/raw_responses/INMUTV7L__stage2_v2_raw_response.json`
  - `source_raw_response_schema` is `stage2_live_v2_raw_response_minimal_contract`
  - `source_mode` is `saved_raw_live_v2_replay_to_stage2_v2`

Representative semantic-object fragment:

```json
{
  "candidate_id": "Table1_Formulation_Family",
  "candidate_kind": "formulation_family",
  "source_table_id": "Table 15",
  "label_hint": "Formulations from Table 1",
  "instance_role": "synthesis_core",
  "status": "reported",
  "confidence": "high"
}
```

**Compare S2-4b vs S2-5**
This is case `A`:
- The LLM never provided row-level information.
- `S2-5` did not collapse twelve explicit rows into a family.
- Instead, `S2-5` preserved an already-family-level raw response.

So the appropriate classification is:
- `S2_4B_LIMITED`

**Collapse Mechanism**
The decisive mechanism is in the `S2-4a` task contract and the replay parser behavior.

1. The `S2-4a` prompt explicitly instructs the LLM to stay at understanding level and avoid materialized rows.
   Evidence from the frozen prompt:
   [s2_4a_prompts_v1.jsonl](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_43ed145/01_s2_4a/analysis/s2_4a_prompts_v1.jsonl)
   - `formulation_candidates should describe likely formulation units only, using concise hints rather than materialized rows`
   - `Do not perform relation resolution, inheritance closure, or final-row materialization`

2. The `S2-5` replay path does not add row splitting for the minimal contract.
   Code:
   [extract_semantic_stage2_objects_v2.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py:7087)
   - `build_live_v2_document(...)` detects the minimal raw-response contract and passes through:
     - `normalize_shrunken_table_scopes(parsed.get("table_scopes"))`
     - `normalize_shrunken_semantic_signals(parsed.get("semantic_signals"))`
     - `normalize_shrunken_formulation_candidates(parsed.get("formulation_candidates"))`
   Code:
   [extract_semantic_stage2_objects_v2.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py:340)
   - `normalize_shrunken_formulation_candidates(...)` only normalizes existing candidate fields
   - it does not create per-row candidates
   - it does not split a family candidate into numbered row instances

So the exact mechanism is:
- the prompt contract asked for high-level formulation units instead of materialized rows
- the raw response complied with that contract
- the `S2-5` replay parser preserved that high-level shape

**Final Classification**
`S2_4B_LIMITED`

**FACTS**
- The frozen raw response already contains only four high-level formulation candidates.
- The main formulation table is represented as `Table1_Formulation_Family`, not as twelve row-level formulations.
- The replayed `S2-5` semantic document matches that high-level shape.
- The `S2-5` replay code path normalizes minimal-contract candidates but does not generate row-level candidates from a family candidate.
- The `S2-4a` prompt explicitly asks for concise understanding-level candidates rather than materialized rows.

**INFERENCES**
- The under-enumeration for `INMUTV7L` is primarily a limitation of the LLM-facing semantic contract at `S2-4b`, not a downstream semantic parser collapse.
- The weak `Table 1` summary likely makes it even harder for the LLM to emit row-level structure, but the immediate boundary of collapse is still the raw response itself.

**UNCERTAINTIES**
- This audit does not determine whether a different prompt contract, or a stronger summary surface for `Table 1`, would have caused the same LLM to emit full or partial row-level candidates.
- This audit also does not address the later `S2-7` row-emission blockage; it only isolates the `S2-4b` vs `S2-5` collapse boundary.
