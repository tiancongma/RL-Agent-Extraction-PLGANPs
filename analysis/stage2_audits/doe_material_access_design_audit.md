# DOE Material Access Design Audit

## Executive Conclusion

The maintained Stage2 path is best classified as:

- **`C_PARTIAL_REOPEN`**

The repository evidence does **not** support Design A (`explicit handoff`), and it does **not** show a fully stable Design B (`governed authority reopen`) either. The implemented pattern is:

1. `S2-2` preserves row-bearing table authority in normalized payload artifacts.
2. `S2-4a`/`S2-4b` remain summary-only / semantic-facing for the LLM.
3. `S2-5` preserves semantic authorization plus some path handles.
4. `S2-7` then **tries** to reopen upstream row-bearing authority indirectly from `source_raw_response_path`.

That reopen is not an explicit stable downstream contract. It is derived from the current raw-response lineage shape. In the concrete lineage `data/results/20260421_43ed145`, the derived root does **not** match the actual upstream authority root that contains `normalized_table_payloads`. So the design intent looks like authority reopen, but the maintained implementation is only a **partial / broken authority reopen**.

## Intended Design from Docs

The docs imply this high-level architecture:

- `S2-2` is the freeze point for evidence construction and persists:
  - `semantic_stage2_objects/normalized_table_payloads/<paper_key>/normalized_table_payloads_v1.json`
  - `semantic_stage2_objects/evidence_blocks/<paper_key>/evidence_blocks_v1.json`
  Source: [project/2_ARCHITECTURE.md](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/project/2_ARCHITECTURE.md:93)
- `S2-3` / `S2-4a` are summary-only for LLM-facing tables; full tables must not enter the prompt surface.
  Source: [project/2_ARCHITECTURE.md](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/project/2_ARCHITECTURE.md:116)
- Stage2 includes:
  - LLM semantic discovery
  - deterministic post-LLM completion into the downstream-ready Stage2 artifact
  Source: [project/ACTIVE_PIPELINE_FLOW.md](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/project/ACTIVE_PIPELINE_FLOW.md:161)
- README says:
  - deterministic DOE row expansion is preserved, but in normal `llm_first_composite` mode it is lawful only within LLM-declared DOE scope
  - full tables are preserved in `S2-2` authority artifacts for deterministic execution, but not allowed back into the LLM prompt
  Source: [README.md](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/README.md:174)

### Documentation-level interpretation

The docs most naturally imply **Design B**:

- `S2-2` remains the unique row-bearing authority
- downstream LLM-facing surfaces remain summary-only
- deterministic downstream completion should be able to act after LLM authorization without making the prompt surface row-bearing

The docs do **not** imply Design A strongly. They do not say `S2-7` receives normalized payloads explicitly as a declared runner input.

## Implemented Design from Code

### 1. What `S2-7` explicitly accepts

The maintained `S2-7` runner consumes:

- passing `S2-6` validation report
- `source_s2_5_run_dir`
- `semantic_jsonl`

It explicitly says it consumes the `S2-6`-validated `S2-5` semantic-intermediate surface only.

Evidence:
- [run_stage2_s2_7_compatibility_projection_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/run_stage2_s2_7_compatibility_projection_v1.py:139)
- [run_stage2_s2_7_compatibility_projection_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/run_stage2_s2_7_compatibility_projection_v1.py:158)
- [run_stage2_s2_7_compatibility_projection_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/run_stage2_s2_7_compatibility_projection_v1.py:241)

It does **not** explicitly accept:

- `normalized_table_payloads`
- `evidence_blocks_v1.json`
- `candidate_blocks_v1.json`
- Stage1 table manifests

So Design A is **not implemented** in the maintained `S2-7` input contract.

### 2. What `S2-7` actually does

`build_stage2_compatibility_projection_v1.py` preserves:

- `source_text_path`
- `source_raw_response_path`
- `source_table_files`

from the semantic document, then calls:

- `run_doe_row_expansion_function_unit(...)`
- `run_table_row_expansion(...)`

Evidence:
- [build_stage2_compatibility_projection_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py:679)
- [build_stage2_compatibility_projection_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py:1325)

### 3. Where the function units get row-bearing material

Both DOE and non-DOE table expansion units try to reopen normalized payload authority:

- DOE:
  - `_resolve_semantic_stage2_root(document)` reads `source_raw_response_path`
  - `_load_normalized_table_payloads(document)` loads:
    `normalized_table_payloads/<paper_key>/normalized_table_payloads_v1.json`
  - DOE authorization then requires a matching normalized payload with `normalized_csv_path`
  Evidence:
  - [doe_row_expansion_function_unit_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/function_units/doe_row_expansion_function_unit_v1.py:154)
  - [doe_row_expansion_function_unit_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/function_units/doe_row_expansion_function_unit_v1.py:166)
  - [doe_row_expansion_function_unit_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/function_units/doe_row_expansion_function_unit_v1.py:204)
- non-DOE table expansion:
  - same pattern for `_resolve_semantic_stage2_root(...)`
  - same pattern for `_load_normalized_table_payloads(...)`
  - row assignments come from `authority_row_entries(authority_payload)`
  Evidence:
  - [table_row_expansion_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/table_row_expansion_v1.py:97)
  - [table_row_expansion_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/table_row_expansion_v1.py:109)
  - [table_row_expansion_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/table_row_expansion_v1.py:728)

Stage1 table CSVs are also preserved in `source_table_files`, but they are only a limited helper surface. The core authority reopen path for both function units is still `normalized_table_payloads`.

### 4. Is the reopen explicit and stable?

No.

The reopen mechanism is:

- derive a semantic root from `source_raw_response_path`
- if the raw response is under `raw_responses/`, use `parent.parent`
- then expect `normalized_table_payloads/...` under that derived root

This is not an explicit upstream authority pointer. It is a **derived path rule**.

Evidence:
- [doe_row_expansion_function_unit_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/function_units/doe_row_expansion_function_unit_v1.py:154)
- [table_row_expansion_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/table_row_expansion_v1.py:97)

That makes the reopen lineage-sensitive and fragile.

## Concrete Lineage Trace: `20260421_43ed145`

### Where the real row-bearing authority lives

The accepted pre-LLM root used by this lineage contains row-bearing normalized payloads:

- `data/results/20260421_3579206/09_selector_contract_dev15_prellm/semantic_stage2_objects/normalized_table_payloads/UFXX9WXE/normalized_table_payloads_v1.json`
- likewise for `WIVUCMYG`, `WFDTQ4VX`, and `5GIF3D8W`

So upstream S2-2 authority is real and present.

### What root `S2-7` actually derives

In `03_s2_5/semantic_stage2_objects/semantic_stage2_v2_objects.jsonl`, the target documents carry:

- `source_raw_response_path = data/results/20260421_43ed145/02_s2_4b/raw_responses/<paper>__stage2_v2_raw_response.json`
- `source_text_path = ...`
- `source_table_files = [...]`

The function-unit reopen logic therefore derives:

- semantic root = `data/results/20260421_43ed145/02_s2_4b`

But there is no `normalized_table_payloads` family there.

Observed:

- `data/results/20260421_43ed145/02_s2_4b/normalized_table_payloads` does not exist
- `data/results/20260421_43ed145/03_s2_5/semantic_stage2_objects/normalized_table_payloads` does not exist
- `data/results/20260421_43ed145/05_s2_7/semantic_to_widerow_adapter/normalized_table_payloads` does not exist

So the derived root and the real authority root do **not** match.

### Is the mismatch incidental or structural?

This audit supports “structural”, not “incidental”.

Reason:

- the maintained reopen mechanism is path-derived from `source_raw_response_path`
- the maintained `S2-7` input contract does not explicitly pass the true upstream authority root
- the lineage only works if the derived raw-response root happens to co-locate normalized payloads

That is a fragile coincidence requirement, not a stable contract.

### Concrete DOE behavior

For DOE-sensitive papers like `UFXX9WXE`, `WIVUCMYG`, and `5GIF3D8W`, the semantic documents preserve DOE authorization:

- `scope_kind = doe_table_row_enumeration_scope`
- `authorizes_row_materialization_modes = [llm_semantic_discovery, deterministic_row_expansion_within_llm_scope]`

But `compatibility_projection_summary_v1.json` shows:

- DOE `authorized = true`
- DOE `called = false`
- `skip_reason = authorized_target_unresolved`
- `normalized_payload_used = no`

This is exactly the signature of a broken reopen path rather than a pure semantic absence.

## Classification Rationale

### Why not `A_IMPLEMENTED`

Because `S2-7` does not accept explicit row-bearing normalized/full-table payload input.

### Why not `B_IMPLEMENTED`

Because the reopen is not a stable governed pointer to upstream authority. It is reconstructed indirectly from `source_raw_response_path`, and in the audited lineage it resolves to the wrong root.

### Why `C_PARTIAL_REOPEN`

Because all of the following are true:

- the design intent is clearly reopen, not explicit handoff
- the code explicitly tries to reopen upstream normalized authority
- the reopen relies on derived path logic instead of an explicit authority contract
- in the concrete maintained lineage, that derived root does not match the real authority root
- as a result, general DOE expansion is not reliably available in practice

## Does the Evidence Support the Prior Interpretation?

**Yes.**

The evidence supports the prior interpretation that:

- the intended design is authority reopen rather than explicit handoff
- the implemented design is only a partial / broken authority reopen

That prior interpretation is consistent with both the maintained code path and the concrete lineage behavior.

## FACTS

- `S2-2` persists `normalized_table_payloads` and `evidence_blocks`.
- `S2-4a` is summary-only for table evidence.
- `S2-7` explicitly consumes `S2-6` + `S2-5 semantic_jsonl`, not explicit normalized payload input.
- `build_stage2_compatibility_projection_v1.py` preserves `source_raw_response_path`, `source_text_path`, and `source_table_files`.
- DOE and table expansion units both attempt to load `normalized_table_payloads` by deriving a semantic root from `source_raw_response_path`.
- In lineage `20260421_43ed145`, `source_raw_response_path` points into `02_s2_4b/raw_responses/...`.
- The true normalized payloads for the target papers live instead under `20260421_3579206/09_selector_contract_dev15_prellm/...`.
- `compatibility_projection_summary_v1.json` records DOE authorization but unresolved execution targets and `normalized_payload_used = no`.

## INFERENCES

- The maintained implementation was aiming for an authority-reopen model.
- That model is not contract-complete because it depends on reconstructing the authority root from the raw-response lineage shape.
- Because the reopen root is not explicit, the design is lineage-fragile and not reliable for general DOE expansion.

## UNCERTAINTIES

- Some special-case downstream recovery may still be possible through `source_table_files` or text-path helpers for specific papers.
- This audit does not claim every historical lineage fails in the same way; it claims the maintained design, as currently implemented and audited here, is not a stable full authority-reopen contract.

