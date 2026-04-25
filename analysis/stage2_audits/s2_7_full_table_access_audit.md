# S2-7 Full Table Access Audit

## Executive Conclusion

The current maintained design is **partially implemented, but not operationally complete**.

Under the maintained code path, downstream Stage2 completion does **not** receive an explicit S2-2 table-authority handoff. Instead, `S2-7` consumes the `S2-5` semantic JSONL and then tries to **re-open** normalized table authority indirectly by deriving a semantic root from `source_raw_response_path`. In the concrete lineage `data/results/20260421_43ed145`, that derived root points to the `02_s2_4b` run directory, but the normalized table payloads actually live in the accepted pre-LLM source root `data/results/20260421_3579206/09_selector_contract_dev15_prellm/...`.

So the downstream code is written as if it can recover full row-bearing table material, but the maintained handoff does not reliably provide that material. In this lineage, the row-bearing authority is **not actually reachable** at `S2-7`. For general DOE expansion, that makes true downstream row recovery **structurally unavailable in practice**.

## Intended Design vs Implemented Reality

### Intended design from governance/docs

- `S2-2` is the first engineering freeze point and persists:
  - `semantic_stage2_objects/normalized_table_payloads/<paper_key>/normalized_table_payloads_v1.json`
  - `semantic_stage2_objects/evidence_blocks/<paper_key>/evidence_blocks_v1.json`
  Source: [project/2_ARCHITECTURE.md](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/project/2_ARCHITECTURE.md:93)
- `S2-3` / `S2-4a` are summary-only for LLM-facing table evidence.
  Source: [project/2_ARCHITECTURE.md](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/project/2_ARCHITECTURE.md:116)
- The canonical Stage2 path includes LLM semantic discovery plus deterministic post-LLM completion.
  Source: [project/ACTIVE_PIPELINE_FLOW.md](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/project/ACTIVE_PIPELINE_FLOW.md:161)

### Implemented reality

- `S2-4a` is correctly summary-only and builds frozen prompts from S2-2 evidence artifacts, not full tables.
  Source: [run_stage2_s2_4a_prompt_construction_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/run_stage2_s2_4a_prompt_construction_v1.py:186)
- `S2-7` is invoked from a passing `S2-6` report and explicitly consumes only the upstream `S2-5` semantic JSONL path.
  Source: [run_stage2_s2_7_compatibility_projection_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/run_stage2_s2_7_compatibility_projection_v1.py:158) and [run_stage2_s2_7_compatibility_projection_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/run_stage2_s2_7_compatibility_projection_v1.py:241)
- `S2-7` does not take an explicit path to S2-2 `evidence_blocks_v1.json`, S2-2 normalized payload manifests, or Stage1 table manifests as runner inputs.
- The projection code preserves some source-path metadata from the semantic document:
  - `source_text_path`
  - `source_raw_response_path`
  - `source_table_files`
  Source: [build_stage2_compatibility_projection_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py:679)
- DOE expansion and table row expansion are then attempted from those semantic documents.
  Source: [build_stage2_compatibility_projection_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py:1325)

## Code Path Trace

### 1. What inputs are passed into S2-7?

The maintained `S2-7` runner receives:

- passing `S2-6` validation report
- `source_s2_5_run_dir`
- `semantic_jsonl`

It does **not** explicitly receive:

- S2-2 `evidence_blocks_v1.json`
- S2-2 `candidate_blocks_v1.json`
- S2-2 `normalized_table_payloads/.../normalized_table_payloads_v1.json`
- Stage1 table manifest handles

Evidence:
- [run_stage2_s2_7_compatibility_projection_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/run_stage2_s2_7_compatibility_projection_v1.py:158)
- [run_stage2_s2_7_compatibility_projection_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/run_stage2_s2_7_compatibility_projection_v1.py:241)

### 2. Where does DOE or formulation-table expansion get row-bearing material from?

`build_stage2_compatibility_projection_v1.py` calls two downstream units:

- `run_doe_row_expansion_function_unit(...)`
- `run_table_row_expansion(...)`

Those units do **not** expand from prompt summaries alone. They attempt to recover row-bearing authority by reopening normalized table payloads through helpers:

- `doe_row_expansion_function_unit_v1.py::_load_normalized_table_payloads(...)`
- `table_row_expansion_v1.py::_load_normalized_table_payloads(...)`

Both helpers:

1. read `source_raw_response_path` from the semantic document
2. derive a semantic root from its parent of `raw_responses/`
3. look for:
   `normalized_table_payloads/<paper_key>/normalized_table_payloads_v1.json`

Evidence:
- [doe_row_expansion_function_unit_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/function_units/doe_row_expansion_function_unit_v1.py:154)
- [doe_row_expansion_function_unit_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/function_units/doe_row_expansion_function_unit_v1.py:166)
- [table_row_expansion_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/table_row_expansion_v1.py:97)
- [table_row_expansion_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/table_row_expansion_v1.py:109)

Classification:

- DOE expansion row source: **normalized table payload**
- non-DOE table row expansion row source: **normalized table payload**
- Stage1 table CSVs are only a **limited auxiliary handle**, not the main authority source

### 3. Does the code explicitly re-enter S2-2 artifact families?

Yes, but only **implicitly** and only for normalized payloads.

Observed behavior:

- explicit read of normalized payload manifest through derived root:
  - yes
- explicit read of `evidence_blocks_v1.json`:
  - no
- explicit read of `candidate_blocks_v1.json`:
  - no
- explicit read of Stage1 table CSV files:
  - limited helper support through `source_table_files`
- explicit read of Stage1 table manifests / table path lookup:
  - no maintained runner-level handoff found

Evidence:
- normalized payload reopen:
  - [doe_row_expansion_function_unit_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/function_units/doe_row_expansion_function_unit_v1.py:166)
  - [table_row_expansion_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/table_row_expansion_v1.py:109)
- direct Stage1 CSV helper:
  - [table_row_expansion_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/table_row_expansion_v1.py:126)

## Artifact Handoff Trace

### S2-2 and S2-4a

- S2-2 persists normalized table payloads and evidence blocks.
- S2-4a prompt construction freezes summary-only prompt material from evidence blocks.
- The prompt surface does not carry full table rows.

Evidence:
- [project/2_ARCHITECTURE.md](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/project/2_ARCHITECTURE.md:93)
- [project/2_ARCHITECTURE.md](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/project/2_ARCHITECTURE.md:120)
- [run_stage2_s2_4a_prompt_construction_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/run_stage2_s2_4a_prompt_construction_v1.py:187)

### S2-5 semantic document

The semantic document preserves:

- `source_raw_response_path`
- `source_text_path`
- `source_table_files`
- `semantic_scope_declarations`

It does **not** embed the normalized payload manifest itself.

Evidence:
- [extract_semantic_stage2_objects_v2.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py:6949)
- [extract_semantic_stage2_objects_v2.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py:7011)
- [extract_semantic_stage2_objects_v2.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py:7233)

### S2-7 compatibility projection

`S2-7` receives those semantic documents and then tries to derive the normalized payload location indirectly from `source_raw_response_path`.

This means the artifact contract is:

- explicit semantic handoff: yes
- explicit row-bearing table-authority handoff: no
- best-effort side lookup for row-bearing table authority: yes

## Concrete Lineage Trace: `20260421_43ed145`

### Upstream accepted pre-LLM root

The accepted pre-LLM source root for this lineage does contain normalized payload manifests for the target papers:

- `UFXX9WXE`: yes
- `WIVUCMYG`: yes
- `WFDTQ4VX`: yes
- `5GIF3D8W`: yes

Example location:
- `data/results/20260421_3579206/09_selector_contract_dev15_prellm/semantic_stage2_objects/normalized_table_payloads/UFXX9WXE/normalized_table_payloads_v1.json`

### Downstream lineage roots actually searched by S2-7

In the concrete lineage:

- `source_raw_response_path` points into:
  - `data/results/20260421_43ed145/02_s2_4b/raw_responses/...`
- `_resolve_semantic_stage2_root(...)` therefore resolves the semantic root to:
  - `data/results/20260421_43ed145/02_s2_4b`

But there is no normalized payload family there:

- `data/results/20260421_43ed145/02_s2_4b/normalized_table_payloads` does not exist
- `data/results/20260421_43ed145/03_s2_5/semantic_stage2_objects/normalized_table_payloads` does not exist
- `data/results/20260421_43ed145/05_s2_7/semantic_to_widerow_adapter/normalized_table_payloads` does not exist

So the concrete lineage preserves table handles and authorization signals, but **not a reachable normalized authority payload root**.

### Concrete paper behavior

For the primary DOE-sensitive papers:

- `UFXX9WXE`
  - DOE authorized: yes
  - DOE called: no
  - skip: `authorized_target_unresolved`
  - `normalized_payload_used`: `no`
- `WIVUCMYG`
  - same pattern
- `5GIF3D8W`
  - same pattern
- `WFDTQ4VX`
  - no DOE scope authorized in this lineage
  - table row expansion still does not get row-bearing authority

Evidence:
- [compatibility_projection_summary_v1.json](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_43ed145/05_s2_7/semantic_to_widerow_adapter/compatibility_projection_summary_v1.json)

This is consistent with the function-unit summaries:

- DOE: `authorized_target_unresolved`, `normalized_payload_used = no`
- table row expansion: `missing_table_authority_payload` or `blocked_by_doe_boundary`

## Contract Mismatch Assessment

Yes, there is a contract mismatch between the intended design and the maintained implementation.

### What the docs imply

- S2-2 preserves full-table authority.
- S2-4a remains summary-only for the LLM.
- downstream deterministic completion exists after LLM authorization.

### What the implementation actually does

- downstream deterministic completion does **not** receive explicit S2-2 full-table authority as a runner input
- instead, downstream attempts a side lookup from semantic metadata
- in the concrete maintained lineage, that side lookup resolves to the wrong run root for normalized payload access

So the docs imply a lawful downstream completion path from semantic authorization back to full-table authority, but the maintained runner contract does not actually provide that handoff robustly.

## Final Verdict: B

**B. The design is only partially implemented: downstream can access limited material but not enough for general DOE expansion.**

More precisely:

- downstream has semantic authorization signals
- downstream preserves `source_text_path` and `source_table_files`
- downstream code exists to reopen normalized payload authority
- but the maintained `S2-7` contract does not pass the S2-2 normalized payload family explicitly
- and in the concrete lineage, the side lookup fails to reach the real S2-2 payload root

Therefore, general DOE expansion is **not reliably implementable end-to-end under the current maintained path**. In practice, for the audited lineage, it is structurally unavailable because the row-bearing authority cannot actually be reached.

## FACTS

- S2-2 persists normalized table payloads and evidence blocks.  
- S2-4a is summary-only and prompt-facing tables are not raw full tables.  
- S2-7 runner input contract includes S2-6 report plus S2-5 semantic JSONL, not explicit S2-2 payload roots.  
- S2-5 semantic documents preserve `source_raw_response_path`, `source_text_path`, and `source_table_files`.  
- DOE and table expansion units try to load normalized payloads from a semantic root derived from `source_raw_response_path`.  
- In lineage `20260421_43ed145`, `source_raw_response_path` points into `02_s2_4b/raw_responses/...`.  
- `normalized_table_payloads` does not exist under `20260421_43ed145/02_s2_4b`, `03_s2_5`, or `05_s2_7`.  
- The accepted upstream pre-LLM root `20260421_3579206/09_selector_contract_dev15_prellm` does contain normalized payloads for the primary target papers.  
- `compatibility_projection_summary_v1.json` records DOE skip reasons such as `authorized_target_unresolved` and `normalized_payload_used = no`.  

## INFERENCES

- The code was designed with the expectation that downstream completion could reopen row-bearing table authority after LLM authorization.  
- That expectation depends on the derived semantic root containing normalized payloads.  
- Because the maintained lineage does not carry normalized payloads under the derived root, the row-bearing recovery contract is incomplete in practice.  
- DOE expansion failure in this lineage is therefore not just semantic weakness; it is also an artifact-handoff failure.  

## UNCERTAINTIES

- The code preserves `source_table_files`, and some helper logic can inspect Stage1 CSVs directly. It is possible that limited special-case recovery can still occur in some papers without normalized payload manifests.  
- This audit did not test every paper or every historical lineage; it is focused on the maintained path and the concrete lineage `20260421_43ed145`.  
- This audit does not establish whether a different maintained lineage layout could accidentally satisfy the side lookup.  

