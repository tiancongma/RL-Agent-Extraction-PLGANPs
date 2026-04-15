# AGENTS.md

Agent execution contract for this repository.

This file defines how AI coding agents (Codex, Claude Code, etc.)
must initialize context and behave when modifying this repository.

This repository enforces a **strict governance structure**.
Agents must read this file before making any modification.

---

# 1. Mandatory startup context

Before making any change, the agent MUST read the following files:

project/0_PROJECT_CHARTER.md  
project/1_REQUIREMENTS.md  
project/2_ARCHITECTURE.md  
project/PIPELINE_SCRIPT_MAP.md  
project/ACTIVE_PIPELINE_FLOW.md  
project/ACTIVE_PIPELINE_RUNBOOK.md  
project/FILE_NAMING_AND_VERSIONING.md  

These files define:

- project goals
- system architecture
- active pipeline
- execution runbook
- repository conventions

Agents must **not infer architecture without reading them first**.

Before running any benchmark, alignment, comparison, workbook-generation, or
audit workflow that consumes `data/results/` artifacts, the agent must also
read:

project/ACTIVE_DATA_SOURCE_CONTRACT.md

Agents must resolve the authoritative source explicitly before execution.
They must not assume latest by directory name, timestamp, parent fallback, glob
matching, or modification time.
If a workflow source is ambiguous, the agent must inspect the active data
source contract and print the resolved run directory plus exact source file
paths before execution.
For execution-facing benchmark, alignment, comparison, workbook-generation, and
audit workflows, agents must select only maintained entrypoint scripts listed
in `project/ACTIVE_PIPELINE_RUNBOOK.md` and `docs/maintained_script_surface.tsv`.
They must not choose scripts by name similarity, recency, or convenience, and
they must not auto-select legacy, deprecated, wrapper-only, or diagnostic
scripts unless the user explicitly requests them.

When the repo-local MCP server `repo-mcp` is available, Codex must use it
before making repository-governed execution decisions:

- call `resolve_active_run_source` before results-based benchmark, alignment,
  comparison, workbook, or audit work
- call `check_script_selection_legality` before choosing an execution-facing
  script
- call `validate_new_file_path` before creating a new file
- call `get_authoritative_gt` before GT-consuming work
- call `check_lawful_resume_boundary` before reusing an intermediate artifact
  as a downstream resume boundary

---

# 2. Governance layer rule

`project/` is the **governance layer**.

It contains only authoritative project definitions.

Agents MUST NOT:

- create new files in `project/`
- duplicate governance documents
- introduce alternative specifications

unless the user explicitly instructs it.

Maximum governance files allowed: **9**

---

# 3. Repository structure

The repository is organized into four layers:


project/ → governance documents
src/ → executable pipeline code
data/ → datasets and run artifacts
docs/ → explanations, audits, historical material


Agents must keep these layers strictly separated.

Never mix:

- documentation with pipeline code
- historical methods with active stage scripts
- governance definitions with experiment notes

---

# 4. Source code identity rules

Every script in `src/` must fall into one of these categories:

ACTIVE_ENTRYPOINT  
→ main pipeline scripts referenced in `ACTIVE_PIPELINE_FLOW.md`

STABLE_TOOL  
→ reusable utilities with clear input/output contracts

ARCHIVED_METHOD  
→ historical or comparative methods stored under:


archive/code/
archive/delete_candidates_pending_confirmation/


Agents must **not treat archived methods as active pipeline components**.

Archived methods exist for:

- historical comparison
- methodology reference
- reproducibility

They must not be moved back into stage directories.

Every script in `src/` must also have an explicit recorded registry entry with:

- path
- status
- architecture layer
- function
- primary inputs
- primary outputs
- upstream dependencies
- downstream consumers
- current pipeline role

Agents must not leave a script in `src/` as an implicit or mysterious asset.
If the script cannot be classified clearly enough to record that metadata with confidence, it is a governance failure and must be flagged for archive/delete disposition rather than silently retained.

---

# 4a. Run reproducibility rule

Every `data/results/run_*` directory must contain a reproducibility-grade run specification.

That specification must record:

- run purpose
- run type
- exact starting inputs
- exact script execution order
- exact script paths
- intermediate artifacts
- final outputs
- benchmark-valid or diagnostic-only status

Agents must not create or leave undocumented run directories.
If a run lacks this specification, treat it as non-compliant.

---

# 5. Pipeline authority

The active pipeline definition lives in:


project/ACTIVE_PIPELINE_FLOW.md


The execution instructions live in:


project/ACTIVE_PIPELINE_RUNBOOK.md


Agents must **not invent alternative pipelines or entrypoints**.

New pipeline logic requires explicit user approval.

---

# 5a. Benchmark reporting rule

Agents must not present partial-layer outputs as final benchmark evidence.

Hard rule:
- Formal GT comparison results may be reported only from the complete intended pipeline final-output layer.
- Stage2 candidate rows, candidate graphs, partial extraction outputs, and other intermediate artifacts may be compared to GT only for debugging or failure localization.
- If the complete intended pipeline has not been executed, the agent must label the result as `diagnostic-only, not benchmark-valid final output`.
- Agents must not repeatedly run one isolated layer and frame direct GT comparison as the final system result.

Before reporting benchmark performance, the agent must verify whether all intended downstream guardrail, normalization, filtering, and final-output layers for that workflow have been executed.

For Layer3 workbook generation and review surfaces:
- the latest Stage5 final table and audit-ready export are the canonical source of truth for current-system formulation presence and identity resolution
- historical alignment scaffolds, prior workbook bridge rows, and trusted annotation carry-forward files are advisory only
- advisory artifacts may help map GT rows to system rows, but they must not downgrade a canonically present row to `missing_in_system`

---

# 6. File creation rules

Default behavior:

**append to existing files**

Creating new files requires explicit user instruction.

Agents must not create:

- new governance documents
- duplicate specifications
- snapshot-style notes
- alternative runbooks
- temporary debug documentation

---

# 7. Archive rules

Historical material must be stored in one of these locations:


docs/archive_project/
archive/code/
archive/delete_candidates_pending_confirmation/


Agents must not place historical material in:


project/
src/
src/stage*


---

# 8. Code safety rule

Agents must not modify runtime behavior unless:

- explicitly requested by the user
- fixing broken references
- aligning code with governance files

Structural refactors must preserve pipeline behavior.

---

# 9. When uncertain

If repository structure or authority is unclear:

1. read `PIPELINE_SCRIPT_MAP.md`
2. read `ACTIVE_PIPELINE_RUNBOOK.md`
3. read `ACTIVE_PIPELINE_FLOW.md`
4. ask the user

Agents must **never guess pipeline structure**.

---

# 9a. Supporting memory layer

Governed long-term memory is a **supporting layer**, not an active pipeline
stage.

Rules:

- memory artifacts must stay under `data/mem/v1/`
- memory files must remain flat row-based TSV or schema-manifest assets
- for complex debugging, regression investigation, run comparison, pipeline
  modification, GT mismatch analysis, or lineage tracing, agents should
  identify the task type and query memory before reading local source files or
  acting
- preferred bootstrap pattern:
  - `python src/utils/mem_bootstrap_v1.py --query "..."`
  - or `python src/utils/query_mem_v1.py --query "..."`
- rebuild memory when governed source documents or `RUN_CONTEXT.md` artifacts
  change materially
- use targeted memory updates only for small manual corrections or additions;
  do not create alternative memory trees

---

# 10. Design intent reminder

Current architecture direction:

- LLM performs **semantic formulation identification**
- rules perform **validation, audit, and evidence checking**

Older rule-heavy reconstruction paths and multimodel comparison pipelines
are considered **historical methods** unless explicitly reactivated.
