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

Before running any alignment, comparison, workbook-generation, audit, or other
workflow that consumes `data/results/` artifacts, the agent must also read:

project/ACTIVE_DATA_SOURCE_CONTRACT.md

Agents must resolve the authoritative source explicitly before execution.
They must not assume latest by directory name, timestamp, parent fallback, glob
matching, or modification time.
If a workflow source is ambiguous, the agent must inspect the active data
source contract and print the resolved run directory plus exact source file
paths before execution.
For execution-facing workflows, agents must select only maintained entrypoint
scripts listed in `project/ACTIVE_PIPELINE_RUNBOOK.md` and
`docs/maintained_script_surface.tsv`.
They must not choose scripts by name similarity, recency, or convenience, and
they must not auto-select legacy, deprecated, wrapper-only, or diagnostic
scripts unless the user explicitly requests them.

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
- declared run status and intended downstream use

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
  modification, mismatch analysis, or lineage tracing, agents should
  identify the task type and query memory before reading local source files or
  acting
- preferred bootstrap pattern:
  - `python src/utils/mem_bootstrap_v1.py --query "..."`
  - or `python src/utils/query_mem_v1.py --query "..."`
- rebuild memory when governed source documents or `RUN_CONTEXT.md` artifacts
  change materially
- use targeted memory updates only for small manual corrections or additions;
  do not create alternative memory trees

### Repair-Pattern-Guided Regression Workflow (MANDATORY)

Trigger on baseline regression, capability restoration, missing formulations or
missing rows, post-baseline audit, "historical success not in mainline", or
single-paper failure analysis.

Step 1 - Repair Index Lookup
- Read `docs/repair_index/success_pattern_index_v1.tsv`
- Identify matching pattern rows
- Do not assume mainline capability unless `activation_evidence_strength =
  explicit_governed_activation`

Step 2 - Memory Query
- Query `data/mem/v1/`
- Retrieve prior runs, lineage, decisions, and error traces
- Do not skip memory

Step 3 - Baseline + Run Evidence Check
- Resolve baseline from explicit `--run-dir` or `data/results/ACTIVE_RUN.json`
- Inspect `RUN_CONTEXT.md`, Feature Unit Activation, and Boundary Governance
- Confirm capability status as active, partially restored, or historical only

Step 4 - Minimal Repair + Replay
- Propose the minimal patch only
- Run a single-paper or otherwise bounded replay
- Do not modify unrelated stages
- Do not introduce new semantics

Hard prohibitions
- no skipping repair index
- no skipping memory
- no assuming code presence = capability
- no promoting pattern without governed activation evidence

---

# 10. Design intent reminder

Current architecture direction:

- LLM performs **semantic formulation identification**
- rules perform **validation, audit, and evidence checking**
- pre-LLM table handling may irreversibly remove only **confirmed pure noise**
  tables; if a table is not confirmed noise, preserve it into the pre-LLM
  authority surface
- table summary remains a semantic-facing surface only; deterministic authority
  handles such as authority roots or payload locators are execution-side
  metadata and must not be treated as LLM semantic content
- when LLM task, prompt content, evidence content, model, and generation
  settings are unchanged, prefer replay from frozen raw responses over fresh
  live calls if the deterministic replay path can lawfully reattach required
  execution-side metadata

Older rule-heavy reconstruction paths and multimodel pipelines are considered
**historical methods** unless explicitly reactivated.

---

# 11. Boundary legality (hard enforcement)

This section defines **non-negotiable failure conditions** for pipeline
execution and debugging.

These rules are **not guidance**.  
They define conditions under which results are invalid.

---

## 11.1 Diagnostic boundary misuse

Artifacts classified as diagnostic-only must not be used as downstream
execution inputs or authoritative outputs.

Diagnostic-only artifacts include (non-exhaustive):

- Stage2 raw responses
- Stage2 semantic-intermediate objects
- Stage4 evaluation outputs
- reviewer workbooks and audit exports

Hard rule:

- Diagnostic artifacts must not be treated as:
  - lawful Stage3 inputs
  - lawful Stage5 inputs
  - authoritative downstream outputs

If a diagnostic artifact is used as if it were a valid pipeline boundary,
the result is **invalid**.

---

## 11.2 Lawful resume boundary rule

Not every replayable artifact is a valid resume point.

Only the following are lawful downstream inputs:

- completed Stage2 artifact → Stage3 input
- Stage3 relation artifacts → Stage5 input

Hard rule:

- If an artifact does not preserve the full upstream contract required by the
  next stage, it must not be used as a resume boundary.

Agents must explicitly verify boundary legality before resuming execution.

---

## 11.3 Stage2 authority drift

Stage2 semantic authority belongs to the LLM semantic discovery layer.

Deterministic components inside Stage2 may:

- expand
- normalize
- project
- repair within declared semantic scope

They must not:

- define the candidate universe
- replace LLM semantic discovery as mainline authority

Deterministic semantic emitters, semantic lifts, and reconstruction paths:

- are allowed only as:
  - fallback
  - comparator
  - migration-support
  - diagnostic infrastructure

Hard rule:

- If a deterministic path is used or described as active Stage2 mainline
  authority without explicit governed fallback declaration,
  this is a **Stage2 authority violation**.

---

## 11.4 Failure handling rule

If any rule in Section 11 is violated:

- the result must be marked as **invalid**
- the agent must not proceed to:
  - downstream execution based on the invalid artifact
  - conclusions based on the invalid artifact

The agent must instead:

1. identify the violated rule
2. explain the violation
3. stop further interpretation

No silent fallback is allowed.
