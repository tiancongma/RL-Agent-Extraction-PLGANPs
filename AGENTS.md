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

# 10. Design intent reminder

Current architecture direction:

- LLM performs **semantic formulation identification**
- rules perform **validation, audit, and evidence checking**

Older rule-heavy reconstruction paths and multimodel comparison pipelines
are considered **historical methods** unless explicitly reactivated.
