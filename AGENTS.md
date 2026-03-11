# AGENTS.md

Agent execution contract for this repository.

This file defines how AI coding agents (Codex, Claude Code, etc.)
should initialize context before modifying the repository.
This repository enforces a strict governance layer.
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

Do not infer architecture without reading them.

---

# 2. Governance layer rule

`project/` is a governance layer.

It contains **only authoritative project definitions**.

Agents MUST NOT create new files inside `project/`
unless explicitly instructed by the user.

Maximum governance files allowed: **9**

---

# 3. Repository structure

The repository is organized as four layers:

project/  
→ governance documents

src/  
→ executable pipeline code

data/  
→ datasets and run artifacts

docs/  
→ explanations, benchmarks, historical material

Agents must keep these layers separated.

---

# 4. File creation rules

Default behavior:

append to existing files.

Creating new files requires explicit user instruction.

Never create:

- new governance documents
- duplicate specifications
- snapshot-style files

---

# 5. Pipeline authority

The active pipeline definition lives in:

project/ACTIVE_PIPELINE_FLOW.md

The execution instructions live in:

project/ACTIVE_PIPELINE_RUNBOOK.md

Agents must not invent alternative pipelines.

---

# 6. Archive rule

Historical documents must go under:

docs/archive_project/

They must not remain in `project/`.

---

# 7. Code safety rule

Agents must not modify runtime behavior unless:

- requested by the user
- required to fix broken references
- required to align with governance files

---

# 8. When uncertain

If repository structure or authority is unclear:

1. read `PIPELINE_SCRIPT_MAP.md`
2. read `ACTIVE_PIPELINE_RUNBOOK.md`
3. ask the user

Never guess pipeline structure.