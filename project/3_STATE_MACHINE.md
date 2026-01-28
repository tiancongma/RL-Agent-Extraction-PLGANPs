# Project State Machine

This document defines the **current lifecycle state of the project**, the allowed
operations within each state, and the explicit criteria for transitioning to the
next state.

The goal is to prevent scope drift, accidental rework, and ambiguity about
"where the project currently is."

---

## STATE_0 — Scope Locked

### Description
The overall project goal, scope, and non-goals are fixed.

### Allowed
- Documentation updates
- Minor refactoring without behavior change

### Forbidden
- Adding new objectives
- Expanding problem scope

### Exit Criteria
- `project/0_PROJECT_CHARTER.md` finalized
- `project/1_REQUIREMENTS.md` finalized

---

## STATE_1 — Data Ready

### Description
Raw and cleaned data pipelines are functional and stable.

### Allowed
- Regenerating raw or cleaned data
- Fixing data parsing bugs

### Forbidden
- Changing downstream evaluation logic

### Exit Criteria
- `manifest_current.tsv` exists and is validated
- `key2txt.tsv` exists and is validated

---

## STATE_2 — Pipeline Frozen (CURRENT)

### Description
The extraction pipeline structure is frozen.
Only stability and correctness are addressed.

### Allowed
- Bug fixes
- Prompt wording refinements (no new fields)
- Logging and robustness improvements

### Forbidden
- Adding new extraction fields
- Changing data schemas
- Adding new models

### Exit Criteria
- Sample20 fully processed
- No schema changes across two consecutive runs

---

## STATE_3 — Results Stable

### Description
Extraction outputs and evaluation metrics are stable enough for reporting.

### Allowed
- Figure/table generation
- Result summarization

### Forbidden
- Pipeline logic changes
- Data regeneration

### Exit Criteria
- Final tables generated
- Metrics documented

---

## STATE_4 — Writing and Release

### Description
Focus shifts to writing, sharing, and publication.

### Allowed
- Documentation
- Manuscript or arXiv preparation

### Forbidden
- New experiments
- Pipeline changes

### Exit Criteria
- arXiv v1 submitted or shared
