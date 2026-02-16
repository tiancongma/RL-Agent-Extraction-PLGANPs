# Project State Machine

This document defines the current lifecycle state of the project, the allowed operations within each state, and the explicit criteria for transitioning to the next state.

The goal is to prevent scope drift, accidental rework, and ambiguity about where the project currently is.

---

## STATE_0 — Scope Locked

### Description

The overall project goal, scope, and non-goals are fixed.

### Allowed

- Documentation updates  
- Clarifying requirements  
- Non-behavioral refactoring  

### Forbidden

- Adding new research objectives  
- Expanding scientific scope  
- Changing evaluation philosophy  

### Exit Criteria

- `project/0_PROJECT_CHARTER.md` finalized  
- `project/1_REQUIREMENTS.md` finalized  

---

## STATE_1 — Data Ready

### Description

Raw ingestion and cleaned text generation pipelines are stable and reproducible.

### Allowed

- Regenerating cleaned text  
- Fixing parsing errors  
- Manifest validation  

### Forbidden

- Modifying extraction schema  
- Adding evaluation logic  

### Exit Criteria

- `manifest_current.tsv` validated  
- `key2txt.tsv` validated  
- Cleaned text reproducible across reruns  

---

## STATE_2 — Architecture Revision (CURRENT)

### Description

Controlled redesign of the extraction and evaluation architecture.

This state allows structural changes to `stage4_eval` and `stage3_gt`,
provided they remain within the original project scope.

**Current focus:**  
Transition from dual-model extraction to evidence-grounded verification.

### Allowed

- Adding verifier layer  
- Modifying extraction output schema to include evidence  
- Introducing multi-span evidence contract  
- Updating aggregation logic  
- Updating state documentation  

### Forbidden

- Expanding field list  
- Adding new scientific tasks  
- Introducing new external data sources  
- Changing project scope defined in STATE_0  

### Exit Criteria

- Evidence contract finalized  
- Verifier script implemented  
- Aggregator implemented  
- Sample20 successfully processed under new architecture  
- No schema change across two consecutive evidence-based runs  

---

## STATE_3 — Pipeline Frozen

### Description

The evidence-grounded extraction pipeline structure is frozen.  
Only stability, bug fixes, and prompt refinement are allowed.

### Allowed

- Prompt wording refinement (no schema change)  
- Robustness improvements  
- Logging enhancements  
- Performance tuning  

### Forbidden

- Adding new extraction fields  
- Changing evidence schema  
- Changing verification logic  
- Reintroducing independent second extractor  

### Exit Criteria

- Two consecutive stable runs  
- Evidence-based GT workflow validated  
- No structural change required for 2 weeks  

---

## STATE_4 — Results Stable

### Description

Extraction outputs and evaluation metrics are stable enough for reporting.

### Allowed

- Figure and table generation  
- Metric analysis  
- Ablation comparisons (dual-model vs evidence-based)  

### Forbidden

- Architecture changes  
- Data regeneration unless bug fix  

### Exit Criteria

- Final tables generated  
- Evaluation documented  
- Reproducibility verified  

---

## STATE_5 — Writing and Release

### Description

Focus shifts to writing, sharing, and publication.

### Allowed

- Documentation updates  
- Manuscript preparation  
- arXiv or submission packaging  

### Forbidden

- New experiments  
- Pipeline changes  
- Schema modifications  

### Exit Criteria

- arXiv v1 submitted or shared  
