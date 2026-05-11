# Governance Reference Audit

## 1. Scan scope

This was a read-only integrity audit for stale references to removed or merged
`project/` governance files after governance reduction.

Excluded from scan:

- `.git/`
- `.venv/`
- `data/results/`
- large binary files

Search targets:

- `project/3_STATE_MACHINE.md`
- `project/AGENT_RUNBOOK.md`
- `project/6_AGENT_RUNBOOK.md`
- `project/7_DATASET_LAYOUT_CONVENTION.md`
- `project/8_EVAL_SPLITS_REGISTRY.md`
- `project/project_specification_UPDATED_20260130_v5.txt`
- `project/project_specification_UPDATED_20260131_v6.txt`
- `project/project_specification_UPDATED_20260201_v7.txt`
- `project/ACTIVE_PIPELINE_FLOW.tsv`

Scanned text file types:

- `.md`
- `.py`
- `.tsv`
- `.txt`
- `.json`
- `.yaml`
- `.yml`
- `.sh`
- `.ps1`

## 2. Files scanned

Scanned file count by extension:

- `.md`: 75
- `.py`: 135
- `.tsv`: 165
- `.txt`: 3403
- `.json`: 1360
- `.yaml`: 0
- `.yml`: 0
- `.sh`: 0
- `.ps1`: 1

Total text files scanned: 5139

## 3. Detected stale references

FILE: `AGENTS.md`  
LINE: `16`  
REFERENCE: `project/AGENT_RUNBOOK.md`  
RECOMMENDED FIX:  
  → replace with `project/ACTIVE_PIPELINE_RUNBOOK.md`

FILE: `docs/archive_project/REPO_REDUCTION_REVIEW.md`  
LINE: `13`  
REFERENCE: `project/3_STATE_MACHINE.md`  
RECOMMENDED FIX:  
  → replace with `project/2_ARCHITECTURE.md`

FILE: `docs/archive_project/REPO_REDUCTION_REVIEW.md`  
LINE: `20`  
REFERENCE: `project/AGENT_RUNBOOK.md`  
RECOMMENDED FIX:  
  → replace with `project/ACTIVE_PIPELINE_RUNBOOK.md`

FILE: `docs/archive_project/REPO_REDUCTION_REVIEW.md`  
LINE: `21`  
REFERENCE: `project/7_DATASET_LAYOUT_CONVENTION.md`  
RECOMMENDED FIX:  
  → replace with `project/FILE_NAMING_AND_VERSIONING.md`

FILE: `docs/archive_project/REPO_REDUCTION_REVIEW.md`  
LINE: `22`  
REFERENCE: `project/8_EVAL_SPLITS_REGISTRY.md`  
RECOMMENDED FIX:  
  → replace with `project/ACTIVE_PIPELINE_FLOW.md`

FILE: `docs/archive_project/REPO_REDUCTION_REVIEW.md`  
LINE: `32`  
REFERENCE: `project/project_specification_UPDATED_20260201_v7.txt`  
RECOMMENDED FIX:  
  → replace with `project/1_REQUIREMENTS.md` and `project/2_ARCHITECTURE.md`

FILE: `docs/archive_project/REPO_REDUCTION_REVIEW.md`  
LINE: `32`  
REFERENCE: `project/AGENT_RUNBOOK.md`  
RECOMMENDED FIX:  
  → replace with `project/ACTIVE_PIPELINE_RUNBOOK.md`

FILE: `docs/archive_project/REPO_REDUCTION_REVIEW.md`  
LINE: `33`  
REFERENCE: `project/project_specification_UPDATED_20260131_v6.txt`  
RECOMMENDED FIX:  
  → replace with `project/1_REQUIREMENTS.md` and `project/2_ARCHITECTURE.md`

FILE: `docs/archive_project/REPO_REDUCTION_REVIEW.md`  
LINE: `33`  
REFERENCE: `project/AGENT_RUNBOOK.md`  
RECOMMENDED FIX:  
  → replace with `project/ACTIVE_PIPELINE_RUNBOOK.md`

FILE: `docs/archive_project/REPO_REDUCTION_REVIEW.md`  
LINE: `34`  
REFERENCE: `project/project_specification_UPDATED_20260130_v5.txt`  
RECOMMENDED FIX:  
  → replace with `project/1_REQUIREMENTS.md` and `project/2_ARCHITECTURE.md`

FILE: `docs/archive_project/REPO_REDUCTION_REVIEW.md`  
LINE: `34`  
REFERENCE: `project/AGENT_RUNBOOK.md`  
RECOMMENDED FIX:  
  → replace with `project/ACTIVE_PIPELINE_RUNBOOK.md`

FILE: `docs/archive_project/REPO_REDUCTION_REVIEW.md`  
LINE: `39`  
REFERENCE: `project/ACTIVE_PIPELINE_FLOW.tsv`  
RECOMMENDED FIX:  
  → replace with `project/ACTIVE_PIPELINE_FLOW.md`

FILE: `docs/archive_project/REPO_REDUCTION_REVIEW.md`  
LINE: `42`  
REFERENCE: `project/6_AGENT_RUNBOOK.md`  
RECOMMENDED FIX:  
  → replace with `project/ACTIVE_PIPELINE_RUNBOOK.md`

FILE: `docs/archive_project/REPO_REDUCTION_REVIEW.md`  
LINE: `42`  
REFERENCE: `project/AGENT_RUNBOOK.md`  
RECOMMENDED FIX:  
  → replace with `project/ACTIVE_PIPELINE_RUNBOOK.md`

FILE: `docs/archive_project/REPO_REDUCTION_REVIEW.md`  
LINE: `79`  
REFERENCE: `project/ACTIVE_PIPELINE_FLOW.tsv`  
RECOMMENDED FIX:  
  → replace with `project/ACTIVE_PIPELINE_FLOW.md`

FILE: `docs/archive_project/PROJECT_REDUCTION_PLAN.md`  
LINE: `8`  
REFERENCE: `project/3_STATE_MACHINE.md`  
RECOMMENDED FIX:  
  → replace with `project/2_ARCHITECTURE.md`

FILE: `docs/archive_project/PROJECT_REDUCTION_PLAN.md`  
LINE: `11`  
REFERENCE: `project/6_AGENT_RUNBOOK.md`  
RECOMMENDED FIX:  
  → replace with `project/ACTIVE_PIPELINE_RUNBOOK.md`

FILE: `docs/archive_project/PROJECT_REDUCTION_PLAN.md`  
LINE: `12`  
REFERENCE: `project/7_DATASET_LAYOUT_CONVENTION.md`  
RECOMMENDED FIX:  
  → replace with `project/FILE_NAMING_AND_VERSIONING.md`

FILE: `docs/archive_project/PROJECT_REDUCTION_PLAN.md`  
LINE: `13`  
REFERENCE: `project/8_EVAL_SPLITS_REGISTRY.md`  
RECOMMENDED FIX:  
  → replace with `project/ACTIVE_PIPELINE_FLOW.md`

FILE: `docs/archive_project/PROJECT_REDUCTION_PLAN.md`  
LINE: `15`  
REFERENCE: `project/ACTIVE_PIPELINE_FLOW.tsv`  
RECOMMENDED FIX:  
  → replace with `project/ACTIVE_PIPELINE_FLOW.md`

FILE: `docs/archive_project/PROJECT_REDUCTION_PLAN.md`  
LINE: `17`  
REFERENCE: `project/AGENT_RUNBOOK.md`  
RECOMMENDED FIX:  
  → replace with `project/ACTIVE_PIPELINE_RUNBOOK.md`

FILE: `docs/archive_project/PROJECT_REDUCTION_PLAN.md`  
LINE: `25`  
REFERENCE: `project/project_specification_UPDATED_20260130_v5.txt`  
RECOMMENDED FIX:  
  → replace with `project/1_REQUIREMENTS.md` and `project/2_ARCHITECTURE.md`

FILE: `docs/archive_project/PROJECT_REDUCTION_PLAN.md`  
LINE: `26`  
REFERENCE: `project/project_specification_UPDATED_20260131_v6.txt`  
RECOMMENDED FIX:  
  → replace with `project/1_REQUIREMENTS.md` and `project/2_ARCHITECTURE.md`

FILE: `docs/archive_project/PROJECT_REDUCTION_PLAN.md`  
LINE: `27`  
REFERENCE: `project/project_specification_UPDATED_20260201_v7.txt`  
RECOMMENDED FIX:  
  → replace with `project/1_REQUIREMENTS.md` and `project/2_ARCHITECTURE.md`

FILE: `docs/archive_project/PROJECT_REDUCTION_PLAN.md`  
LINE: `36`  
REFERENCE: `project/3_STATE_MACHINE.md`  
RECOMMENDED FIX:  
  → replace with `project/2_ARCHITECTURE.md`

FILE: `docs/archive_project/PROJECT_REDUCTION_PLAN.md`  
LINE: `37`  
REFERENCE: `project/7_DATASET_LAYOUT_CONVENTION.md`  
RECOMMENDED FIX:  
  → replace with `project/FILE_NAMING_AND_VERSIONING.md`

FILE: `docs/archive_project/PROJECT_REDUCTION_PLAN.md`  
LINE: `38`  
REFERENCE: `project/8_EVAL_SPLITS_REGISTRY.md`  
RECOMMENDED FIX:  
  → replace with `project/ACTIVE_PIPELINE_FLOW.md`

FILE: `docs/archive_project/PROJECT_REDUCTION_PLAN.md`  
LINE: `39`  
REFERENCE: `project/AGENT_RUNBOOK.md`  
RECOMMENDED FIX:  
  → replace with `project/ACTIVE_PIPELINE_RUNBOOK.md`

FILE: `docs/archive_project/PROJECT_REDUCTION_PLAN.md`  
LINE: `40`  
REFERENCE: `project/6_AGENT_RUNBOOK.md`  
RECOMMENDED FIX:  
  → replace with `project/ACTIVE_PIPELINE_RUNBOOK.md`

FILE: `docs/archive_project/PROJECT_REDUCTION_PLAN.md`  
LINE: `41`  
REFERENCE: `project/project_specification_UPDATED_20260130_v5.txt`  
RECOMMENDED FIX:  
  → replace with `project/1_REQUIREMENTS.md` and `project/2_ARCHITECTURE.md`

FILE: `docs/archive_project/PROJECT_REDUCTION_PLAN.md`  
LINE: `42`  
REFERENCE: `project/project_specification_UPDATED_20260131_v6.txt`  
RECOMMENDED FIX:  
  → replace with `project/1_REQUIREMENTS.md` and `project/2_ARCHITECTURE.md`

FILE: `docs/archive_project/PROJECT_REDUCTION_PLAN.md`  
LINE: `43`  
REFERENCE: `project/project_specification_UPDATED_20260201_v7.txt`  
RECOMMENDED FIX:  
  → replace with `project/1_REQUIREMENTS.md` and `project/2_ARCHITECTURE.md`

FILE: `docs/archive_project/PROJECT_REDUCTION_PLAN.md`  
LINE: `55`  
REFERENCE: `project/ACTIVE_PIPELINE_FLOW.tsv`  
RECOMMENDED FIX:  
  → replace with `project/ACTIVE_PIPELINE_FLOW.md`

FILE: `docs/ee_coverage_rl/2026-02-26_stage1_html_first_tables_manifest.md`  
LINE: `31`  
REFERENCE: `project/6_AGENT_RUNBOOK.md`  
RECOMMENDED FIX:  
  → replace with `project/ACTIVE_PIPELINE_RUNBOOK.md`

FILE: `project/2_ARCHITECTURE.md`  
LINE: `172`  
REFERENCE: `project/3_STATE_MACHINE.md`  
RECOMMENDED FIX:  
  → replace with `project/2_ARCHITECTURE.md`

FILE: `project/2_ARCHITECTURE.md`  
LINE: `178`  
REFERENCE: `project/AGENT_RUNBOOK.md`  
RECOMMENDED FIX:  
  → replace with `project/ACTIVE_PIPELINE_RUNBOOK.md`

FILE: `project/2_ARCHITECTURE.md`  
LINE: `349`  
REFERENCE: `project/3_STATE_MACHINE.md`  
RECOMMENDED FIX:  
  → replace with `project/2_ARCHITECTURE.md`

FILE: `project/ACTIVE_PIPELINE_FLOW.md`  
LINE: `90`  
REFERENCE: `project/8_EVAL_SPLITS_REGISTRY.md`  
RECOMMENDED FIX:  
  → replace with `project/ACTIVE_PIPELINE_FLOW.md`

FILE: `project/ACTIVE_PIPELINE_RUNBOOK.md`  
LINE: `264`  
REFERENCE: `project/AGENT_RUNBOOK.md`  
RECOMMENDED FIX:  
  → replace with `project/ACTIVE_PIPELINE_RUNBOOK.md`

FILE: `project/ACTIVE_PIPELINE_RUNBOOK.md`  
LINE: `264`  
REFERENCE: `project/6_AGENT_RUNBOOK.md`  
RECOMMENDED FIX:  
  → replace with `project/ACTIVE_PIPELINE_RUNBOOK.md`

FILE: `project/ACTIVE_PIPELINE_RUNBOOK.md`  
LINE: `300`  
REFERENCE: `project/6_AGENT_RUNBOOK.md`  
RECOMMENDED FIX:  
  → replace with `project/ACTIVE_PIPELINE_RUNBOOK.md`

FILE: `project/FILE_NAMING_AND_VERSIONING.md`  
LINE: `127`  
REFERENCE: `project/7_DATASET_LAYOUT_CONVENTION.md`  
RECOMMENDED FIX:  
  → replace with `project/FILE_NAMING_AND_VERSIONING.md`

FILE: `project/FILE_NAMING_AND_VERSIONING.md`  
LINE: `170`  
REFERENCE: `project/7_DATASET_LAYOUT_CONVENTION.md`  
RECOMMENDED FIX:  
  → replace with `project/FILE_NAMING_AND_VERSIONING.md`

FILE: `src/utils/split_registry_v1.py`  
LINE: `7`  
REFERENCE: `project/8_EVAL_SPLITS_REGISTRY.md`  
RECOMMENDED FIX:  
  → replace with `project/ACTIVE_PIPELINE_FLOW.md`

## 4. Recommended replacements

Canonical replacement map used in this audit:

- `project/3_STATE_MACHINE.md` -> `project/2_ARCHITECTURE.md`
- `project/AGENT_RUNBOOK.md` -> `project/ACTIVE_PIPELINE_RUNBOOK.md`
- `project/6_AGENT_RUNBOOK.md` -> `project/ACTIVE_PIPELINE_RUNBOOK.md`
- `project/7_DATASET_LAYOUT_CONVENTION.md` -> `project/FILE_NAMING_AND_VERSIONING.md`
- `project/8_EVAL_SPLITS_REGISTRY.md` -> `project/ACTIVE_PIPELINE_FLOW.md`
- `project/project_specification_UPDATED_20260130_v5.txt` -> `project/1_REQUIREMENTS.md` and `project/2_ARCHITECTURE.md`
- `project/project_specification_UPDATED_20260131_v6.txt` -> `project/1_REQUIREMENTS.md` and `project/2_ARCHITECTURE.md`
- `project/project_specification_UPDATED_20260201_v7.txt` -> `project/1_REQUIREMENTS.md` and `project/2_ARCHITECTURE.md`
- `project/ACTIVE_PIPELINE_FLOW.tsv` -> `project/ACTIVE_PIPELINE_FLOW.md`

Priority fix set outside archived documents:

- `AGENTS.md`
- `docs/ee_coverage_rl/2026-02-26_stage1_html_first_tables_manifest.md`
- `project/2_ARCHITECTURE.md`
- `project/ACTIVE_PIPELINE_FLOW.md`
- `project/ACTIVE_PIPELINE_RUNBOOK.md`
- `project/FILE_NAMING_AND_VERSIONING.md`
- `src/utils/split_registry_v1.py`

## 5. Verification conclusion

Verification did not pass. Stale references to removed governance files remain.

Observed result summary:

- Active/current repository files still contain stale references.
- Archived documents under `docs/archive_project/` also preserve stale references by design.
- The highest-priority non-archived fixes are in `AGENTS.md`, the surviving `project/` governance files, `docs/ee_coverage_rl/2026-02-26_stage1_html_first_tables_manifest.md`, and `src/utils/split_registry_v1.py`.
