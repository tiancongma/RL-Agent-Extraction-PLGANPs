# Agent Runbook v1.1

This document defines the default execution contract for automated agents working in this repository.

---

## Mandatory Environment Preflight (Run First, Always)

Before doing anything else, run an environment preflight in the repo root.

### 1) Print interpreter + pip paths

Windows PowerShell:

where python  
python -V  
where pip  
pip -V  

Confirm:
- `python` and `pip` resolve to the same virtual environment  
- The repo `.venv` (if used) is active  

---

### 2) Dependency probe (must succeed)

python -c "import pandas as pd, fitz, trafilatura, bs4, lxml; pd.read_html; print('deps_ok')"

Notes:
- `PyMuPDF` provides module `fitz`
- `pandas.read_html` requires `lxml`
- This probe verifies both HTML and PDF extraction paths

---

### 3) If probe fails

pip install -r requirements.txt

Re-run the probe.

If `deps_ok` is not printed, STOP.  
Do not proceed with any Stage1/Stage2 execution.  
Fix environment first.

Only after `deps_ok`:
Proceed with requested Stage1/Stage2 command(s).

---

## Preflight Read Order

Before starting implementation work, read in this order:

1. `README.md`
2. `project/2_ARCHITECTURE.md`
3. `project/FILE_NAMING_AND_VERSIONING.md`
4. `project/FEATURE_EE_COVERAGE_RL_SCOPE.md` (when working on EE coverage branch tasks)
5. `docs/tool_index.md`
6. Relevant diagnostics under `docs/ee_coverage_rl/`

---

## Branch Discipline (EE Validation Branch)

This branch serves two purposes:

1. EE coverage validation
2. Controlled upgrade of general extraction logic

Before writing or modifying any file, agents must classify the task as:

- **Type A – General Extraction Upgrade (future merge candidate)**
- **Type B – EE / Benchmark-Specific Logic (branch-only)**

---

### Type A (May Merge to Main)

Allowed locations:

- `src/stage0_relevance/`
- `src/stage1_cleaning/`
- `src/stage2_sampling_labels/`
- `src/utils/`
- Project-level documentation updates

Constraints:

- Must preserve directory contracts
- Must preserve manifest and key2txt invariants
- Must not introduce breaking schema changes
- Must use deterministic `run_id` behavior
- Must not introduce EE-specific filtering
- Must not silently alter extraction semantics

---

### Type B (Branch-Only)

Must remain isolated under:

- `src/stage4_eval/`
- `src/stage5_benchmark/`
- `docs/ee_coverage_rl/`
- `data/benchmark/`
- `data/db/`

EE-specific logic must not be embedded into Stage1 or Stage2 scripts.

---

## Invariants

- Benchmark/view logic must not constrain Full DB upstream.
- Evidence gating must not reduce Full DB row counts unless explicitly fixing splitting/dedup.
- All human-review debug outputs must include DOI metadata:
  - `reference_normalized_doi`
  - `doi_url`
- Avoid committing `data/results/` outputs by default.
- Require regression checks (before/after) for any behavior change.
- Keep commits small and scoped.
- No hard-coded paths. Use canonical directory contracts.
- All run outputs must follow deterministic `run_id` naming.
- `run_id` generation must use `python -m src.utils.run_id ...` (or the same underlying function), never manual string concatenation.
- `run_id` must match: `^run_\d{8}_\d{4}_[0-9a-f]{7}_.+$`.
- Every pipeline run must update `runs/latest.txt`: first line is exactly `run_id`; metadata lines must start with `# `.

### Preflight-First Run Discipline

- All entry scripts that write under `data/results/` require explicit `--run-id` (no internal run_id generation).
- The reuse/new decision must be made only via: `python -m src.utils.run_preflight ...`.
- All reused work must specify `--out-subdir` and write to `data/results/<run_id>/<out-subdir>/...`.

Canonical workflow:
1. `python -m src.utils.run_preflight --subset <subset> --stage <stage> --version <v> --input-path <path> [--input-path <path2>] [--note "..."]`
2. Run entry script with `--run-id <chosen_run_id> --out-subdir <stage_or_variant_subdir>`.

### HTML Extraction Invariant

If HTML exists and HTML extraction fails:

- Failure reason must be explicitly recorded.
- Exception type/message must be preserved in logs or manifest.
- Silent fallback to PDF is not allowed without an explicit reason field.
- Environment dependency failures must not be masked as structural HTML failures.
- For table extraction manifests, default selected tables are HTML-first; PDF tables are fallback-only unless HTML yields zero tables.

---

## Workflow

1. Identify which layer the issue belongs to:
   - extraction
   - evidence
   - instance
   - confidence
   - view
2. Classify task as Type A or Type B.
3. Modify only that layer.
4. When diagnosing, generate a human-auditable debug matrix.
5. Write a step report under `docs/ee_coverage_rl/`.
6. Update system documentation if behavior changes.
7. If change qualifies as Type A:
   - Record it in `FEATURE_EE_COVERAGE_RL_SCOPE.md`
   - Do not merge automatically without explicit review.

---

## Formulation Core Signature v1 (Type B)

Entry script:
- `src/stage5_benchmark/run_formulation_core_signature_v1.py`

Library module:
- `src/stage5_benchmark/formulation_core_signature_v1.py`

Output contract (run-scoped only):

data/results/<run_id>/formulation_core_signature_v1/formulation_core_v1.tsv  
data/results/<run_id>/formulation_core_signature_v1/instance_assignment_v1.tsv  
data/results/<run_id>/formulation_core_signature_v1/signature_trace_v1.tsv  
data/results/<run_id>/formulation_core_signature_v1/build_log.json  

Execution example:

python src/stage5_benchmark/run_formulation_core_signature_v1.py --input-tsv data/results/<run_id>/weak_labels__gemini.tsv

Rules:

- Keep logic in Stage5 benchmark scope.
- Use `src/utils/paths.py` for default path resolution.
- Do not write non-run-scoped artifacts.
- Do not introduce cross-stage coupling.

---

## Merge Candidate Recording

When a general extraction upgrade is validated and considered stable:

1. Confirm it is Type A.
2. Ensure no EE-only dependency.
3. Record file path under “Merge Candidate Policy” in `FEATURE_EE_COVERAGE_RL_SCOPE.md`.
4. Do not merge automatically without explicit review.

---

End of Runbook v1.1
