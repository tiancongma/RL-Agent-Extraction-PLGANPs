# File Naming and Versioning Contract

This document defines **mandatory rules** for naming files, organizing versions,
and determining what is "current" versus "legacy" in this project.

The purpose is to eliminate ambiguity, memory-based decisions, and accidental
use of outdated artifacts.

---

## Core Principle

> File names must never encode time, emotion, or trial-and-error history.

Such information belongs in:
- git commits
- run_id
- decision logs

---

## Prohibited Naming Patterns

The following patterns are **not allowed**:

- *_final.*
- *_final_v2.*
- *_new.*
- *_latest.*
- *_fix.*

Any file following these patterns must be renamed or archived.

---

## Index Files (Single Source of Truth)

### Rules
- Only one active index file is allowed
- All alternatives must be archived

### Structure
```
data/cleaned/index/
├── key2txt.tsv
├── manifest_current.tsv
└── legacy/
```

---

### Global vs Dataset Manifest Naming

- Unique global index: `data/cleaned/index/manifest__zotero_all.tsv`
- Dataset manifest path is fixed: `data/cleaned/<dataset_id>/index/manifest.tsv`
- Run slices or run-scoped outputs must not be named `manifest*.tsv`

### Split Naming (Dataset Index)

- Split files must live under: `data/cleaned/<dataset_id>/index/splits/`
- Reserved DEV contract names:
  - `dev_keys_v{N}.tsv`
  - `dev_manifest_v{N}.tsv`
- TEST split names must be distinct and must exclude keys in current DEV file.

---

## Sample Definitions

### Rules
- Samples describe selection only
- No version suffixes

### Examples
```
sample10.tsv
sample20.tsv
sample30.jsonl
```

---

## Label Files

### Manual Labels
```
data/cleaned/labels/manual/
├── manual_labels_v2.tsv
├── manual_labels_v3.tsv
└── manual_labels_v4.tsv
```

### Weak Labels
```
data/cleaned/labels/weak/
├── v2/
├── v3/
└── v4/
```

Version number reflects **generation logic**, not quality.

---

## Results and run_id

### Rule
All results must be organized by run_id.
Run IDs must be generated via `python -m src.utils.run_id ...` (or the same underlying function in code), never by manual string concatenation.
Valid run_id regex: `^run_\d{8}_\d{4}_[0-9a-f]{7}_.+$`.

Run-root uniqueness rule:

- each run has exactly one run root directory named by `run_id`
- artifact subdirectories under that run root must not repeat:
  - the full `run_id`
  - timestamp/hash fragments such as `20260320_1317_f54824a`
- subdirectories under a run root must describe functional layout only, for
  example:
  - `analysis/`
  - `outputs/`
  - `audit/`
  - `fgt_v3_dev15_v2/`
- if an output unit needs its own independent rerun identity, it must be a
  separate run with its own `run_id`, not a nested artifact folder

## Cleaned Asset Layout (Dataset-Scoped)

Canonical rule: cleaned assets are dataset-scoped and reusable across runs, under:

`data/cleaned/<dataset_id>/...`

Run outputs are run-scoped and must remain under:

`data/results/<run_id>/...`

Do not mix run outputs into cleaned assets.

Do not create new top-level directories under `data/cleaned/<dataset_id>/` outside:
`index`, `content`, `text`, `sections`, `tables`, `analysis`.

Reference: consolidated dataset layout convention below.

### Structure
```
data/results/
└── run_YYYYMMDD_HHMM_<commit>_<sample>/
```

### Latest Result
The only valid definition of "latest" is:
```
runs/latest.txt
```
Compatibility rule: first line must be exactly the run_id. Additional metadata lines must start with `# `.

### Entry Script Discipline

- All entry scripts that write to `data/results/` must require explicit `--run-id`.
- Entry scripts must not generate run_id internally.
- Reuse/new policy is determined only by `python -m src.utils.run_preflight ...`.
- Reused work must specify `--out-subdir`; outputs must go under `data/results/<run_id>/<out-subdir>/...`.
- `--out-subdir` is a functional artifact path only and must not encode a
  nested run identifier or timestamp/hash token.

---

## Safe Experimentation Rule

To try a new idea:
1. Do not overwrite current files
2. Create a new run_id
3. Log the decision in `project/4_DECISIONS_LOG.md`

---

## Enforcement

If a file cannot be explained in one sentence,
it must not exist in the repository.

---

## Consolidated Dataset Layout Convention

This section consolidates the durable dataset layout policy.

### Dataset-Scoped Cleaned Asset Rule

- Cleaned assets are dataset-scoped and reusable across runs.
- Dataset assets must live under `data/cleaned/<dataset_id>/...`.
- Run outputs must live under `data/results/<run_id>/...`.

### Allowed Dataset Roots

Top-level dataset directories under `data/cleaned/<dataset_id>/` are limited to:

- `index`
- `content`
- `text`
- `sections`
- `tables`
- `analysis`

### Provenance Requirements

Manifest and artifact-level outputs must preserve:

- document identifiers and DOI where available,
- source format and source path,
- generator identity,
- generated timestamp,
- artifact-level counts,
- evidence locators needed for downstream audit.

### Legacy Layout Rule

- Legacy layouts such as `data/cleaned/content_goren_2025/...` are compatibility structures, not preferred dataset roots.
- New outputs should not expand legacy layouts when a dataset-scoped root is available.
