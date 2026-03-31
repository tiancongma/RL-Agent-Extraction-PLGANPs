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
- governed path identity
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
```text
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
```text
sample10.tsv
sample20.tsv
sample30.jsonl
```

---

## Label Files

### Manual Labels
```text
data/cleaned/labels/manual/
├── manual_labels_v2.tsv
├── manual_labels_v3.tsv
└── manual_labels_v4.tsv
```

### Weak Labels
```text
data/cleaned/labels/weak/
├── v2/
├── v3/
└── v4/
```

Version number reflects **generation logic**, not quality.

---

## Results Naming And Lineage Governance

### Future-Facing Rule

Per MDEC084, future `data/results/` lineages must separate:

- bucket identity
- child execution identity
- rich execution meaning
- active authority

Future top-level run bucket naming:

- `YYYYMMDD_<short_hash>`

Future child execution folder naming:

- `NN_<cue>`
- `NNN_<cue>`

Examples:

- `data/results/20260331_03e5d25/`
- `data/results/20260331_03e5d25/01_stage2/`
- `data/results/20260331_03e5d25/02_relation/`

Future lineage rules:

- rich semantics must not be encoded in folder names
- rich semantics must live in `RUN_CONTEXT.md`
- child execution identity is represented by ordinal child folders, not by
  repeated full nested run IDs
- future lineage must not use nested repeated full `run_id` directories inside
  a governed run bucket
- artifact folders below a child execution root must remain functional only,
  for example:
  - `analysis/`
  - `outputs/`
  - `audit/`
  - `formulation_relation_v1/`

Future authority rule:

- `data/results/ACTIVE_RUN.json` remains the only authority surface for active
  results workflows
- active authority must not be inferred from recency, lexical sort order,
  parent fallback, or folder-name parsing
- `ACTIVE_RUN.json` may reference governed future bucket/child paths and is
  not limited to historical `run_*` directory names

### Legacy Compatibility Rule

Historical runs remain frozen legacy surfaces.

- do not rename or restructure existing historical `run_*` directories only to
  satisfy the future naming scheme
- preserve backward compatibility for historical references
- the old style remains readable as historical compatibility:
  - `run_YYYYMMDD_HHMM_<short_hash>_<suffix>`
- historical nested lineage paths may remain in place until an explicit
  governed migration is approved

Legacy compatibility regex:

- `^run_\d{8}_\d{4}_[0-9a-f]{7}_.+$`

Utility migration note:

- current writer utilities may still expose legacy `run_id` interfaces until
  the governed utility migration is completed
- future naming must not be enabled in writer code until the utility updates
  declared in MDEC084 are complete
- the governed future writer contract for bucket creation and child allocation
  is documented in `docs/mdec084_writer_contract_v1.md`

## Cleaned Asset Layout (Dataset-Scoped)

Canonical rule: cleaned assets are dataset-scoped and reusable across runs, under:

`data/cleaned/<dataset_id>/...`

Run outputs are run-scoped and must remain under governed `data/results/`
lineage paths.

Future default:

- `data/results/<YYYYMMDD_<short_hash>>/<NN_<cue>>/...`

Historical compatibility:

- `data/results/<run_id>/...`

Do not mix run outputs into cleaned assets.

Do not create new top-level directories under `data/cleaned/<dataset_id>/` outside:
`index`, `content`, `text`, `sections`, `tables`, `analysis`.

Reference: consolidated dataset layout convention below.

### Future Structure
```text
data/results/
└── YYYYMMDD_<short_hash>/
    ├── LINEAGE.md
    ├── 01_<cue>/
    │   ├── RUN_CONTEXT.md
    │   └── analysis/
    └── 02_<cue>/
```

### Historical Compatibility Structure
```text
data/results/
└── run_YYYYMMDD_HHMM_<short_hash>_<suffix>/
```

### Latest Result
The only valid definition of "latest" is:
```text
runs/latest.txt
```
Compatibility rule: first line must be exactly the run_id. Additional metadata lines must start with `# `.

Current architecture note:

- `runs/latest.txt` is legacy compatibility only.
- It must not be treated as the sole authority for current
  `data/results/run_*` benchmark, alignment, comparison, workbook, or audit
  workflows.
- Those workflows must use the active data-source contract in
  `project/ACTIVE_DATA_SOURCE_CONTRACT.md` and the repository pointer
  `data/results/ACTIVE_RUN.json`.

### Entry Script Discipline

- Until the utility migration is complete, entry scripts that write to
  `data/results/` must require explicit identity inputs rather than silently
  inventing paths.
- Historical compatibility writers may still require explicit `--run-id` until
  the new bucket/child writer model is enabled.
- Entry scripts must not generate governed path identity internally without the
  maintained utility path.
- Reuse/new policy is determined only by `python -m src.utils.run_preflight ...`.
- Reused work must specify a functional output location.
- `--out-subdir` is a functional artifact path only and must not encode a
  nested run identifier or timestamp/hash token.
- Future child execution folders must be ordinal child folders such as
  `01_stage2`, not nested repeated full `run_id` directories.

---

## Safe Experimentation Rule

To try a new idea:
1. Do not overwrite current files
2. Create a new governed future bucket/child path or an explicit legacy
   compatibility run only when required by current utilities
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
- Run outputs must live under governed `data/results/` lineage paths.

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
