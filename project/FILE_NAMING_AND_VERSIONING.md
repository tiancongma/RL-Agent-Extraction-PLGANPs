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
