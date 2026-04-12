# data/cleaned/index

This directory contains **pipeline-critical index files**.
Any change here requires re-running downstream extraction.

## Files

- `key2txt.tsv`  
  Mapping from paper key to cleaned text file path.
  **Single source of truth.**

- `manifest_current.tsv`  
  Current manifest used by the main pipeline.
  Do NOT modify without recording decision in `project/4_DECISIONS_LOG.md`.

## Legacy

- `legacy/`  
  Historical manifests and index files.
  Not used by the current pipeline.
## Note

- data/cleaned/content/key2txt.tsv is a transient byproduct of cleaning; the authoritative mapping is promoted to data/cleaned/index/key2txt.tsv and all downstream stages must use the index copy.