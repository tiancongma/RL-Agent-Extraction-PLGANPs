# Dataset Layout Convention (Cleaned Assets)

This document is the source of truth for dataset-scoped cleaned assets and provenance.

## Core Separation

- Cleaned assets are dataset-scoped and reusable across runs.
- Run outputs are run-scoped and must live under `data/results/<run_id>/...`.
- Do not mix run outputs into `data/cleaned/`.

## Dataset ID

`dataset_id` identifies one cleaned corpus layout, for example:

- `goren_2025`
- `goren18`
- `sample20`
- `full_413`
- `gt_random50_v1`

All cleaned assets for a dataset must resolve under `data/cleaned/<dataset_id>/...`.

## Required Directory Convention

```text
data/
  cleaned/
    <dataset_id>/
      index/
        manifest.tsv (or manifest_<dataset_id>.tsv)
        key2txt.tsv (if used)
      content/
        html/<zotero_key>/...        (optional)
        pdf/<zotero_key>/...         (optional)
      text/<zotero_key>/
        cleaned.txt
        cleaned_manifest.json
      sections/<zotero_key>/
        sections.jsonl or sections.tsv
        sections_manifest.json
      tables/<zotero_key>/
        *.csv
        tables_manifest.json
        tables_index.tsv
      analysis/
        coverage and validation reports
  results/
    <run_id>/
      ... run-scoped outputs only
```

## Binding Rules

- `text`, `sections`, and `tables` are parallel cleaned asset types.
- Per-document assets must be paper-local under `<asset_root>/<zotero_key>/...`.
- Do not create new top-level directories under `data/cleaned/<dataset_id>/` outside:
  `index`, `content`, `text`, `sections`, `tables`, `analysis`.
- Scripts producing cleaned assets must accept `--dataset-id` or explicit cleaned roots
  (`--tables-root`, `--text-root`, etc.), and resolve to dataset-scoped paths.
- Scripts must not write cleaned assets to `data/results/`.

## Provenance Chain (Required)

### 1) Document-level (index / manifest)

Required fields:

- `key` (zotero key)
- `has_html`
- `has_pdf`
- `html` path (when present)
- `pdf` path (when present)
- `doi` (if available)

### 2) Artifact-level (per-key manifests)

Each artifact manifest (for example `tables_manifest.json`, `cleaned_manifest.json`,
`sections_manifest.json`) must capture:

- `source_format` (`html` or `pdf`)
- `source_path`
- `generator` (script + version)
- `generated_at` timestamp
- artifact counts (for example number of rows/tables/sections)

### 3) Field-level (extraction and audit outputs)

Each extracted field should carry:

- `value_source_format` (`html` / `pdf`)
- `evidence_kind`
- `evidence_locator` that can be traced back to artifact-level manifests

## Legacy Note

Existing layouts such as `data/cleaned/content_goren_2025/...` are legacy structures.
They are not migrated by this document. Future migration should map legacy content into
`data/cleaned/<dataset_id>/` without changing extraction logic semantics.

## Legacy Is Not Dataset Root

- Legacy roots like `data/cleaned/content_goren_2025/` are not valid `dataset_id` roots.
- `dataset_id` must be semantic-only (for example `goren_2025`), never asset-prefixed (`content_*`).
- Recommended approach: create `data/cleaned/<dataset_id>/` and map/copy assets from legacy.
- Do not expand legacy layouts with new outputs.
