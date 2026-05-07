# manifest_current.tsv Hydration/Mainline Integration Notes — 2026-05-07

## Why this note exists

The project has repeatedly hit a failure mode where a repair works in tests, parser bakeoff, or a run-local `data/results/.../targeted_manifest.tsv`, but the next baseline still misses it because future maintained entrypoints do not consume the repaired surface. This note records the manifest authority, generation path, current state, rollback approach, and promotion checklist so future agents can search for `manifest_current hydration mainline integration` and avoid repeating that error.

## Authority

The unique corpus-level Stage1 authority table is:

```text
data/cleaned/index/manifest_current.tsv
```

Governance references:

- `project/2_ARCHITECTURE.md` lines 266-283: `manifest_current.tsv` is the Stage1 single source of truth and is contract-complete only after explicit hydration populates asset-binding and scope-overlay fields.
- `project/ACTIVE_PIPELINE_RUNBOOK.md` lines 306-338: Stage1 core scripts include `zotero_raw_to_manifest.py`, `hydrate_manifest_v1.py`, `clean_manifest_to_text.py`, and `run_tables_extraction_for_dataset_v1.py`; completion artifacts include `manifest_current.tsv` and `key2txt.tsv`.
- `docs/maintained_script_surface.tsv`: `src/stage1_cleaning/hydrate_manifest_v1.py` is the explicit Stage1 manifest hydration helper.

Sidecar inventories such as `stage1_table_cells_manifest_v1.tsv` are asset indexes only. They may feed hydration but must not become competing corpus manifests.

## Original generation path before this repair

The maintained Stage1 flow is:

```text
raw Zotero records / assembled manifest
  -> src/stage1_cleaning/zotero_raw_to_manifest.py
  -> data/cleaned/index/manifest_current.tsv or assembled manifest
  -> src/stage1_cleaning/hydrate_manifest_v1.py
  -> hydrated manifest_current.tsv
```

Before this repair, `hydrate_manifest_v1.py` hydrated only:

```text
text_path
text_source_type
text_available
table_dir
table_available
dataset_id
split_tag
benchmark_tag
```

It did not hydrate:

```text
structure_path
structure_available
stage1_table_cell_sidecar_path
stage1_table_cell_sidecar_available
```

Therefore Marker/HTML/PDF structure/table-cell improvements could pass diagnostic smokes but remain absent from `manifest_current.tsv`, meaning future baselines would not reliably consume them.

## Current mainline-integration target

Extend `hydrate_manifest_v1.py` so governed Stage1 indexes can be joined into the unique manifest:

```text
data/cleaned/index/key2structure.tsv
  -> structure_path / structure_available

tables_cell_sidecar/stage1_table_cells_manifest_v1.tsv
  -> stage1_table_cell_sidecar_path / stage1_table_cell_sidecar_available
```

A repair is not complete until a maintained Stage2 dryrun derived from the hydrated authority surface shows explicit sidecar status such as:

```text
stage1_structure_sidecar_status = loaded
stage1_cell_sidecar_status = consumed
```

## Rollback plan

Before overwriting `data/cleaned/index/manifest_current.tsv`, create a timestamped backup in the same directory or a governed run/analysis directory, for example:

```text
data/cleaned/index/manifest_current.pre_stage1_sidecar_hydration_YYYYMMDD_HHMMSS.tsv
```

Also record:

- command used to generate the new manifest;
- input manifest path;
- `key2txt.tsv` path;
- `key2structure.tsv` path;
- table-cell sidecar manifest path;
- metadata JSON path;
- diff summary of header and counts.

Rollback command shape:

```bash
cp data/cleaned/index/manifest_current.pre_stage1_sidecar_hydration_YYYYMMDD_HHMMSS.tsv \
   data/cleaned/index/manifest_current.tsv
```

Do not manually edit `manifest_current.tsv` for this operation. Regenerate via `hydrate_manifest_v1.py` and keep metadata.

## Promotion checklist

- [ ] TDD test proves hydration of structure/table-cell fields.
- [ ] `hydrate_manifest_v1.py` accepts explicit governed sidecar index paths.
- [ ] Dryrun hydration writes a candidate manifest and metadata without touching `manifest_current.tsv`.
- [ ] If promoted, backup current `manifest_current.tsv` before overwrite.
- [ ] No-live Stage2 smoke derives from the hydrated manifest, not a synthetic run-local manifest.
- [ ] Progress ledger records status as one of `mainline_integrated`, `maintained_code_only_not_hydrated`, or `diagnostic_only_not_mainline`.
