# Stage2 Table-Path Contract Fix Choice

Date: 2026-04-17

## Chosen Fix

Chosen option: `A`

- hydrate `table_dir` when building `targeted_manifest.tsv`

## Why This Choice

### Smallest

- The stale `table_dir` is copied into Stage2 exactly where `targeted_manifest.tsv`
  is written.
- Repairing the run-scoped Stage2 manifest avoids broader Stage1 changes and
  avoids pushing more hidden fallback behavior into the extractor.

### Safest

- `manifest_current.tsv` stays the only authority for row universe and scope.
- No machine-specific path is introduced.
- The repair follows the existing Stage1 table-binding rule:
  `data/cleaned/<dataset_id>/tables/<paper_key>`.
- The extractor can keep its current contract of trusting explicit
  `table_dir` in the manifest it receives.

### Symmetric With The Text-Path Repair

The successful text fix refreshed `text_path` in the Stage2 run-scoped manifest
from the maintained Stage1 authority surface before execution.

This table fix follows the same philosophy:

- refresh execution bindings at Stage2 scope materialization
- keep the generated `targeted_manifest.tsv` truthful
- preserve extractor behavior
- avoid creating a second authority surface

## Plain-Language Contract Statement

Stage2 should still execute from a manifest-driven scope surface, but before
the composite wrapper writes `targeted_manifest.tsv` it should refresh both:

- `text_path` from `key2txt.tsv`
- `table_dir` from the maintained dataset-local tables root for the selected
  row's `dataset_id`

That keeps the run-scoped Stage2 manifest aligned with the current maintained
Stage1 asset surfaces without redesigning Stage1 or Stage2 authority.
