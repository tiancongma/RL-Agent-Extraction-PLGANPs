# weak_labels_v7pilot_r3_fixparse note (2026-03-06)

- `fixflat` addressed JSON -> TSV loss when `fields` arrived as a list and were dropped by dict-only flattening.
- `fixparse` addressed parser/object-construction loss where `value` stayed empty even when the raw response carried explicit value signals in alternate keys or strict `value_text` patterns (`la_ga_ratio`, `plga_mw_kDa`).
- Pilot runs now persist raw model responses under each run folder at `raw_responses/` for direct layer-by-layer diagnostics.
