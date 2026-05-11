# weak_labels_v7pilot_r3_fixflat note (2026-03-06)

- Root cause was in pilot flattening (`auto_extract_weak_labels_v7pilot_r3.py`), not prompt semantics.
- For DOI `10.2147/ijn.s130908`, the LLM returned `formulation.fields` as a list of field objects.
- The previous flattening path accepted dict-only fields and replaced list-style payloads with `{}`, which dropped values/scopes/membership/evidence into blank or `unknown` TSV cells.
- Bugfix was applied in pilot-only script `auto_extract_weak_labels_v7pilot_r3_fixflat.py` before further prompt evaluation.
