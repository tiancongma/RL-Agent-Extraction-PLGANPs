# v7pilot_r3_fixparse Input Assembly Audit (2026-03-10)

## Direct answer

No, current implementation is not truly table-first.

## Conclusion

`src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py` sends a generic paper text blob to the model, not a table-first packed evidence bundle. The current table-heavy behavior is prompt-side only: the script detects sweep/table-heavy text patterns and prepends extra row-enumeration instructions, but it does not reorder the underlying evidence so that tables are guaranteed to appear first.

## Real assembly order

The actual order is:

1. load manifest row with `key`, `doi`, `title`, `text_path`
2. read the file at `text_path` as one raw text string
3. apply `--max-chars` as a single front-slice truncation on that raw string
4. build the prompt as:
   - `LLM_PROMPT_TEMPLATE`
   - optional `ENUMERATION_HEAVY_TABLE_HINT` if `is_table_heavy_sweep_candidate(text)` returns `True`
   - the truncated raw text appended after `TEXT:\n`
5. send that final string to `call_gemini(...)`

There is no code path here that:

- extracts tables separately
- serializes tables into a protected block
- places tables before body text
- attaches captions next to tables
- budgets characters per block
- prioritizes methods/results/table regions before the rest of the text

## Code locations supporting this

Primary evidence:

- [auto_extract_weak_labels_v7pilot_r3_fixparse.py](/c:/Users/tianc/Downloads/GitHub/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py#L778)
  `load_manifest(...)` requires `text_path`; no table asset path is loaded.
- [auto_extract_weak_labels_v7pilot_r3_fixparse.py](/c:/Users/tianc/Downloads/GitHub/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py#L950)
  `txt = paper.text_path.read_text(...)`
- [auto_extract_weak_labels_v7pilot_r3_fixparse.py](/c:/Users/tianc/Downloads/GitHub/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py#L951)
  `txt = txt[: args.max_chars]`
- [auto_extract_weak_labels_v7pilot_r3_fixparse.py](/c:/Users/tianc/Downloads/GitHub/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py#L326)
  `build_prompt(text)` starts with `LLM_PROMPT_TEMPLATE`, optionally appends `ENUMERATION_HEAVY_TABLE_HINT`, then appends `text`
- [auto_extract_weak_labels_v7pilot_r3_fixparse.py](/c:/Users/tianc/Downloads/GitHub/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py#L334)
  `is_table_heavy_sweep_candidate(text)` is a detector over the already assembled raw text, not an evidence packer

Prompt-side behavior:

- [auto_extract_weak_labels_v7pilot_r3_fixparse.py](/c:/Users/tianc/Downloads/GitHub/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py#L276)
  `ENUMERATION_HEAVY_TABLE_HINT` forces row-level enumeration before abstraction

## Current prompt-side behavior vs current evidence-packing behavior

Current prompt-side behavior:

- Detects whether the raw paper text looks table-heavy or sweep-heavy
- Adds stricter instructions such as:
  - treat each table row/run as a potential formulation instance
  - enumerate row-by-row before parent/variant abstraction
  - do not collapse rows into family-level summaries
  - do not emit global/shared pseudo-formulations

Current evidence-packing behavior:

- Uses one pre-existing text asset from `text_path`
- Keeps that asset in its original order
- Applies one global character cutoff
- Offers no special protection for tables, captions, or formulation rows

## How `--max-chars` is applied

`--max-chars` is applied as one global truncation after file read and before prompt construction:

- `txt = paper.text_path.read_text(...)`
- `if args.max_chars > 0 and len(txt) > args.max_chars: txt = txt[: args.max_chars]`

This means:

- one global cutoff, not per-block budgeting
- no table protection
- no caption protection
- no methods/results protection
- later parts of the raw text are dropped if they fall beyond the limit

## Practical LLM-visible order for table-heavy papers

For the current implementation, the LLM sees:

- prompt schema/instructions first
- optional table-heavy row-enumeration instructions second
- then whatever order already exists inside the source text file

That practical order depends on the source asset, not on a table-first policy.

Examples:

- `L3H2RS2H` text asset is body-first. In `data/cleaned/content_goren_2025/text/L3H2RS2H.pdf.txt`, `Table 1` first appears around character `14176`, `Table 2` around `15506`, `Table 3` around `21343`, and `Table 4` around `25427`. So the model sees abstract/body discussion first and table content later.
- `WIVUCMYG` text asset happens to list table references very early in `data/cleaned/content_goren_2025/text/WIVUCMYG.html.txt` because the HTML-derived text starts with an outline and table links. That is an artifact of the source text file, not an intentional table-first packing strategy in the extractor.

## Recommendation

A future table-first packing change is still needed if the goal is true table-priority extraction under a character budget. The current prompt-side row-enumeration rule helps, but it does not guarantee that table evidence is surfaced early or protected from truncation. If table rows are meant to drive formulation-instance enumeration reliably, a later change should explicitly assemble tables and nearby captions first, then add selected body text with remaining budget.
