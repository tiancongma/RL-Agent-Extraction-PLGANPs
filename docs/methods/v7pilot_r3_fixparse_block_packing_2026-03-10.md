# v7pilot_r3_fixparse Block-Based Evidence Packing (2026-03-10)

## Old behavior

The extractor previously used:

1. `paper.text_path.read_text(...)`
2. a global front slice: `txt = txt[: args.max_chars]`
3. `build_prompt(txt)`

That meant evidence order was inherited from the raw text file. Tables were not explicitly prioritized, captions were not attached near tables, and body text could consume budget before row-dense formulation evidence.

## New behavior

The extractor now assembles evidence blocks before prompt construction. The packing unit is:

- `table_block`
- `caption_block`
- `paragraph_block`

The implementation is local to `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py` and uses simple heuristics over the existing raw text asset.

## Packing priorities

The packing order is:

1. high-value `table_block`
2. nearby `caption_block`
3. `paragraph_block` containing formulation/run labels, sweep/design language, inheritance language, control/blank/empty variants, or explicit parameter changes
4. methods/results `paragraph_block` defining preparation variables
5. lower-priority narrative `paragraph_block` only if budget remains

The assembled text is emitted with visible markers such as:

- `[METADATA]`
- `[TABLE_BLOCK]`
- `[CAPTION_BLOCK]`
- `[PARAGRAPH_BLOCK]`

This makes the LLM-visible order auditable.

## Why these block types

Paragraphs, captions, and tables are a better fit than sentence slices or whole-section dumps for this pilot because:

- formulation instances are often encoded row-wise in tables
- captions provide local semantic labels for tables
- paragraph-sized blocks preserve nearby explanatory context without flooding the prompt with low-value narrative

## Current heuristic implementation

The stage2 extractor now:

- normalizes raw text into block candidates
- extracts `table_block` candidates from `Table N ...` spans
- splits each table-like span into a `caption_block` plus table body when possible
- derives `paragraph_block` candidates from paragraph-sized text groups
- scores blocks by formulation-instance evidence value
- appends blocks in priority order under `--max-chars`

This replaces raw front truncation as the main assembly strategy.

## Limitations

This is still a heuristic packer over noisy cleaned text files, not a full document-layout parser. Some extracted table blocks still include nearby narrative spillover, especially in PDF-derived text. The current change is intentionally minimal and reversible; if cleaner table isolation is needed later, that should be done as a separate refactor.
