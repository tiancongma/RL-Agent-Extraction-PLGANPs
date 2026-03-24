# mem_v1 audit

## Scope

This audit reviewed the governed `data/mem/v1/` supporting layer for retrieval
quality, schema consistency, row usability, and default Codex workflow fit.

Audit queries:

- `collapse`
- `BB3JUVW7`
- `family variant`
- `DOE`
- `table-first`
- `identity mismatch`
- `stage2 parsing`
- `run lineage`

## Current strengths

- Flat memory layout stays within governance constraints.
- Schema and ID structure validate cleanly with `check_mem_v1.py`.
- `DOE`, `identity mismatch`, `BB3JUVW7`, and `run lineage` now return useful
  decisions, lineage rows, and recurring failure patterns.
- Workflow prompt recipes in `prm.tsv` give a governed way to bootstrap complex
  debugging and comparison tasks.

## Weaknesses found

- `err.tsv` previously contained many repeated run-context fragments for the
  same collapse or regression symptoms.
- `dec.tsv` previously admitted weak rows such as section scaffolding and
  non-decision fragments.
- Some `run.tsv` summaries fell back to `RUN_CONTEXT` headings because the
  builder was not reading `run_purpose:` bullets.
- Prompt retrieval could surface generic sections like `Conclusion` or
  `Output summary` instead of reusable workflow guidance.
- `stage2 parsing` remains the weakest tested query because the source corpus
  still contains more regression prose than compact canonical parsing recipes.

## Changes made

- Tightened decision extraction to reject weak section scaffolding.
- Added canonical workflow prompt recipes for debugging, regression
  investigation, run comparison, pipeline modification, GT mismatch analysis,
  and lineage tracing.
- Merged clearly repeated error fragments into more canonical error rows.
- Parsed flat `run_purpose:` and `run_type:` bullets from `RUN_CONTEXT.md`.
- Filtered low-signal prompt rows with generic titles.
- Tightened query ranking and added a workflow helper:
  `src/utils/mem_bootstrap_v1.py`.

## Remaining risks

- `stage2 parsing` still benefits from manual source reading after memory
  lookup because the governed markdown corpus does not yet contain many compact
  parsing-specific decision summaries.
- Some retrieved rows remain near-duplicates when multiple child runs document
  the same issue from slightly different angles.

## Recommended Codex pattern

For complex tasks:

1. Identify the task class.
2. Run `python src/utils/mem_bootstrap_v1.py --query "..."`
   or `python src/utils/query_mem_v1.py --query "..."`.
3. Read the top memory-linked files.
4. Then inspect local source code and artifacts.
5. Only then modify code or rerun workflows.

This keeps memory as a supporting layer and avoids treating it as a new
pipeline stage.
