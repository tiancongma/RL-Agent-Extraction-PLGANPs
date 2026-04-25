# Capability Audit

## New capabilities

- Local role-aware full-table selection is now active for some full-table papers.
  - Evidence:
    - baseline `QLYKLPKT`, `UFXX9WXE`, `WFDTQ4VX` used `selector_strategy=sorted_csv_first_4`
    - current `01_s2_2` uses `selector_strategy=role_aware_selector_v1` for those papers
  - Effect:
    - prompts are now built from curated evidence packs instead of raw-prefix-plus-first-four tables for those cases

- Sequential-optimization local evidence packing now exists.
  - Evidence:
    - `QLYKLPKT` now has selector-side local optimization logic and evidence-pack prompts
  - Effect:
    - the system can intentionally keep optimization tables plus bridge text instead of relying on broad document fallback

- Trimmed context fallback now exists.
  - Effect:
    - front matter and some assay-heavy context can be removed before prompt assembly
    - prompt construction is less dependent on a raw text prefix

- Evidence-pack prompt construction is now activated by selector strategy, not only by explicit ordered-packing mode.
  - Effect:
    - current `QLYKLPKT`, `UFXX9WXE`, and `WFDTQ4VX` use `ordered_controlled_evidence_pack` even with `input_packing_mode=off`

- Fresh-live stepwise restart path now exists and is governed.
  - Evidence:
    - current lineage has explicit `01_s2_2`, `02_s2_3`, `03_s2_4`, `30_s2_5`, `31_s2_6`, `32_s2_7`, `28_compare`
  - Effect:
    - the repo can localize failures by sub-boundary instead of rerunning the whole composite stage

## Regressed capabilities

- WFDTQ4VX restored table-recovery behavior regressed badly in downstream outputs.
  - Baseline final rows: `27`
  - Current final rows: `2`
  - Interpretation:
    - the current selector/prompt path no longer preserves the baseline WFDTQ4VX capability in a downstream-stable way

- Whole-pipeline completion capability regressed because fresh live completion is incomplete.
  - Baseline:
    - completed Stage2, Stage3, and Stage5 for 14 papers in the final table
  - Current:
    - only 10 papers reached completed Stage2
    - final table contains only 10 papers
  - Missing entirely in current final table:
    - `5GIF3D8W`
    - `BXCV5XWB`
    - `PA3SPZ28`
    - `YGA8VQKU`

- Stable bounded-output behavior regressed on some papers.
  - `UFXX9WXE`: `1 -> 17` final rows
  - `WIVUCMYG`: `6 -> 23`
  - These are not simple coverage recoveries; they indicate formulation-boundary inflation or different row-materialization behavior downstream of current Stage2 outputs.

- V99GKZEI did not inherit the new safer prompt path.
  - It still uses `selector_strategy=sorted_csv_first_4`
  - Its prompt layout remains `raw_prefix_then_table_excerpts`
  - Prompt preview records truncation at `103485` chars
  - So the new selector improvements are not generalized across top-risk cases

## Mixed/conditional capabilities

- UFXX9WXE DOE capability is improved but still incomplete.
  - Positive:
    - current evidence selection is cleaner than baseline
    - current final rows improved from `1` to `17`
  - Negative:
    - GT target is `27`
    - this is still under by `10`
  - Assessment:
    - partially restored, not baseline-equivalent

- QLYKLPKT sequential optimization capability is improved only modestly.
  - Positive:
    - final rows improved from `3` to `4`
    - dedicated local optimization pack capability now exists
  - Negative:
    - current evidence still contains assay/noisy table leakage
    - GT target remains `7`
  - Assessment:
    - capability exists, but selector precision is still weak

- Prompt size control improved structurally but not enough operationally.
  - Positive:
    - evidence-pack prompts replaced raw-prefix layout for several papers
    - trimmed context fallback exists
  - Negative:
    - `QLYKLPKT`, `UFXX9WXE`, and `WFDTQ4VX` prompt lengths remain large in the current preview
    - fresh live call still completed only `9/15` at the first restart pass
  - Assessment:
    - prompt-control capability improved, but runtime robustness remains conditional

## Bottom line

- New ability:
  - targeted local evidence selection and evidence-pack prompt assembly
- Main regressions:
  - incomplete fresh-live completion
  - loss of WFDTQ4VX downstream preservation
  - strong row-count inflation on some papers
- Overall status:
  - capability set is broader than baseline, but baseline stability was not preserved

## Caveat

- This audit is diagnostic-only.
- Findings should be peer-reviewed before major action.
