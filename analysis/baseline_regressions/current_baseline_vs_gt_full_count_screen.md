# Current Baseline vs GT Full Count Screen

## Executive summary

Current replay baseline inspected:

- `data/results/20260421_c8f4b61`

Reference older diagnostic baseline:

- `data/results/20260421_43ed145`

Count headline:

- current replay final rows: `117`
- GT rows: `210`
- matched papers: `1 / 15`
- mismatched papers: `14 / 15`

Biggest replay-era improvements versus the older diagnostic baseline:

- `UFXX9WXE`: `2 -> 28` (`+26`)
- `WIVUCMYG`: `3 -> 29` (`+26`)
- `YGA8VQKU`: `3 -> 19` (`+16`)
- `5GIF3D8W`: `1 -> 11` (`+10`)
- `PA3SPZ28`: `1 -> 4` (`+3`)

Highest remaining undercounts:

- `WFDTQ4VX`: `3 / 30` (`-27`)
- `L3H2RS2H`: `5 / 21` (`-16`)
- `5GIF3D8W`: `11 / 26` (`-15`)
- `BB3JUVW7`: `3 / 12` (`-9`)
- `INMUTV7L`: `3 / 12` (`-9`)

Short screening view:

- strongest additional pre-LLM screening candidates from counts:
  `WFDTQ4VX`, `L3H2RS2H`, `BB3JUVW7`, `INMUTV7L`, `BXCV5XWB`
- proven pre-LLM anchor already established:
  `5ZXYABSU`
- strongest downstream / identity / compare-pattern candidates from counts:
  `UFXX9WXE`, `WIVUCMYG`, `RHMJWZX8`, `YGA8VQKU`

Important scope note:

- `prior_localization_if_any` means a boundary/failure family was already
  proven in prior audits.
- `screening_label` is count-based only unless a prior localization is
  explicitly named.

## Full All-Paper Table

| paper_key | final_rows | gt_rows | delta_vs_gt | status_vs_gt | prior_localization_if_any | screening_label | notes |
|---|---:|---:|---:|---|---|---|---|
| WFDTQ4VX | 3 | 30 | -27 | under |  | likely pre-LLM candidate | Large unchanged undercount after downstream replay improvements; good screening candidate for S2-2b/S2-3/S2-4a review. |
| L3H2RS2H | 5 | 21 | -16 | under |  | likely pre-LLM candidate | Large unchanged undercount after downstream replay improvements; good screening candidate for S2-2b/S2-3/S2-4a review. |
| 5GIF3D8W | 11 | 26 | -15 | under | proven downstream: non-DOE single-variable recovery family / post-reopen row-emission family | likely downstream candidate | Large replay improvement but still under GT; proven post-reopen/non-DOE recovery family rather than pre-LLM. |
| BB3JUVW7 | 3 | 12 | -9 | under |  | likely pre-LLM candidate | Large unchanged undercount after downstream replay improvements; good screening candidate for S2-2b/S2-3/S2-4a review. |
| INMUTV7L | 3 | 12 | -9 | under |  | likely pre-LLM candidate | Large unchanged undercount after downstream replay improvements; good screening candidate for S2-2b/S2-3/S2-4a review. |
| 5ZXYABSU | 1 | 9 | -8 | under | proven pre-LLM: S2-2b table-loss/preservation failure (Table 1/2 dropped) | likely pre-LLM candidate | Proven S2-2b preservation loss; unchanged vs prior baseline. |
| BXCV5XWB | 1 | 9 | -8 | under |  | likely pre-LLM candidate | Large unchanged undercount after downstream replay improvements; good screening candidate for S2-2b/S2-3/S2-4a review. |
| QLYKLPKT | 2 | 7 | -5 | under |  | unclear | Remaining undercount is moderate/small; count pattern alone does not isolate pre-LLM vs downstream. |
| V99GKZEI | 4 | 6 | -2 | under |  | unclear | Improved undercount suggests some downstream gains, but remaining gap is not localized from counts alone. |
| PA3SPZ28 | 4 | 5 | -1 | under |  | unclear | Improved undercount suggests some downstream gains, but remaining gap is not localized from counts alone. |
| 7ZS858NS | 1 | 1 | 0 | match |  | unclear | Match or weak signal; no pre-LLM screening priority from counts alone. |
| RHMJWZX8 | 3 | 2 | 1 | over |  | likely downstream candidate | Over GT on final counts; count pattern points away from pre-LLM loss and toward downstream retention/identity/compare review. |
| UFXX9WXE | 28 | 27 | 1 | over | proven downstream: post-reopen row-emission family; recovered rows survive replay | likely downstream candidate | Proven post-reopen downstream family; current issue is not a pre-LLM screening target. |
| YGA8VQKU | 19 | 17 | 2 | over |  | likely downstream candidate | Over GT on final counts; count pattern points away from pre-LLM loss and toward downstream retention/identity/compare review. |
| WIVUCMYG | 29 | 26 | 3 | over | proven downstream: DOE/post-reopen row-emission family; recovered rows survive replay | likely downstream candidate | Proven post-reopen downstream family; current issue is not a pre-LLM screening target. |

## Highest Remaining Undercount

The highest remaining undercounts are:

- `WFDTQ4VX`: `-27`
- `L3H2RS2H`: `-16`
- `5GIF3D8W`: `-15`
- `BB3JUVW7`: `-9`
- `INMUTV7L`: `-9`
- `5ZXYABSU`: `-8`
- `BXCV5XWB`: `-8`

For screening purposes:

- `WFDTQ4VX`, `L3H2RS2H`, `BB3JUVW7`, `INMUTV7L`, and `BXCV5XWB` are the best
  count-based candidates for additional pre-LLM review because they stayed
  substantially under GT and did not benefit from the recent downstream replay
  repairs.
- `5GIF3D8W` remains heavily under GT, but that paper already has a proven
  downstream recovery-family localization, so it should not be treated as a
  fresh pre-LLM screening candidate from counts alone.

## Likely Pre-LLM Candidates

Proven:

- `5ZXYABSU`
  - proven `S2-2b` preservation failure-family anchor
  - formulation-bearing `Table 1` and `Table 2` survived `S2-2a` and were lost
    at `S2-2b`

Count-based suspicion only:

- `WFDTQ4VX`
- `L3H2RS2H`
- `BB3JUVW7`
- `INMUTV7L`
- `BXCV5XWB`

Reason for suspicion:

- substantial remaining undercount
- no meaningful replay-era gain from the recent downstream deterministic
  repairs
- count shape is therefore compatible with unresolved pre-LLM loss, but not
  yet proven

## Likely Downstream Candidates

Proven:

- `UFXX9WXE`
  - proven downstream post-reopen row-emission family
  - replay moved it from severe undercount to slight overcount
- `WIVUCMYG`
  - proven downstream DOE/post-reopen row-emission family
  - replay moved it from severe undercount to overcount
- `5GIF3D8W`
  - proven downstream non-DOE single-variable recovery family
  - replay improved counts materially, but a large downstream gap remains

Count-based suspicion only:

- `RHMJWZX8`
- `YGA8VQKU`

Reason for suspicion:

- both are now over GT on final counts
- overcount patterns point away from pre-LLM loss and more toward downstream
  retention, identity, projection, or compare interpretation review

## Proven Localization vs Count-Based Suspicion

Proven localization already exists for:

- `5ZXYABSU`
- `5GIF3D8W`
- `UFXX9WXE`
- `WIVUCMYG`

For all other papers in this screen:

- the label is a count-based screening label only
- it is not a claim that the first failure boundary is known
- further localization would still need artifact-backed tracing before naming
  `S2-2b`, `S2-4a`, `S2-7`, `Stage5`, or compare as the true first blocker
