# WFDTQ4VX Fix Proof Boundary

## Chosen boundary

- `raw response`

## Why this is the strongest proven boundary

The repo directly proves that the two lineages already differ at the raw
response layer:

- Lineage A raw response:
  - `data/results/20260402_5c1e7a4/01_dev15_raw_llm_freeze/frozen_raw_responses/WFDTQ4VX__stage2_v2_raw_response.json`
  - contains only `3` formulation candidates
- Lineage B raw response:
  - `data/results/20260414_0011ee7/32_full_dev15_rebuilt_from_current_stage2_v1/semantic_stage2_objects/raw_responses/WFDTQ4VX__stage2_v2_raw_response.json`
  - textually contains batch-level DOE candidates from `F_T2_B1` onward and a
    much larger formulation universe

That is stronger than saying only:

- `S2-5 semantic objects recovered`
- or `Stage5 final rows recovered`

because the row-universe split is already visible before later deterministic
stages run.

## What exactly is proven

Proven:

- the newer live/current recovery lineage already has the richer WFDTQ4VX row
  universe in its raw-response artifact
- the older replay lineage does not
- S2-5, S2-7, Stage3, and Stage5 mostly preserve that upstream difference

## What is not proven

Not proven:

- that the newer recovery was frozen into a replay-grade raw-response authority
  suitable for later operational replay
- that the newer recovery is benchmark-verified
- that every live/current WFDTQ4VX raw-response success automatically yields the
  same downstream final count in other lineages

## Why today's replay does not inherit that proof automatically

Today's replay started from the older raw-response authority:

- `data/results/20260402_5c1e7a4/01_dev15_raw_llm_freeze/frozen_raw_responses/WFDTQ4VX__stage2_v2_raw_response.json`

The stronger recovery proof belongs to a different raw-response artifact:

- `data/results/20260414_0011ee7/32_full_dev15_rebuilt_from_current_stage2_v1/semantic_stage2_objects/raw_responses/WFDTQ4VX__stage2_v2_raw_response.json`

Those are different governed boundary artifacts.

So today's replay cannot inherit the stronger recovery automatically because it
did not consume the same raw-response boundary.

## Final statement

The strongest repo-proven WFDTQ4VX recovery boundary is:

- `raw response`

More precise wording:

- the repo proves a richer WFDTQ4VX candidate universe exists in the newer
  live/current raw-response artifact
- it does **not** prove that this richer raw-response boundary was frozen and
  adopted as the replay-grade operational source used today
