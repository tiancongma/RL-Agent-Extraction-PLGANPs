# WFDTQ4VX Next Action Recommendation

## Chosen action

1. `freeze the newer live/current WFDTQ4VX recovery into a replay-grade raw-response boundary`

## Why this is the single best next action

The dual-lineage audit shows that the decisive divergence is already present at
the raw-response layer:

- older replay raw response:
  - `3` formulation candidates
- newer live/current raw response:
  - batch-level DOE candidate universe visible directly in the raw artifact

The later stages then mostly preserve that difference:

- S2-5:
  - `3` vs `34`
- S2-7:
  - `3` vs `34`
- Stage5:
  - `2` vs `33`

That makes the next step a boundary-governance problem before it is a code
change problem.

## Why the other choices are weaker

### 2. port the newer live/current recovery logic into the maintained replay-consumed path

- weaker as the immediate next action because the audit already proves the
  recovered row universe exists in the newer raw response
- first we need a replay-grade freeze of that successful boundary so the repo
  can prove inheritance across replay

### 3. change S2-5 parsing

- not justified by current evidence
- S2-5 preserves the upstream raw-response difference; it is not the strongest
  proven loss point

### 4. change S2-6 validation

- not justified
- S2-6 passes in both lineages with no WFDTQ4VX-specific suppression evidence

### 5. change S2-7 projection

- not justified as first action
- S2-7 row counts mirror S2-5 row counts in both lineages

### 6. change Stage5

- not justified
- Stage5 only performs a small explicit non-formulation drop in both lineages
- it is not causing the large `2 vs 33` split

### 7. no code change yet; first add paper-specific replay regression infrastructure

- valuable, but secondary
- the repo already has enough evidence to justify first freezing the successful
  raw-response boundary that today’s replay failed to use

## Direct evidence behind the recommendation

- Lineage A raw artifact:
  - `data/results/20260402_5c1e7a4/01_dev15_raw_llm_freeze/frozen_raw_responses/WFDTQ4VX__stage2_v2_raw_response.json`
  - only `3` formulation candidates
- Lineage B raw artifact:
  - `data/results/20260414_0011ee7/32_full_dev15_rebuilt_from_current_stage2_v1/semantic_stage2_objects/raw_responses/WFDTQ4VX__stage2_v2_raw_response.json`
  - batch-level DOE candidates already present
- downstream counts preserve the same split:
  - `3 -> 2` in A
  - `34 -> 33` in B

## Final recommendation

Freeze the newer live/current WFDTQ4VX recovery into a replay-grade raw-response
boundary, then replay from that exact frozen boundary through the maintained
downstream path.

That is the smallest action that directly tests the currently proven boundary
of recovery.
