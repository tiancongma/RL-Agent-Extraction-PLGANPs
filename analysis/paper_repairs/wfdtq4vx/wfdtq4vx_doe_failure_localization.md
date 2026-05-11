# WFDTQ4VX DOE Failure Localization

## Executive conclusion

First failing boundary: `LOST_AT_S2_2B`

`WFDTQ4VX` does contain a row-bearing DOE design surface upstream, and that DOE surface survives `S2-2a` candidate segmentation. The first practical failure happens at `S2-2b`, where every DOE-like table candidate is hard-dropped as `hard_drop_table_noise`, no formulation-surface floor is added, and only a malformed downstream `Table 8/9` surface is preserved into the pre-LLM evidence pack. After that point, DOE expansion is already impossible in practice because the prompt, raw LLM response, `S2-5` semantic objects, and `S2-7` execution all inherit the wrong table family.

One-sentence root cause:

`WFDTQ4VX` loses its actual DOE design table at `S2-2b` selector/evidence preservation, where preserved DOE candidates such as the recovered `Table 1/2` design-layout table are irreversibly replaced by a weak downstream `Table 8/9` surface.

## DOE table presence check

Upstream DOE table presence: yes.

Artifact-backed evidence:

- [WFDTQ4VX__table_12__pdf_table.csv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/cleaned/goren_2025/tables/WFDTQ4VX/WFDTQ4VX__table_12__pdf_table.csv) contains:
  - `Table 1. Factorial design parameters and experimental conditions.`
  - `Table 2. Full factorial design layout of lopinavir loaded NPs...`
  - explicit row-bearing design entries with `Sr. No.` and rows `1` through `27`
  - coded factors `X1`, `X2`, `X3` and responses `EE` and `PS`
- This is sufficient DOE row-bearing material in principle.

## S2-2a / S2-2b status

### S2-2a

DOE table survives `S2-2a`: yes.

Key candidate surfaces:

- `WFDTQ4VX__candidate_table__08`
  - source: `data/cleaned/goren_2025/tables/WFDTQ4VX/WFDTQ4VX__table_12__pdf_table.csv`
  - role hint: `design matrix`
  - title/caption recovery includes `Table 1. Factorial design parameters and experimental conditions.`
  - `selector_readiness: ready`
  - evidence: [candidate_segmentation_debug_v1.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_3579206/09_selector_contract_dev15_prellm/analysis/candidate_segmentation_debug_v1.tsv)
- Additional DOE-like candidates also survive `S2-2a`:
  - `WFDTQ4VX__candidate_table__02` from `WFDTQ4VX__table_15__pdf_table.csv`
  - `WFDTQ4VX__candidate_table__06` from `WFDTQ4VX__table_14__pdf_table.csv`

### S2-2b

DOE table survives `S2-2b`: no.

Artifact-backed evidence:

- In [evidence_blocks_v1.json](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_3579206/09_selector_contract_dev15_prellm/semantic_stage2_objects/evidence_blocks/WFDTQ4VX/evidence_blocks_v1.json):
  - `WFDTQ4VX__candidate_table__02` -> `hard_drop_table_noise`
  - `WFDTQ4VX__candidate_table__06` -> `hard_drop_table_noise`
  - `WFDTQ4VX__candidate_table__08` -> `hard_drop_table_noise`
  - `minimal_evidence_floor_applied = no`
  - `floor_added_formulation_surface = no`
- Only one table evidence block is preserved:
  - `WFDTQ4VX__table__01`
  - selected from `WFDTQ4VX__candidate_table__07`
  - normalized payload path resolves to `WFDTQ4VX__table_06__pdf_table__normalized.csv`
  - rendered table label becomes `Table 8`

This is the first boundary where DOE recovery becomes impossible in practice.

## S2-4a prompt sufficiency

Prompt contains a DOE table: no.

Artifact-backed evidence:

- [s2_4a_prompt_audit_v1.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_43ed145/01_s2_4a/analysis/s2_4a_prompt_audit_v1.tsv) shows the prompt for `WFDTQ4VX` is built from the `S2-2b` evidence artifact.
- [s2_4a_prompts_v1.jsonl](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_43ed145/01_s2_4a/analysis/s2_4a_prompts_v1.jsonl) shows the only table summary included is:
  - `data/cleaned/goren_2025/tables/WFDTQ4VX/WFDTQ4VX__table_06__pdf_table.csv`
  - caption-recovered as `Table 8 ... Table 9 ...`
- The prompt still contains supporting narrative about `Twenty seven batches ... 3^3 factorial design`, but the DOE design table itself is absent.

This is a downstream consequence of the earlier `S2-2b` loss, not the first failure.

## S2-4b semantic recognition

LLM DOE recognition: partial but mis-scoped.

Artifact-backed evidence from [WFDTQ4VX__stage2_v2_raw_response.json](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_43ed145/02_s2_4b/raw_responses/WFDTQ4VX__stage2_v2_raw_response.json):

- `formulation_candidates` includes:
  - `Lopinavir-PLGA NPs family (DOE)`
  - `core_change_hint: Varied drug concentration, polymer concentration, and surfactant concentration in a 3x3 factorial design`
- But `table_scopes` includes only:
  - `Table 8` as `downstream_variant_table`, `is_doe=false`
  - `Table 9` as `downstream_variant_table`, `is_doe=false`

So the LLM recognizes paper-level DOE semantics from text, but the DOE table itself is not present in the evidence pack it receives.

## S2-5 semantic structure

`S2-5` preserves the same mis-scoped structure.

Artifact-backed evidence from [semantic_stage2_v2_objects.jsonl](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_c8f4b61/01_s2_5/semantic_stage2_objects/semantic_stage2_v2_objects.jsonl):

- `table_scopes` count is `2`
- scopes are only:
  - `Table 8`, `downstream_variant_table`, `is_doe=false`, `is_formulation_bearing=false`
  - `Table 9`, `downstream_variant_table`, `is_doe=false`, `is_formulation_bearing=false`
- The semantic object still carries a DOE family candidate, but no DOE table scope survives.

This is downstream inheritance of the `S2-2b` table loss.

## S2-7 DOE execution trace

DOE expansion attempted: no.

Artifact-backed evidence from [execution_ledger_v2.tsv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_c8f4b61/03_s2_7/analysis/execution_ledger_v2.tsv):

- `WFDTQ4VX` has only:
  - `Table 8`, `table_type=sequential_child`, `was_unit_authorized=no`, `was_unit_called=no`, `skip_reason=not_formulation_table`
  - `Table 9`, `table_type=sequential_child`, `was_unit_authorized=no`, `was_unit_called=no`, `skip_reason=not_formulation_table`

There is no surviving DOE table scope to route into the DOE emitter.

## Final classification

`LOST_AT_S2_2B`

## FACTS

- Upstream cleaned table asset [WFDTQ4VX__table_12__pdf_table.csv](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/cleaned/goren_2025/tables/WFDTQ4VX/WFDTQ4VX__table_12__pdf_table.csv) contains explicit DOE design rows `1..27`.
- `S2-2a` candidate segmentation preserves multiple DOE-like table candidates, including `WFDTQ4VX__candidate_table__08`.
- `S2-2b` evidence preservation marks DOE-like candidates `WFDTQ4VX__candidate_table__02`, `__06`, and `__08` as `hard_drop_table_noise`.
- `S2-2b` preserves only `WFDTQ4VX__candidate_table__07`, which corresponds to the malformed downstream `Table 8/9` surface.
- The frozen `S2-4a` prompt contains only that downstream table summary.
- The frozen `S2-4b` raw response contains DOE family semantics but only `Table 8` and `Table 9` as non-DOE table scopes.
- `S2-7` never attempts DOE expansion for `WFDTQ4VX`.

## INFERENCES

- Because a row-bearing DOE design table is available upstream and explicitly candidateized, DOE recovery would still be possible in principle before `S2-2b`.
- Once `S2-2b` drops all DOE-like table candidates and preserves only the downstream `Table 8/9` surface, the later LLM and deterministic stages no longer have a lawful DOE table target.

## UNCERTAINTIES

- The selector debug artifact does not explain why these DOE-like candidates were judged `hard_drop_table_noise`; it records the drop reason label but not a richer justification for this paper.
- `WFDTQ4VX__table_12__pdf_table.csv` is noisy and merged, but it still clearly contains DOE design rows; this audit does not quantify whether additional Stage1 repair would be beneficial, only that the first practical loss is later at `S2-2b`.
