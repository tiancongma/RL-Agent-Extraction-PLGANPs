# DEV15 LLM Capability & Function-Unit Coverage Audit

## Scope

This is a Stage2-only, diagnostic audit based only on governed existing Stage2 outputs. It is not benchmark-valid final-output reporting.

## Source Resolution

`ACTIVE_RUN.json` was not used as the primary source for this audit because it points to `data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1`, which is the semantic-emitter lineage rather than the maintained `llm_first_composite` Stage2 surface requested here.

Primary audit baseline:

- run dir: `data/results/20260402_5c1e7a4/12_dev15_stage2_livev2_raw_rehydration`
- reason: this is the only governed maintained composite Stage2 run that covers all 15 DEV15 papers in one scope
- exact Stage2 semantic objects: `data/results/20260402_5c1e7a4/12_dev15_stage2_livev2_raw_rehydration/semantic_stage2_objects/semantic_stage2_v2_objects.jsonl`
- exact Stage2 semantic summary: `data/results/20260402_5c1e7a4/12_dev15_stage2_livev2_raw_rehydration/semantic_stage2_objects/semantic_stage2_v2_summary.tsv`
- exact completed Stage2 artifact: `data/results/20260402_5c1e7a4/12_dev15_stage2_livev2_raw_rehydration/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`

Supplemental governed trigger evidence:

- `data/results/20260406_ced19d6/06_doe_fu_wiv_5gif_final`
- `data/results/20260406_ced19d6/07_doe_fu_ufxx_scopefix`
- `data/results/20260406_ced19d6/08_doe_trigger_path_audit/doe_trigger_diagnostics_v1.json`

These April 6 runs were used only to answer Module 7 precisely for the three DOE-sensitive papers because the full DEV15 April 2 run had DOE enumeration disabled.

## Overall Distribution

Overall status counts:

- `fully_actionable`: 10
- `actionable_with_existing_fu`: 2
- `needs_new_function_unit`: 2
- `insufficient_llm_signal`: 0
- `out_of_scope_correctly`: 1

Primary gap counts:

- `none`: 12
- `missing_scope_construction`: 0
- `missing_variable_detection`: 0
- `missing_relation_marking`: 1
- `insufficient_numeric_grounding`: 0
- `missing_evidence_anchor`: 0
- `function_unit_gap`: 1

## Paper Groups

Already workable with current pipeline behavior:

- `5ZXYABSU`
- `L3H2RS2H`
- `7ZS858NS`
- `BB3JUVW7`
- `BXCV5XWB`
- `PA3SPZ28`
- `RHMJWZX8`
- `V99GKZEI`
- `WFDTQ4VX`
- `YGA8VQKU`

Already workable when an existing function unit is enabled and legally triggered:

- `UFXX9WXE`
- `WIVUCMYG`

Clearly requiring a new function-unit type or equivalent structured handler:

- `INMUTV7L`
- `QLYKLPKT`

Insufficient LLM signal:

- none in this DEV15 set

Correctly excluded from DOE-style governed expansion:

- `5GIF3D8W`

## Module-Level Readout

High-confidence strengths observed in the maintained Stage2 outputs:

- Document-level formulation framing is usually strong. All 15 papers were recognized as formulation-relevant, and biology or in vivo material was usually left as secondary context rather than becoming the formulation universe.
- Explicit formulation-row studies perform well. Papers such as `WIVUCMYG`, `YGA8VQKU`, `BB3JUVW7`, and `5ZXYABSU` already produce a usable formulation universe from text-plus-table evidence.
- Existing DOE handling works when the governed preconditions are met. `UFXX9WXE` and `WIVUCMYG` both show legal DOE trigger sufficiency once governed DOE scope is present.

Observed weak areas:

- Numbered variant studies are still fragile when the LLM emits formulation numbers without binding the row-level factor assignments back onto those numbered instances. `INMUTV7L` is the clearest example.
- Non-DOE optimization factor sweeps remain under-covered. `QLYKLPKT` exposes factor ranges and an optimal formulation, but the current Stage2 surfaces do not construct an enumerable universe from that pattern.
- Some papers are actionably present but still label-bound in the completed projection. `7ZS858NS` and `PA3SPZ28` are usable, but some variant identity is carried mainly by raw formulation labels rather than richly normalized structured fields.

## Function Unit Expansion Candidates

Candidate 1: numbered variable-study linker

- pattern: the LLM identifies `Formulation 1 ... Formulation n` and separately identifies varying formulation factors, but does not bind factor assignments to the numbered formulations strongly enough for downstream deterministic use
- supporting papers: `INMUTV7L`
- why existing units cannot handle it: the current DOE unit requires governed DOE scope plus explicit row-enumeration legality, and the compatibility projection cannot safely infer factor-to-row bindings from numbered labels alone

Candidate 2: non-DOE optimization/factor-sweep handler

- pattern: the LLM captures varying optimization factors and a best or optimal formulation, but the paper does not satisfy current governed DOE-scope construction rules and no existing unit turns that factor sweep into a governed formulation universe
- supporting papers: `QLYKLPKT`
- boundary evidence: `5GIF3D8W` shows a weaker version of this pattern, but the April 6 DOE trigger audit indicates it is correctly excluded under the current DOE contract rather than being an existing-unit regression
- why existing units cannot handle it: current DOE expansion is restricted to governed DOE scope with explicit table-enumeration legality, and no other maintained unit covers non-DOE optimization sweeps

## Direct Answer

### Is current LLM output sufficient to proceed to full-corpus LLM execution?

No.

The Stage2 LLM output is strong enough to justify continued targeted execution work, but not strong enough to treat the current maintained system as full-corpus ready without additional structured coverage. The exact blocking categories exposed by DEV15 are:

- `missing_relation_marking` for numbered formulation studies like `INMUTV7L`
- `function_unit_gap` for non-DOE optimization or factor-sweep studies like `QLYKLPKT`

Important non-blocking clarification:

- `5GIF3D8W` is not the reason for the no-go decision. Its DOE-like signal is currently too weak to become governed DOE scope, and the April 6 trigger audit shows that exclusion is consistent with the current governed contract.

## Strict Engineering Conclusion

The maintained Stage2 LLM surface already covers most explicit formulation-row papers in DEV15, and the existing DOE unit is viable when governed scope is present. The current no-go comes from coverage gaps in structured handling of numbered variant studies and non-DOE optimization sweeps, not from a broad collapse of LLM semantic quality.

## Uncertainties

- The memory bootstrap helper could not be executed successfully in this session, so this audit relied directly on governed local artifacts rather than the supporting memory layer.
- This audit intentionally stayed at the existing Stage2 surface plus governed DOE trigger diagnostics. It does not claim benchmark-valid final-output performance because downstream Stage3 and Stage5 were not re-run as part of this task.
