# 1. Executive summary

The current repo state still has important unresolved problems across architecture, benchmarking, identity binding, audit usability, and modeling readiness. The most serious issue is an architecture/runtime mismatch: the governed architecture now freezes Stage2 authority to an LLM semantic-discovery boundary, but the current machine-readable active run and parts of the human-facing docs still resolve to a deterministic semantic-emitter lineage. The active DEV15 semantic run also shows significant row-identity drift once identity-bearing variables are preserved, and the benchmark comparison workflow is not fully governed because the maintained compare entrypoint could not consume the current GT workbook shape and a read-only custom evaluator was used instead. Downstream, Layer2 binding remains only partially generalized, Stage5 closure rules remain intentionally narrow, Layer3 is still fragmented, evidence binding is still coarse in key places, and normalization / chemical-entity modeling surfaces remain incomplete.

Evidence used for this audit came from governed docs, governed memory, and the current repository authority pointer in [data/results/ACTIVE_RUN.json](../../data/results/ACTIVE_RUN.json). The active lineage inspected was [run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1](../../data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1), and no child `RUN_CONTEXT.md` files were present under that lineage root.

# 2. Resolved vs unresolved boundary

Closed or materially decided:

- The Stage2 architecture split is now explicitly frozen again: LLM for open semantic discovery, deterministic logic for downstream relation resolution, normalization, filtering, audit, and materialization. Evidence: [project/4_DECISIONS_LOG.md](../../project/4_DECISIONS_LOG.md#L2274), [data/mem/v1/dec.tsv](../../data/mem/v1/dec.tsv#L31).
- Stage2.5 remains retired / archived, not active mainline. Evidence: [project/2_ARCHITECTURE.md](../../project/2_ARCHITECTURE.md), [project/PIPELINE_SCRIPT_MAP.md](../../project/PIPELINE_SCRIPT_MAP.md).
- Identity freeze is no longer just advisory; it is an enforced Stage5 gate. Evidence: [data/mem/v1/dec.tsv](../../data/mem/v1/dec.tsv#L30).
- Identity-variable preservation itself is implemented. Evidence: [data/mem/v1/dec.tsv](../../data/mem/v1/dec.tsv#L27).

Still unresolved:

- The frozen Stage2 contract is not yet matched by a compliant restored LLM Stage2 runtime entrypoint in `src/`.
- The repository-default active run still points to a deterministic semantic-emitter lineage.
- The active semantic DEV15 run changes row identity surfaces enough to require manual caution and custom comparison logic.
- Layer2/Layer3 audit and comparison surfaces remain only partially generalized and partially unified.
- Several open-problem rows in governed memory remain active, and at least some now conflict with newer active-run evidence rather than being cleanly narrowed.

# 3. Unresolved issues table

| ID | Title | Category | Current status | Evidence file(s) | Why it remains unresolved | Impact if ignored | Suggested next action | Priority | Blocker flags |
|---|---|---|---|---|---|---|---|---|---|
| U1 | Frozen LLM Stage2 authority is not yet reflected in runtime authority everywhere | architecture / governance | unresolved with direct doc-memory-run conflict | [project/2_ARCHITECTURE.md](../../project/2_ARCHITECTURE.md#L55), [project/ACTIVE_PIPELINE_FLOW.md](../../project/ACTIVE_PIPELINE_FLOW.md#L219), [README.md](../../README.md#L115), [data/results/ACTIVE_RUN.json](../../data/results/ACTIVE_RUN.json#L2), [data/mem/v1/dec.tsv](../../data/mem/v1/dec.tsv#L31) | The contract was frozen, but the active run and parts of the repo surface still resolve to deterministic semantic-emitter authority, and the flow doc explicitly says a compliant restored LLM Stage2 entrypoint is “not yet re-established”. | Future work can silently keep using the wrong Stage2 authority boundary. | Decide whether to restore a compliant LLM Stage2 entrypoint or temporarily demote `ACTIVE_RUN.json` from benchmark authority until that exists; then align README and memory rows. | critical | architecture blocker; benchmark blocker; scaling blocker |
| U2 | Maintained GT comparison entrypoint is incompatible with the current GT workbook authority | benchmark / workflow contract | unresolved | [data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/RUN_CONTEXT.md](../../data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/RUN_CONTEXT.md#L46), [data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/RUN_CONTEXT.md](../../data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/RUN_CONTEXT.md#L94) | The maintained compare script expects a `review_formulations` worksheet, but the explicit GT authority for the active experiment was `value_gt_annotation_workbook_representation_repaired_v4.xlsx`, so the run fell back to a custom read-only evaluator. | Benchmark comparisons are not fully governed and reproducible through one maintained comparison surface. | Add or restore a maintained comparison path that can consume the current GT authority workbook shape without ad hoc logic. | critical | benchmark blocker; audit-usability blocker |
| U3 | Identity-variable preservation changes row identity surfaces in benchmark-significant ways | Stage2 / Stage3 / Stage5 contract gap | unresolved, implemented under evaluation | [data/mem/v1/dec.tsv](../../data/mem/v1/dec.tsv#L27), [data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/RUN_CONTEXT.md](../../data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/RUN_CONTEXT.md#L96), [changed_papers_detailed_report.md](../../data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/audit_identity_variable_diff/changed_papers_detailed_report.md#L12) | The preservation carrier works technically, but the active DEV15 experiment shows major row-id drift, split/merge effects, and ambiguous mapping on multiple papers. | The project can over-split or rename formulations while believing it preserved identity better. | Freeze this path as experimental only until paper-by-paper identity effects are reconciled against Layer2 and GT row binding. | critical | benchmark blocker; scaling blocker |
| U4 | Layer2 formulation-row binding is still only partially generalized | Layer2 boundary integrity | unresolved | [binding_priority_proposal.md](../../data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/audit_binding_risk/binding_priority_proposal.md#L13), [binding_priority_proposal.md](../../data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/audit_binding_risk/binding_priority_proposal.md#L134), [data/mem/v1/dec.tsv](../../data/mem/v1/dec.tsv#L28), [data/mem/v1/dec.tsv](../../data/mem/v1/dec.tsv#L30) | The proposal says the maintained compare path is count-only and several papers still need Layer2-style identity repair even after article-native and namespaced binding. MDEC087 also says initial validation scope is only WIVUCMYG and 5ZXYABSU. | Value-level and identity-level comparisons can remain untrustworthy for several DEV15 papers. | Promote a maintained, benchmark-safe binding ladder from the proposal into a deterministic Layer2 helper, then extend beyond the initial validation papers. | critical | benchmark blocker; scaling blocker; audit-usability blocker |
| U5 | Stage5 closure remains intentionally narrow and does not yet cover broader identity closure cases | Stage5 contract gap | unresolved by design | [final_output_summary_v1.md](../../data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/final_output_summary_v1.md#L80), [final_output_summary_v1.md](../../data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/final_output_summary_v1.md#L84) | The active summary explicitly says broader core-signature fields remain unresolved and DOE-aware coordinate closure still needs a later explicit contract beyond the narrow WFDTQ4VX rule. | Final-table closure may stay paper-specific and fragile outside validated special cases. | Define the next explicit Stage5 closure contract before broadening benchmark claims beyond currently validated narrow rules. | high | benchmark blocker; scaling blocker |
| U6 | Layer3 remains formulation-centered in intent but fragmented in implementation | Layer3 audit usability | unresolved | [docs/methods/layer3_field_gt_protocol_v1.md](../methods/layer3_field_gt_protocol_v1.md#L37), [data/mem/v1/dec.tsv](../../data/mem/v1/dec.tsv#L22) | The method spec and decision memory both say the current capability is only partially present and not yet unified into one formulation-centered audit system contract. | Reviewer workflows remain harder to trust and harder to scale. | Consolidate the existing Layer3 pieces into one governed audit entry contract instead of more ad hoc exports. | high | audit-usability blocker; scaling blocker |
| U7 | Evidence binding and evidence ownership are still too coarse in important places | evidence binding / audit integrity | unresolved | [project/5_PARKING_LOT.md](../../project/5_PARKING_LOT.md), [docs/methods/stage2_llm_field_audit_and_db_redesign_2026-03-25.md](../methods/stage2_llm_field_audit_and_db_redesign_2026-03-25.md), [docs/methods/layer3_field_gt_protocol_v1.md](../methods/layer3_field_gt_protocol_v1.md#L108), [docs/methods/layer3_field_gt_protocol_v1.md](../methods/layer3_field_gt_protocol_v1.md#L590) | Evidence alignment granularity is still deferred; Stage2 still emits coarse evidence metadata; Layer3 still relies on `unresolved_table`, missing anchors, and non-local-support guards. | Audit surfaces can stay reviewer-hostile and field support can remain ambiguous. | Build the deterministic evidence-binding layer that the Stage2 field audit recommends, starting from high-risk fields and table-local anchors. | high | audit-usability blocker |
| U8 | Final outputs are still not fully normalized or model-ready | normalization / modeling readiness | unresolved | [data/mem/v1/dec.tsv](../../data/mem/v1/dec.tsv#L24), [docs/methods/layer3_field_gt_protocol_v1.md](../methods/layer3_field_gt_protocol_v1.md#L457), [docs/methods/layer3_field_gt_protocol_v1.md](../methods/layer3_field_gt_protocol_v1.md#L459) | Memory explicitly says Stage5 is not yet a fully normalized modeling surface, and the Layer3 protocol still lists unresolved concentration and phase-denominator ambiguities. | Downstream ML or database use can silently encode inconsistent units and meanings. | Prioritize a deterministic normalization layer with explicit rule IDs before any broader modeling export claims. | high | modeling-readiness blocker; scaling blocker |
| U9 | Chemical-entity representation is still too weak for modeling use | chemical registry / modeling readiness | unresolved | [data/mem/v1/dec.tsv](../../data/mem/v1/dec.tsv#L24) | MDEC083 says `drug_name` alone is insufficient and calls for a registry with full names, abbreviations, CAS, and physicochemical properties. No governed implementation exists yet. | EE modeling and chemistry-aware analysis will remain underpowered and error-prone. | Define the minimal chemical-entity registry contract before building more model-facing exports. | medium | modeling-readiness blocker |
| U10 | Schema-extensible variable handling is only partially resolved and the governed story is inconsistent | identity-variable handling | unresolved with active conflict | [data/mem/v1/dec.tsv](../../data/mem/v1/dec.tsv#L24), [data/mem/v1/dec.tsv](../../data/mem/v1/dec.tsv#L27), [data/mem/v1/err.tsv](../../data/mem/v1/err.tsv#L370), [changed_papers_detailed_report.md](../../data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/audit_identity_variable_diff/changed_papers_detailed_report.md#L139) | One memory row says pH disappears before workbook materialization, but a later decision says pH now survives through `identity_variables_json`, and the active identity-variable diff report shows pH-like variables driving row changes. The governed narrative has not been narrowed cleanly. | Future work can either over-trust or under-trust pH and other schema-extensible variables. | Update memory and workflow docs so they distinguish “variable preserved in JSON carrier” from “variable safely promoted into identity or reviewer surfaces”. | high | benchmark blocker; scaling blocker; modeling-readiness blocker |
| U11 | Measurement-field coverage and problem framing for PDI/zeta are still not settled | Stage2 extraction coverage / memory consistency | unresolved with stale framing risk | [data/mem/v1/dec.tsv](../../data/mem/v1/dec.tsv#L26), [weak_labels__v7pilot_r3_fixparse.tsv](../../data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv) | Governed memory still carries an active “not investigated” PDI/zeta omission problem, while the current active semantic run does contain `pdi_value` and `zeta_mV_value` on a subset of rows (`77/199` and `64/199`). The issue now appears narrower than the current memory wording. | Measurement-coverage debugging can start from an outdated diagnosis. | Narrow MDEC085 to current-state wording: not total omission, but incomplete or paper-specific measurement retention and review-surface coverage. | medium | audit-usability blocker; scaling blocker |

# 4. Detailed notes by issue

## U1. Frozen LLM Stage2 authority is not yet reflected in runtime authority everywhere

Facts from repo evidence:

- The architecture now says “Stage2 authority belongs to LLM semantic discovery, not deterministic semantic reconstruction” in [project/2_ARCHITECTURE.md](../../project/2_ARCHITECTURE.md#L55).
- The flow doc says deterministic semantic emitters “must not be treated as active Stage2 mainline authority” in [project/ACTIVE_PIPELINE_FLOW.md](../../project/ACTIVE_PIPELINE_FLOW.md#L48).
- The same flow doc also says the compliant restored LLM Stage2 entrypoint is “not yet re-established” and the deterministic emitter “remains available only for fallback, comparator, migration-support, or diagnostic runs” in [project/ACTIVE_PIPELINE_FLOW.md](../../project/ACTIVE_PIPELINE_FLOW.md#L219).
- `README.md` still says “Current active Stage 2 script” is `emit_semantic_objects_from_cleaned_papers_v1.py` in [README.md](../../README.md#L115).
- `ACTIVE_RUN.json` still points to the deterministic semantic-emitter lineage in [data/results/ACTIVE_RUN.json](../../data/results/ACTIVE_RUN.json#L2).

Inference:

- The architecture freeze repaired the principle, but the executable/default authority chain is still not fully harmonized with it.

Uncertainty:

- The repo does not yet contain the restored compliant LLM Stage2 entrypoint that the architecture freeze presumes.

## U2. Maintained GT comparison entrypoint is incompatible with the current GT workbook authority

Facts from repo evidence:

- The active run explicitly says the maintained compare script could not consume the repaired-v4 workbook because it lacked the `review_formulations` sheet, so a read-only custom evaluator was used instead in [RUN_CONTEXT.md](../../data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/RUN_CONTEXT.md#L46).
- The same run says its final-table outputs should be interpreted together with the active baseline comparison because row identity changed materially in [RUN_CONTEXT.md](../../data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/RUN_CONTEXT.md#L94).

Inference:

- The compare node contract is not yet robust to the current GT authority surface.

Uncertainty:

- The repo evidence does not show whether the compare script should be broadened or whether GT workbook authority should be narrowed to a maintained shape.

## U3. Identity-variable preservation changes row identity surfaces in benchmark-significant ways

Facts from repo evidence:

- Memory labels the capability “IMPLEMENTED / UNDER EVALUATION” and says “evaluation misalignment is observed due to row-binding differences” in [data/mem/v1/dec.tsv](../../data/mem/v1/dec.tsv#L27).
- The changed-papers report says `5GIF3D8W`, `BXCV5XWB`, and `WFDTQ4VX` “appear to drift ids without scientific benefit” in [changed_papers_detailed_report.md](../../data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/audit_identity_variable_diff/changed_papers_detailed_report.md#L12), [changed_papers_detailed_report.md](../../data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/audit_identity_variable_diff/changed_papers_detailed_report.md#L48), and [changed_papers_detailed_report.md](../../data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/audit_identity_variable_diff/changed_papers_detailed_report.md#L171).
- The same report shows many `split_in_new_run`, `merged_in_new_run`, and `needs_human_review` cases on `L3H2RS2H`, `WIVUCMYG`, `YGA8VQKU`, `UFXX9WXE`, and `BB3JUVW7`.

Inference:

- Identity-variable preservation improves signal availability but is not yet safe as a benchmark-default identity policy.

Uncertainty:

- The repo does not yet contain a governed per-paper acceptance threshold for when a split is scientifically justified versus benchmark-harmful.

## U4. Layer2 formulation-row binding is still only partially generalized

Facts from repo evidence:

- The binding proposal states: “The maintained `compare_final_table_to_gt_v1.py` entrypoint is a count-level comparison only. It does not perform formulation-row binding” in [binding_priority_proposal.md](../../data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/audit_binding_risk/binding_priority_proposal.md#L13).
- The same proposal names papers that “still need Layer2-style identity repair even after Priority 2” in [binding_priority_proposal.md](../../data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/audit_binding_risk/binding_priority_proposal.md#L134).
- MDEC087 says initial validation scope is only WIVUCMYG and 5ZXYABSU in [data/mem/v1/dec.tsv](../../data/mem/v1/dec.tsv#L28).

Inference:

- Layer2 binding logic exists as a design and partial contract, but not yet as a generalized benchmark-safe binding layer across DEV15.

Uncertainty:

- The repo does not yet prove whether Priority 2 normalization alone is enough for most papers or only a subset.

## U5. Stage5 closure remains intentionally narrow

Facts from repo evidence:

- The active final-output summary says “exact core-signature fields for broader collapse remain unresolved” and “DOE-aware coordinate closure still needs a later explicit contract” in [final_output_summary_v1.md](../../data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/final_output_summary_v1.md#L80) and [final_output_summary_v1.md](../../data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/final_output_summary_v1.md#L84).

Inference:

- The current closure logic is intentionally conservative and therefore incomplete for broader deployment.

Uncertainty:

- The repo does not define the next safe expansion of the closure rule family beyond the current narrow cases.

## U6. Layer3 remains formulation-centered in intent but fragmented in implementation

Facts from repo evidence:

- The Layer3 method says the “current repo capability is partially present but not yet unified into one formulation-centered audit system contract” in [docs/methods/layer3_field_gt_protocol_v1.md](../methods/layer3_field_gt_protocol_v1.md#L37).
- MDEC081 repeats the same state in [data/mem/v1/dec.tsv](../../data/mem/v1/dec.tsv#L22).

Inference:

- Layer3 is governed, but still not one cohesive audit surface.

Uncertainty:

- The repo does not yet define which current Layer3 export should be the single reviewer entry contract.

## U7. Evidence binding and evidence ownership are still too coarse

Facts from repo evidence:

- The parking lot explicitly records “Evidence Alignment Granularity (Deferred)” in [project/5_PARKING_LOT.md](../../project/5_PARKING_LOT.md).
- The Stage2 field audit says the LLM is already doing too much coarse evidence-binding and arbitration work in [docs/methods/stage2_llm_field_audit_and_db_redesign_2026-03-25.md](../methods/stage2_llm_field_audit_and_db_redesign_2026-03-25.md).
- The Layer3 protocol still needs `unresolved_table`, `missing_evidence_anchor`, and other downgrade states in [docs/methods/layer3_field_gt_protocol_v1.md](../methods/layer3_field_gt_protocol_v1.md#L108) and [docs/methods/layer3_field_gt_protocol_v1.md](../methods/layer3_field_gt_protocol_v1.md#L590).

Inference:

- The repo has guards against bad evidence binding, but not a clean, deterministic, audit-grade evidence-binding layer yet.

Uncertainty:

- The repo does not yet show which fields should be first migrated into deterministic evidence binding.

## U8. Final outputs are still not fully normalized or model-ready

Facts from repo evidence:

- MDEC083 says “Stage5 outputs are not yet a fully normalized modeling surface” in [data/mem/v1/dec.tsv](../../data/mem/v1/dec.tsv#L24).
- The Layer3 protocol says “concentration vs absolute amount ambiguity is unresolved” and “phase-volume denominator is unclear” in [docs/methods/layer3_field_gt_protocol_v1.md](../methods/layer3_field_gt_protocol_v1.md#L457) and [docs/methods/layer3_field_gt_protocol_v1.md](../methods/layer3_field_gt_protocol_v1.md#L459).

Inference:

- The final table is still benchmark-facing first and model-facing second.

Uncertainty:

- The repo does not yet define the exact governed normalized data product that would count as model-ready.

## U9. Chemical-entity representation is still too weak for modeling use

Facts from repo evidence:

- MDEC083 says `drug_name` alone is insufficient and calls for a chemical-entity registry with more identifiers and properties in [data/mem/v1/dec.tsv](../../data/mem/v1/dec.tsv#L24).

Inference:

- Current chemistry representation is still task-oriented rather than registry-oriented.

Uncertainty:

- The repo does not yet specify whether the chemical registry should live in Stage5 exports, a sidecar reference table, or a later modeling layer.

## U10. Schema-extensible variable handling is only partially resolved and the governed story is inconsistent

Facts from repo evidence:

- MDEC083 says formulation-defining variable detection and identity integrity remain open in [data/mem/v1/dec.tsv](../../data/mem/v1/dec.tsv#L24).
- MDEC086 says identity-bearing variables such as pH now survive to final output through `identity_variables_json` in [data/mem/v1/dec.tsv](../../data/mem/v1/dec.tsv#L27).
- MERR593 still says pH disappears before workbook materialization in [data/mem/v1/err.tsv](../../data/mem/v1/err.tsv#L370).
- The active identity-variable diff report shows many row changes explicitly driven by `ph` and other identity variables, for example in [changed_papers_detailed_report.md](../../data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/audit_identity_variable_diff/changed_papers_detailed_report.md#L139).

Inference:

- The variable is no longer simply “missing”; it is partially preserved but not yet cleanly integrated into benchmark-safe identity and reviewer surfaces.

Uncertainty:

- The repo does not yet define when a preserved variable should remain audit-only metadata versus become an identity-defining split field.

## U11. Measurement-field coverage and problem framing for PDI/zeta are still not settled

Facts from repo evidence:

- MDEC085 remains active and says PDI and zeta are “OPEN PROBLEM / NOT INVESTIGATED” in [data/mem/v1/dec.tsv](../../data/mem/v1/dec.tsv#L26).
- The current active semantic run’s compatibility TSV contains nonblank `pdi_value` for `77/199` rows and nonblank `zeta_mV_value` for `64/199` rows in [weak_labels__v7pilot_r3_fixparse.tsv](../../data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv).

Inference:

- The current live issue is probably not total absence, but incomplete / paper-specific retention plus stale problem framing in memory.

Uncertainty:

- The repo does not yet contain a governed narrowed replacement for MDEC085 that reflects the current semantic run state.

# 5. Suggested priority order

1. Resolve the Stage2 authority mismatch by choosing one truthful active state: either restore a compliant LLM Stage2 entrypoint or stop treating the deterministic semantic run as repository-default active authority.
2. Repair the benchmark comparison contract so the maintained compare workflow can consume the current GT workbook authority without a custom evaluator.
3. Generalize Layer2 row binding beyond count-only comparison and the current partial binding proposal.
4. Freeze the identity-variable preservation path as experimental until the row-drift papers are reconciled paper by paper.
5. Define the next explicit Stage5 closure contract beyond the current narrow conservative rules.
6. Narrow and update governed memory around pH and PDI/zeta so current-state problem framing matches active-run evidence.
7. Consolidate Layer3 into one governed formulation-centered audit entry surface.
8. Build the deterministic evidence-binding layer for high-risk fields and table-local anchors.
9. Define the model-ready normalization contract and chemical-entity registry contract before expanding downstream modeling exports.

# 6. Open uncertainties

- `README.md` still names the deterministic semantic emitter as the “Current active Stage 2 script” while the architecture freeze says deterministic Stage2 authority is forbidden. This conflict is explicit, but the repo does not yet show which file the maintainers consider the final authority for human-facing summary wording.
- The current active lineage is a top-level root with no child `RUN_CONTEXT.md` files beneath it, so there is no later child run in that lineage proving that the row-drift and custom-comparison issues were subsequently resolved.
- `MDEC076` and `MDEC082` still preserve the earlier semantic-emitter-as-authority story, while `MDEC090` restores the frozen LLM-centered contract. The repo does not yet contain a governed normalization row that explicitly narrows or supersedes those older authority statements.
- The repo shows active evidence that pH is preserved in the identity-variable carrier, but it does not yet show a governed decision for whether pH should become a first-class benchmark / reviewer column versus remain an auxiliary identity carrier.
- The repo shows that PDI and zeta are present in the active semantic run on a subset of rows, but it does not yet contain a current governed measurement-coverage audit replacing the still-active “not investigated” framing in MDEC085.
