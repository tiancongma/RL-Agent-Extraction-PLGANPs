# Phase 1 Regression Guard Plan

## Scope
- Start with `UFXX9WXE` and `5GIF3D8W` because both are already documented as real capability-loss cases, both have explicit historical recovery evidence in repo records, and both localize to Stage2 without needing a full benchmark rerun.
- Keep this phase limited to two guards so the first pass stays auditably narrow: one selector/authority-handoff guard (`UFXX9WXE`) and one formulation-universe preservation guard (`5GIF3D8W`).
- Fact: `UFXX9WXE` is a confirmed strong DOE under-enumeration case with preserved Stage1 table structure and later successful recovery evidence in repo records.
- Fact: `5GIF3D8W` is a confirmed formulation-universe loss case with fixed GT authority count `26`, explicit optimized-table rows, and historically preserved sweep-expansion capability in repo records.
- Inference: these two papers cover the two most valuable phase-1 failure classes for the current baseline:
  - critical authority table dropped before the LLM call
  - formulation-universe activation collapsing from a proven richer scope to a tiny subset

## Guard 1: UFXX9WXE
### Historical capability proven
- Fact: Stage1 preserves the authority table: `UFXX9WXE__table_13__pdf_table.csv` contains numbered rows `1.` through `26.` and the cleaned text preserves the same `26`-run DOE framing.
- Fact: current baseline S2-2a still detects the correct DOE candidates. The audited `candidate_blocks_v1.json` records `table_13` and `table_14` as top-ranked ready formulation tables.
- Fact: a later repaired run proved the capability still exists in maintained repo artifacts: the governed DOE recovery path emitted and retained `26` numbered DOE rows for `UFXX9WXE`.

### Current regression class
- Fact: the current failure is not missing source structure and not primarily Stage5 collapse.
- Fact: the audited baseline loses the authority table at S2-2b because the canonical evidence handoff reverts to `sorted_csv_first_4` and excludes the ranked DOE authority table from the evidence package.
- Inference: this is an integration failure between preserved S2-2a candidate discovery and the maintained evidence-handoff / Stage2 completion path, not a missing capability.

### Earliest stable guard boundary
- `S2-2b selector / evidence prioritization`

### Proposed minimal invariant
- Primary capability guard:
  - If S2-2a contains a ready high-priority DOE formulation table candidate for a paper and that candidate is the authority table for the preserved DOE universe, S2-2b must preserve that authority table across the canonical evidence handoff.
  - For `UFXX9WXE`, this means the authority table corresponding to `table_13` must survive from candidate space into the canonical evidence package and must remain available as a non-empty execution-grade table authority payload for downstream governed activation.
- Optional secondary sanity check:
  - The downstream Stage2 completion path should not fall through to a trivial single-row UFXX surface when the preserved DOE authority table is present and activatable.

### Why this invariant would have caught the regression
- Fact: the current baseline still detects `table_13` at S2-2a but drops it from `evidence_blocks_v1.json`.
- Fact: the same baseline later records `missing_table_authority_payload` and never activates the already-proven numbered DOE recovery path.
- Inference: a hard failure at the handoff boundary would have stopped the run at the exact point where capability continuity was broken:
  - authority continuity failed because the preserved DOE table was dropped
  - activation continuity failed because downstream DOE recovery never became legally callable

### Implementation surface
- Existing artifacts are already sufficient for a deterministic guard:
  - `semantic_stage2_objects/candidate_blocks/UFXX9WXE/candidate_blocks_v1.json`
  - `semantic_stage2_objects/evidence_blocks/UFXX9WXE/evidence_blocks_v1.json`
  - `semantic_stage2_objects/normalized_table_payloads/UFXX9WXE/normalized_table_payloads_v1.json`
- Minimal helper logic could:
  - find ready DOE / formulation table candidates at S2-2a
  - resolve the selected table origins in S2-2b
  - fail if no selected authority table survives or if the corresponding normalized payload set is empty
- This can remain stage-local and does not need Stage3, Stage5, or GT comparison.

### What this guard intentionally does not check
- It does not require full final-count agreement against GT.
- It does not require Stage5 replay.
- It does not authorize broad DOE logic redesign.
- It does not require generic table widening.
- It does not assert generalized DOE recovery on every paper; it only blocks the known failure mode where a preserved authority table is dropped before downstream recovery can legally activate.
- It does not make exact row count the primary pass/fail condition.

## Guard 2: 5GIF3D8W
### Historical capability proven
- Fact: Stage1 preserves both sources needed for formulation existence:
  - `Table 1` with `8` explicit optimized formulations
  - narrative / figure-backed one-variable sweep semantics for stabilizer concentration, polymer amount, and drug amount
- Fact: repo decisions fix the current Layer1 GT authority for `5GIF3D8W` at `26`.
- Fact: repo history proves broader sweep-preservation capability existed:
  - earlier targeted Stage2 fixes restored missing sweep members
  - the `2026-03-18` regression run reduced the paper from `38` to `26`
  - the `2026-03-29` preserved experiment still materializes `24` sweep rows and treats sweep semantics as row-authorizing identity carriers

### Current regression class
- Fact: the current repaired baseline preserves the critical sweep-bearing paragraph and optimized-table text at S2-2a.
- Fact: the earliest clear collapse is S2-2b: evidence packaging falls back to `sorted_csv_first_4`, has no `VARIABLE_TABLE` role, and emits an empty `normalized_table_payloads_v1.json`.
- Fact: the replayed raw response then degrades even further and only produces `3` generic formulation candidates.
- Inference: this is a formulation-universe preservation / activation loss, not a missing-value problem and not primarily a downstream suppression problem.

### Earliest stable guard boundary
- `S2-7 completed Stage2 artifact`, with supporting observability at `S2-2b`

### Proposed minimal invariant
- Primary capability guard:
  - If S2-2a preserves both:
    - the explicit optimized formulation table surface for `5GIF3D8W`
    - and sweep-bearing one-variable formulation evidence
    then the Stage2 path must preserve blank-tolerant sweep-expansion authorization continuity through the completed Stage2 boundary.
  - Concretely, the completed Stage2 artifact must still expose that sweep-derived formulation existence is legally activatable even when some downstream numeric values remain blank.
  - Missing numeric completeness must not revoke authorization for sweep-derived formulation existence.
- Optional secondary sanity check:
  - The completed Stage2 surface should retain the explicit optimized-table floor and should extend beyond a tiny generic subset such as `3` rows.

### Why this invariant would have caught the regression
- Fact: the current baseline ends with only `3` Stage2 rows for `5GIF3D8W`.
- Fact: earlier repo runs preserved `8` explicit optimized rows even under a weak selector contract, and stronger historical paths preserved much more of the sweep universe.
- Fact: the current baseline loses sweep authorization earlier than final counting:
  - S2-2b emits no `VARIABLE_TABLE` role
  - execution-grade normalized table payload is empty
  - the replayed raw response collapses sweep semantics into generic optimized categories
- Inference: a capability-preservation guard would have caught the true regression earlier:
  - authorization continuity failed because sweep-bearing evidence no longer remained execution-usable
  - activation continuity failed because blank-tolerant sweep expansion was no longer legally reachable in the completed Stage2 path

### Implementation surface
- Existing artifacts are sufficient for a bounded completed-Stage2 guard:
  - `semantic_stage2_objects/candidate_blocks/5GIF3D8W/candidate_blocks_v1.json`
  - `semantic_stage2_objects/evidence_blocks/5GIF3D8W/evidence_blocks_v1.json`
  - `semantic_stage2_objects/normalized_table_payloads/5GIF3D8W/normalized_table_payloads_v1.json`
  - `semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`
- Minimal helper logic could:
  - verify the optimized-table and sweep-bearing evidence survived S2-2a
  - verify that S2-2b preserves execution-usable sweep-bearing evidence rather than collapsing it into raw-prefix-plus-lossy-table excerpts only
  - verify that completed Stage2 still preserves sweep-derived formulation authorization under blank-tolerant semantics instead of treating missing numeric values as loss of formulation existence
  - optionally report retained optimized-table floor and whether any sweep-derived members survived as a sanity signal
- This remains smaller than a full benchmark rerun and does not require image extraction.

### What this guard intentionally does not check
- It does not require exact recovery of all `26` rows in phase 1.
- It does not require numeric completeness for sweep-derived rows.
- It does not require figure numeric extraction.
- It does not treat missing measurement values as missing formulation identity.
- It does not redesign sweep semantics or Stage2 prompt structure in this task.
- It does not make exact Stage2 output counts the primary pass/fail condition.

## Recommended implementation order
- first: add the `UFXX9WXE` S2-2b authority-table handoff guard because it is the clearest earliest-boundary invariant and the failure class is already sharply localized.
- second: add the `5GIF3D8W` completed-Stage2 floor guard because it blocks the current `26 -> 3` universe collapse without requiring a full exact-universe solver on day one.
- optional immediate follow-up: add a small shared audit helper that reads paper-local `candidate_blocks`, `evidence_blocks`, `normalized_table_payloads`, and completed Stage2 TSV rows for a bounded paper list.

## Expansion strategy
- Expand one failure class at a time, not one whole benchmark split at a time.
- Prefer guards at the earliest lawful boundary where the intended capability is already supposed to be preserved:
  - selector / evidence handoff for authority-table loss
  - completed Stage2 artifact for formulation-universe preservation
  - Stage5 deterministic closure only when the upstream candidate surface is already sufficient
- Reuse frozen historical proof cases where repo records already show:
  - prior success
  - later regression
  - a bounded root-cause class
- Keep each added guard paper-backed, stage-local where possible, and artifact-driven so the suite grows by explicit failure classes instead of by ad hoc benchmark reruns.
