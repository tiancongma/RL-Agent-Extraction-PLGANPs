# Repo Sync Audit - 2026-04-21 DEV15 Baseline

## Executive conclusion

The newly completed lineage at
`data/results/20260421_43ed145/` is a governed, reproducible diagnostic
baseline lineage, but it is not benchmark-valid authority because:

- identity freeze failed
- compare was downgraded to diagnostic
- `benchmark_valid = no`

It should therefore be treated as local reproducibility evidence only, not as a
tracked authoritative run artifact and not as the new `ACTIVE_RUN.json`
authority target.

The repo is safe to prepare for one clean local-to-remote sync by:

1. committing maintained code, governance, calibration, and repair-index files
2. committing this sync-audit report pair
3. keeping the new baseline lineage local and ignored
4. keeping machine-local observer/memory churn unstaged

## Baseline lineage treatment

Recommended treatment for `data/results/20260421_43ed145`:

- class: local governed diagnostic lineage
- GitHub policy: do not track
- authority status: do not promote
- reason:
  - current `.gitignore` already ignores all new `data/results/**` surfaces by
    default except the single allowlisted authority lineage
  - this lineage is diagnostic-only under current governance
  - tracking it would pollute the repo with non-authoritative run outputs

Small run metadata inside that lineage should also remain local:

- `LINEAGE.md`
- child `RUN_CONTEXT.md`
- stage-local analysis summaries

Reason:

- they are meaningful only inside the ignored lineage
- they do not belong on GitHub unless governance later explicitly allowlists
  this lineage as tracked authority, which is not justified here

## ACTIVE_RUN.json decision

`data/results/ACTIVE_RUN.json` should remain unchanged.

Reason:

- the new lineage is not benchmark-valid authority
- the current tracked authority pointer already targets the allowlisted run
  `run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1`
- promoting a diagnostic-only lineage would violate the active data-source
  contract and blur the difference between authoritative benchmark inputs and
  local experimental outputs

## What should be committed now

### Commit-worthy maintained source / governance / docs

Track and commit:

- `.gitignore`
- `AGENTS.md`
- `README.md`
- `docs/maintained_script_surface.tsv`
- `docs/src_script_registry.tsv`
- `docs/selector_calibration/5ZXYABSU.json`
- `docs/selector_calibration/INMUTV7L.json`
- `docs/selector_calibration/L3H2RS2H.json`
- `docs/selector_calibration/QLYKLPKT.json`
- `docs/selector_calibration/UFXX9WXE.json`
- `docs/selector_calibration/V99GKZEI.json`
- `docs/selector_calibration/readme.md`
- `docs/repair_index/success_pattern_index_v1.md`
- `docs/repair_index/success_pattern_index_v1.tsv`
- `project/S2_4A_AUDIT_STANDARD.md`
- `project/2_ARCHITECTURE.md`
- `project/4_DECISIONS_LOG.md`
- `project/ACTIVE_PIPELINE_FLOW.md`
- `project/ACTIVE_PIPELINE_RUNBOOK.md`
- `project/PIPELINE_SCRIPT_MAP.md`
- `project/feature_units/FEATURE_UNIT_GOVERNANCE.md`
- `project/feature_units/feature_intervention_matrix.tsv`
- `project/feature_units/feature_unit_registry.json`
- `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`
- `src/stage2_sampling_labels/evaluate_s2_4a_hard_gate_v1.py`
- `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`
- `src/stage2_sampling_labels/run_stage2_composite_v1.py`
- `src/stage2_sampling_labels/run_stage2_s2_4a_prompt_construction_v1.py`
- `src/stage2_sampling_labels/run_stage2_s2_5_semantic_parsing_v1.py`
- `src/stage2_sampling_labels/run_stage2_s2_6_contract_validation_v1.py`
- `src/stage2_sampling_labels/validate_stage2_semantic_authority_contract_v1.py`
- `src/utils/build_feature_activation_report_v1.py`
- `src/utils/build_mem_v1.py`
- `src/utils/run_repair_intake_v1.py`

### Commit-worthy sync audit outputs

Track and commit:

- `analysis/repo_governance/repo_sync_audit_2026-04-21_baseline.md`
- `analysis/repo_governance/repo_sync_audit_2026-04-21_baseline.tsv`

## What must remain local

Keep local and unstaged:

- `data/results/20260421_43ed145/`
- `.codex_supervisor/candidate_change.txt`
- `data/mem/v1/dec.tsv`
- `data/mem/v1/err.tsv`
- `data/mem/v1/idx.tsv`
- `data/mem/v1/lin.tsv`
- `data/mem/v1/prm.tsv`
- `data/mem/v1/run.tsv`
- the existing large set of ad hoc untracked analysis notes under `analysis/`
  that are not part of this sync audit

Reason:

- results lineage is ignored by policy and diagnostic-only
- memory TSV churn is machine-local support state, not part of the governed
  baseline sync being prepared here
- `.codex_supervisor/candidate_change.txt` is a local assistant bookkeeping
  file and should not be included in a scientific or governance sync
- unrelated untracked analysis notes would add noise without improving the
  maintained pipeline contract

## Whether .gitignore should change

Yes, but only minimally.

Recommended change:

- add `.codex_observer/`

Reason:

- it is a machine-local observer directory
- it is not part of any governed repo layer
- it currently appears as untracked local noise

No other `.gitignore` change is recommended:

- `data/results/20260421_43ed145/` is already correctly ignored
- the `analysis/` directory already contains intentionally tracked audit files,
  so broad ignore rules there would be too blunt

## Risk of polluting the repo

Main pollution risks if the wrong files are synced:

1. tracking `data/results/20260421_43ed145/`
   - would blur local diagnostic lineage vs tracked authority lineage
2. updating `ACTIVE_RUN.json`
   - would incorrectly promote a diagnostic-only run
3. committing raw responses / request metadata
   - would add bulky replay-heavy artifacts without changing the maintained
     contract
4. committing memory TSV churn
   - would mix local memory refresh state into a baseline sync
5. committing unrelated ad hoc analysis notes
   - would bury the actual maintained code/governance changes

## Recommended commit scope split

### Commit 1: code / governance / maintained docs

Suggested scope:

- `.gitignore`
- maintained `src/` changes
- `project/` governance updates
- maintained registry/docs updates
- selector calibration JSON updates
- repair-index files

Suggested message:

- `Align Stage2 S2-4a governance and contract enforcement with maintained baseline path`

### Commit 2: sync audit only

Suggested scope:

- `analysis/repo_governance/repo_sync_audit_2026-04-21_baseline.md`
- `analysis/repo_governance/repo_sync_audit_2026-04-21_baseline.tsv`

Suggested message:

- `Add 2026-04-21 baseline repo sync audit`

No separate commit for the new baseline lineage is justified.

## Final sync recommendation

Proceed with a GitHub sync only for maintained code/governance/docs plus this
audit report pair.

Do not:

- push `data/results/20260421_43ed145/`
- change `ACTIVE_RUN.json`
- stage memory TSV churn
- stage `.codex_supervisor/candidate_change.txt`
- stage unrelated ad hoc `analysis/` notes
