# S2-2 Evidence Quality

Comparison surfaces used:

- Baseline:
  - `data/results/20260417_385b6e1/09_dev15_stage2_baseline_repaired_contractfix_v1/semantic_stage2_objects/evidence_blocks/<paper>/evidence_blocks_v1.json`
- Current:
  - `data/results/20260418_9538ec2/01_s2_2/semantic_stage2_objects/evidence_blocks/<paper>/evidence_blocks_v1.json`

## QLYKLPKT

- table quality:
  - `noisy`
- extra noise present:
  - `yes`
- selector correctness:
  - `partial`

Notes:

- Baseline used `sorted_csv_first_4` with broad raw-prefix fallback and obvious assay/front-matter leakage.
- Current switched to `role_aware_selector_v1`, but selected table evidence still includes noisy assay-bearing tables.
- Observed current noise still includes:
  - `lc-ms`
  - `lc-ms/ms`
  - `pharmacokinetic`
  - `rat plasma`
  - `tcpdf`
- Current quality is somewhat better structured than baseline, but not clean.

## UFXX9WXE

- table quality:
  - `clean`
- extra noise present:
  - `partial`
- selector correctness:
  - `good`

Notes:

- Baseline used the first four sorted CSV tables, including obvious biodistribution/non-core content.
- Current selector picked a compact DOE-style variable table and a formulation-result table.
- Noise signals still exist in the evidence artifact, but the chosen table set is much cleaner and more formulation-relevant than baseline.
- This is the clearest S2-2 improvement among the audited papers.

## WFDTQ4VX

- table quality:
  - `noisy`
- extra noise present:
  - `yes`
- selector correctness:
  - `poor`

Notes:

- Current switched to `role_aware_selector_v1`, but the resulting pack still contains assay-heavy and non-core material.
- Current artifact has more narrative blocks than baseline, but they are not cleanly formulation-focused.
- Noise signals still include:
  - `caco-2`
  - `pharmacokinetic`
  - `pharmacokinetics`
- Downstream collapse from `27 -> 2` final rows is consistent with a failed or mis-targeted WFDTQ4VX evidence pack.

## V99GKZEI

- table quality:
  - `partial clean`
- extra noise present:
  - `no`
- selector correctness:
  - `partial`

Notes:

- Current did not inherit the new role-aware selector path.
- It still uses `sorted_csv_first_4`, but only one table is present.
- The selected table itself is comparatively clean.
- The problem is not table contamination; the problem is that the overall prompt still falls back to a large raw-prefix style and later truncates.

## Summary

- `QLYKLPKT`: still noisy, only partial selector success
- `UFXX9WXE`: clearly cleaner and mostly correct
- `WFDTQ4VX`: still noisy and likely mis-selected
- `V99GKZEI`: table itself is fine, but the paper remains on the weaker fallback path

## Overall verdict

- S2-2 evidence quality is not uniformly clean.
- It improved for `UFXX9WXE`.
- It remains noisy or unstable for `QLYKLPKT` and especially `WFDTQ4VX`.
- `V99GKZEI` is not primarily an S2-2 cleanliness failure; it is a prompt-path fallback problem.

## Caveat

- Diagnostic-only.
- Peer review recommended before action.
