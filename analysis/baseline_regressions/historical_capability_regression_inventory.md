# Historical Capability Regression Inventory

## Method
- Cases were collected from:
  - `project/4_DECISIONS_LOG.md`
  - `project/ACTIVE_PIPELINE_FLOW.md`
  - `project/ACTIVE_PIPELINE_RUNBOOK.md`
  - `docs/methods/*.md` regression and audit notes
  - `docs/snapshots/*.md` where they preserved regression framing
  - the current baseline analysis artifacts under `data/results/20260417_385b6e1/09_dev15_stage2_baseline_repaired_contractfix_v1/analysis/`
- A case counts as a capability regression only when repo records show all three:
  - the capability previously worked or was explicitly recovered
  - a later run or rule change regressed that capability
  - the likely failure boundary or failure class was explicitly discussed
- When only part of that chain is documented, the case is kept in the lower-confidence section instead of being promoted into a near-term guard candidate.

## High-confidence candidate cases
### paper or feature
`QLYKLPKT`
### previously proven capability
- Stage2 already preserved the comparative/commercial semantics for the external `Sporanox®` row.
- Repo decisions document a successful Stage5 fix that filtered the commercial comparator and moved the reviewed-boundary DEV15 comparison from `14/15` to `15/15`.
### later regression
- Stage5 retained the commercial comparator because the older filter only excluded rows explicitly marked `candidate_non_formulation`.
### likely boundary
- `Stage5 final formulation closure`
### recommended future guard priority
- `high`
- Good next candidate because the fix is deterministic, paper-backed, and cheaper than a new Stage2 guard.

### paper or feature
`BB3JUVW7`
### previously proven capability
- Benchmark-facing sweep-style variant rows should remain in final output when they are still formulation members.
- Repo decisions document restoration of the missing `F2.x` rows and a deterministic regression checker that now asserts `BB3JUVW7` retains all `12` benchmark-facing final rows.
### later regression
- Five valid benchmark-facing rows were filtered when the early Stage5 descendant branch treated `post_processing` as a universal exclusion signal.
### likely boundary
- `Stage5 duplicate / descendant governance`
### recommended future guard priority
- `high`
- Strong candidate because the regression checker surface already exists and the failure class is sharply defined.

### paper or feature
`L3H2RS2H`
### previously proven capability
- Older candidate / review workflow preserved the `22`-instance expectation.
- Later row-enumeration work restored the paper from `8` collapsed family summaries to `21` or `24` row-level candidates depending on the bounded validation path, including later recovery of the missing nanocapsule row in a replacement-path validation slice.
### later regression
- The LLM collapsed table rows into family-level summaries before enumeration, producing only `8` instances.
### likely boundary
- `Stage2 LLM extraction / prompt evidence behavior`
### recommended future guard priority
- `high`
- Valuable because it is a clean non-DOE Stage2 under-enumeration case and complements `UFXX9WXE`.

### paper or feature
`WFDTQ4VX`
### previously proven capability
- Repo decisions record a validated coordinate-signature identity rule for checkpoint / validation rows and later integration of that rule into the benchmark-valid Stage5 path.
### later regression
- Without that rule on mainline, checkpoint / validation rows were over-counted as independent formulation identities.
### likely boundary
- `Stage5 benchmark-valid identity closure`
### recommended future guard priority
- `medium`
- Strongly evidenced, but slightly less attractive than the first three because the current fix is intentionally narrow and paper-local.

### paper or feature
`BXCV5XWB`
### previously proven capability
- Repo decisions document that Stage2 already preserved enough helper/control semantics downstream to distinguish benchmark-facing KGN rows from helper descendants.
- Later Stage5 governance validation retained only the intended `3` benchmark-facing rows.
### later regression
- Blank / FITC helper descendants were over-retained because Stage5 relied too heavily on a narrow upstream routing field.
### likely boundary
- `Stage5 helper-descendant governance`
### recommended future guard priority
- `medium`
- Useful, but slightly lower priority than `BB3JUVW7` because the class overlaps and the history is entangled with later GT-boundary tightening.

## Lower-confidence or incomplete cases
- `YGA8VQKU`
  - Evidence exists for identity drift / split-merge concerns in later audits, but the currently loaded repo records do not show as clean a previously-worked -> regressed -> repaired chain as the cases above.
  - Likely boundary: unresolved identity / closure behavior across Stage2 to Stage5.
  - Recommended status: keep in inventory, do not promote to a guard until a sharper artifact trail is assembled.
- `INMUTV7L`
  - Repo records show it as a successful no-regression validation target inside later Stage5 governance checks, but the currently inspected material does not expose the original regression class with enough detail for immediate guard authoring.
  - Likely boundary: Stage5 duplicate / descendant governance.
  - Recommended status: keep as a bounded follow-up candidate once the original blocker material is re-opened.

## Suggested next 3 additions after phase 1
- `QLYKLPKT`
  - Highest implementation efficiency after phase 1: clear Stage5 rule, explicit regression run, and no need to reopen Stage2.
- `BB3JUVW7`
  - Best next regression-protection target for the descendant / variant filter class because the repo already names it as the validated anchor and ships a deterministic checker surface.
- `L3H2RS2H`
  - Best next Stage2 guard after the phase-1 pair because it captures table-heavy non-DOE family-collapse behavior that `UFXX9WXE` does not cover.
