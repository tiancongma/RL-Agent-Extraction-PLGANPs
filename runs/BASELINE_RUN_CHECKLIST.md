# Baseline Run Checklist (sample10)

## 0) Preconditions
- [ ] On `main` branch (or note the branch)
- [ ] `data/cleaned/index/manifest_current.tsv` exists and is the *only* active manifest
- [ ] `data/cleaned/index/key2txt.tsv` exists and is the *only* active key2txt
- [ ] `data/cleaned/samples/` contains the sample10 definition file
- [ ] `.env` is NOT tracked by git
- [ ] Large outputs (data/results) are ignored by git; run metadata (runs/) is tracked

## 1) Create run_id
Recommended format:
`run_YYYYMMDD_HHMM_<shortcommit>_sample10`

## 2) Create folders
- `data/results/<run_id>/`
- `runs/<run_id>/`

## 3) Capture provenance
- [ ] Write git commit + branch into `runs/<run_id>/meta.json`
- [ ] Save a copy of the effective config (optional but recommended):
  - `runs/<run_id>/config_used.yaml` (copy from configs/)

## 4) Execute pipeline (sample10)
- [ ] Run your stage entry scripts with sample10 inputs
- [ ] Redirect console logs to: `runs/<run_id>/logs.txt` (optional)

## 5) Validate outputs
- [ ] Expected output files appear in `data/results/<run_id>/`
- [ ] No errors in logs
- [ ] QC report exists (if your pipeline produces one)

## 6) Update latest pointer
- [ ] Set `runs/latest.txt` = `<run_id>` (one line)

## 7) Commit only metadata
- [ ] Commit:
  - `runs/latest.txt`
  - `runs/<run_id>/RUN.md`
  - `runs/<run_id>/meta.json`
  - `runs/<run_id>/config_used.yaml` (if exists)
- [ ] Do NOT commit large data outputs unless intentionally curated

Suggested commit message:
`chore: add baseline run metadata (<run_id>)`
