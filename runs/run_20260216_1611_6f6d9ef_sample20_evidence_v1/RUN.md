# Run Report  
run_20260216_<shortsha>_sample20_evidence_v1

---

## Objective

Validate the first implementation of the evidence-grounded extraction schema.

This run extends the weak label extraction output to include:

- evidence_spans_json (TSV)
- text_sha256 (TSV)
- evidence object (JSONL)

No changes to extraction field set.

---

## Architecture Context

STATE_2 — Architecture Revision

Transitioning from:

Dual-model extraction

To:

Extractor  
→ Evidence spans  
→ Verifier (future)  
→ Aggregator (future)  
→ Human GT  

This run focuses only on extractor-level evidence support.

---

## Command Executed

```powershell
python src/stage2_sampling_labels/auto_extract_weak_labels_v6.py `
--sample-jsonl data/cleaned/samples/sample20_stratified.jsonl `
--key2txt data/cleaned/index/key2txt.tsv `
--sections-dir data/cleaned/content/sections `
--model gemini-2.5-flash `
--max-chars 60000 `
--sleep 1.0 `
--out-tsv data/results/run_20260216_<shortsha>_sample20_evidence_v1/weak_labels__gemini.tsv `
--out-jsonl data/results/run_20260216_<shortsha>_sample20_evidence_v1/weak_labels__gemini.jsonl `
--verbose
