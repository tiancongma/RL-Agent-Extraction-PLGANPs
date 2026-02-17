# Run Report  
run_20260217_1003_6f6d9ef_sample1_evidence_v2

---

## Objective

Prototype and validate field-level evidence extraction on a single document.

This run extends the weak label extraction outputs to include:

- field_evidence attached to each formulation field (multi-span per field).
- Evaluation of field-level evidence generation with early pattern matching.

This run is on `sample1_debug.jsonl` to minimize token cost and test correctness before scaling.

---

## Architecture Context

STATE_2 — Architecture Revision

Transitioning from:

Document-level evidence

To:

Field-level multi-span evidence  
Verifier pipeline (future)  
Aggregator (future)  
Human GT

The primary goal here is to confirm field-level evidence attachment for one sample.

---

## Command Executed

```powershell
python src/stage2_sampling_labels/auto_extract_weak_labels_v6.py --sample-jsonl data/cleaned/samples/sample1_debug.jsonl --key2txt data/cleaned/index/key2txt.tsv --sections-dir data/cleaned/content/sections --model gemini-2.5-flash --max-chars 60000 --sleep 1.0 --out-tsv data/results/run_20260217_1003_6f6d9ef_sample1_evidence_v2/weak_labels__gemini.tsv --out-jsonl data/results/run_20260217_1003_6f6d9ef_sample1_evidence_v2/weak_labels__gemini.jsonl --verbose

Output Files

TSV:

data/results/run_20260217_1003_6f6d9ef_sample1_evidence_v2/weak_labels__gemini.tsv


JSONL:

data/results/run_20260217_1003_6f6d9ef_sample1_evidence_v2/weak_labels__gemini.jsonl

Validation Checklist

 TSV generated successfully with field_evidence_json column

 JSONL with field_evidence per formulation present

 field_evidence lists per field have one or more spans

 Evidence offsets valid with respect to the cleaned text view

 No changes to existing extraction fields

Observations (to be completed after inspection)

Field-level evidence coverage

Unresolved fields and patterns

Start/end validity

Quality of extracted excerpt windows

Next Step

Review and fix edge cases in matching logic.

Extend to multi-document small batch (3–5 samples).

Plan for verifier layer integration.