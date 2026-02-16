# Project Requirements  
RL-Agent-Extraction-PLGANPs

This document defines functional and non-functional requirements for the extraction pipeline.

---

# 1. Functional Requirements

## 1.1 Data Ingestion

- The system must ingest literature records from Zotero and WoS exports.
- The system must generate a validated manifest_current.tsv.
- The system must generate a validated key2txt.tsv mapping.

---

## 1.2 Cleaned Text View

- Cleaned text must be stored under data/cleaned/content/text/.
- Cleaned text must be reproducible.
- Extraction must reference cleaned text only.
- Each cleaned file must correspond to a zotero_key.

---

## 1.3 Extraction

- The system must support formulation-level extraction.
- A single article may produce multiple formulation records.
- Extraction output must include structured field values.
- Extraction must include evidence spans.

---

## 1.4 Evidence Contract

Each extracted field must include:

- formulation identifier
- field name
- raw extracted value
- canonicalized value (if applicable)
- one or more evidence spans

Each evidence span must include:

- start_char
- end_char
- excerpt
- reference to cleaned text source

Evidence spans must allow traceability to the frozen cleaned text view.

---

## 1.5 Verification Layer

- A verifier model must evaluate extracted fields using only provided evidence.
- Verifier output must be one of:
  - supported
  - insufficient
  - contradicted
- Verifier must not access full document text directly.

---

## 1.6 Aggregation

- Only supported fields enter final output tables.
- Insufficient fields enter uncertainty summary.
- Contradicted fields enter conflict table.
- Human GT may adjudicate unresolved fields.

---

## 1.7 Ground Truth (GT)

- GT decisions must be recorded at field level.
- GT must reference evidence spans.
- GT schema must remain stable across runs.

---

# 2. Non-Functional Requirements

## 2.1 Reproducibility

- Runs must be versioned under runs/.
- Each run must include metadata.
- Schema changes must be explicitly documented.

---

## 2.2 Stability

- Directory structure must not be restructured.
- Existing stage definitions must remain intact.
- Changes must be additive unless explicitly documented.

---

## 2.3 Auditability

- All extracted fields must be traceable to cleaned text.
- Evidence must survive re-evaluation within the same frozen text version.
- Human adjudication must not depend on PDF files.

---

# 3. Constraints

- Project scope must remain limited to PLGA nanoparticle literature extraction.
- No new scientific objectives may be introduced without charter update.
- Downstream modeling is out of scope for the current phase.
