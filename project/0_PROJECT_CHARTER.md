# Project Charter  
RL-Agent-Extraction-PLGANPs

## 1. Project Purpose

This project develops a structured, reproducible pipeline for extracting formulation-level data from PLGA nanoparticle literature using large language models (LLMs), with explicit evidence grounding and human adjudication.

The system converts cleaned scientific text into structured records while preserving traceability to the source text.

---

## 2. Scientific Scope

The project focuses on:

- PLGA-based nanoparticle formulation literature
- Extraction of formulation-level experimental parameters
- Field-level evidence grounding
- Human-in-the-loop adjudication for unresolved fields
- Reproducible multi-run evaluation

The project does not include:

- Experimental wet-lab validation
- Downstream predictive modeling
- Biological outcome modeling
- Automated hypothesis generation
- External data integration beyond literature text

---

## 3. Data Reference Model

All extraction and adjudication must reference the cleaned text view under:

data/cleaned/content/text/

The cleaned text view is the authoritative reference frame for:

- Extraction
- Evidence span indexing
- Verification
- Human GT adjudication

PDF files are not considered authoritative sources during extraction.

---

## 4. Architectural Philosophy

The pipeline follows a staged architecture:

- Stage0: Relevance filtering
- Stage1: Text cleaning
- Stage2: Sampling and weak labeling
- Stage3: Human GT tooling
- Stage4: Extraction and evaluation
- Stage5: Merging and publication outputs

Architecture changes are allowed only within the project scope defined above.

---

## 5. Non-Goals

The project is not intended to:

- Replace domain experts
- Automatically generate biological interpretations
- Perform statistical meta-analysis
- Build predictive models at this stage

---

## 6. Success Criteria

The project is considered successful when:

- Extraction is reproducible across runs
- Evidence spans are traceable to frozen cleaned text
- Human GT adjudication is supported at field level
- Output tables are stable and version-controlled
