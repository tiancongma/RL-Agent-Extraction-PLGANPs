# Project Snapshot  
RL-Agent-Extraction-PLGANPs  
Last Updated: 2026-02-16  

---

## 1. Project Objective

Build a reproducible, evidence-grounded LLM pipeline to extract formulation-level structured data from PLGA nanoparticle literature.

The system converts cleaned scientific text into structured records with field-level traceable evidence and human adjudication support.

---

## 2. Historical Evolution

### Phase 1 — Dual-Model Extraction

Architecture:
Model1 extraction  
Model2 extraction  
Conflict merge  
Human GT adjudication  

Outcome:
- Multi-model extraction functional  
- Sample20 processed  
- Human GT performed  

Limitation identified:
- Evidence insufficient for formulation-level adjudication  
- Independent second model introduced hallucinated conflicts  

---

### Phase 2 — Evidence-Grounded Redesign (Current)

New Architecture:

Extractor  
→ Field-level evidence (multi-span)  
→ Verifier (evidence-only evaluation)  
→ Aggregator  
→ Human GT (only unresolved fields)

Key change:
Replace independent second extractor with evidence-based verification.

---

## 3. Current State

STATE_2 — Architecture Revision  

In progress:
- Finalizing evidence contract
- Implementing verifier layer
- Implementing aggregation logic
- Migrating Sample20 to new pipeline

Legacy dual-model outputs remain for historical comparison.

---

## 4. Authoritative Data Reference

All extraction and adjudication reference:

data/cleaned/content/text/

This cleaned text view is the frozen auditable reference frame.

PDF files are not used during adjudication.

---

## 5. Current Constraints

- No expansion of scientific scope
- No new extraction fields
- No new external data sources
- Directory structure remains frozen
- Changes must be additive

---

## 6. Immediate Next Steps

1. Finalize evidence span schema  
2. Implement verifier script in stage4_eval  
3. Implement aggregator  
4. Process Sample20 under new architecture  
5. Confirm schema stability across two consecutive runs  

---

## 7. Definition of Stability

Pipeline considered frozen when:

- Evidence contract finalized  
- Two consecutive runs produce identical schema  
- GT workflow validated under evidence-based design  

---

## 8. Long-Term Direction (Out of Current Scope)

- Large-scale extraction  
- Predictive modeling  
- Design rule discovery  

These are explicitly outside the current phase.
