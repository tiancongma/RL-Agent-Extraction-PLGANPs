### Evidence Alignment Granularity (Deferred)

**Observation**  
Evidence spans are aligned post-hoc and may be coarse-grained (e.g., section-level or missing).  
Some extracted values cannot be mapped to uniquely identifiable textual spans.

**Current Handling**  
Such cases are explicitly labeled as `unclear` during manual GT annotation to preserve traceability.

**Deferred Improvements**  
- Field-aware span extraction  
- Sentence-level evidence re-alignment  
- Evidence confidence scoring

**Status**  
Out of scope for the current phase; recorded for future methodological refinement.

### Stage2 Contract Minimization And `db_v2` Adapter (2026-03-25)

**Fact**
- The current authoritative Stage2 artifact is a wide-row TSV that mixes:
  - formulation identity
  - component-like material fields
  - measurement outputs
  - coarse evidence hints
- Current code and authoritative artifact are not on the same emitted schema
  version.

**Inference**
- The current wide-row contract is serviceable for benchmark continuity, but it
  is not the cleanest long-term surface for interpretable formulation-to-EE
  modeling.
- Evidence-related metadata in Stage2 is broader than the downstream Stage3 and
  Stage5 core logic actually needs.

**Recommendation**
- Build a deterministic Stage2-to-`db_v2` adapter before changing the active
  extractor contract.
- Adapter-first target tables:
  - `document`
  - `formulation_identity`
  - `formulation_measurement`
  - coarse `evidence_binding`
- Second-wave target tables:
  - `formulation_component`
  - `formulation_phase`
  - `formulation_process`
- Only after the adapter stabilizes should the LLM prompt/contract be narrowed.

**Non-change**
- No active pipeline code change is proposed by this parking-lot note alone.
