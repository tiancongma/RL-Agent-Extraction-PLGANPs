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
