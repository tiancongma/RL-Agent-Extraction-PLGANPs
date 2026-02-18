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


## 20260218 Multi-span Evidence & Composite Method Representation (Deferred)

During audit of the sample20 evidence-verifier pipeline, we observed that certain procedural fields (e.g., emul_method) often span multiple sentences and contain composite operational details (e.g., base architecture + homogenizer parameters + duration). The current verifier enforces single-span evidence grounding for conservatism, which may flag such cases as insufficient even when the full procedural description is distributed across the Methods section.

We acknowledge that a future enhancement could support multi-span evidence aggregation and structured composite representations (e.g., decomposing procedural methods into architecture + energy steps + parameter tuples). However, introducing multi-span grounding would require redesign of the evidence contract, verifier logic, and downstream field schema.

For the current manuscript (Phase I publication), we intentionally retain the strict single-span evidence constraint to ensure methodological clarity, auditability, and minimal architectural complexity. Multi-span grounding and composite method modeling are deferred to a future iteration after validation of the core evidence-grounded pipeline.