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
- The legacy compatibility surface is a wide-row TSV that mixes:
  - formulation identity
  - component-like material fields
  - measurement outputs
  - coarse evidence hints
- The authoritative Stage2 boundary is now semantic-object emission, with a
  deterministic projection back into the legacy wide-row surface for unchanged
  downstream consumers.

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

### Literature-Extraction / Experiment-Loop Survey Comparison (2026-05-04)

**Source materials reviewed**
- `Claude_systematic_review_LLM_PLGA_extraction.md`
- `DeepSeek_调研.docx`
- `Gemini_文献抽取与实验闭环.docx`
- `从文献到实验_构建PLGA纳米颗粒配方的可信AI抽取与闭环验证框架.pdf`
- `面向科研文献实验数据抽取与闭环验证的系统性调研.docx`

**Cross-survey consensus**
- The recommended architecture is not pure LLM extraction and not pure rule extraction.
- The common mature pattern is:
  - document parsing and section/table/SI segmentation;
  - high-recall evidence/candidate localization;
  - schema-guided LLM semantic extraction;
  - deterministic unit/range/schema/physical-plausibility validation;
  - formulation identity resolution across labels such as `F1`, optimized formulation, drug-loaded NPs, and table/prose aliases;
  - evidence-grounded output with source quote, table cell, page/location, and provenance;
  - human-in-the-loop review prioritized by uncertainty/conflict;
  - modeling-ready export with extraction confidence;
  - optional active-learning / Bayesian-optimization experimental loop.

**Comparison to current PLGA architecture**
- Current Stage0-Stage5 governance already matches the recommended hybrid architecture:
  - Stage2 keeps LLM semantic discovery authority.
  - Deterministic Stage2 completion, Stage3 relation materialization, and Stage5 closure/validation perform the rule-governed work.
  - Existing surfaces already include evidence packages, normalized table payloads, table-cell/grid bindings, identity scaffold/freeze diagnostics, final-output decision traces, review workbooks, Layer3 compare, and modeling-ready helpers.
- The surveys therefore support the current architecture direction rather than requiring a pipeline redesign.
- Current residual gaps are mostly maturity/integration gaps:
  - field-level evidence validation and quote enforcement;
  - unified PLGA dictionary / ontology / lexicon usage across Stage2 and Stage5;
  - confidence and risk scoring that can feed review queues and downstream modeling;
  - benchmark-valid identity-freeze closure;
  - error-aware modeling export;
  - future experimental-loop selection logic.

**Mature functions worth borrowing directly**
1. Evidence-mandatory value candidate validation.
   - Require source quote/table cell/source file/scope/direct-vs-derived classification for every LLM or deterministic value candidate.
   - Reject candidates with missing evidence, ambiguous scope, derived-as-direct leakage, or conflict with higher authority.
   - Natural home: Stage5 S5-4 value authority validation / merge.
2. Shared PLGA dictionary and table-structure lexicon.
   - Centralize header aliases, material aliases, drug/surfactant/polymer normalization, and paper-local abbreviations.
   - Immediate aliases include `E.E.`, `EE (%)`, `E.E.% ± S.D.`, `P.I.`, `ZP`, `Size`, `Average Size`, PLGA commercial names, PVA/Pluronic/Tween/Lutrol surfaces.
   - Natural home: `table_structure_dictionary_v1.py` plus `value_normalization_lexicon_v1.tsv`.
3. Field-level confidence / risk scoring.
   - Combine evidence strength, source type, deterministic-vs-LLM origin, validator status, identity-freeze status, ambiguity/conflict count, and field source type.
   - Candidate artifacts: `field_confidence_score_v1.tsv`, `field_review_queue_v1.tsv`, `row_confidence_summary_v1.tsv`.
4. Priority human-review queue.
   - Sort review by identity risk, high-impact modeling fields, missing/ambiguous evidence, and validation conflicts instead of asking reviewers to inspect all rows equally.
5. Confidence-aware modeling-ready export.
   - Extend modeling-ready outputs with row confidence, field confidence, evidence coverage, direct/derived flags, and missingness reasons.
   - Use these as sample weights or uncertainty metadata for downstream RF/XGBoost/GPR/BO workflows.
6. Active-learning / Bayesian-optimization experiment-loop design.
   - Treat this as downstream design after extraction confidence and modeling-ready exports stabilize.
   - Candidate selection should consider predictive uncertainty, extraction uncertainty, feature-space diversity, expected improvement, and feasibility.

**Borrow with caution / not immediate mainline changes**
- RAG or agentic retrieval should not redefine Stage2 row universe or formulation membership; it can support fixed-row evidence verification and S5-3 direct value candidate extraction.
- ChemDataExtractor / SciBERT / BioBERT / MatBERT-style NER can be auxiliary baselines or dictionary/NER helpers, but should not replace current table-authority, DOE, identity, and Stage5 governance logic.
- Fine-tuned LLMs are lower priority than fixing evidence validation, dictionary normalization, identity freeze, and deterministic value materialization.
- A wet-lab closed loop should remain a later downstream design target, not a reason to modify active Stage0-Stage5 contracts now.

**Recommended future improvement order**
1. Complete Stage5 S5-3/S5-4 evidence-backed direct value candidate and validation workflow over fixed rows.
2. Finish and wire the shared table-structure / PLGA dictionary layer across Stage2 and Stage5.
3. Add field-level confidence and prioritized review queue outputs.
4. Extend modeling-ready exports with confidence/error metadata.
5. Only after those are stable, design active-learning / experiment-selection sidecars.

**Non-change**
- This note records survey-derived design implications only.
- It does not authorize runtime behavior changes, new Stage2 semantic authority, new row-discovery mechanisms, benchmark-valid claims, or wet-lab closed-loop execution.
