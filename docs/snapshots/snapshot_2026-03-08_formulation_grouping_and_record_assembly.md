# Snapshot: Formulation Grouping and Record Assembly (2026-03-08)

# Project goal
The project builds a clean, auditable, and reproducible tabular formulation database extracted from scientific literature.  
The target is downstream scientific analysis (statistical analysis, predictive modeling, and empirical pattern discovery), and the scope is not limited to EE-only use cases.

# Final output constraint
The final release output remains tabular: one row per formulation and one column per structured parameter.  
Intermediate representations may be richer, but released artifacts must stay table-compatible for scientific workflows.

# Current pipeline
Current architecture after cleaning:
- document preprocessing
- semantic extraction (LLM)
- formulation hypothesis
- evidence binding
- formulation assembly
- formulation-level audit
- final table export

# Current position
Stabilized:
- Environment/path contracts and cleaned-content boundaries
- Layered responsibility split (LLM semantic extraction vs deterministic binding/audit/export)
- Requirement that final outputs are auditable and tabular

Still evolving:
- Upstream extraction schema quality for formulation-level structure
- Formulation grouping robustness in complex multi-instance papers
- Record-level evidence package completeness for high-confidence verification

# Current main bottleneck
The primary bottleneck is formulation grouping quality: instance boundary detection and inheritance interpretation (what is shared vs overridden across formulations).  
The dominant failure mode is incorrect record reconstruction from distributed evidence, not second-model agreement rate.

# Second-model status
Parallel second-model extraction is currently deprioritized as the main development focus.  
A second model may later be used as a selective verifier for high-risk formulation hypotheses after record-level evidence packages and assembly audit signals are more stable.

# Practical instruction for future AI sessions
Read these files first, in order:
1. `project_specification.txt`
2. `project/2_ARCHITECTURE.md`
3. `project/4_DECISIONS_LOG.md`
4. `project/PIPELINE_SCRIPT_MAP.md`
5. `docs/snapshots/snapshot_2026-03-08_formulation_grouping_and_record_assembly.md`
