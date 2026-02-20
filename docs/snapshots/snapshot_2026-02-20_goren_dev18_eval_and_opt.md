# snapshot_2026-02-20_goren_dev18_eval_and_opt

## ENGINEERING HANDOFF (for ChatGPT/Codex)

### Repo / Branch
- Branch: `feature/ee-coverage-rl`
- HEAD: `a509fc2`
- Working tree status summary:
  - Modified tracked: `src/stage2_sampling_labels/auto_extract_weak_labels_v6.py`
  - New untracked: `data/benchmark/goren_2025/overlap_goren18_v1/**`, multiple `src/stage4_eval/*.py`

### Pipeline Structure (Today)
- Core inputs:
  - `data/cleaned/samples/sample_goren18.jsonl` (key/doi list for 18-paper dev set)
  - `data/cleaned/index/key2txt_goren_2025.tsv` and `data/cleaned/content_goren_2025/text/` (text retrieval)
  - `data/benchmark/goren_2025/NP_dataset_formulations.csv` (curated external benchmark)
  - extracted weak labels runs under `data/results/*`.
- Evaluation layer flow:
  1. overlap scaffold build (DOI subset + EE scaffold)
  2. DOI-level metrics tables (coverage, EE agreement, error concentration)
  3. top-DOI audit exports + hypotheses
  4. dedup and formulation grouping experiments
  5. formulation-level alignment / sensitivity / set-level matching
  6. per-DOI diagnostics and failure profiles
  7. drug-name normalization diagnostics and v3 re-alignment
  8. precision-recovery sweep by row-completeness filtering.

### Entry Scripts Used
- Extraction entry:
  - `src/stage2_sampling_labels/auto_extract_weak_labels_v6.py`
- Evaluation scripts added/used today:
  - `src/stage4_eval/build_goren_overlap_scaffold_v1.py`
  - `src/stage4_eval/compute_goren_metrics_tables_v1.py`
  - `src/stage4_eval/audit_top3_doi_root_cause_v1.py`
  - `src/stage4_eval/apply_extracted_ee_dedup_v1.py`
  - `src/stage4_eval/apply_formulation_grouping_v1.py`
  - `src/stage4_eval/compute_set_level_ee_match_v1.py`
  - `src/stage4_eval/compute_formulation_alignment_v1.py`
  - `src/stage4_eval/apply_global_baseline_inheritance_and_rerun_alignment_v1.py`
  - `src/stage4_eval/compute_alignment_sensitivity_v1.py`
  - `src/stage4_eval/build_per_doi_diagnostics_v1.py`
  - `src/stage4_eval/build_failure_profile_v1.py`
  - `src/stage4_eval/compare_drugname_sets_v1.py`
  - `src/stage4_eval/run_alignment_v3_surfactant_drugnorm.py`
  - `src/stage4_eval/precision_recovery_experiment_v1.py`

### Key Directories
- Benchmark/eval artifacts root:
  - `data/benchmark/goren_2025/overlap_goren18_v1/`
- Iterative alignment outputs:
  - `.../formulation_group_v1/`
  - `.../formulation_group_v2_corefields/`
  - `.../formulation_group_v3_surfactant_drugnorm/`
- Dev extraction runs:
  - `data/results/dev18_corefields_prompt_v1/`
  - `data/results/dev18_surfactant_schema_v1/`
  - validation runs: `data/results/run_20260220_1413_a509fc2_dev18_force_default_weaklabels_v1/`, `data/results/dev18_force_explicit/`.

### Representative Commands Used
- Overlap scaffold:
  - `python src/stage4_eval/build_goren_overlap_scaffold_v1.py --weak-tsv data/results/run_20260219_1623_780eb83_goren18_weaklabels_v1/weak_labels__gemini.tsv`
- Metrics tables:
  - `python src/stage4_eval/compute_goren_metrics_tables_v1.py`
- Formulation grouping/alignment baseline:
  - `python src/stage4_eval/apply_formulation_grouping_v1.py ...`
  - `python src/stage4_eval/compute_formulation_alignment_v1.py ...`
- Dev18 extraction re-run (core fields then surfactant schema):
  - `python src/stage2_sampling_labels/auto_extract_weak_labels_v6.py --sample-jsonl ... --key2txt ... --model gemini-2.5-flash --out-tsv data/results/dev18_corefields_prompt_v1/weak_labels__gemini.tsv --out-jsonl data/results/dev18_corefields_prompt_v1/weak_labels__gemini.jsonl --verbose`
  - `python src/stage2_sampling_labels/auto_extract_weak_labels_v6.py --sample-jsonl ... --key2txt ... --model gemini-2.5-flash --out-tsv data/results/dev18_surfactant_schema_v1/weak_labels__gemini.tsv --out-jsonl data/results/dev18_surfactant_schema_v1/weak_labels__gemini.jsonl --verbose`
- V3 alignment with drug-name normalization:
  - `python src/stage4_eval/run_alignment_v3_surfactant_drugnorm.py`
- Precision recovery sweep:
  - `python src/stage4_eval/precision_recovery_experiment_v1.py`

### Key Metrics (Baseline -> Final)
| Stage | Recall | Precision | Source |
|---|---:|---:|---|
| v1 formulation alignment | 0.3228346456692913 | 0.65625 | `.../formulation_group_v1/formulation_alignment_summary.tsv` |
| v2 corefields prompt | 0.3543307086614173 | 0.6428571428571429 | `.../formulation_group_v2_corefields/formulation_alignment_summary.tsv` |
| v3 surfactant+drug normalization | 0.5511811023622047 | 0.3978494623655914 | `.../formulation_group_v3_surfactant_drugnorm/formulation_alignment_summary_v3.tsv` |
| precision recovery best filter (`n_core_present >= 2`) | 0.5354330708661418 | 0.3956043956043956 | `.../precision_recovery_v1/precision_recovery_sweep.tsv` |

Additional benchmark-level metrics:
- DOI overlap: 18, curated rows: 127, extracted rows: 217 (`metrics_tables_v1/table1_structure_coverage.tsv`).
- EE mean abs diff summary: mean 4.907824074074075, median 0.2466666666666697, `n<=5`: 15/18 (`metrics_tables_v1/table2_ee_agreement.tsv`).

Drug-name diagnostics:
- Before v3 normalization layer (set compare): `exact_match=15, overlap_partial=2, no_overlap=1` (`diagnostics_v3_drugname/drugname_build_log.json`).
- After v3 normalization in alignment script: `exact_match=18` (`formulation_group_v3_surfactant_drugnorm/build_log.json`).
- Target DOI diagnostics after normalization:
  - `10.1016/j.ejpb.2004.09.002`: exact_match
  - `10.2147/ijn.s130908`: exact_match
  - `10.2147/ijn.s77498`: exact_match

### Current Status + Next Steps
- Current status:
  - External benchmark evaluation stack is complete and reproducible for dev18.
  - Recall improved significantly from v1 to v3, with a precision drop due to higher extracted formulation count.
  - Precision-recovery filter sweep did not recover precision without recall loss in current setup.
- Next steps (engineering handoff):
  1. Add stricter row-level dedup/consolidation keyed by condition columns before alignment scoring.
  2. Introduce explicit drug/polymer structured field in extraction schema (currently proxy via text) and rerun v3.
  3. Add ablation run to isolate impact of surfactant schema vs drug normalization separately.

## USER-FACING SUMMARY (minimal bullets)

这个外部基准（Goren 2025）用于检验系统在真实文献上的结构化提取是否可靠，特别是配方字段和包封率（EE）的一致性。今天的工作重点是把评估链路补齐，并逐步定位“为什么匹配不上”。

我们先搭建了 DOI 级别的重叠子集与 EE 一致性指标，再扩展到配方级对齐、失败类型拆解和低召回 DOI 的诊断卡。之后又做了两轮优化：一轮是核心字段提示增强，另一轮是药名归一化（去括号别名、剔除示踪剂）。

结果上，配方级召回率从 0.3228 提升到 0.5512，但精确率从 0.6563 下降到 0.3978。这说明系统找到了更多可能匹配项，但也引入了更多候选行，导致误匹配空间变大。当前未解决的问题是“高召回下的精确率回收”，下一步应优先做更强的去重/合并策略与字段级约束，而不是继续放宽匹配规则。

## APPENDIX: FILE-BY-FILE ARTIFACT INDEX

> 注：以下为 2026-02-20 当天在本工作流中“相关且被创建/修改”的脚本与产物（只列与 Goren dev18 评估/优化直接相关）。

### A. Scripts (src)

| timestamp | path | type | generated by | contains | useful for | status |
|---|---|---|---|---|---|---|
| 2026-02-20 11:44:50 | `src/stage4_eval/build_goren_overlap_scaffold_v1.py` | script | manual coding | overlap subset + DOI EE scaffold builder | baseline benchmark scaffolding | frozen prototype |
| 2026-02-20 12:28:49 | `src/stage4_eval/compute_goren_metrics_tables_v1.py` | script | manual coding | weekly-report metrics tables | standardized KPI reporting | frozen output generator |
| 2026-02-20 12:41:53 | `src/stage4_eval/audit_top3_doi_root_cause_v1.py` | script | manual coding | top3 outlier audit exports | evidence-ready row comparison | intermediate/frozen |
| 2026-02-20 12:47:35 | `src/stage4_eval/apply_extracted_ee_dedup_v1.py` | script | manual coding | DOI EE dedup experiment | quantify over-splitting effect | experiment |
| 2026-02-20 12:52:15 | `src/stage4_eval/apply_formulation_grouping_v1.py` | script | manual coding | DOI内按配方签名分组 | reduce condition mixing | experiment |
| 2026-02-20 12:57:25 | `src/stage4_eval/compute_set_level_ee_match_v1.py` | script | manual coding | set-level EE matching metrics | alternative to mean-only comparison | experiment |
| 2026-02-20 13:02:20 | `src/stage4_eval/compute_formulation_alignment_v1.py` | script | manual coding | formulation recall/precision evaluator | core alignment metric | frozen baseline evaluator |
| 2026-02-20 13:19:04 | `src/stage4_eval/apply_global_baseline_inheritance_and_rerun_alignment_v1.py` | script | manual coding | global baseline inheritance + rerun alignment | inheritance effect validation | experiment |
| 2026-02-20 13:24:10 | `src/stage4_eval/compute_alignment_sensitivity_v1.py` | script | manual coding | strict/core/minimal sensitivity sweep | matching rule robustness check | experiment |
| 2026-02-20 13:35:03 | `src/stage4_eval/build_per_doi_diagnostics_v1.py` | script | manual coding | per-DOI diagnostic merge/rollup/queue | triage and audit prioritization | frozen diagnostics |
| 2026-02-20 13:41:47 | `src/stage4_eval/build_failure_profile_v1.py` | script | manual coding | missing/conflict/failure-type profiling | low-recall root-cause labeling | frozen diagnostics |
| 2026-02-20 14:53:05 | `src/stage4_eval/compare_drugname_sets_v1.py` | script | manual coding | curated vs extracted drug set diff | identify name mismatch artifacts | frozen diagnostics |
| 2026-02-20 15:05:40 | `src/stage4_eval/run_alignment_v3_surfactant_drugnorm.py` | script | manual coding | v3 alignment with drug normalization | main improved-recall eval track | current main experiment |
| 2026-02-20 15:15:55 | `src/stage4_eval/precision_recovery_experiment_v1.py` | script | manual coding | completeness filter + consolidation sweep | precision recovery analysis | experiment |
| 2026-02-20 14:30:36 | `src/stage2_sampling_labels/auto_extract_weak_labels_v6.py` | script (modified) | prompt/schema updates | extraction schema + output policy changes | produced dev18 reruns used by eval | modified tracked |

### B. Core benchmark outputs (overlap_goren18_v1 root)

| timestamp | path | type | generated by | contains | useful for | status |
|---|---|---|---|---|---|---|
| 2026-02-20 11:44:54 | `data/benchmark/goren_2025/overlap_goren18_v1/goren18_curated_overlap_subset.tsv` | tsv | `build_goren_overlap_scaffold_v1.py` | curated overlap rows for 18 DOI | benchmark reference slice | frozen |
| 2026-02-20 11:44:54 | `data/benchmark/goren_2025/overlap_goren18_v1/rows_per_doi.tsv` | tsv | same | row count per DOI | coverage diagnostics | frozen |
| 2026-02-20 11:44:54 | `data/benchmark/goren_2025/overlap_goren18_v1/coverage_report.json` | json | same | overlap DOI/row totals | quick coverage check | frozen |
| 2026-02-20 11:44:54 | `data/benchmark/goren_2025/overlap_goren18_v1/doi_level_ee_scaffold.tsv` | tsv | same | DOI EE agreement scaffold | upstream for many analyses | frozen |
| 2026-02-20 11:44:54 | `data/benchmark/goren_2025/overlap_goren18_v1/audit_priority__doi.tsv` | tsv | same | prioritized DOI audit list | root-cause queue | frozen |
| 2026-02-20 11:44:54 | `data/benchmark/goren_2025/overlap_goren18_v1/agreement_summary.json` | json | same | compact DOI EE summary stats | reporting | frozen |
| 2026-02-20 11:44:54 | `data/benchmark/goren_2025/overlap_goren18_v1/build_log.json` | json | same | columns/counts/provenance | reproducibility | frozen |

### C. Metrics + audit outputs

| timestamp | path | type | generated by | contains | useful for | status |
|---|---|---|---|---|---|---|
| 2026-02-20 12:28:56 | `data/benchmark/goren_2025/overlap_goren18_v1/metrics_tables_v1/table1_structure_coverage.tsv` | tsv | `compute_goren_metrics_tables_v1.py` | structure coverage metrics | weekly report table 1 | frozen |
| 2026-02-20 12:28:56 | `.../metrics_tables_v1/table2_ee_agreement.tsv` | tsv | same | EE agreement metrics | weekly report table 2 | frozen |
| 2026-02-20 12:28:56 | `.../metrics_tables_v1/table3_error_distribution.tsv` | tsv | same | error concentration metrics | weekly report table 3 | frozen |
| 2026-02-20 12:28:56 | `.../metrics_tables_v1/top_doi_for_audit.tsv` | tsv | same | top DOI audit shortlist | manual audit entrypoint | frozen |
| 2026-02-20 12:28:56 | `.../metrics_tables_v1/metrics_build_log.json` | json | same | metrics provenance | reproducibility | frozen |
| 2026-02-20 12:41:59 | `.../audit_top3_v1/audit_top3_summary.tsv` | tsv | `audit_top3_doi_root_cause_v1.py` | top3 DOI summary | focused audit report | frozen |
| 2026-02-20 12:41:59 | `.../audit_top3_v1/audit_10_1016_j_ejpb_2004_09_002.tsv` | tsv | same | curated+extracted evidence dump | DOI-specific debugging | frozen |
| 2026-02-20 12:41:59 | `.../audit_top3_v1/audit_10_1039_c5ra27386b.tsv` | tsv | same | curated+extracted evidence dump | DOI-specific debugging | frozen |
| 2026-02-20 12:41:59 | `.../audit_top3_v1/audit_10_1089_jamp_2009_0759.tsv` | tsv | same | curated+extracted evidence dump | DOI-specific debugging | frozen |
| 2026-02-20 12:41:59 | `.../audit_top3_v1/audit_top3_hypotheses.tsv` | tsv | same | root-cause hypothesis tags | triage and reporting | frozen |

### D. EE and formulation experiments

| timestamp | path | type | generated by | contains | useful for | status |
|---|---|---|---|---|---|---|
| 2026-02-20 12:47:43 | `.../dedup_ee_v1/extracted_ee_dedup_rows.tsv` | tsv | `apply_extracted_ee_dedup_v1.py` | deduped EE rows | over-splitting test | intermediate |
| 2026-02-20 12:47:43 | `.../dedup_ee_v1/doi_level_ee_scaffold__dedup.tsv` | tsv | same | DOI before/after dedup stats | dedup impact analysis | intermediate |
| 2026-02-20 12:47:43 | `.../dedup_ee_v1/dedup_impact_summary.tsv` | tsv | same | per-DOI delta abs diff | identify benefited DOI | intermediate |
| 2026-02-20 12:52:20 | `.../formulation_group_v1/extracted_formulation_level_v1.tsv` | tsv | `apply_formulation_grouping_v1.py` | grouped extracted formulations | alignment input baseline | frozen baseline |
| 2026-02-20 12:52:20 | `.../formulation_group_v1/doi_level_ee_scaffold__formulation_grouped.tsv` | tsv | same | grouped DOI EE view | grouping effect | intermediate |
| 2026-02-20 12:52:20 | `.../formulation_group_v1/formulation_group_impact_summary.tsv` | tsv | same | before/after grouping impact | quick scan | intermediate |
| 2026-02-20 12:57:31 | `.../formulation_group_v1/doi_level_set_match.tsv` | tsv | `compute_set_level_ee_match_v1.py` | set-level EE matching outcomes | alternative metric | intermediate |
| 2026-02-20 12:57:31 | `.../formulation_group_v1/set_match_summary.tsv` | tsv | same | aggregate set-match summary | reporting | intermediate |
| 2026-02-20 13:02:35 | `.../formulation_group_v1/formulation_alignment.tsv` | tsv | `compute_formulation_alignment_v1.py` | formulation-level matches | baseline evaluator output | frozen baseline |
| 2026-02-20 13:02:35 | `.../formulation_group_v1/formulation_alignment_summary.tsv` | tsv | same | baseline recall/precision | key baseline metric | frozen baseline |
| 2026-02-20 13:02:35 | `.../formulation_group_v1/per_doi_recall.tsv` | tsv | same | per-DOI recall | downstream diagnostics | frozen baseline |
| 2026-02-20 13:19:10 | `.../global_inherit_v1/*` | mixed | `apply_global_baseline_inheritance_and_rerun_alignment_v1.py` | global baseline inheritance experiment outputs | inheritance ablation | intermediate |
| 2026-02-20 13:24:15 | `.../formulation_group_v1/alignment_sensitivity.tsv` | tsv | `compute_alignment_sensitivity_v1.py` | strict/core/minimal sensitivity | matching-rule check | intermediate |

### E. Diagnostics packs

| timestamp | path | type | generated by | contains | useful for | status |
|---|---|---|---|---|---|---|
| 2026-02-20 13:35:09 | `.../diagnostics_v1/per_doi_diagnostic_v1.tsv` | tsv | `build_per_doi_diagnostics_v1.py` | merged per-DOI diagnostic table | root-cause pivot | frozen |
| 2026-02-20 13:35:09 | `.../diagnostics_v1/diagnostic_rollup_v1.tsv` | tsv | same | bucket rollups and contingency | summary reporting | frozen |
| 2026-02-20 13:35:09 | `.../diagnostics_v1/audit_queue_by_recall_v1.tsv` | tsv | same | top low-recall queue | audit prioritization | frozen |
| 2026-02-20 13:35:09 | `.../diagnostics_v1/diagnostics_build_log.json` | json | same | diagnostics provenance | reproducibility | frozen |
| 2026-02-20 13:41:52 | `.../diagnostics_v2_failure_profile/per_doi_extracted_missingness.tsv` | tsv | `build_failure_profile_v1.py` | field missingness by DOI | failure decomposition | frozen |
| 2026-02-20 13:41:52 | `.../diagnostics_v2_failure_profile/per_doi_extracted_conflicts.tsv` | tsv | same | field conflict rates by DOI | failure decomposition | frozen |
| 2026-02-20 13:41:52 | `.../diagnostics_v2_failure_profile/per_doi_alignment_failure_fields.tsv` | tsv | same | unmatched field never-match fractions | failure decomposition | frozen |
| 2026-02-20 13:41:52 | `.../diagnostics_v2_failure_profile/per_doi_failure_type.tsv` | tsv | same | A/B/C failure labels | action planning | frozen |
| 2026-02-20 13:41:52 | `.../diagnostics_v2_failure_profile/low_recall_action_cards.tsv` | tsv | same | top low-recall action cards | manual review guide | frozen |
| 2026-02-20 13:41:52 | `.../diagnostics_v2_failure_profile/diagnostics_build_log.json` | json | same | provenance | reproducibility | frozen |
| 2026-02-20 14:53:10 | `.../diagnostics_v3_drugname/per_doi_drugname_sets.tsv` | tsv | `compare_drugname_sets_v1.py` | per-DOI drug set compare | diagnose name mismatch artifacts | frozen |
| 2026-02-20 14:53:10 | `.../diagnostics_v3_drugname/per_doi_drugname_problem_cases.tsv` | tsv | same | problematic DOI subset | focused cleanup targets | frozen |
| 2026-02-20 14:53:10 | `.../diagnostics_v3_drugname/drugname_build_log.json` | json | same | status counts and provenance | reproducibility | frozen |

### F. Dev extraction runs + v2/v3 alignment tracks

| timestamp | path | type | generated by | contains | useful for | status |
|---|---|---|---|---|---|---|
| 2026-02-20 14:02:58 | `data/results/dev18_corefields_prompt_v1/weak_labels__gemini.jsonl` | jsonl | `auto_extract_weak_labels_v6.py` | dev18 core-fields run raw output | v2 eval input | intermediate |
| 2026-02-20 14:02:58 | `data/results/dev18_corefields_prompt_v1/weak_labels__gemini.tsv` | tsv | same | flattened extraction rows | v2 eval input | intermediate |
| 2026-02-20 14:03:10 | `.../formulation_group_v2_corefields/extracted_formulation_level_v1.tsv` | tsv | `apply_formulation_grouping_v1.py` | v2 grouped extracted formulations | v2 alignment input | intermediate |
| 2026-02-20 14:03:10 | `.../formulation_group_v2_corefields/doi_level_ee_scaffold__formulation_grouped.tsv` | tsv | same | v2 grouped DOI EE view | v2 analysis | intermediate |
| 2026-02-20 14:03:10 | `.../formulation_group_v2_corefields/formulation_group_impact_summary.tsv` | tsv | same | v2 grouping impact | v2 analysis | intermediate |
| 2026-02-20 14:03:23 | `.../formulation_group_v2_corefields/formulation_alignment.tsv` | tsv | `compute_formulation_alignment_v1.py` | v2 alignment details | v2 evaluation | intermediate |
| 2026-02-20 14:03:23 | `.../formulation_group_v2_corefields/per_doi_recall.tsv` | tsv | same | v2 per-DOI recall | v2 diagnostics | intermediate |
| 2026-02-20 14:03:23 | `.../formulation_group_v2_corefields/formulation_alignment_summary.tsv` | tsv | same | v2 recall/precision headline | baseline for v3 delta | frozen baseline |
| 2026-02-20 14:14:48 | `data/results/run_20260220_1413_a509fc2_dev18_force_default_weaklabels_v1/*` | jsonl/tsv | `auto_extract_weak_labels_v6.py` | run_id-default-path validation run | output-policy verification | intermediate |
| 2026-02-20 14:16:03 | `data/results/dev18_force_explicit/*` | jsonl/tsv | `auto_extract_weak_labels_v6.py` | explicit-path validation run | output-policy verification | intermediate |
| 2026-02-20 14:46:01 | `data/results/dev18_surfactant_schema_v1/weak_labels__gemini.jsonl` | jsonl | `auto_extract_weak_labels_v6.py` | dev18 run with surfactant schema | v3 main input | current candidate |
| 2026-02-20 14:46:01 | `data/results/dev18_surfactant_schema_v1/weak_labels__gemini.tsv` | tsv | same | flattened extraction rows (new schema) | v3 main input | current candidate |
| 2026-02-20 15:05:46 | `.../formulation_group_v3_surfactant_drugnorm/extracted_formulation_level_v3.tsv` | tsv | `run_alignment_v3_surfactant_drugnorm.py` | v3 grouped rows with drug normalization | v3 alignment input | current output |
| 2026-02-20 15:05:46 | `.../formulation_group_v3_surfactant_drugnorm/formulation_alignment_v3.tsv` | tsv | same | v3 alignment details | current evaluation | current output |
| 2026-02-20 15:05:46 | `.../formulation_group_v3_surfactant_drugnorm/formulation_alignment_summary_v3.tsv` | tsv | same | v3 recall/precision headline | current headline metric | current output |
| 2026-02-20 15:05:46 | `.../formulation_group_v3_surfactant_drugnorm/per_doi_recall_v3.tsv` | tsv | same | v3 per-DOI recall | v3 diagnostics | current output |
| 2026-02-20 15:05:46 | `.../formulation_group_v3_surfactant_drugnorm/per_doi_failure_type_v3.tsv` | tsv | same | v3 failure-type labels | triage | current output |
| 2026-02-20 15:05:46 | `.../formulation_group_v3_surfactant_drugnorm/low_recall_action_cards_v3.tsv` | tsv | same | v3 low-recall action cards | triage | current output |
| 2026-02-20 15:05:46 | `.../formulation_group_v3_surfactant_drugnorm/build_log.json` | json | same | normalization rules + status counts | provenance | current output |
| 2026-02-20 15:16:01 | `.../precision_recovery_v1/diagnostics_row_completeness.tsv` | tsv | `precision_recovery_experiment_v1.py` | row-level core-field completeness diagnostics | precision recovery analysis | experiment |
| 2026-02-20 15:16:02 | `.../precision_recovery_v1/precision_recovery_sweep.tsv` | tsv | same | filter sweep metrics | choose best tradeoff | experiment |
| 2026-02-20 15:16:02 | `.../precision_recovery_v1/extracted_best_filter_consolidated.tsv` | tsv | same | consolidated filtered set | post-filter alignment test | experiment |
| 2026-02-20 15:16:02 | `.../precision_recovery_v1/build_log.json` | json | same | experiment provenance | reproducibility | experiment |
