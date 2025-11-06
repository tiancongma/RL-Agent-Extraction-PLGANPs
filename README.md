# RL-Agent-Extraction-PLGANPs
This project is an automated reinforcement learning-based prompt engineering workflow.

It's designed as a closed-loop system to automatically evaluate and optimize the performance of a Large Language Model (LLM). The core of the project is a Rule-based Grader that accurately calculates key metrics like extraction accuracy and hallucination rates based on pre-annotated data.

This workflow provides a scalable framework not just for validating initial prompts but also for continuously improving them through an iterative process. The ultimate goal is to boost the LLM's performance on specific tasks, such as extracting structured information from unstructured documents like PDFs and HTML files.

The key value of this project is that it transforms prompt engineering from a manual, trial-and-error process into a systematic, data-driven, and automated workflow, significantly improving development efficiency and model performance.

```mermaid
flowchart TB

A0["**Stage 0 — Pre-filtering and Relevance**"]
A1["Raw CSV export (Zotero or DB)"]
A2["prefilter_regex.py"]
A3["classify_gemini_grouped.py (LLM relevance)"]
A4["auto_tag_plga_gemini.py / auto_tag_plga_openai.py"]
A5["zotero_tag_sync.py"]
A6["zotero_fetch_llm_relevant_pdfs.py"]
A7["fill_missing_snapshots.py (HTML)"]
A0 --> A1 --> A2 --> A3 --> A4 --> A5 --> A6 --> A7

B0["**Stage 1 — CSV → Clean → Manifest (HTML first)**"]
B1["csv2clean_manifest.py (--csv --outroot --pdf2clean-path)"]
B2["pdf2clean.py (PDF → .cleaned.txt / .sections.json)"]
B3["Outputs: text/<stem>.cleaned.txt; sections/<stem>.sections.json"]
B4["Manifest: data/cleaned/manifests/zotero_llm_relevant.tsv|jsonl"]
B0 --> B1 --> B4
B1 -->|HTML found| B3
B1 -->|no HTML → call| B2 --> B3

C0["**Stage 2 — Sampling and Weak Label Extraction**"]
C1["sample_from_manifest_html_first.py"]
C2["Outputs: sampleN.jsonl; key2txt.tsv"]
C3["auto_extract_weak_labels_v4.py"]
C4["Outputs: weak_labels_v4.jsonl; weak_labels_v4.tsv"]
C0 --> C1 --> C2 --> C3 --> C4

D0["**Stage 3 — Manual Label Template (GT)**"]
D1["gt_tool.py / gt_tool_v3.py"]
D2["Outputs: manual_labels_vX.tsv"]
D0 --> D1 --> D2

E0["**Stage 4 — Rule-based Scoring and RL Optimization**"]
E1["regex_grader.py (accuracy, hallucination)"]
E2["Reports"]
E3["Prompt / parser update"]
E0 --> E1 --> E2 --> E3
E3 -.-> C3

F0["**Stage 5 — Merge and Publish**"]
F1["merge_results.py"]
F2["Final dataset: plga_dataset.tsv/json + summary stats"]
F0 --> F1 --> F2

A7 --> B1
B3 --> C1
B4 --> C1
C4 --> D1
D2 --> E1
C4 --> E1
E2 --> F1
D2 --> F1
```
