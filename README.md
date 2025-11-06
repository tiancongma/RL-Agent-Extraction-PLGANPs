# RL-Agent-Extraction-PLGANPs
This project is an automated reinforcement learning-based prompt engineering workflow.

It's designed as a closed-loop system to automatically evaluate and optimize the performance of a Large Language Model (LLM). The core of the project is a Rule-based Grader that accurately calculates key metrics like extraction accuracy and hallucination rates based on pre-annotated data.

This workflow provides a scalable framework not just for validating initial prompts but also for continuously improving them through an iterative process. The ultimate goal is to boost the LLM's performance on specific tasks, such as extracting structured information from unstructured documents like PDFs and HTML files.

The key value of this project is that it transforms prompt engineering from a manual, trial-and-error process into a systematic, data-driven, and automated workflow, significantly improving development efficiency and model performance.

```mermaid
flowchart TB

  %% ---------- L0: prefilter & relevance ----------
  subgraph L0["0) Pre-filtering & Relevance Classification"]
    A1[Raw CSV export (Zotero / DB)]
    A2[prefilter_regex.py (local regex filter)]
    A3[classify_gemini_grouped.py (LLM relevance)]
    A4[auto_tag_plga_gemini.py / auto_tag_plga_openai.py (auto-tag)]
    A5[zotero_tag_sync.py (sync tags)]
    A6[zotero_fetch_llm_relevant_pdfs.py (batch PDFs)]
    A7[fill_missing_snapshots.py (fill HTML)]
    A1 --> A2 --> A3 --> A4 --> A5 --> A6 --> A7
  end

  %% ---------- L1: CSV -> Clean -> Manifest ----------
  subgraph L1["1) CSV → Clean → Manifest (HTML preferred)"]
    B1[csv2clean_manifest.py (--csv --outroot --pdf2clean-path)]
    B2[pdf2clean.py (PDF → .cleaned.txt / .sections.json)]
    B3((Outputs:\ntext/<stem>.cleaned.txt\nsections/<stem>.sections.json))
    B4((Manifest:\ndata/cleaned/manifests/\nzotero_llm_relevant.tsv|jsonl))
    B1 -->|HTML found| B3
    B1 -->|No HTML → call| B2 --> B3
    B1 --> B4
  end

  %% ---------- L2: sampling & weak labels ----------
  subgraph L2["2) Sampling & Weak Label Extraction"]
    C1[sample_from_manifest_html_first.py]
    C2((Outputs:\nsampleN.jsonl\nkey2txt.tsv))
    C3[auto_extract_weak_labels_v4.py (LLM extraction)]
    C4((Outputs:\nweak_labels_v4.jsonl\nweak_labels_v4.tsv))
    C1 --> C2 --> C3 --> C4
  end

  %% ---------- L3: GT ----------
  subgraph L3["3) Manual Label Template (GT)"]
    D1[gt_tool.py / gt_tool_v3.py]
    D2((Outputs:\nmanual_labels_vX.tsv))
    D1 --> D2
  end

  %% ---------- L4: grading & RL ----------
  subgraph L4["4) Rule-based Scoring & RL Optimization"]
    E1[regex-based grader (accuracy / hallucination)]
    E2((Reports))
    E3[prompt / parser update]
    E1 --> E2 --> E3
    E3 -.-> C3
  end

  %% ---------- L5: merge & publish ----------
  subgraph L5["5) Merge & Publish"]
    F1[merge_results.py]
    F2((Final dataset:\nplga_dataset.tsv/json\n+ summary stats))
    F1 --> F2
  end

  %% ---------- cross-layer links ----------
  A7 --> B1
  B3 --> C1
  B4 --> C1
  C4 --> D1
  D2 --> E1
  C4 --> E1
  E2 --> F1
  D2 --> F1
```
