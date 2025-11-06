# RL-Agent-Extraction-PLGANPs
This project is an automated reinforcement learning-based prompt engineering workflow.

It's designed as a closed-loop system to automatically evaluate and optimize the performance of a Large Language Model (LLM). The core of the project is a Rule-based Grader that accurately calculates key metrics like extraction accuracy and hallucination rates based on pre-annotated data.

This workflow provides a scalable framework not just for validating initial prompts but also for continuously improving them through an iterative process. The ultimate goal is to boost the LLM's performance on specific tasks, such as extracting structured information from unstructured documents like PDFs and HTML files.

The key value of this project is that it transforms prompt engineering from a manual, trial-and-error process into a systematic, data-driven, and automated workflow, significantly improving development efficiency and model performance.

```mermaid
flowchart TB
  subgraph L0[0) Pre-filtering & Relevance Classification]
    A1[Raw CSV export\n(Zotero / database)]
    A2[prefilter_regex.py\nLocal regex filtering]
    A3[classify_gemini_grouped.py\nLLM-based relevance classification]
    A4[auto_tag_plga_gemini.py / auto_tag_plga_openai.py\nAutomatic tagging]
    A5[zotero_tag_sync.py\nSync tags to Zotero]
    A6[zotero_fetch_llm_relevant_pdfs.py\nBatch download PDFs]
    A7[fill_missing_snapshots.py\nFill missing HTML snapshots]
    A1 --> A2 --> A3 --> A4 --> A5 --> A6 --> A7
  end

  subgraph L1[1) CSV → Clean → Manifest (HTML preferred)]
    B1[csv2clean_manifest.py\nEntry point (--csv, --outroot, --pdf2clean-path)]
    B2[pdf2clean.py\nPDF → .cleaned.txt & .sections.json]
    B3[(Outputs:\n text/<stem>.cleaned.txt\n sections/<stem>.sections.json)]:::out
    B4[(Manifest:\n data/cleaned/manifests/zotero_llm_relevant.tsv/jsonl)]:::out
    note right of B1
      Logic:
      • If HTML exists → directly clean and save text/sections  
      • Else → call pdf2clean.py  
      • Record source_type, parse_quality, and paths in manifest
    end note
    B1 -->|HTML found| B3
    B1 -->|No HTML → call| B2 --> B3
    B1 --> B4
  end

  subgraph L2[2) Sampling & Weak Label Extraction]
    C1[sample_from_manifest_html_first.py\nSelect sample subset]
    C2[(Outputs:\n sampleN.jsonl\n key2txt.tsv)]:::out
    C3[auto_extract_weak_labels_v4.py\nLLM extraction of formulations]
    C4[(Outputs:\n weak_labels_v4.jsonl / weak_labels_v4.tsv)]:::out
    C1 --> C2 --> C3 --> C4
  end

  subgraph L3[3) Manual Label Template (GT)]
    D1[gt_tool.py / gt_tool_v3.py\nGenerate GT templates]
    D2[(Outputs:\n manual_labels_vX.tsv)]:::out
    D1 --> D2
  end

  subgraph L4[4) Rule-based Scoring & RL Optimization]
    E1[regex-based grader\n(accuracy / hallucination rate)]
    E2[(Reports)]:::out
    E3[prompt / parser update\n(feedback to extraction)]
    E1 --> E2 --> E3
    E3 -.-> C3
  end

  subgraph L5[5) Merge & Publish]
    F1[merge_results.py\nMerge weak + GT labels]
    F2[(Final dataset:\n plga_dataset.tsv/json + summary stats)]:::out
    F1 --> F2
  end

  %% Cross-links
  A7 --> B1
  B3 --> C1
  B4 --> C1
  C4 --> D1
  D2 --> E1
  C4 --> E1
  E2 --> F1
  D2 --> F1

  classDef out fill:#efe,stroke:#484,stroke-width:1px,color:#000;
