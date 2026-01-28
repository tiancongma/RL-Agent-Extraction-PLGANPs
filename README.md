# RL-Agent-Extraction-PLGANPs

This project implements a **systematic, data-driven workflow for LLM-assisted extraction of nanoparticle formulation data** from scientific literature (PDF / HTML), with a focus on **weak supervision, multi-model consensus, and reproducibility**.

Rather than treating prompt engineering as a manual trial-and-error process, this repository structures the task as a **multi-stage pipeline** with explicit data boundaries, versioned artifacts, and measurable performance metrics.

---

## Project Scope and Goal

**Primary goal**

- Build a reproducible pipeline to extract structured PLGA nanoparticle formulation data from unstructured literature using LLMs.
- Support weak labels, partial ground truth, and rule-based grading to evaluate extraction quality.

**Non-goals (current stage)**

- Industrial-scale deployment
- Fully automated ground truth generation
- End-to-end meta-analysis across all nanoparticle types

---

## Pipeline Overview (Conceptual Stages)

This project follows a staged workflow. The implementation is modular, but the **data flow and responsibilities of each stage are fixed**.

### Stage 0 — Relevance Filtering (Raw Metadata)
- Input: Zotero / database CSV exports
- Operations:
  - Regex pre-filtering
  - LLM-based relevance classification
  - Optional Zotero tagging and snapshot fetching
- Output:
  - Candidate pool of *LLM-relevant papers*
- Location:
  - `data/raw/zotero/`
- Notes:
  - Outputs here are **re-runnable and non-binding**
  - No downstream pipeline should depend on these files directly

---

### Stage 1 — Text Cleaning and Manifest Generation (HTML-first)
- Input:
  - Relevant paper metadata
  - Available HTML / PDF snapshots
- Operations:
  - HTML-first text extraction
  - PDF fallback cleaning
  - Section and table parsing
- Outputs:
  - Cleaned text content
  - Sectioned representations
  - A manifest linking papers to extracted content
- Location:
  - Cleaned content: `data/cleaned/content/`
  - Index files: `data/cleaned/index/`
- **Single source of truth**:
  - `data/cleaned/index/manifest_current.tsv`

---

### Stage 2 — Sampling and Weak Label Extraction
- Input:
  - Cleaned manifest
  - Sample definitions
- Operations:
  - Sample selection (e.g. sample10, sample20)
  - Key-to-text index generation
  - LLM-based weak label extraction
- Outputs:
  - Sample lists
  - `key2txt.tsv`
  - Versioned weak labels
- Location:
  - Samples: `data/cleaned/samples/`
  - Index: `data/cleaned/index/key2txt.tsv`
  - Weak labels: `data/cleaned/labels/weak/`

---

### Stage 3 — Manual Annotation (Ground Truth)
- Input:
  - Weak labels
  - Selected samples
- Operations:
  - Human annotation / correction
- Outputs:
  - Versioned manual labels
- Location:
  - `data/cleaned/labels/manual/`
- Notes:
  - Manual labels are **partial and intentionally imperfect**
  - This project does not assume complete GT coverage

---

### Stage 4 — Rule-based Evaluation and Iteration
- Input:
  - Weak labels
  - Manual labels (when available)
- Operations:
  - Regex / rule-based grading
  - Metric calculation (accuracy, hallucination, etc.)
- Outputs:
  - Evaluation reports
- Notes:
  - This stage supports iterative prompt/parser updates
  - Optimization logic may be rule-based or RL-assisted

---

### Stage 5 — Merge and Publication
- Input:
  - Extraction outputs
  - Labels
  - Evaluation summaries
- Outputs:
  - Final structured dataset
  - Aggregated statistics
- Location:
  - `data/results/`
- Notes:
  - Results are organized by **run_id**, not by vague version names

---

## Repository Structure (How to Navigate This Repo)

```text
project/        Project governance (scope, requirements, state machine)
src/            Executable scripts (no results here)
configs/        Run configurations and standard commands
data/raw/       Raw, re-runnable inputs (Zotero, relevance filtering)
data/cleaned/   Cleaned and indexed data used by the pipeline
data/results/   Extraction outputs, organized by run_id
runs/           Run metadata and pointers (latest.txt)
