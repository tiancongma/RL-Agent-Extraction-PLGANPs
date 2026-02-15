```mermaid
flowchart LR
  %% Figure 1: End-to-end pipeline schematic (auditable GT reference frame)

  A["Scientific Literature<br/>(PDF or HTML)"]
  B["Auditable Text View (Frozen)<br/>Sole reference frame for evidence and GT"]

  M1["Model 1 Extraction<br/>Candidate values from local inference<br/>over provided text snapshot"]
  M2["Model 2 Extraction<br/>Candidate values from local inference<br/>over provided text snapshot"]

  Q["Merge and Disagreement Analysis (QC)<br/>Agreement and disagreement as weak supervision signals"]

  E["Post-hoc Evidence Alignment (System)<br/>Deterministic evidence spans from auditable text view<br/>Independent of LLM internal reasoning"]

  G["GT Decision Tool (Human-in-the-loop)<br/>Field-level decisions based strictly on auditable evidence<br/>No expert inference"]

  O["Outputs (Structured and Auditable)<br/>Consensus weak labels<br/>Conflict queue and GT decisions<br/>Uncertainty annotations (unclear)"]

  U["unclear<br/>Epistemic indeterminacy under system constraints<br/>Not a model failure"]

  %% Edges
  A -->|"Deterministic parsing and normalization by system"| B

  B -->|"Identical text snapshots provided as inputs"| M1
  B -->|"Identical text snapshots provided as inputs"| M2

  M1 -->|"Merge and compare field-level outputs"| Q
  M2 -->|"Merge and compare field-level outputs"| Q

  Q -->|"Route candidate extractions for evidence alignment"| E
  B -->|"Evidence source for alignment"| E

  E -->|"Aligned evidence plus model proposals for adjudication"| G

  G -->|"Apply GT only to contested fields"| O
  G -->|"If evidence is insufficient or not reliably exposed"| U
  U --> O
