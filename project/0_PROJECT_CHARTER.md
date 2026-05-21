# Project Charter

This project builds an auditable literature-to-dataset system for PLGA
nanoparticle formulations.

The core goal is to convert heterogeneous scientific papers into
formulation-level structured records that are reusable for data analysis,
machine-learning modeling, and downstream experimental planning.

The project follows a strict division of responsibility:

- LLMs perform open semantic discovery, including formulation-boundary
  recognition, table-scope interpretation, and result-to-formulation binding.
- Deterministic rules preserve source authority, validate contracts, resolve
  relations, materialize source-backed fields, bind evidence, assess risk, and
  prepare audit or modeling outputs.

Every released or reviewed dataset surface must remain reproducible,
traceable to source evidence, and explicit about whether it is a diagnostic,
audit, modeling-ready, or benchmark-valid artifact.
