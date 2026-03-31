#!/usr/bin/env python3
from __future__ import annotations

"""
Compatibility shim for the historical three-paper extractor name.

The governed reusable semantic extractor now lives in:
`src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`

This file is retained so older comparison wrappers and docs do not break
immediately, but scope must be expressed through manifest rows and `--paper-key`
arguments rather than by treating this filename as the Stage2 definition.
"""

from src.stage2_sampling_labels.extract_semantic_stage2_objects_v2 import main


if __name__ == "__main__":
    main()
