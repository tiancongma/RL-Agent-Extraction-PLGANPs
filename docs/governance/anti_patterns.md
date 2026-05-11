**Architectural Anti-Patterns Observed**

**1. Execution boundary drift**
Deterministic execution moved to a later step without hard enforcement that the step must run. This creates valid-but-incomplete Stage2 outputs that are treated as usable.

**2. LLM fallback regains de facto authority**
When deterministic completion is skipped or starved, downstream rows reflect only what the LLM emitted directly. This re-promotes LLM-only behavior, which the architecture explicitly forbids as the completed Stage2 authority.

**3. Execution hooks removed from the primary path**
The composite Stage2 path originally guaranteed functional-unit invocation. Decomposition introduced a path where functional units exist but are no longer invoked by default.

**4. Artifact boundary loss**
The data required by functional units is not always guaranteed to cross S-step boundaries. The completion step may lack the necessary table assets, authorization markers, or manifest context, leading to silent skips.

**5. Observability without enforceability**
Artifacts can show semantic markers without proving that the units ran. This creates a false sense of correctness because observability exists, but execution cannot be demonstrated.

