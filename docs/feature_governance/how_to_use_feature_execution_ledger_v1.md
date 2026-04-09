# How To Use Feature Execution Ledger v1

Diagnostic-only governance-support guide.

1. Open the run's `analysis/feature_execution_ledger_v1.tsv`.
2. Filter `mismatch_flag=yes`.
3. Check `expected_state_for_this_run` versus `actual_state_for_this_run`.
4. Open `analysis/feature_upstream_processing_trace_v1.json` to see which stages already happened upstream.
5. Treat `replay_hidden` as a prompt-visibility limitation, not as proof that the feature was absent.
6. Treat `active_inferred_from_upstream` as upstream-applied behavior, not run-local execution.
7. Only after that, inspect raw prompts, semantic rows, or GT deltas.
