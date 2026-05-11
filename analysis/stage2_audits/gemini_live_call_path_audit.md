# Gemini Live Call Path Audit

## Scope
- determine why a prior blocker conclusion reported `GEMINI_API_KEY` as missing
- inspect the maintained Stage2 Gemini live-call path only
- verify repo-local `.env` visibility without printing the secret
- run the smallest bounded maintained Gemini connectivity probe that reuses the maintained initializer and live-call helper

## Prior live-call reference checked
- prior successful maintained reference surface:
  - `data/frozen/dev15_stage2_freeze_v1/s2_4b/RUN_CONTEXT.md`
  - `data/frozen/dev15_stage2_freeze_v1/s2_4b/stage2_s2_4b_run_metadata_v1.json`
  - `data/frozen/dev15_stage2_freeze_v1/s2_4b/analysis/s2_4b_request_summary_v1.tsv`
  - `data/frozen/dev15_stage2_freeze_v1/s2_4b/request_metadata/5GIF3D8W__stage2_v2_request_metadata.json`
- maintained runner used:
  - `src/stage2_sampling_labels/run_stage2_s2_4b_live_llm_call_v1.py`
- maintained live helper used:
  - `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::call_gemini_stream_collect`
- model:
  - `gemini-2.5-flash`
- request mode:
  - `stream_collect`
- prior run type:
  - true live Gemini raw-response persistence, not replay
- prior credential expectation:
  - same maintained helper path that loads `PROJECT_ROOT / ".env"` before calling `genai.configure(...)`

## Current maintained call path inspected
- `src/stage2_sampling_labels/run_stage2_s2_4b_live_llm_call_v1.py`
  - the frozen S2-4b runner calls `call_gemini_stream_collect(...)` at [run_stage2_s2_4b_live_llm_call_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/run_stage2_s2_4b_live_llm_call_v1.py:276)
- `src/stage2_sampling_labels/run_stage2_composite_v1.py`
  - composite Stage2 loads repo-local `.env` up front at [run_stage2_composite_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/run_stage2_composite_v1.py:354)
- `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`
  - maintained Gemini initialization lives in `ensure_genai(...)` at [extract_semantic_stage2_objects_v2.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py:619)
  - both `call_gemini(...)` and `call_gemini_stream_collect(...)` invoke `ensure_genai(...)` before client creation at [extract_semantic_stage2_objects_v2.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py:640) and [extract_semantic_stage2_objects_v2.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py:724)
- shared path authority:
  - `PROJECT_ROOT = Path(__file__).resolve().parents[2]` at [paths.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/utils/paths.py:31)

## Credential loading findings
- the maintained Gemini path does explicitly load `.env`
  - `ensure_genai()` calls `load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)` at [extract_semantic_stage2_objects_v2.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py:622)
- the variable name expected by the maintained Gemini path is exactly `GEMINI_API_KEY`
  - `ensure_genai()` reads exactly `os.getenv("GEMINI_API_KEY")`
  - if missing, it raises `RuntimeError("GEMINI_API_KEY is missing in environment.")`
- there is no reliance on shell `cwd` for `.env` lookup
  - lookup is anchored to `PROJECT_ROOT`, not `load_dotenv()` discovery from the current working directory
- there is no silent fallback to another Gemini variable name in the maintained Stage2 path
  - unlike some older scripts, this maintained path does not accept `GOOGLE_API_KEY`
- the maintained path can still look blocked if an audit checks only the ambient process environment before the maintained code executes `load_dotenv(...)`

## Runtime visibility findings
- `.env` exists at the repository root
- `GEMINI_API_KEY` is present in that file
- in a plain repo-root Python process, `os.getenv("GEMINI_API_KEY")` is false before `load_dotenv(...)`
- in the same process, `os.getenv("GEMINI_API_KEY")` becomes true immediately after the maintained `ensure_genai(...)` runs
- the same maintained `ensure_genai(...)` test also succeeded when run from `/tmp` with repo code on `PYTHONPATH`
  - that confirms there is no `cwd`-sensitive `.env` resolution bug in the maintained loader

## Minimal connectivity check
- probe used:
  - direct call to `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::call_gemini_stream_collect`
  - model: `gemini-2.5-flash`
  - prompt: minimal JSON echo
  - retries: `0`
  - helper timeout parameter: `20`
- result:
  - credential loading succeeded
  - Gemini client initialization reached the outbound request phase
  - the bounded subprocess probe did not receive an auth success or auth failure result within 25 seconds
  - observed classification: `transport_or_network_hang_before_auth_confirmation`
- therefore:
  - this environment did not reproduce a `.env` or `GEMINI_API_KEY` loading failure
  - authentication success could not be positively confirmed because the request did not complete

## Root cause classification
- prior “missing key” blocker classification:
  - false negative from checking the ambient process environment before the maintained code loaded `PROJECT_ROOT / ".env"`
- current maintained code-path classification:
  - `.env not loaded`: no
  - env var not visible after maintained load: no
  - wrong variable name: no
  - client initialization failure: no
  - API auth failure: not observed
  - network/connectivity failure: most likely current blocker for live verification
- exact current blocker:
  - maintained live request entered `stream_collect` but did not return an auth result within the bounded subprocess window

## Smallest fix or next action
- do not treat the maintained Stage2 Gemini path as blocked on a missing key
- if the goal is only to run the maintained path, no manual `export GEMINI_API_KEY=...` is required as long as repo-root `.env` remains present
- for positive auth verification, rerun the same minimal maintained probe in an environment with confirmed outbound Gemini API connectivity or wrap the maintained runner with a stronger outer process timeout so transport hangs are reported cleanly
