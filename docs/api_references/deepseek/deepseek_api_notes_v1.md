# DeepSeek API Notes for PLGA S2-4b Integration

Accessed: 2026-05-08

Source pages read successfully:

- https://api-docs.deepseek.com/
- https://api-docs.deepseek.com/guides/thinking_mode
- https://api-docs.deepseek.com/guides/json_mode
- https://api-docs.deepseek.com/guides/kv_cache
- https://api-docs.deepseek.com/quick_start/error_codes

Purpose: local, queryable reference for future DeepSeek API work in this repository. This file is documentation only; it does not change the active pipeline or authorize any live LLM call.

## Repository Integration Boundary

Current PLGA governance requires model selection to remain an explicit S2-4b live-call boundary parameter. DeepSeek integration must therefore follow these constraints:

- Do not restore a repository-wide default model.
- Do not change S2-4a prompt construction when switching models.
- Do not change S2-5+ parsing, validation, compatibility projection, Stage3, or Stage5 because of model selection.
- Any DeepSeek live-call output should be persisted as S2-4b raw-response lineage with request metadata sufficient for replay/audit.
- DeepSeek backend support, if implemented later, should be an explicit backend path such as `--llm-backend deepseek --model deepseek-v4-pro` or `--model deepseek-v4-flash`, not a global default.

## Base API and Models

From the DeepSeek docs home page:

- OpenAI-compatible base URL: `https://api.deepseek.com`
- Anthropic-compatible base URL: `https://api.deepseek.com/anthropic`
- API key environment variable used in local checks: `DEEPSEEK_API_KEY`
- Current models visible from `/models` during local connectivity check:
  - `deepseek-v4-flash`
  - `deepseek-v4-pro`
- Older model names noted by docs:
  - `deepseek-chat`
  - `deepseek-reasoner`
  - docs state these are deprecated on `2026/07/24` and map compatibly to `deepseek-v4-flash` non-thinking / thinking modes.

Local connectivity result recorded before this file was written:

```text
DEEPSEEK_API_KEY_present=yes
status=200
models=deepseek-v4-flash,deepseek-v4-pro
```

## Thinking Mode

Source: https://api-docs.deepseek.com/guides/thinking_mode

DeepSeek supports thinking mode. Before final answer output, the model can output chain-of-thought reasoning to improve final-answer accuracy.

### Parameters

OpenAI-format controls:

```json
{"thinking": {"type": "enabled"}}
```

or:

```json
{"thinking": {"type": "disabled"}}
```

Effort control:

```json
{"reasoning_effort": "high"}
```

or:

```json
{"reasoning_effort": "max"}
```

Anthropic-format effort control:

```json
{"output_config": {"effort": "high"}}
```

or:

```json
{"output_config": {"effort": "max"}}
```

Docs notes:

- Thinking toggle defaults to `enabled`.
- In thinking mode, default effort is `high` for regular requests.
- Some complex agent requests such as Claude Code / OpenCode may automatically use `max`.
- For compatibility, `low` and `medium` map to `high`; `xhigh` maps to `max`.
- With the OpenAI SDK, `thinking` must be passed in `extra_body`.

OpenAI SDK example from docs:

```python
response = client.chat.completions.create(
  model="deepseek-v4-pro",
  # ...
  reasoning_effort="high",
  extra_body={"thinking": {"type": "enabled"}}
)
```

### Unsupported sampling controls in thinking mode

Docs state that thinking mode does not support:

- `temperature`
- `top_p`
- `presence_penalty`
- `frequency_penalty`

For compatibility, sending these parameters does not trigger an error, but they have no effect.

### Response fields

In thinking mode:

- `content` contains the final answer.
- `reasoning_content` contains chain-of-thought content and is returned at the same level as `content`.

Multi-turn handling:

- If the model did not perform a tool call, intermediate assistant `reasoning_content` does not need to be returned in subsequent context. If passed back, the API ignores it.
- If the model performed a tool call, assistant `reasoning_content` must be passed back in all subsequent requests; otherwise the API returns `400`.

PLGA S2-4b implication:

- For one-shot S2-4b prompt calls with no tools, raw payload capture should preserve both `content` and `reasoning_content` if thinking mode is enabled.
- Downstream JSON parsing should use final `content`, not `reasoning_content`.
- Request metadata should record `thinking.type` and `reasoning_effort`.

## JSON Output

Source: https://api-docs.deepseek.com/guides/json_mode

DeepSeek provides JSON Output to ensure the model outputs valid JSON strings.

To enable JSON Output, docs require:

1. Set `response_format` to:

```json
{"type": "json_object"}
```

2. Include the word `json` in the system or user prompt.
3. Provide an example of the desired JSON format to guide the model.
4. Set `max_tokens` reasonably to avoid truncating the JSON string.

Important docs warning:

- JSON Output may occasionally return empty content.
- DeepSeek says they are actively optimizing this issue.
- Prompt modification may mitigate it.

Docs example:

```python
import json
from openai import OpenAI

client = OpenAI(
    api_key="<your api key>",
    base_url="https://api.deepseek.com",
)

system_prompt = """
The user will provide some exam text. Please parse the "question" and "answer" and output them in JSON format.

EXAMPLE INPUT:
Which is the highest mountain in the world? Mount Everest.

EXAMPLE JSON OUTPUT:
{
    "question": "Which is the highest mountain in the world?",
    "answer": "Mount Everest"
}
"""

user_prompt = "Which is the longest river in the world? The Nile River."

messages = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]

response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
    response_format={
        'type': 'json_object'
    }
)

print(json.loads(response.choices[0].message.content))
```

PLGA S2-4b implication:

- If DeepSeek is used for S2-4b semantic extraction, prefer `response_format={"type":"json_object"}` only if the existing S2-4a prompt contains the word `json` and a desired JSON schema/example.
- Capture empty-content responses as explicit failed raw responses, not silent successes.
- Set token budget high enough for full Stage2 response JSON.

## Context Caching

Source: https://api-docs.deepseek.com/guides/kv_cache

DeepSeek Context Caching on Disk is enabled by default for all users and does not require code changes.

Behavior:

- Each request may trigger construction of a hard disk cache.
- If later requests have overlapping prefixes with previous requests, the overlapping part may be fetched from cache and counted as a cache hit.
- A cache hit requires the corresponding prefix to have been persisted.
- Due to Sliding Window Attention, cached prefixes are independent, complete cache prefix units.
- A later request can hit only if it fully matches a cache prefix unit.

When cache prefixes are persisted:

1. Request-boundary persistence
   - Each request produces two cache prefix units:
     - at the end position of user input
     - at the end position of model output
   - Later requests can hit if they fully match these units.

2. Common-prefix detection persistence
   - When the system detects a common prefix across multiple requests, it persists that common prefix as an independent unit.
   - Later requests can hit if they fully reuse that unit.

3. Fixed-token-interval persistence
   - For long inputs or outputs, the system carves cache prefix units at fixed token intervals so long prefixes are not completely uncacheable.

Usage fields added by DeepSeek:

- `prompt_cache_hit_tokens`: number of input tokens that hit cache.
- `prompt_cache_miss_tokens`: number of input tokens that missed cache.

Randomness note:

- Disk cache only matches input prefix; output is still generated by computation/inference and remains affected by randomness parameters such as temperature.

Additional docs notes:

- Cache works on a best-effort basis and does not guarantee 100% hit rate.
- Cache construction takes seconds.
- Once no longer used, cache is automatically cleared, usually within a few hours to a few days.

PLGA S2-4b implication:

- No implementation is needed to enable DeepSeek cache.
- Reusing identical S2-4a prompt prefixes may reduce effective input cost/latency, but cache hit is best-effort and must not be treated as reproducibility authority.
- Persist `usage.prompt_cache_hit_tokens` and `usage.prompt_cache_miss_tokens` in raw-response/request metadata if present.

## Error Codes

Source: https://api-docs.deepseek.com/quick_start/error_codes

| Code | Meaning | Cause | Suggested handling |
|---|---|---|---|
| 400 | Invalid Format | Invalid request body format. | Modify request body according to error message/API docs. In thinking-mode tool-call workflows, missing required `reasoning_content` can also cause 400 per Thinking Mode docs. |
| 401 | Authentication Fails | Wrong API key. | Check `DEEPSEEK_API_KEY` and key creation. Do not print the key. |
| 402 | Insufficient Balance | Account has run out of balance. | Check account balance/top up. Treat as non-retryable until balance fixed. |
| 422 | Invalid Parameters | Request contains invalid parameters. | Modify parameters according to error message/API docs. |
| 429 | Rate Limit Reached | Requests are too fast. | Pace requests reasonably. Backoff and/or reduce parallelism. |
| 500 | Server Error | DeepSeek server issue. | Retry after brief wait; contact DeepSeek if persistent. |
| 503 | Server Overloaded | High-traffic overload. | Retry after brief wait. |

PLGA S2-4b implication:

- 400/422 should be captured as request-shape/parameter failures and should not be blindly retried without inspection.
- 401/402 are credential/account failures and should stop the run.
- 429/500/503 are retry/backoff candidates, subject to the governed S2-4b call-layer policy. Current S2-4b Gemini runner has `request_retries=0`; any DeepSeek retry behavior must be explicit and recorded in metadata/governance before benchmark use.

## Minimum Metadata to Persist for DeepSeek S2-4b Raw Responses

If DeepSeek backend is implemented later, persist at least:

- provider/backend: `deepseek`
- endpoint base URL or endpoint family: OpenAI-compatible `https://api.deepseek.com`
- model, e.g. `deepseek-v4-pro` or `deepseek-v4-flash`
- request body minus secrets
- `response_format`, especially `{"type":"json_object"}` when enabled
- `thinking.type`
- `reasoning_effort`
- whether streaming was used
- response `content`
- response `reasoning_content` when present
- raw response object or lossless JSON-compatible projection
- usage fields, including `prompt_cache_hit_tokens` and `prompt_cache_miss_tokens` when present
- HTTP status and error body on failed requests
- retry/backoff decisions, if any

## Do Not Store Secrets

Never store `DEEPSEEK_API_KEY` value in repo artifacts, logs, raw response sidecars, or RUN_CONTEXT files. It is acceptable to record whether the key was present (`yes/no`) and which env var name was used.

## Recommended Initial S2-4b Trial Configuration (2026-05-08 discussion)

This section records the agreed starting position for a small three-paper DeepSeek S2-4b trial using the already frozen S2-4a prompt artifacts.

### Model choice

- Start with `deepseek-v4-flash`.
- Rationale: the current S2-4b task is a bounded semantic-discovery and compact JSON-emission task over frozen S2-4a evidence packs, not downstream row materialization or Stage3/Stage5 reasoning.
- Do not assume `flash` is sufficient until validated on representative papers.
- Keep `deepseek-v4-pro` as the comparison/fallback candidate if flash shows failures on:
  - table-scope classification,
  - DOE / variable-sweep formulation boundary,
  - parent-child / sequential optimization semantics,
  - sparse controls / downstream or measurement-only variants,
  - `shared_semantics` stability.

### Thinking mode

- Initial trial default: `thinking.type=disabled`.
- Rationale: S2-4b needs final structured JSON, not multi-turn/tool reasoning; disabling thinking keeps raw-response lineage simpler and avoids storing `reasoning_content` unnecessarily.
- If complex papers fail under no-thinking mode, run a bounded comparison with:
  - `thinking.type=enabled`
  - `reasoning_effort=high`
- Do not default to `reasoning_effort=max` unless `high` still fails on complex diagnostic samples.
- If thinking is enabled, persist `reasoning_content` in raw-response metadata, but downstream S2-5 parsing must use final `content` only.

### JSON output

- Initial trial should use DeepSeek JSON Output:
  - `response_format={"type":"json_object"}`
- Rationale: current S2-4a prompts already instruct the model to return JSON and include a compact skeleton/schema; DeepSeek JSON Output should strengthen structural compliance without changing S2-4a.
- Do not change S2-4a prompts for this trial.
- Treat HTTP 200 with empty `content` as an explicit failed raw response, not as a successful S2-4b output.
- Ensure output token budget is large enough to avoid truncating the Stage2 JSON.

### Context cache

- No cache parameter or code path is required.
- DeepSeek context caching is enabled by default by the provider.
- Cache hits may help cost/latency for repeated prompt prefixes, but are best-effort and must not be treated as reproducibility authority.
- Persist `usage.prompt_cache_hit_tokens` and `usage.prompt_cache_miss_tokens` when present.

### Trial boundary

- Use frozen S2-4a artifacts as input.
- The trial changes only S2-4b live-call provider/model/request parameters and raw-response lineage.
- Do not alter S2-4a prompt construction or S2-5+ parsing/validation/projection during this trial unless a separate explicit repair is approved.

