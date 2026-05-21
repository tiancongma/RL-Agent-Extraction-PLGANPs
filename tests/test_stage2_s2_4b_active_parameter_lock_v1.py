import argparse
import tempfile
import unittest
from pathlib import Path

from src.stage2_sampling_labels.run_stage2_s2_4b_live_llm_call_v1 import (
    discover_campaign_active_parameter_lock,
    parse_run_context_active_parameters,
    validate_against_active_parameter_lock,
)
from src.utils.paths import DATA_RESULTS_DIR


ACTIVE_RUN_CONTEXT = """# RUN_CONTEXT

## 6. Live Call Settings
- llm_backend: `deepseek`
- model: `deepseek-v4-flash`
- generation_config:
  - gemini: `temperature=0`, `response_mime_type=application/json`
  - deepseek: `response_format=json_object`, `thinking.type=disabled`, `streaming=disabled`, `max_tokens=8192`
- request_timeout_seconds: `180`
- request_retries: `0`
- retry_sleep_sec: `3.0`
- max_parallel_requests: `1`
- inter_request_sleep_seconds: `0.5`
"""


def make_args(**overrides):
    values = {
        "llm_backend": "deepseek",
        "model": "deepseek-v4-flash",
        "deepseek_response_format": "json_object",
        "deepseek_thinking": "disabled",
        "deepseek_streaming": "disabled",
        "max_tokens": 8192,
        "request_timeout_seconds": 180,
        "request_retries": 0,
        "retry_sleep_sec": 3.0,
        "max_parallel_requests": 1,
        "inter_request_sleep_seconds": 0.5,
        "allow_active_parameter_deviation": False,
        "active_parameter_deviation_reason": "",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class Stage2S24bActiveParameterLockTest(unittest.TestCase):
    def test_parse_campaign_active_deepseek_lock(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context_path = Path(temp_dir) / "RUN_CONTEXT.md"
            context_path.write_text(ACTIVE_RUN_CONTEXT, encoding="utf-8")
            parsed = parse_run_context_active_parameters(context_path)

        self.assertEqual(parsed["llm_backend"], "deepseek")
        self.assertEqual(parsed["model"], "deepseek-v4-flash")
        self.assertEqual(parsed["deepseek_response_format"], "json_object")
        self.assertEqual(parsed["deepseek_thinking"], "disabled")
        self.assertEqual(parsed["deepseek_streaming"], "disabled")
        self.assertEqual(parsed["max_tokens"], 8192)
        self.assertEqual(parsed["inter_request_sleep_seconds"], 0.5)

    def test_active_lock_rejects_gemini_before_live_call(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context_path = Path(temp_dir) / "RUN_CONTEXT.md"
            context_path.write_text(ACTIVE_RUN_CONTEXT, encoding="utf-8")
            lock = parse_run_context_active_parameters(context_path)

        with self.assertRaisesRegex(ValueError, "active parameter lock mismatch"):
            validate_against_active_parameter_lock(make_args(llm_backend="gemini", model="gemini-2.5-flash"), lock)

    def test_active_lock_allows_matching_deepseek_parameters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context_path = Path(temp_dir) / "RUN_CONTEXT.md"
            context_path.write_text(ACTIVE_RUN_CONTEXT, encoding="utf-8")
            lock = parse_run_context_active_parameters(context_path)

        validate_against_active_parameter_lock(make_args(), lock)

    def test_discovery_allows_multiple_identical_campaign_locks(self):
        with tempfile.TemporaryDirectory(dir=DATA_RESULTS_DIR) as temp_dir:
            bucket = Path(temp_dir)
            prompts_run = bucket / "105_stage2_s2_4a"
            target_run = bucket / "106_stage2_s2_4b"
            for child in [
                bucket / "08_stage2_s2_4b_active_params_pass_prompts_only",
                bucket / "34_stage2_s2_4b_deepseek_active_params_10paper_recovered_probe",
            ]:
                child.mkdir(parents=True)
                (child / "RUN_CONTEXT.md").write_text(ACTIVE_RUN_CONTEXT, encoding="utf-8")
            prompts_run.mkdir(parents=True)
            prompts_jsonl = prompts_run / "s2_4a_prompts_v1.jsonl"
            prompts_jsonl.write_text("", encoding="utf-8")

            lock = discover_campaign_active_parameter_lock(prompts_jsonl, target_run)

        self.assertIsNotNone(lock)
        self.assertEqual(lock["model"], "deepseek-v4-flash")
        self.assertIn("08_stage2_s2_4b_active_params_pass_prompts_only", lock["lock_sources"])
        self.assertIn("34_stage2_s2_4b_deepseek_active_params_10paper_recovered_probe", lock["lock_sources"])

    def test_discovery_rejects_multiple_conflicting_campaign_locks(self):
        conflicting_context = ACTIVE_RUN_CONTEXT.replace(
            "- model: `deepseek-v4-flash`",
            "- model: `gemini-2.5-flash`",
        )
        with tempfile.TemporaryDirectory(dir=DATA_RESULTS_DIR) as temp_dir:
            bucket = Path(temp_dir)
            prompts_run = bucket / "105_stage2_s2_4a"
            target_run = bucket / "106_stage2_s2_4b"
            child_a = bucket / "08_stage2_s2_4b_active_params_pass_prompts_only"
            child_b = bucket / "34_stage2_s2_4b_deepseek_active_params_10paper_recovered_probe"
            child_a.mkdir(parents=True)
            child_b.mkdir(parents=True)
            (child_a / "RUN_CONTEXT.md").write_text(ACTIVE_RUN_CONTEXT, encoding="utf-8")
            (child_b / "RUN_CONTEXT.md").write_text(conflicting_context, encoding="utf-8")
            prompts_run.mkdir(parents=True)
            prompts_jsonl = prompts_run / "s2_4a_prompts_v1.jsonl"
            prompts_jsonl.write_text("", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Multiple campaign-local S2-4b active parameter locks"):
                discover_campaign_active_parameter_lock(prompts_jsonl, target_run)

    def test_deviation_requires_reason(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context_path = Path(temp_dir) / "RUN_CONTEXT.md"
            context_path.write_text(ACTIVE_RUN_CONTEXT, encoding="utf-8")
            lock = parse_run_context_active_parameters(context_path)

        with self.assertRaisesRegex(ValueError, "requires --active-parameter-deviation-reason"):
            validate_against_active_parameter_lock(
                make_args(
                    llm_backend="gemini",
                    model="gemini-2.5-flash",
                    allow_active_parameter_deviation=True,
                ),
                lock,
            )


if __name__ == "__main__":
    unittest.main()
