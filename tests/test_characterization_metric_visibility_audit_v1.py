import unittest

from src.stage5_benchmark.audit_characterization_metric_cleantext_selector_visibility_v1 import (
    find_token_visibility,
    normalize_numeric_tokens,
    row_field_value,
    selector_has_metric_context,
)


class CharacterizationMetricVisibilityAuditTests(unittest.TestCase):
    def test_numeric_token_visibility_handles_mean_sd_metric_text(self):
        tokens = normalize_numeric_tokens("234.1 ± 0.5")
        self.assertEqual(tokens, ["234.1", "0.5"])
        visible, snippet = find_token_visibility("Size (nm) 234.1 ± 0.5 with PDI 0.12", tokens)
        self.assertTrue(visible)
        self.assertIn("234.1", snippet)

    def test_numeric_token_visibility_normalizes_unicode_minus(self):
        tokens = normalize_numeric_tokens("−21.23")
        self.assertEqual(tokens, ["-21.23"])
        visible, snippet = find_token_visibility("ZP (mV) −21.23 ± 1.04", tokens)
        self.assertTrue(visible)
        self.assertIn("-21.23", snippet)

    def test_numeric_token_visibility_does_not_match_digit_substrings(self):
        tokens = normalize_numeric_tokens("34")
        visible, _ = find_token_visibility("The value was 134 nm, not the target token.", tokens)
        self.assertFalse(visible)

    def test_field_alias_maps_layer3_particle_size_to_stage2_size_nm(self):
        row = {"size_nm_value": "151.2", "particle_size_nm_value": ""}
        self.assertEqual(row_field_value(row, "particle_size_nm"), "151.2")

    def test_field_alias_maps_layer3_ee_to_stage2_encapsulation(self):
        row = {"encapsulation_efficiency_percent_value": "83", "ee_percent_value": ""}
        self.assertEqual(row_field_value(row, "ee_percent"), "83")
    def test_selector_metric_context_detects_summary_without_numeric_value(self):
        text = "[TABLE_SUMMARY] Characterization-only table. Mean diameter, PI and zeta potential were summarized."
        self.assertTrue(selector_has_metric_context(text, "particle_size_nm"))
        self.assertTrue(selector_has_metric_context(text, "pdi"))
        self.assertTrue(selector_has_metric_context(text, "zeta_mV"))


if __name__ == "__main__":
    unittest.main()
