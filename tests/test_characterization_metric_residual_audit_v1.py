import unittest

from src.stage5_benchmark.audit_characterization_metric_residuals_v1 import classify_measurement_residual


class CharacterizationMetricResidualAuditTests(unittest.TestCase):
    def test_missing_metric_with_missing_surface_is_projection_gap(self):
        row = {
            "field_name": "particle_size_nm",
            "compare_status": "missing_in_system",
            "system_value_source_type": "missing_system_field_surface",
            "evidence_status_detail": "missing_system_field_surface",
            "alignment_rule": "direct",
        }
        self.assertEqual(classify_measurement_residual(row), "missing_system_field_surface")

    def test_blocked_alignment_is_classified_before_projection(self):
        row = {
            "field_name": "zeta_mV",
            "compare_status": "blocked_alignment",
            "system_value_source_type": "",
            "evidence_status_detail": "",
            "alignment_rule": "blocked_alignment",
        }
        self.assertEqual(classify_measurement_residual(row), "alignment_blocked_before_metric_projection")

    def test_present_mismatch_is_endpoint_or_value_policy(self):
        row = {
            "field_name": "ee_percent",
            "compare_status": "present_but_mismatch",
            "system_value_source_type": "direct_extracted",
            "evidence_status_detail": "supported",
            "alignment_rule": "direct",
        }
        self.assertEqual(classify_measurement_residual(row), "present_but_mismatch_endpoint_or_value_policy")

    def test_extra_metric_requires_review(self):
        row = {
            "field_name": "dl_percent",
            "compare_status": "extra_in_system",
            "system_value_source_type": "row_local_table_cell_grid_binding",
            "evidence_status_detail": "supported",
            "alignment_rule": "direct",
        }
        self.assertEqual(classify_measurement_residual(row), "extra_metric_surface_requires_review")


if __name__ == "__main__":
    unittest.main()
