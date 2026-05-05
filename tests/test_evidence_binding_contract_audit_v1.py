from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.stage5_benchmark import audit_evidence_binding_contract_v1 as audit


class EvidenceBindingContractAuditTests(unittest.TestCase):
    def write_tsv(self, path: Path, rows: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(rows[0].keys()) if rows else ["empty"]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_classifies_direct_anchor_relation_and_legacy_surfaces_from_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed = root / "seed.tsv"
            self.write_tsv(
                seed,
                [
                    {
                        "paper_key": "P1",
                        "formulation_id": "F1",
                        "field_name": "polymer_mw_kDa",
                        "evidence_text": "direct sentence",
                        "evidence_anchor_text": "broad row anchor",
                        "relation_resolution_rule": "shared_polymer",
                    }
                ],
            )
            rows = audit.audit_tsv_surface(
                surface_name="layer3_workbook_seed_rows",
                path=seed,
                expected_component="Layer3 workbook seed rows",
            )
            classes = {row["evidence_logic_class"] for row in rows}
            self.assertIn("direct_evidence", classes)
            self.assertIn("broad_anchor", classes)
            self.assertIn("stage3_relation_provenance", classes)

    def test_missing_optional_surface_is_not_consumed(self):
        missing = Path("/definitely/missing/evidence.tsv")
        rows = audit.audit_tsv_surface(
            surface_name="audit_ready_export",
            path=missing,
            expected_component="audit-ready export",
            required=False,
        )
        self.assertEqual(rows[0]["surface_status"], "missing_optional")
        self.assertEqual(rows[0]["evidence_logic_class"], "not_consumed")

    def test_manifest_audit_uses_authority_paths_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            risk = root / "risk.tsv"
            self.write_tsv(
                risk,
                [
                    {
                        "paper_id": "P1",
                        "field_name": "drug_mass_mg",
                        "evidence_status": "unsupported",
                        "evidence_snippet": "snippet",
                        "source_paths": "source.tsv",
                    }
                ],
            )
            manifest = {
                "active_run_id": "run_test",
                "active_run_dir": str(root / "run_test"),
                "pointer_path": str(root / "ACTIVE_RUN.json"),
                "artifacts": [
                    {
                        "semantic_name": "layer3_risk_review_queue",
                        "selected_path": str(risk),
                        "status": "resolved",
                    }
                ],
            }
            manifest_path = root / "authority.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            out_dir = root / "out"
            result = audit.run_contract_audit(authority_manifest_path=manifest_path, out_dir=out_dir)
            self.assertTrue((out_dir / "analysis" / "evidence_binding_contract_audit_v1.tsv").exists())
            self.assertTrue((out_dir / "analysis" / "evidence_binding_contract_audit_v1.md").exists())
            self.assertTrue((out_dir / "RUN_CONTEXT.md").exists())
            classes = {row["evidence_logic_class"] for row in result["rows"]}
            self.assertIn("legacy_fallback", classes)


if __name__ == "__main__":
    unittest.main()
