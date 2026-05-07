import tempfile
import unittest
from pathlib import Path

from src.stage1_cleaning.audit_source_anchor_cleantext_visibility_v1 import (
    DEV15_ANCHOR_KEYS,
    AnchorSection,
    audit_anchor_clean_text_visibility,
    normalize_for_visibility,
    audit_anchor_table_authority_visibility,
    classify_table_authority_first_failure,
    parse_key2txt,
    parse_manifest_sources,
    parse_user_source_anchor_sections,
)


class TestSourceAnchorParser(unittest.TestCase):
    def setUp(self):
        self.protocol_path = Path("docs/methods/layer3_field_gt_protocol_v1.md")

    def test_finds_all_dev15_anchor_keys_in_order(self):
        anchors = parse_user_source_anchor_sections(self.protocol_path)
        self.assertEqual([a.paper_key for a in anchors], DEV15_ANCHOR_KEYS)

    def test_inmutv7l_anchor_preserves_method_and_table_text(self):
        anchors = {a.paper_key: a for a in parse_user_source_anchor_sections(self.protocol_path)}
        text = anchors["INMUTV7L"].raw_text
        self.assertIn("90 mg of PLGA", text)
        self.assertIn("5 mg of dexibuprofen", text)
        self.assertIn("Table 1. Characterization of the different formulations developed", text)
        self.assertTrue(anchors["INMUTV7L"].has_method_marker)
        self.assertTrue(anchors["INMUTV7L"].has_table_marker)

    def test_bb3juvw7_anchor_preserves_nanosphere_and_nanorod_tables(self):
        anchors = {a.paper_key: a for a in parse_user_source_anchor_sections(self.protocol_path)}
        text = anchors["BB3JUVW7"].raw_text
        self.assertIn("Preparation of artemether loaded PLGA nanospheres", text)
        self.assertIn("Preparation of artemether loaded PLGA nanorods", text)
        self.assertIn("Table 1. Particle size, PDI, %EE, %DL and zeta potential", text)
        self.assertIn("Table 2. Physicochemical parameters of nanorods", text)
        self.assertTrue(anchors["BB3JUVW7"].has_table_marker)

    def test_diagnostic_notes_after_last_anchor_are_not_included(self):
        anchors = {a.paper_key: a for a in parse_user_source_anchor_sections(self.protocol_path)}
        anchor = anchors["5GIF3D8W"]
        self.assertEqual(anchor.start_line, 1814)
        self.assertEqual(anchor.end_line, 1864)
        self.assertIn("cEncapsulation efficiency", anchor.raw_text)
        self.assertNotIn("Stage5 shared loaded-drug identity carrythrough", anchor.raw_text)
        self.assertNotIn("PAT_STAGE5_SHARED_DRUG_IDENTITY_CARRYTHROUGH_V1", anchor.raw_text)

    def test_rhmjwzX8_prose_numeric_anchor_is_not_table_marker(self):
        anchors = {a.paper_key: a for a in parse_user_source_anchor_sections(self.protocol_path)}
        anchor = anchors["RHMJWZX8"]
        self.assertTrue(anchor.has_method_marker)
        self.assertFalse(anchor.has_table_marker)
        self.assertIn("145.0", anchor.raw_text)
        self.assertIn("1.3", anchor.raw_text)
        self.assertIn("90.51", anchor.raw_text)
        self.assertIn("0.28%", anchor.raw_text)

    def test_parser_rejects_short_expected_key_prefix(self):
        with self.assertRaises(ValueError):
            parse_user_source_anchor_sections(self.protocol_path, expected_keys=["INMUTV7L"])

    def test_parser_rejects_unexpected_paper_key_header(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "protocol.md"
            p.write_text(
                "## User-Provided Original Source Excerpts For Field-GT Debugging\n"
                "\n"
                "### INMUTV7L\n"
                "source\n"
                "### XXXXXXXX\n"
                "unexpected paper-key-like anchor\n"
                "### 2026-04-27 — Diagnostic Note\n"
                "not an anchor\n"
            )
            with self.assertRaises(ValueError):
                parse_user_source_anchor_sections(p, expected_keys=["INMUTV7L"])

    def test_parser_rejects_missing_anchor_section(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "protocol.md"
            p.write_text("# no governed anchors here\n")
            with self.assertRaises(ValueError):
                parse_user_source_anchor_sections(p)
    def test_clean_text_visibility_normalizes_unicode_without_claiming_binding(self):
        anchors = {a.paper_key: a for a in parse_user_source_anchor_sections(self.protocol_path)}
        anchor = anchors["RHMJWZX8"]
        with tempfile.TemporaryDirectory() as td:
            clean_text = Path(td) / "RHMJWZX8.pdf.txt"
            clean_text.write_text(
                "particle size of 145.0 ± 1.3 nm and EE was 90.51 ± 0.28%; "
                "zeta potential was -14.81 ± 1.39 mV and μm pore size was normalized."
            )
            result = audit_anchor_clean_text_visibility(anchor, [clean_text])
        self.assertEqual(result.anchor_visibility, "partial")
        self.assertGreaterEqual(result.matched_fragment_count, 1)
        self.assertIn("visibility_only_not_row_binding", result.audit_note)

    def test_clean_text_visibility_reports_absent_without_stage5_loss(self):
        anchors = {a.paper_key: a for a in parse_user_source_anchor_sections(self.protocol_path)}
        anchor = anchors["INMUTV7L"]
        with tempfile.TemporaryDirectory() as td:
            clean_text = Path(td) / "INMUTV7L.pdf.txt"
            clean_text.write_text("unrelated article text")
            result = audit_anchor_clean_text_visibility(anchor, [clean_text])
        self.assertEqual(result.anchor_visibility, "absent")
        self.assertGreater(result.missing_fragment_count, 0)
        self.assertNotIn("stage5", result.first_missing_fragment.lower())

    def test_unicode_normalizer_handles_minus_mu_nbsp_and_thin_spaces(self):
        text = "−14.81\u2009±\u20091.39 mV; 0.8-μm pore; 90\u00a0mg"
        normalized = normalize_for_visibility(text)
        self.assertIn("-14.81 ± 1.39 mv", normalized)
        self.assertIn("0.8-µm pore", normalized)
        self.assertIn("90 mg", normalized)

    def test_parse_key2txt_keeps_multiple_paths_per_key(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            key2txt = root / "key2txt.tsv"
            key2txt.write_text(
                "ABC12345\tdata/cleaned/content/text/ABC12345.html.txt\n"
                "ABC12345\tdata/cleaned/content/text/ABC12345.pdf.txt\n"
            )
            mapping = parse_key2txt(key2txt, repo_root=root)
        self.assertEqual(len(mapping["ABC12345"]), 2)
        self.assertTrue(str(mapping["ABC12345"][0]).endswith("ABC12345.html.txt"))
    def test_parse_manifest_sources_records_primary_secondary_lineage(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "manifest.tsv"
            manifest.write_text(
                "key\tpaper_key\tpdf\thtml\ttext_path\ttext_source_type\n"
                "ABC12345\tABC12345\tdata/raw/a.pdf\tdata/raw/a.html\tdata/cleaned/content/text/ABC12345.html.txt\thtml\n"
            )
            sources = parse_manifest_sources(manifest, repo_root=root)
        self.assertEqual(sources["ABC12345"]["primary_source_type"], "html")
        self.assertEqual(sources["ABC12345"]["secondary_source_available"], "yes")
        self.assertEqual(sources["ABC12345"]["pdf_path"], "data/raw/a.pdf")
        self.assertEqual(sources["ABC12345"]["html_path"], "data/raw/a.html")

    def test_numeric_fallback_is_counted_separately_from_exact_fragment(self):
        anchors = {a.paper_key: a for a in parse_user_source_anchor_sections(self.protocol_path)}
        anchor = anchors["RHMJWZX8"]
        with tempfile.TemporaryDirectory() as td:
            clean_text = Path(td) / "RHMJWZX8.pdf.txt"
            clean_text.write_text("145.0 1.3 90.51 0.28 -14.81 1.39")
            result = audit_anchor_clean_text_visibility(anchor, [clean_text])
        self.assertGreater(result.numeric_token_fallback_count, 0)
        self.assertEqual(result.exact_fragment_match_count, 0)
        self.assertIn("not_row_binding", result.audit_note)
    def test_table_authority_visibility_uses_payload_and_grid_without_stage5_claim(self):
        anchors = {a.paper_key: a for a in parse_user_source_anchor_sections(self.protocol_path)}
        anchor = anchors["INMUTV7L"]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            payload_root = root / "payloads"
            paper_dir = payload_root / "INMUTV7L"
            paper_dir.mkdir(parents=True)
            payload_json = paper_dir / "normalized_table_payloads_v1.json"
            payload_json.write_text(
                '{"normalized_table_payloads":[{"table_id":"Table 1","normalized_csv_path":"payloads/INMUTV7L/table.csv"}]}'
            )
            (paper_dir / "table.csv").write_text(
                "Formulation Number,Average Size (nm),EE (%)\n"
                "1,153.5,59.4\n"
            )
            grid_tsv = root / "table_cell_grid_v1.tsv"
            grid_tsv.write_text(
                "paper_key\traw_cell_value\traw_header_text\tsource_locator\n"
                "INMUTV7L\t2.1. Preparation of Polymeric Nanoparticles Table 1. Characterization of the different formulations developed. PLGA nanoparticles (NPs) containing DXI were prepared by using the solvent displacement method.\tpreparation\tTable 1::row_1::col_1\n"
            )
            result = audit_anchor_table_authority_visibility(
                anchor,
                payload_root=payload_root,
                grid_tsv_path=grid_tsv,
                repo_root=root,
            )
        self.assertIn(result.anchor_visibility, {"partial", "full"})
        self.assertGreater(result.matched_fragment_count, 0)
        self.assertIn("table_authority_visibility_only", result.audit_note)
        self.assertIn("not_stage5", result.audit_note)
        self.assertGreaterEqual(result.payload_json_count, 1)
        self.assertGreaterEqual(result.grid_cell_count, 1)

    def test_table_authority_visibility_absent_when_no_payload_or_grid(self):
        anchors = {a.paper_key: a for a in parse_user_source_anchor_sections(self.protocol_path)}
        anchor = anchors["INMUTV7L"]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result = audit_anchor_table_authority_visibility(
                anchor,
                payload_root=root / "missing_payloads",
                grid_tsv_path=root / "missing_grid.tsv",
                repo_root=root,
            )
        self.assertEqual(result.anchor_visibility, "absent")
        self.assertEqual(result.payload_json_count, 0)
        self.assertEqual(result.grid_cell_count, 0)
        self.assertEqual(result.table_authority_first_failure_class, "raw_table_asset_missing")
        self.assertEqual(result.exact_visibility_proof, "no")

    def test_table_authority_numeric_only_signal_does_not_make_anchor_visible(self):
        anchors = {a.paper_key: a for a in parse_user_source_anchor_sections(self.protocol_path)}
        anchor = anchors["RHMJWZX8"]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            payload_root = root / "payloads"
            paper_dir = payload_root / "RHMJWZX8"
            paper_dir.mkdir(parents=True)
            (paper_dir / "normalized_table_payloads_v1.json").write_text('{"normalized_table_payloads":[]}')
            grid_tsv = root / "table_cell_grid_v1.tsv"
            grid_tsv.write_text(
                "paper_key\traw_cell_value\traw_header_text\tsource_locator\n"
                "RHMJWZX8\t145.0 1.3 90.51 0.28 -14.81 1.39\tmetrics\tTable::row\n"
            )
            result = audit_anchor_table_authority_visibility(
                anchor,
                payload_root=payload_root,
                grid_tsv_path=grid_tsv,
                repo_root=root,
            )
        self.assertGreater(result.numeric_token_fallback_count, 0)
        self.assertEqual(result.matched_fragment_count, 0)
        self.assertEqual(result.anchor_visibility, "absent")
        self.assertEqual(result.exact_visibility_proof, "no")
        self.assertEqual(
            result.table_authority_first_failure_class,
            "source_excerpt_method_prose_not_expected_in_table_payload",
        )

    def test_table_authority_first_failure_detects_payload_geometry_degradation(self):
        anchor = AnchorSection(
            paper_key="TESTKEY1",
            start_line=1,
            end_line=2,
            raw_text="### TESTKEY1\nTable 1. Formulation A had PLGA 90 mg and drug 5 mg",
            has_table_marker=True,
            has_method_marker=False,
        )
        failure_class, repair_hint, proof = classify_table_authority_first_failure(
            anchor=anchor,
            fragments=["Table 1. Formulation A had PLGA 90 mg and drug 5 mg"],
            normalized_table_text=normalize_for_visibility("Formulation A\nPLGA\n90\nDrug\n5"),
            exact_matches=0,
            numeric_token_fallbacks=1,
            payload_json_count=1,
            normalized_csv_count=1,
            grid_cell_count=5,
            raw_table_asset_exists=True,
        )
        self.assertEqual(failure_class, "payload_exists_but_row_header_geometry_degraded")
        self.assertIn("geometry", repair_hint)
        self.assertEqual(proof, "no")

    def test_table_authority_first_failure_detects_normalization_mismatch(self):
        anchor = AnchorSection(
            paper_key="TESTKEY2",
            start_line=1,
            end_line=2,
            raw_text="### TESTKEY2\nTable 2. PLGA-PEG 10 mg",
            has_table_marker=True,
            has_method_marker=False,
        )
        failure_class, repair_hint, proof = classify_table_authority_first_failure(
            anchor=anchor,
            fragments=["Table 2. PLGA-PEG 10 mg"],
            normalized_table_text=normalize_for_visibility("Table 2 PLGA PEG 10 mg"),
            exact_matches=0,
            numeric_token_fallbacks=0,
            payload_json_count=1,
            normalized_csv_count=1,
            grid_cell_count=1,
            raw_table_asset_exists=True,
        )
        self.assertEqual(failure_class, "payload_exists_but_text_normalization_mismatch")
        self.assertIn("normalization", repair_hint)
        self.assertEqual(proof, "no")


if __name__ == "__main__":
    unittest.main()
