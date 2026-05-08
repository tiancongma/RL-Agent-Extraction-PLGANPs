import json
import tempfile
import unittest
from pathlib import Path

from src.stage1_cleaning.extract_tables_for_keys_v1 import (
    AttachmentChoice,
    discover_html_table_asset_links,
    extract_tables_from_html_table_asset_links,
    write_tables_for_key,
)


class TestStage1TableAssetManifestV1(unittest.TestCase):
    def test_discovers_and_extracts_full_size_table_link_as_stage1_table(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            html_path = root / "article.html"
            asset_path = root / "tables" / "table2.csv"
            asset_path.parent.mkdir(parents=True)
            asset_path.write_text("Formulation,Size\nNP1,166.6\n", encoding="utf-8")
            html_path.write_text(
                '<html><body><p>Table 2. KGN-loaded nanoparticle properties.</p>'
                '<a href="tables/table2.csv">View full size table</a></body></html>',
                encoding="utf-8",
            )

            links = discover_html_table_asset_links(html_path)
            tables, parsed_links = extract_tables_from_html_table_asset_links(html_path)

        self.assertEqual(len(links), 1)
        link = links[0]
        self.assertEqual(link["href_raw"], "tables/table2.csv")
        self.assertEqual(link["local_path"], str(asset_path.resolve()))
        self.assertEqual(link["caption_or_title"], "Table 2. KGN-loaded nanoparticle properties.")
        self.assertEqual(link["table_source_kind"], "html_full_size_table_asset")
        self.assertEqual(parsed_links[0]["local_extraction_status"], "parsed")
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0]["source_type"], "html_table_asset")
        self.assertEqual(tables[0]["df"].iloc[0]["Size"], "166.6")

    def test_write_tables_manifest_preserves_table_asset_links_when_no_dom_tables(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            html_path = root / "article.html"
            asset_path = root / "table2.csv"
            asset_path.write_text("Formulation,Drug loading\nNP-HA,0.467\n", encoding="utf-8")
            html_path.write_text(
                '<html><body><p>Table 2. KGN-loaded nanoparticle properties.</p>'
                '<a href="table2.csv">Download table</a></body></html>',
                encoding="utf-8",
            )
            tables_root = root / "out_tables"
            chosen = AttachmentChoice(html_path=html_path, pdf_path=None, html_found=True, pdf_found=False)

            write_tables_for_key(
                "PAPER1",
                extracted=[],
                chosen=chosen,
                html_reason="no_html_table_tags",
                pdf_reason="",
                n_tables_html_extracted=0,
                n_tables_pdf_extracted=0,
                tables_root=tables_root,
            )

            manifest = json.loads((tables_root / "PAPER1" / "tables_manifest.json").read_text(encoding="utf-8"))
            extracted_csv = tables_root / "PAPER1" / "PAPER1__table_01__html_table_asset.csv"
            extracted_csv_text = extracted_csv.read_text(encoding="utf-8")

        self.assertEqual(manifest["n_tables_html_extracted"], 0)
        self.assertEqual(manifest["n_tables_html_asset_extracted"], 1)
        self.assertEqual(manifest["html_table_reason"], "html_table_assets_extracted_no_dom_table")
        self.assertEqual(manifest["total_tables"], 1)
        self.assertEqual(manifest["tables"][0]["source_type"], "html_table_asset")
        self.assertEqual(len(manifest["html_table_asset_links"]), 1)
        self.assertEqual(manifest["selected_table_assets"][0]["local_path"], str(asset_path.resolve()))
        self.assertEqual(manifest["selected_table_assets"][0]["caption_or_title"], "Table 2. KGN-loaded nanoparticle properties.")
        self.assertEqual(manifest["selected_table_files"], ["PAPER1__table_01__html_table_asset.csv"])
        self.assertIn("NP-HA,0.467", extracted_csv_text)


if __name__ == "__main__":
    unittest.main()
