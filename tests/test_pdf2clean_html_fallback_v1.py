import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.stage1_cleaning.pdf2clean import extract_bs4_blocks, extract_text_from_html, make_block


class Pdf2CleanHtmlFallbackTests(unittest.TestCase):
    def test_bs4_fallback_recovers_content_bearing_div_paragraphs(self):
        html = """
        <html><body>
          <nav><div>Journals & Books</div><div>Help</div></nav>
          <section id="article-body">
            <h2>Preparation of nanospheres</h2>
            <div class="section-paragraph">
              PLGA nanospheres were prepared by solvent displacement. Briefly,
              90 mg PLGA and 15 mg drug were dissolved in acetone and added to
              an aqueous phase containing Poloxamer 188.
            </div>
          </section>
        </body></html>
        """
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "paper.html"
            path.write_text(html, encoding="utf-8")
            blocks, warnings = extract_bs4_blocks(path)
        texts = [b["text"] for b in blocks]
        joined = "\n".join(texts)
        self.assertEqual(warnings, [])
        self.assertIn("Preparation of nanospheres", joined)
        self.assertIn("90 mg PLGA", joined)
        self.assertNotIn("Journals & Books", joined)
    def test_bs4_fallback_does_not_duplicate_nested_div_container_text(self):
        html = """
        <html><body>
          <article>
            <div class="method-section">
              <div class="method-body">
                Microparticles were fabricated by solvent evaporation with polymer
                dissolved in organic solvent and emulsified into an aqueous stabilizer
                phase before washing and collection.
              </div>
            </div>
          </article>
        </body></html>
        """
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "paper.html"
            path.write_text(html, encoding="utf-8")
            blocks, warnings = extract_bs4_blocks(path)
        texts = [b["text"] for b in blocks]
        self.assertEqual(warnings, [])
        self.assertEqual(len(texts), 1)
        self.assertEqual(
            texts[0],
            "Microparticles were fabricated by solvent evaporation with polymer dissolved in organic solvent and emulsified into an aqueous stabilizer phase before washing and collection.",
        )

    def test_bs4_fallback_preserves_dom_order_for_mixed_heading_and_div_body(self):
        html = """
        <html><body>
          <article>
            <h2>Methods</h2>
            <div class="method-body">
              Particles were prepared using polymer solution added into aqueous
              stabilizer with mixing and solvent removal before collection.
            </div>
            <h2>Results</h2>
          </article>
        </body></html>
        """
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "paper.html"
            path.write_text(html, encoding="utf-8")
            blocks, warnings = extract_bs4_blocks(path)
        self.assertEqual(warnings, [])
        self.assertEqual([block["type"] for block in blocks], ["heading", "paragraph", "heading"])
        self.assertEqual(blocks[0]["text"], "Methods")
        self.assertIn("Particles were prepared", blocks[1]["text"])
        self.assertEqual(blocks[2]["text"], "Results")

    def test_trafilatura_supplement_uses_bs4_dom_order_when_trafilatura_keeps_later_heading(self):
        html = """
        <html><body>
          <article>
            <h2>Introduction</h2>
            <div class="intro-body">
              This article introduces a polymer delivery system and describes why
              formulation choices affect release behavior and experimental design.
            </div>
            <h2>Methods</h2>
            <div class="method-body">
              Particles were prepared by adding polymer solution into an aqueous
              stabilizer phase followed by stirring, solvent evaporation, washing,
              and collection for downstream characterization.
            </div>
          </article>
        </body></html>
        """
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "paper.html"
            path.write_text(html, encoding="utf-8")
            sparse_blocks = [make_block("heading", "Methods")]
            with patch(
                "src.stage1_cleaning.pdf2clean.extract_trafilatura_blocks",
                return_value=(sparse_blocks, "trafilatura_native", []),
            ):
                result = extract_text_from_html(path)

        block_texts = [block["text"] for block in result["sidecar"]["blocks"] if block["type"] != "table"]
        self.assertEqual(block_texts[0], "Introduction")
        self.assertIn("This article introduces", block_texts[1])
        self.assertEqual(block_texts[2], "Methods")
        self.assertIn("Particles were prepared", block_texts[3])

    def test_trafilatura_success_with_sparse_heading_is_supplemented_by_bs4_body(self):
        html = """
        <html><body>
          <header><div>Search Download PDF</div></header>
          <article>
            <h2>Preparation of PLGA nanoparticles</h2>
            <div class="article-section__content">
              Nanoparticles were prepared by emulsification solvent evaporation using
              PLGA dissolved in dichloromethane and an aqueous polyvinyl alcohol phase.
              The emulsion was stirred overnight to remove solvent before collection.
            </div>
          </article>
        </body></html>
        """
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "paper.html"
            path.write_text(html, encoding="utf-8")
            sparse_blocks = [make_block("heading", "Preparation of PLGA nanoparticles")]
            with patch(
                "src.stage1_cleaning.pdf2clean.extract_trafilatura_blocks",
                return_value=(sparse_blocks, "trafilatura_native", []),
            ):
                result = extract_text_from_html(path)

        text = result["text"]
        warnings = result["sidecar"]["metadata"]["warnings"]
        self.assertIn("Preparation of PLGA nanoparticles", text)
        self.assertIn("PLGA dissolved in dichloromethane", text)
        self.assertNotIn("Search Download PDF", text)
        self.assertIn("trafilatura_insufficient_prose_bs4_supplemented", warnings)
        self.assertEqual(result["sidecar"]["metadata"]["parser"], "trafilatura_plus_beautifulsoup_supplement")


if __name__ == "__main__":
    unittest.main()
