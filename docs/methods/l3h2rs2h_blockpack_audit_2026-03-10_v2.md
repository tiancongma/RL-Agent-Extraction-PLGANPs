# L3H2RS2H Block Packing Audit (2026-03-10)

## Overall readout
- Selected table blocks: 9 total; likely true formulation tables: 9; likely noisy table blocks: 0.
- First selected paragraph block appears at packing rank 20.
- Best current hypothesis: noisy PDF-derived table fragments are being promoted before clarifying paragraphs, which biases the LLM toward over-splitting row variants and away from parent/variant interpretation.

## Suspicious selected blocks
- `table_1` (table, score=71): contains many narrative sentences for a table block. Preview: XAN (33.0G4.1%) or for 3-MeOXAN (41.5G7.6%) corresponding to a theoretical concentration of 60 mg/mL. For concentrations above 70 mg/mL both xanthones precipitated in the form o...
- `paragraph_125` (paragraph, score=22): generic paragraph with weak formulation linkage. Preview: PLGA nanoparticles: preparation, physicochemical characterization and in vitro anti-tumoral activity, J. Control Release 83 (2002) 273–286. [13] M.J. Alonso, Nanoparticulate dru...
- `table_3` (table, score=18): contains many narrative sentences for a table block. Preview: empty and drug-loaded nanosphere formulations exhibited a negative charge with values ranging from K38.9 to K36.0 mV, typically observed for these types of systems [27]. The sur...
- `paragraph_128` (paragraph, score=10): generic paragraph with weak formulation linkage. Preview: nanoparticles prepared by nanoprecipitation: drug loading and release studies of a water soluble drug, J. Control Release 57 (1999) 171–185. [31] S.S. Guterres, H. Fessi, G. Bar...
- `paragraph_7` (paragraph, score=10): generic paragraph with weak formulation linkage. Preview: PLGA was selected for the preparation of the nanocarriers because of its adequate biocompatibility and biodegrad- ability [17]. Both systems, nanospheres and nanocapsules were p...
- `paragraph_129` (paragraph, score=6): generic paragraph with weak formulation linkage. Preview: nanoemulsions, as ocular drug carries, J. Pharm. Sci. 85 (1996) 530–536. [36] H. Marchais, S. Benali, J. Irache, C. Thrasse-Bloch, O. Lafont, A.M. Orecchioni, Entrapment efﬁcien...
- `paragraph_16` (paragraph, score=4): generic paragraph with weak formulation linkage. Preview: A is the drug concentration (mg/mL) in the nanosphere dispersion and B is the theoretical drug concentration (mg/mL).
- `paragraph_93` (paragraph, score=4): generic paragraph with weak formulation linkage. Preview: [20,35,38], the major components of nanocapsules, which can affect their zeta potential are lecithins, oil core, polymer and poloxamer. Lecithins and oils have compounds such as...

## Before vs after suspicious blocks
- `table_5`: shorter (3934 -> 419); quality noisy -> likely_true_formulation_table; packing_rank 2 -> 4.
- `table_9`: shorter (6709 -> 696); quality noisy -> likely_true_formulation_table; packing_rank 3 -> 2.
- `table_8`: shorter (3888 -> 437); quality noisy -> likely_true_formulation_table; packing_rank 4 -> 5.
- `table_6`: shorter (3395 -> 663); quality noisy -> likely_true_formulation_table; packing_rank 5 -> 8.

## Audit answers
- True formulation-table candidates among selected table blocks: table_9, table_1, table_5, table_8, table_7, table_4, table_6, table_2, table_3.
- Noisy/fragmentary selected table blocks: none strongly flagged.
- Explanatory paragraphs appear late in the packing order: first paragraph at rank 20 (previously 20).
- Final packed text likely biases toward row over-splitting when table blocks contain mixed captions, headers, results prose, or repeated row fragments without nearby inheritance explanations.

## Files
- Block inventory: `data\cleaned\labels\manual\l3h2rs2h_blockpack_audit_2026-03-10_v2\block_inventory.tsv`
- Packed order: `data\cleaned\labels\manual\l3h2rs2h_blockpack_audit_2026-03-10_v2\packed_block_order.tsv`
- Packed evidence text: `data\cleaned\labels\manual\l3h2rs2h_blockpack_audit_2026-03-10_v2\packed_evidence_text.txt`
