# L3H2RS2H Block Packing Audit (2026-03-10)

## Overall readout
- Selected table blocks: 9 total; likely true formulation tables: 5; likely noisy table blocks: 4.
- First selected paragraph block appears at packing rank 20.
- Best current hypothesis: noisy PDF-derived table fragments are being promoted before clarifying paragraphs, which biases the LLM toward over-splitting row variants and away from parent/variant interpretation.

## Suspicious selected blocks
- `table_5` (table, score=287): contains running paper header/footer text; table block is unusually long and likely includes narrative spillover; contains many narrative sentences for a table block. Preview: PI) and zeta potential (z) of PLGA empty and loaded nanospheres Empty nanospheres XAN nanospheresa 3-MeOXAN nanospheresb Diameter (nm) 154G6 164G8 164G9 PI 0.06G0.03 0.06G0.03 0...
- `table_9` (table, score=255): contains running paper header/footer text; table block is unusually long and likely includes narrative spillover; contains many narrative sentences for a table block. Preview: PI), zeta potential (z) and incorporation parameters of various nanocapsule formulations: empty nanocapsules (0.6 mL Myritol 318 and without xanthones), XAN-loaded nanocapsules...
- `table_8` (table, score=189): contains running paper header/footer text; table block is unusually long and likely includes narrative spillover; contains many narrative sentences for a table block. Preview: PI) and zeta potential (z) of empty and loaded PLGA nanocapsules Empty nanocapsules XAN nanocapsulesa 3-MeOXAN nanocapsulesb Diameter (nm) 274G3 278G15 280G19 PI 0.455G0.130 0.4...
- `table_6` (table, score=99): table block is unusually long and likely includes narrative spillover; contains many narrative sentences for a table block. Preview: XAN crystals could be observed, indicating that the maximum loading capacity of the nanocapsules had been reached. For the 3-MeOXAN-loaded nanocapsules, the incorporation efﬁcie...
- `table_1` (table, score=95): contains many narrative sentences for a table block. Preview: XAN (33.0G4.1%) or for 3-MeOXAN (41.5G7.6%) corresponding to a theoretical concentration of 60 mg/mL. For concentrations above 70 mg/mL both xanthones precipitated in the form o...
- `paragraph_125` (paragraph, score=22): generic paragraph with weak formulation linkage. Preview: PLGA nanoparticles: preparation, physicochemical characterization and in vitro anti-tumoral activity, J. Control Release 83 (2002) 273–286. [13] M.J. Alonso, Nanoparticulate dru...
- `table_3` (table, score=18): contains many narrative sentences for a table block. Preview: empty and drug-loaded nanosphere formulations exhibited a negative charge with values ranging from K38.9 to K36.0 mV, typically observed for these types of systems [27]. The sur...
- `paragraph_128` (paragraph, score=10): generic paragraph with weak formulation linkage. Preview: nanoparticles prepared by nanoprecipitation: drug loading and release studies of a water soluble drug, J. Control Release 57 (1999) 171–185. [31] S.S. Guterres, H. Fessi, G. Bar...

## Audit answers
- True formulation-table candidates among selected table blocks: table_1, table_7, table_4, table_2, table_3.
- Noisy/fragmentary selected table blocks: table_5, table_9, table_8, table_6.
- Explanatory paragraphs appear late in the packing order: first paragraph at rank 20.
- Final packed text likely biases toward row over-splitting when table blocks contain mixed captions, headers, results prose, or repeated row fragments without nearby inheritance explanations.

## Files
- Block inventory: `data\cleaned\labels\manual\l3h2rs2h_blockpack_audit_2026-03-10\block_inventory.tsv`
- Packed order: `data\cleaned\labels\manual\l3h2rs2h_blockpack_audit_2026-03-10\packed_block_order.tsv`
- Packed evidence text: `data\cleaned\labels\manual\l3h2rs2h_blockpack_audit_2026-03-10\packed_evidence_text.txt`
