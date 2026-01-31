# Pipeline Spec Update: Strata-tagged Sampling (Sample20)

This update refines sampling to support stratified sampling without LLM calls by introducing a deterministic
tagging step that scans cleaned texts.

----------------------------------------------------------------------  
Step 3. Clean → Tags → Sample (select a subset of papers)  
----------------------------------------------------------------------

Purpose:
- Create a small, reproducible subset (sample10, sample20, sample30, …)
- Enable stratified sampling (nano vs micro; O/W vs W/O/W; table vs text) using deterministic, rule-based tags
- Reduce cost for downstream weak-labeling and evaluation

Authoritative inputs:
- Manifest:  data/cleaned/index/manifest_current.tsv
- key2txt:   data/cleaned/index/key2txt.tsv   (2-column mapping: key → repo-relative text path)

3A) Build strata tags (deterministic, rule-based)
Script:
- src/stage2_sampling_labels/build_strata_tags_from_text.py

Command:
```bash
python src/stage2_sampling_labels/build_strata_tags_from_text.py \
  --key2txt data/cleaned/index/key2txt.tsv \
  --summary data/cleaned/content/key2txt.tsv \
  --out data/cleaned/index/strata_tags.tsv \
  --max-chars 120000 \
  --overwrite \
  --verbose
```

Expected output:
- data/cleaned/index/strata_tags.tsv

Notes:
- Tags are best-effort and intentionally coarse:
  - particle_scale_tag: nano | micro | mixed | unknown
  - emulsion_route_tag: O/W | W/O/W | mixed | unknown
  - reporting_style_tag: table | text | unknown   (proxy via table_detected)

3B) Sample (HTML-first random or stratified20)
Primary script:
- src/stage2_sampling_labels/sample_from_manifest_html_first.py

Example command (sample10, htmlfirst):
```bash
python src/stage2_sampling_labels/sample_from_manifest_html_first.py \
  --manifest data/cleaned/index/manifest_current.tsv \
  --out-jsonl data/cleaned/samples/sample10_htmlfirst.jsonl \
  --n 10 \
  --seed 42 \
  --mode htmlfirst \
  --overwrite \
  --verbose
```

Example command (sample20, stratified20):
```bash
python src/stage2_sampling_labels/sample_from_manifest_html_first.py \
  --manifest data/cleaned/index/manifest_current.tsv \
  --out-jsonl data/cleaned/samples/sample20_stratified.jsonl \
  --seed 42 \
  --mode stratified20 \
  --strata-tags data/cleaned/index/strata_tags.tsv \
  --overwrite \
  --verbose
```

Expected outputs:
- data/cleaned/samples/sample10_htmlfirst.jsonl
- data/cleaned/samples/sample10_htmlfirst.tsv (if the script emits TSV)
- data/cleaned/samples/sample20_stratified.jsonl
- data/cleaned/samples/sample20_stratified.tsv (if the script emits TSV)

Notes:
- Sampling is reproducible via --seed.
- Keep sample artifacts under data/cleaned/samples/ only.

----------------------------------------------------------------------  
Step 4. Sample → key2txt (optional helper for sample-local runs)  
----------------------------------------------------------------------

Purpose:
- Some stage2 extraction scripts operate on a sample jsonl and a key2txt mapping.
- If you want a sample-specific key2txt (instead of full key2txt.tsv), build it.

Script:
- src/stage2_sampling_labels/build_key2txt_from_sample_manifest.py

Example:
```bash
python src/stage2_sampling_labels/build_key2txt_from_sample_manifest.py \
  --sample-jsonl data/cleaned/samples/sample10_htmlfirst.jsonl \
  --key2txt data/cleaned/index/key2txt.tsv \
  --out data/cleaned/samples/sample10_for_key2txt.tsv \
  --overwrite \
  --verbose
```

Expected outputs:
- data/cleaned/samples/sample10_for_key2txt.tsv
