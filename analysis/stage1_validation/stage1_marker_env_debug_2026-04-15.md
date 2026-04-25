# Stage1 Marker Environment Debug (2026-04-15)

## Scope

Narrow environment repair and validation for Stage1 Marker parsing only. No Stage2+ work. No pipeline redesign.

## Stage1 runtime environment observed from current shell

- `sys.executable`: `C:\Program Files\Python314\python.exe`
- Python version: `3.14.4`
- Working directory: `D:\tiancong\GitHub\RL-Agent-Extraction-PLGANPs`
- `VIRTUAL_ENV`: `None`
- `CONDA_PREFIX`: `None`
- `CONDA_DEFAULT_ENV`: `None`
- `pip --version`: `pip 26.0.1 from C:\Program Files\Python314\Lib\site-packages\pip (python 3.14)`
- `python -m pip --version`: `pip 26.0.1 from C:\Program Files\Python314\Lib\site-packages\pip (python 3.14)`

## Package availability audit

- Current shell `python` (`3.14`):
  - `marker`: not importable before install
  - `fitz`: not importable before install
  - `pandas`: not importable before install
  - `bs4`: not importable before install
- Alternate interpreter on machine:
  - `C:\Program Files\Python313\python.exe`
  - after install, `marker`, `fitz`, `pandas`, and `bs4` are importable there
- Marker CLI:
  - installed under `C:\Users\tiancong\AppData\Roaming\Python\Python313\Scripts`
  - not on PATH in the current shell

## Install commands executed

### Failed in current shell Python 3.14

```powershell
python -m pip index versions marker
python -m pip index versions marker-pdf
python -m pip install marker-pdf==1.10.2 pandas beautifulsoup4 PyMuPDF pypdf
```

Result:

- `marker-pdf==1.10.2` is available on PyPI
- install failed on Windows/Python 3.14 because dependencies fell back to source builds:
  - `Pillow 10.4.0` does not support Python 3.14 on Windows here and failed with missing `zlib`
  - `regex` failed with `Microsoft Visual C++ 14.0 or greater is required`

### Succeeded in Python 3.13

```powershell
py -3.13 -m pip index versions marker-pdf
py -3.13 -m pip install marker-pdf==1.10.2 pandas beautifulsoup4 PyMuPDF pypdf
```

Result:

- install succeeded in `C:\Program Files\Python313\python.exe`
- Marker module became importable there
- `src/stage1_cleaning/pdf2clean.py` imports successfully there without code changes

## Stage1 import/API compatibility audit

`pdf2clean.py` expects:

```python
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
```

Observed result in Python 3.13:

- `src.stage1_cleaning.pdf2clean` imports successfully
- `extract_marker_pdf_blocks` exists
- no Stage1 import/API compatibility patch was required

## Minimal validation

Test key: `5GIF3D8W`

- Original manifest PDF path:
  - `C:\Users\tianc\Downloads\00_Data\UMD\1 Project 1 CRNPs for Breast CSC\Tiancong\Zotero\storage\6KQNS8XQ\Snehalatha et al. - 2008 - Etoposide-Loaded PLGA and PCL Nanoparticles I Preparation and Effect of Formulation Variables.pdf`
- Resolved through current Stage1 path logic:
  - `D:\tiancong\Zotero\storage\6KQNS8XQ\Snehalatha et al. - 2008 - Etoposide-Loaded PLGA and PCL Nanoparticles I Preparation and Effect of Formulation Variables.pdf`
- Resolved path exists: `True`
- Temporary one-page test PDF:
  - `analysis/tmp_marker_5GIF3D8W_page1.pdf`

### First live Marker run in Python 3.13

- direct `extract_marker_pdf_blocks(...)`
- first attempt inside sandbox failed before parsing with:
  - `marker_model_init_failed:PermissionError:[WinError 5] 拒绝访问。: 'C:\\Users\\tiancong\\AppData\\Local\\datalab'`
- rerun with elevated access succeeded
- total runtime: `173.15242409706116 s`
- page count: `1`
- blocks: `9`
- tables: `0`
- warnings: `[]`
- sample blocks:
  - `paragraph` -> `![](_page_0_Picture_0.jpeg)`
  - `paragraph` -> `## **Drug Delivery**`

Interpretation:

- cold start included model downloads into `C:\Users\tiancong\AppData\Local\datalab\datalab\Cache\...`
- Marker itself worked once it could initialize/cache models

### Cached rerun

- direct `extract_marker_pdf_blocks(...)`
  - runtime: `26.457234382629395 s`
  - page count: `1`
  - blocks: `9`
  - tables: `0`
  - warnings: `[]`
- wrapper `extract_text_from_pdf(...)`
  - runtime: `11.97872018814087 s`
  - `metadata.parser`: `marker`
  - `metadata.warnings`: `[]`
  - `metadata.page_count`: `1`
  - final blocks: `10`
  - final tables: `1`
  - sample blocks:
    - `paragraph` -> `![](_page_0_Picture_0.jpeg)`
    - `paragraph` -> `## **Drug Delivery**`

## Exact fix applied

- No Stage1 code patch was required for Marker import/API compatibility.
- The environment repair outcome is:
  - current shell default `python` (`3.14`) is not a viable Marker host on this machine for this package set
  - Python `3.13` is a viable Marker host and successfully exercised the current Stage1 code path

## Current caveat

Bare `python ...` from the current shell still resolves to `C:\Program Files\Python314\python.exe`, so Stage1 will not use the now-working Marker install unless Stage1 is launched with Python 3.13 (or the shell environment is changed to point `python` at that interpreter).
