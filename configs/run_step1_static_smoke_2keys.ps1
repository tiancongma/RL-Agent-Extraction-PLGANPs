param(
    [Parameter(Mandatory = $true)]
    [string]$RunId,
    [string]$OutSubdir = "step1_dev_smoke_2keys",
    [string]$DatasetId = "goren_2025",
    [string]$Keys = "5GIF3D8W,7ZS858NS",
    [string]$SourceWeakLabels = "data/results/run_20260227_1016_a8d884b_goren2025_step1dev_v1/step1_dev/weak_labels__gemini.tsv"
)

$ErrorActionPreference = "Stop"

$base = Join-Path "data/results/$RunId" $OutSubdir
New-Item -ItemType Directory -Force -Path $base | Out-Null

$keysList = $Keys.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }
if ($keysList.Count -lt 1) {
    throw "No keys provided."
}

$keysFile = Join-Path $base "keys_2.txt"
$keysList | Set-Content -Path $keysFile -Encoding UTF8

python src/stage1_cleaning/run_tables_extraction_for_dataset_v1.py `
  --dataset-id $DatasetId `
  --manifest-tsv "data/cleaned/$DatasetId/index/manifest.tsv" `
  --keys-file $keysFile `
  --tables-root "data/cleaned/$DatasetId/tables" `
  --coverage-out (Join-Path $base "step1_tables_extraction_coverage.tsv")

$weakOut = Join-Path $base "weak_labels__gemini.tsv"
@"
from pathlib import Path
import pandas as pd
src = Path(r"$SourceWeakLabels")
out = Path(r"$weakOut")
keys = [x.strip() for x in r"$Keys".split(",") if x.strip()]
if not src.exists():
    raise SystemExit(f"source weak-labels TSV not found: {src}")
df = pd.read_csv(src, sep="\t", dtype=str).fillna("")
sub = df[df["key"].astype(str).isin(keys)].copy()
if sub.empty:
    raise SystemExit("filtered weak-labels rows are empty")
sub.to_csv(out, sep="\t", index=False)
print(f"filtered_rows={len(sub)}")
print(f"weak_labels_out={out}")
"@ | python -

python src/stage5_benchmark/run_formulation_core_signature_v1.py `
  --input-tsv $weakOut `
  --run-id $RunId `
  --out-subdir $OutSubdir

$outXlsx = Join-Path $base "audit_pack/audit_pack__human_evidence_v1__smoke2.xlsx"
python src/stage5_benchmark/build_audit_pack_human_evidence_v1.py `
  --run-id $RunId `
  --out-subdir $OutSubdir `
  --input-tsv $weakOut `
  --out-xlsx $outXlsx

@"
import pandas as pd
from pathlib import Path
p = Path(r"$outXlsx")
df = pd.read_excel(p, sheet_name="audit_cases", dtype=str).fillna("")
status = df["table_selection_status"].astype(str).value_counts().to_dict()
hr = df["human_review_tag"].astype(str).replace("", "(empty)").value_counts().to_dict()
print("smoke_status_counts=" + str(status))
print("smoke_human_review_tag_counts=" + str(hr))
"@ | python -

Write-Host "OK static smoke run complete."
