@echo off
setlocal

set "PY313=C:\Program Files\Python313\python.exe"
set "SCRIPT=%~dp0src\stage1_cleaning\clean_manifest_to_text.py"

echo [INFO] Stage1 launcher Python: %PY313%
"%PY313%" --version
"%PY313%" "%SCRIPT%" %*
