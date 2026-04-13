$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path .\.venv)) {
    python -m venv .venv
}

.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

