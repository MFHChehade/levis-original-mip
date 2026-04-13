$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot
& .\.venv\Scripts\Activate.ps1

python -m levis_original_mip.run_smoke_tests --root_dir . --data_dir data --time_limit 15 --mip_gap 0.05 --max_candidates 1000 --solver cbc
