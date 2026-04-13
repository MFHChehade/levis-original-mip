$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

foreach ($Path in @("checkpoints", "results", "smoke_results", "data", "build", "dist", "src\levis_original_mip.egg-info")) {
    if (Test-Path $Path) {
        Remove-Item $Path -Recurse -Force
    }
}
