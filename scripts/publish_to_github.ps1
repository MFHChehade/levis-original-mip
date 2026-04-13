$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

if (-not (Test-Path ".git")) {
    git init
}

$RemoteUrl = "https://github.com/MFHChehade/levis-original-mip.git"
git remote remove origin 2>$null
git remote add origin $RemoteUrl

git add -A
$HasChanges = git diff --cached --quiet; if ($LASTEXITCODE -ne 0) {
    git commit -m "Replace repo with proper Python project for exact LEVIS MIP handoff"
}
git branch -M main
git push -u origin main
