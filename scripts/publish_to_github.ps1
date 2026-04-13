$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path .git)) {
    git init
}

git remote remove origin 2>$null
git remote add origin https://github.com/MFHChehade/levis-original-mip.git
git branch -M main
git add .
git status

$changes = git status --porcelain
if ($changes) {
    git commit -m "Refactor repo into handoff-ready exact MIP baseline with detailed README"
}

git push -u origin main

