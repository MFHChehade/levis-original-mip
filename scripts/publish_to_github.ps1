param(
    [string]$RemoteUrl = "https://github.com/MFHChehade/levis-original-mip.git"
)

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Split-Path -Parent $Root
Set-Location $Repo

if (-not (Test-Path ".git")) {
    git init
}

$HasOrigin = $false
try {
    git remote get-url origin | Out-Null
    $HasOrigin = $true
} catch {
    $HasOrigin = $false
}

if (-not $HasOrigin) {
    git remote add origin $RemoteUrl
}

git add .
git status

$HasUserName = (git config user.name)
$HasUserEmail = (git config user.email)
if (-not $HasUserName) { throw "git user.name is not set." }
if (-not $HasUserEmail) { throw "git user.email is not set." }

$Staged = git diff --cached --name-only
if ($Staged) {
    git commit -m "Restructure repo as a proper Python project for LEVIS-style exact MIP runs"
}

git branch -M main
git push -u origin main

