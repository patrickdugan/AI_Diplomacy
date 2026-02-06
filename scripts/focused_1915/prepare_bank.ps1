param(
    [string]$SourceDir = "",
    [string]$TargetDir = ""
)

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $SourceDir) {
    $SourceDir = Join-Path $repoRoot "ai_diplomacy\storyworld_bank_extracted"
}
if (-not $TargetDir) {
    $TargetDir = Join-Path $repoRoot "ai_diplomacy\storyworld_bank_focus_1915"
}
$StoryworldSourcesDir = Join-Path $repoRoot "ai_diplomacy\storyworld_sources"
$TemplateDir = Join-Path $TargetDir "templates"

$selectedFiles = @(
    "forecast_false_concession_p.json",
    "forecast_backstab_p.json",
    "forecast_coalition_p.json",
    "forecast_defection_p.json"
)

if (-not (Test-Path $SourceDir)) {
    throw "Source directory not found: $SourceDir"
}

New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
New-Item -ItemType Directory -Path $TemplateDir -Force | Out-Null

foreach ($name in $selectedFiles) {
    $src = Join-Path $SourceDir $name
    $dst = Join-Path $TemplateDir $name
    if (-not (Test-Path $src)) {
        throw "Missing required template in source bank: $src"
    }
    Copy-Item -Path $src -Destination $dst -Force
}

# Keep the focused template bank deterministic.
Get-ChildItem -Path $TemplateDir -File -Filter *.json |
    Where-Object { $selectedFiles -notcontains $_.Name } |
    Remove-Item -Force

# Put full SweepWeave storyworlds at the focus-bank root so they open directly in the editor.
foreach ($name in $selectedFiles) {
    $fullSrc = Join-Path $StoryworldSourcesDir $name
    $fullDst = Join-Path $TargetDir $name
    if (-not (Test-Path $fullSrc)) {
        throw "Missing full storyworld source for $name at $fullSrc"
    }
    Copy-Item -Path $fullSrc -Destination $fullDst -Force
}

# Keep only selected full storyworld files in the root.
Get-ChildItem -Path $TargetDir -File -Filter *.json |
    Where-Object { $selectedFiles -notcontains $_.Name } |
    Remove-Item -Force

# Keep a duplicated full-storyworld folder for compatibility with prior tooling/docs.
$fullDir = Join-Path $TargetDir "full_storyworlds"
New-Item -ItemType Directory -Path $fullDir -Force | Out-Null

foreach ($name in $selectedFiles) {
    $fullSrc = Join-Path $StoryworldSourcesDir $name
    $fullDst = Join-Path $fullDir $name
    if (Test-Path $fullSrc) {
        Copy-Item -Path $fullSrc -Destination $fullDst -Force
    } else {
        Write-Warning "Missing full storyworld source for $name at $fullSrc"
    }
}

Get-ChildItem -Path $fullDir -File -Filter *.json |
    Where-Object { $selectedFiles -notcontains $_.Name } |
    Remove-Item -Force

Write-Output "Prepared focused bank: $TargetDir"
Write-Output ("Templates: " + (($selectedFiles | Sort-Object) -join ", "))
Write-Output "Template bank (harness): $TemplateDir"
Write-Output "Full storyworlds at root: $TargetDir"
Write-Output "Full QC storyworlds: $fullDir"
