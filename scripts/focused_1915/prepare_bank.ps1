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

foreach ($name in $selectedFiles) {
    $src = Join-Path $SourceDir $name
    $dst = Join-Path $TargetDir $name
    if (-not (Test-Path $src)) {
        throw "Missing required template in source bank: $src"
    }
    Copy-Item -Path $src -Destination $dst -Force
}

# Keep the focus bank deterministic: only the selected p-value templates.
Get-ChildItem -Path $TargetDir -File -Filter *.json |
    Where-Object { $selectedFiles -notcontains $_.Name } |
    Remove-Item -Force

Write-Output "Prepared focused bank: $TargetDir"
Write-Output ("Templates: " + (($selectedFiles | Sort-Object) -join ", "))
