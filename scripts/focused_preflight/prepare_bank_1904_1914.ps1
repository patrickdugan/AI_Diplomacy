param(
    [string]$TargetDir = ""
)

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $TargetDir) {
    $TargetDir = Join-Path $repoRoot "ai_diplomacy\storyworld_bank_focus_preflight_1904_1914"
}

$StoryworldSourcesDir = Join-Path $repoRoot "ai_diplomacy\storyworld_sources"
$TemplateDir = Join-Path $TargetDir "templates"
$FullDir = Join-Path $TargetDir "full_storyworlds"

$selectedFiles = @(
    "preflight_1904_alliance_pressure_p.json",
    "preflight_1914_austria_entanglement_p.json"
)

if (-not (Test-Path $StoryworldSourcesDir)) {
    throw "Storyworld source directory not found: $StoryworldSourcesDir"
}

New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
New-Item -ItemType Directory -Path $TemplateDir -Force | Out-Null
New-Item -ItemType Directory -Path $FullDir -Force | Out-Null

foreach ($name in $selectedFiles) {
    $src = Join-Path $StoryworldSourcesDir $name
    if (-not (Test-Path $src)) {
        throw "Missing source storyworld for $name at $src"
    }

    Copy-Item -Path $src -Destination (Join-Path $TargetDir $name) -Force
    Copy-Item -Path $src -Destination (Join-Path $TemplateDir $name) -Force
    Copy-Item -Path $src -Destination (Join-Path $FullDir $name) -Force
}

# Keep the bank deterministic.
Get-ChildItem -Path $TargetDir -File -Filter *.json |
    Where-Object { $selectedFiles -notcontains $_.Name } |
    Remove-Item -Force

Get-ChildItem -Path $TemplateDir -File -Filter *.json |
    Where-Object { $selectedFiles -notcontains $_.Name } |
    Remove-Item -Force

Get-ChildItem -Path $FullDir -File -Filter *.json |
    Where-Object { $selectedFiles -notcontains $_.Name } |
    Remove-Item -Force

Write-Output "Prepared preflight bank: $TargetDir"
Write-Output ("Files: " + (($selectedFiles | Sort-Object) -join ", "))
Write-Output "Templates: $TemplateDir"
Write-Output "Full storyworlds: $FullDir"
