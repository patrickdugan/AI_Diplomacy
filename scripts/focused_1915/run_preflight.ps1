param(
    [string]$Model = "gpt-5-mini",
    [int]$MaxTokens = 700,
    [string]$FocusPowers = "ENGLAND,FRANCE,GERMANY",
    [string]$EndAtPhase = "W1915A",
    [string]$TemplateMap = "",
    [string]$RunDir = "",
    [switch]$Execute
)

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$prepareScript = Join-Path $PSScriptRoot "prepare_bank.ps1"
& $prepareScript

$bankDir = Join-Path $repoRoot "ai_diplomacy\storyworld_bank_focus_1915"
$stateFile = Join-Path $repoRoot "ai_diplomacy\scenarios\forecasting_1915_press.json"
$promptsDir = Join-Path $repoRoot "ai_diplomacy\prompts_forecasting"

if (-not (Test-Path $bankDir)) {
    throw "Focused storyworld bank not found: $bankDir"
}
if (-not (Test-Path $stateFile)) {
    throw "1915 scenario file not found: $stateFile"
}
if (-not (Test-Path $promptsDir)) {
    throw "Forecasting prompts dir not found: $promptsDir"
}

if (-not $RunDir) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $RunDir = Join-Path $repoRoot ("results\focused_1915_pvalue_{0}" -f $stamp)
}

# Default: force first focus power to use the fresh storyworld template.
if (-not $TemplateMap) {
    $firstFocusPower = ($FocusPowers -split ",")[0].Trim().ToUpper()
    if ($firstFocusPower) {
        $TemplateMap = "${firstFocusPower}:forecast_false_concession_p"
    }
}

$maxYear = 1915
if ($EndAtPhase.Length -ge 5) {
    $phaseYear = $EndAtPhase.Substring(1, 4)
    if ($phaseYear -match "^\d{4}$") {
        $maxYear = [int]$phaseYear
    }
}

$env:GPT_STORYWORLD_DIR = "C:\projects\GPTStoryworld"
$env:STORYWORLD_BANK_DIR = $bankDir
$env:STORYWORLD_BANK_ONLY = "1"
$env:STORYWORLD_PLAYBACK = "1"
$env:STORYWORLD_PLAY_MODE = "model"
$env:STORYWORLD_PLAY_MAX_STEPS = "8"
$env:STORYWORLD_PLAY_REASONING = "1"
$env:STORYWORLD_TEMPLATE_MAP = $TemplateMap
$env:PYTHONUTF8 = "1"

$modelList = ((1..7 | ForEach-Object { $Model }) -join ",")

$arguments = @(
    "lm_game.py",
    "--run_dir", $RunDir,
    "--forecasting_analysis_mode", "true",
    "--forecasting_focus_powers", $FocusPowers,
    "--forecasting_state_file", $stateFile,
    "--prompts_dir", $promptsDir,
    "--models", $modelList,
    "--max_year", "$maxYear",
    "--end_at_phase", $EndAtPhase,
    "--num_negotiation_rounds", "1",
    "--max_tokens", "$MaxTokens",
    "--simple_prompts", "false",
    "--generate_phase_summaries", "false"
)

function Quote-Arg([string]$arg) {
    if ($arg -match "\s") {
        return ('"{0}"' -f $arg)
    }
    return $arg
}

$preview = @("python") + ($arguments | ForEach-Object { Quote-Arg $_ })
$estimatedCalls = 60
$estimatedOutputTokenCeiling = $estimatedCalls * $MaxTokens

Write-Output "Focused 1915 preflight is ready."
Write-Output "Run dir: $RunDir"
Write-Output "Focus powers: $FocusPowers"
Write-Output "End phase: $EndAtPhase"
Write-Output "Forced template map: $TemplateMap"
Write-Output "Pinned storyworld bank: $bankDir"
Write-Output "Estimated output-token ceiling (rough): $estimatedOutputTokenCeiling"
Write-Output "Command:"
Write-Output ($preview -join " ")

if (-not $Execute) {
    Write-Output "Dry run mode only. Re-run with -Execute to start the harness."
    exit 0
}

Push-Location $repoRoot
try {
    & python @arguments
    $code = $LASTEXITCODE
} finally {
    Pop-Location
}

if ($code -ne 0) {
    throw "Harness exited with code $code"
}

Write-Output "Run completed: $RunDir"
Write-Output "Next: python scripts/focused_1915/summarize_reasoning.py --run-dir \"$RunDir\""
