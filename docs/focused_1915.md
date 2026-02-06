# Focused 1915 Forecasting Exercise

This pack tees up a low-waste 1915 run that emphasizes negotiation reasoning with the improved p-value diplomacy storyworlds.

## What This Uses

- Scenario: `ai_diplomacy/scenarios/forecasting_1915_press.json`
- Storyworld bank (pinned):
  - `forecast_backstab_p.json`
  - `forecast_coalition_p.json`
  - `forecast_defection_p.json`
- Prompt set: `ai_diplomacy/prompts_forecasting`
- Focus powers: `ENGLAND,FRANCE,GERMANY` (other powers are set to `silent` in forecasting mode)

## Files

- `scripts/focused_1915/prepare_bank.ps1`
- `scripts/focused_1915/run_preflight.ps1`
- `scripts/focused_1915/summarize_reasoning.py`

## Dry Run First

From `C:\projects\AI_Diplomacy`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\focused_1915\run_preflight.ps1
```

This prints the exact harness command and environment setup, but does not start the run.

## Start The Run

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\focused_1915\run_preflight.ps1 -Execute
```

Optional knobs:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\focused_1915\run_preflight.ps1 -Execute -Model gpt-5-mini -MaxTokens 700
```

## Summarize Negotiation Reasoning

After the run finishes:

```powershell
python .\scripts\focused_1915\summarize_reasoning.py --run-dir .\results\focused_1915_pvalue_YYYYMMDD_HHMMSS
```

Outputs:

- `<run_dir>/focused_1915_reasoning_summary.md`
- `<run_dir>/focused_1915_reasoning_summary.json`

## Expected Harness Artifacts

- `storyworld_forecasts.jsonl`
- `storyworld_impact.jsonl`
- `forecast_scores.jsonl`
- `llm_responses.csv`
- `lmvsgame.json`
- `overview.jsonl`
