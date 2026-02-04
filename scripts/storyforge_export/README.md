# Storyforge Diplomacy Trace Exporter

This runner executes headless Diplomacy games and exports JSONL traces for Storyforge ingest.

## Usage

From the repo root:

```
python scripts\storyforge_export\run_and_export.py --iterations 2 --max_year 1902
```

### Output directory

- Default: `D:\storyworld_runs\diplomacy_traces` if `D:` exists
- Fallback: `./runs/diplomacy_traces`

Override with:

```
python scripts\storyforge_export\run_and_export.py --output_dir D:\storyworld_runs\diplomacy_traces
```

### Additional lm_game flags

You can pass flags through via `--extra`:

```
python scripts\storyforge_export\run_and_export.py --iterations 1 --max_year 1902 --extra --planning_phase
```

## JSONL schema

Each line has keys:

- `episode_id`
- `t` (monotonic int)
- `phase`
- `power`
- `event_type` (state, message, order, resolution, result)
- `payload`
