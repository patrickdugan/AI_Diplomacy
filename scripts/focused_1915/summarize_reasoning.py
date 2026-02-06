#!/usr/bin/env python3
"""Summarize focused 1915 negotiation-reasoning outputs.

This script is intentionally lightweight so it can run on partial runs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

SIGNAL_TERMS = ("forecast", "storyworld", "pvalue", "probab", "likelihood", "odds")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def short_text(text: str, limit: int = 220) -> str:
    flat = " ".join((text or "").split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 3] + "..."


def parse_negotiation_diaries(csv_path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "total_negotiation_diaries": 0,
        "signal_diaries": [],
    }
    if not csv_path.exists():
        return result

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("response_type") != "negotiation_diary":
                continue
            result["total_negotiation_diaries"] += 1

            raw_response = row.get("raw_response") or ""
            lowered = raw_response.lower()
            if any(term in lowered for term in SIGNAL_TERMS):
                result["signal_diaries"].append(
                    {
                        "phase": row.get("phase", ""),
                        "power": row.get("power", ""),
                        "snippet": short_text(raw_response),
                    }
                )
    return result


def make_summary(run_dir: Path) -> Dict[str, Any]:
    forecasts = read_jsonl(run_dir / "storyworld_forecasts.jsonl")
    impacts = read_jsonl(run_dir / "storyworld_impact.jsonl")
    scores = read_jsonl(run_dir / "forecast_scores.jsonl")
    diary_info = parse_negotiation_diaries(run_dir / "llm_responses.csv")

    forecast_by_power = Counter()
    forecast_by_storyworld = Counter()
    confidences: List[float] = []

    for row in forecasts:
        power = str(row.get("power", ""))
        artifact = row.get("artifact") or {}
        storyworld_id = str(artifact.get("storyworld_id", "unknown"))
        forecast_by_power[power] += 1
        forecast_by_storyworld[storyworld_id] += 1
        confidence = artifact.get("confidence")
        try:
            confidences.append(float(confidence))
        except (TypeError, ValueError):
            pass

    explicit_impacts = 0
    for row in impacts:
        if str(row.get("impact_flag", "")).lower() == "explicit":
            explicit_impacts += 1

    brier_values: List[float] = []
    for row in scores:
        brier = row.get("brier")
        try:
            brier_values.append(float(brier))
        except (TypeError, ValueError):
            pass

    summary: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "files_present": {
            "storyworld_forecasts.jsonl": (run_dir / "storyworld_forecasts.jsonl").exists(),
            "storyworld_impact.jsonl": (run_dir / "storyworld_impact.jsonl").exists(),
            "forecast_scores.jsonl": (run_dir / "forecast_scores.jsonl").exists(),
            "llm_responses.csv": (run_dir / "llm_responses.csv").exists(),
        },
        "forecast_count": len(forecasts),
        "impact_count": len(impacts),
        "score_count": len(scores),
        "explicit_impact_count": explicit_impacts,
        "mean_confidence": mean(confidences) if confidences else None,
        "mean_brier": mean(brier_values) if brier_values else None,
        "forecast_by_power": dict(forecast_by_power),
        "forecast_by_storyworld": dict(forecast_by_storyworld),
        "total_negotiation_diaries": diary_info["total_negotiation_diaries"],
        "signal_negotiation_diaries": len(diary_info["signal_diaries"]),
        "signal_diary_examples": diary_info["signal_diaries"][:12],
    }
    return summary


def write_markdown(summary: Dict[str, Any], output_path: Path) -> None:
    lines: List[str] = []
    lines.append("# Focused 1915 Reasoning Summary")
    lines.append("")
    lines.append(f"Run dir: `{summary['run_dir']}`")
    lines.append("")
    lines.append("## Availability")
    for name, exists in summary["files_present"].items():
        lines.append(f"- `{name}`: {'yes' if exists else 'no'}")
    lines.append("")

    lines.append("## Storyworld Forecasts")
    lines.append(f"- Forecast artifacts: {summary['forecast_count']}")
    lines.append(f"- Mean confidence: {summary['mean_confidence']}")
    lines.append(f"- Forecast score rows: {summary['score_count']}")
    lines.append(f"- Mean Brier score: {summary['mean_brier']}")
    lines.append(f"- Explicit impact rows: {summary['explicit_impact_count']}")
    lines.append("")

    lines.append("## Forecasts by Power")
    if summary["forecast_by_power"]:
        for power, count in sorted(summary["forecast_by_power"].items()):
            lines.append(f"- {power}: {count}")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Forecasts by Storyworld")
    if summary["forecast_by_storyworld"]:
        for storyworld_id, count in sorted(summary["forecast_by_storyworld"].items()):
            lines.append(f"- {storyworld_id}: {count}")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Negotiation Diary Signal")
    lines.append(f"- Total negotiation diary rows: {summary['total_negotiation_diaries']}")
    lines.append(f"- Rows with forecast/storyworld signal terms: {summary['signal_negotiation_diaries']}")
    lines.append("")

    lines.append("## Diary Examples")
    examples = summary["signal_diary_examples"]
    if examples:
        for row in examples:
            phase = row.get("phase", "")
            power = row.get("power", "")
            snippet = row.get("snippet", "")
            lines.append(f"- {phase} {power}: {snippet}")
    else:
        lines.append("- None")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize focused 1915 storyworld negotiation reasoning output.")
    parser.add_argument("--run-dir", required=True, help="Path to a completed (or partial) lm_game run directory.")
    parser.add_argument(
        "--markdown-out",
        default="",
        help="Optional output path for markdown report. Defaults to <run-dir>/focused_1915_reasoning_summary.md",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional output path for JSON summary. Defaults to <run-dir>/focused_1915_reasoning_summary.json",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    summary = make_summary(run_dir)

    markdown_out = Path(args.markdown_out).resolve() if args.markdown_out else run_dir / "focused_1915_reasoning_summary.md"
    json_out = Path(args.json_out).resolve() if args.json_out else run_dir / "focused_1915_reasoning_summary.json"

    write_markdown(summary, markdown_out)
    json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"Wrote markdown summary: {markdown_out}")
    print(f"Wrote json summary: {json_out}")


if __name__ == "__main__":
    main()
