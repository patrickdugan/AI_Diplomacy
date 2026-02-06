#!/usr/bin/env python3
"""Run headless Diplomacy games and export JSONL traces for Storyforge."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List
from ai_diplomacy.redaction import redact_data, redact_text


def default_output_dir() -> Path:
    if Path("D:/").exists():
        return Path("D:/storyworld_runs/diplomacy_traces")
    return Path("./runs/diplomacy_traces")


def run_game(lm_game: Path, run_dir: Path, max_year: int, extra_args: List[str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(lm_game), "--run_dir", str(run_dir), "--max_year", str(max_year)]
    cmd.extend(extra_args)
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    (run_dir / "console.log").write_text(redact_text(result.stdout), encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"lm_game failed with code {result.returncode}")


def build_unit_owner_map(units: Dict[str, List[str]]) -> Dict[str, str]:
    owner = {}
    for power, unit_list in units.items():
        for unit in unit_list:
            owner[unit] = power
    return owner


def emit_event(out_fh, episode_id: str, t: int, phase: str, power: str, event_type: str, payload: dict) -> int:
    row = {
        "episode_id": episode_id,
        "t": t,
        "phase": phase,
        "power": power,
        "event_type": event_type,
        "payload": payload,
    }
    out_fh.write(json.dumps(redact_data(row)) + "\n")
    return t + 1


def export_lmvs(lmvs_path: Path, out_path: Path) -> None:
    data = json.loads(lmvs_path.read_text(encoding="utf-8"))
    episode_id = str(data.get("id") or lmvs_path.parent.name)
    phases = data.get("phases", [])
    messages = data.get("messages", []) or []

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t = 0
    with out_path.open("w", encoding="utf-8") as out_fh:
        for ph in phases:
            phase_name = ph.get("name") or "UNKNOWN"
            state = ph.get("state", {})
            units = state.get("units", {})
            centers = state.get("centers", {})
            influence = state.get("influence", {})

            for power in centers.keys():
                payload = {
                    "centers_count": len(centers.get(power, [])),
                    "units_count": len(units.get(power, [])),
                    "influence_count": len(influence.get(power, [])),
                }
                t = emit_event(out_fh, episode_id, t, phase_name, power, "state", payload)

            orders = ph.get("orders", {}) or {}
            for power, order_list in orders.items():
                for order in order_list or []:
                    payload = {"order": order}
                    t = emit_event(out_fh, episode_id, t, phase_name, power, "order", payload)

            unit_owner = build_unit_owner_map(units)
            results = ph.get("results", {}) or {}
            for unit, result_list in results.items():
                power = unit_owner.get(unit, "UNKNOWN")
                payload = {"unit": unit, "results": result_list}
                t = emit_event(out_fh, episode_id, t, phase_name, power, "resolution", payload)

            for power in centers.keys():
                payload = {
                    "centers_count": len(centers.get(power, []))
                }
                t = emit_event(out_fh, episode_id, t, phase_name, power, "result", payload)

            for msg in messages:
                if msg.get("phase") != phase_name:
                    continue
                sender = msg.get("sender", "UNKNOWN")
                payload = {
                    "recipient": msg.get("recipient"),
                    "message": msg.get("message")
                }
                t = emit_event(out_fh, episode_id, t, phase_name, sender, "message", payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument("--max_year", type=int, default=1902)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--run_root", type=str, default="")
    parser.add_argument("--lm_game", type=str, default="lm_game.py")
    parser.add_argument("--extra", type=str, nargs="*", default=[])
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    run_root = Path(args.run_root) if args.run_root else (out_dir / "runs")
    lm_game = Path(args.lm_game).resolve()

    for i in range(args.iterations):
        seed = args.seed_base + i
        run_dir = run_root / f"run_{i:05d}"
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(seed)
        run_game(lm_game, run_dir, args.max_year, args.extra)

        lmvs_path = run_dir / "lmvsgame.json"
        if not lmvs_path.exists():
            raise FileNotFoundError(str(lmvs_path))

        out_path = out_dir / f"game_{i:05d}.jsonl"
        export_lmvs(lmvs_path, out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
