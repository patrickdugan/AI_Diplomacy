from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json5
import json_repair

from .utils import get_board_state

try:
    import sys

    _GPT_STORYWORLD_DIR = Path(os.getenv("GPT_STORYWORLD_DIR", "C:/projects/GPTStoryworld"))
    if _GPT_STORYWORLD_DIR.exists():
        sys.path.insert(0, str(_GPT_STORYWORLD_DIR))
    from storyworld.env import DiplomacyStoryworldEnv, load_storyworld
except Exception:
    DiplomacyStoryworldEnv = None  # type: ignore[assignment]
    load_storyworld = None  # type: ignore[assignment]


@dataclass
class StoryworldForecast:
    storyworld_id: str
    mapped_agents: Dict[str, str]
    target_power: str
    forecast: Dict[str, Any]
    confidence: float
    reasoning: str
    raw: Dict[str, Any]


def _default_storyworld_path() -> Path:
    base = Path(os.getenv("GPT_STORYWORLD_DIR", "C:/projects/GPTStoryworld"))
    return base / "storyworld" / "examples" / "diplomacy_min.json"


def _safe_json_load(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return json5.loads(text)
    except Exception:
        pass
    try:
        return json_repair.loads(text)
    except Exception:
        return {}


def _pick_storyworld_agents(active_powers: List[str], focal_power: str) -> Tuple[List[str], Dict[str, str]]:
    active = [p for p in active_powers if p]
    if focal_power not in active:
        active.insert(0, focal_power)
    others = [p for p in active if p != focal_power]
    chosen = [focal_power] + others[:2]
    while len(chosen) < 3:
        chosen.append(focal_power)
    mapping = {chosen[0]: "AgentA", chosen[1]: "AgentB", chosen[2]: "AgentC"}
    return chosen, mapping


def _build_forecast_prompt(
    power_name: str,
    board_state: Dict[str, Any],
    game,
    active_powers: List[str],
    story_state: Dict[str, Any],
    mapping: Dict[str, str],
) -> str:
    units_repr, centers_repr = get_board_state(board_state, game)
    focus_targets = [p for p in active_powers if p != power_name]
    prompt = {
        "task": "Generate a probabilistic forecast for a Diplomacy negotiation turn using the storyworld lens.",
        "power_name": power_name,
        "active_powers": active_powers,
        "focus_targets": focus_targets[:2],
        "mapping": mapping,
        "phase": getattr(game, "current_short_phase", ""),
        "board_summary": {
            "units": units_repr,
            "centers": centers_repr,
        },
        "storyworld_state": {
            "active_node": story_state.get("active_node"),
            "beliefs": story_state.get("beliefs", {}).get(mapping.get(power_name, "AgentA"), {}),
            "coalitions": story_state.get("coalitions", []),
            "world_vars": story_state.get("world_vars", {}),
        },
        "output_schema": {
            "target_power": "POWER_NAME",
            "forecast": {
                "question_id": "q1",
                "likely_outcome": "betrayal|no_betrayal|coalition_formed|stalemate|maneuver",
                "probabilities": {"betrayal": 0.6, "no_betrayal": 0.4},
            },
            "confidence": 0.75,
            "reasoning": "short causal rationale grounded in board state + storyworld state",
        },
    }
    return json.dumps(prompt, ensure_ascii=False, indent=2)


async def generate_storyworld_forecast(
    *,
    power_name: str,
    board_state: Dict[str, Any],
    game,
    active_powers: List[str],
    model_client,
    log_path: Optional[Path] = None,
    storyworld_path: Optional[Path] = None,
    seed: int = 42,
) -> Optional[StoryworldForecast]:
    if DiplomacyStoryworldEnv is None or load_storyworld is None:
        return None

    world_path = storyworld_path or _default_storyworld_path()
    if not world_path.exists():
        return None

    try:
        data = load_storyworld(world_path)
    except Exception:
        return None

    env = DiplomacyStoryworldEnv(data, seed=seed, log_path=None)
    state = env.reset(seed=seed)

    chosen, mapping = _pick_storyworld_agents(active_powers, power_name)
    prompt = _build_forecast_prompt(
        power_name=power_name,
        board_state=board_state,
        game=game,
        active_powers=chosen,
        story_state=state,
        mapping=mapping,
    )

    try:
        try:
            raw = await model_client.generate_response(prompt, temperature=0.2)
        except TypeError:
            raw = await model_client.generate_response(prompt)
    except Exception:
        return None

    data = _safe_json_load(raw)
    if not data:
        return None

    target = data.get("target_power") or (chosen[1] if len(chosen) > 1 else power_name)
    forecast = data.get("forecast", {})
    confidence = float(data.get("confidence", 0.5) or 0.5)
    reasoning = str(data.get("reasoning", "")).strip()

    artifact = StoryworldForecast(
        storyworld_id=str(data.get("storyworld_id", data.get("world_id", data.get("id", "diplomacy_min")))),
        mapped_agents=mapping,
        target_power=str(target),
        forecast=forecast if isinstance(forecast, dict) else {},
        confidence=confidence,
        reasoning=reasoning,
        raw=data,
    )

    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": time.time(),
            "phase": getattr(game, "current_short_phase", ""),
            "power": power_name,
            "storyworld_path": str(world_path),
            "artifact": {
                "storyworld_id": artifact.storyworld_id,
                "mapped_agents": artifact.mapped_agents,
                "target_power": artifact.target_power,
                "forecast": artifact.forecast,
                "confidence": artifact.confidence,
                "reasoning": artifact.reasoning,
            },
            "raw": artifact.raw,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return artifact
