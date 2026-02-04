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


def _storyworld_bank_dir() -> Path:
    return Path(
        os.getenv(
            "STORYWORLD_BANK_DIR",
            str(Path(__file__).resolve().parents[0] / "storyworld_bank"),
        )
    )


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


def _select_template(power_name: str, phase: str, bank_dir: Path) -> Optional[Dict[str, Any]]:
    if not bank_dir.exists():
        return None
    candidates = sorted(bank_dir.glob("*.json"))
    if not candidates:
        return None
    key = f"{power_name}:{phase}"
    idx = abs(hash(key)) % len(candidates)
    try:
        return json.loads(candidates[idx].read_text(encoding="utf-8"))
    except Exception:
        return None


def _choose_threat_and_target(active_powers: List[str], board_state: Dict[str, Any], power_name: str) -> Tuple[str, str]:
    centers = board_state.get("centers", {})

    def center_count(p: str) -> int:
        return len(centers.get(p, []))

    others = [p for p in active_powers if p != power_name]
    if not others:
        return power_name, power_name
    threat = max(others, key=center_count)
    target = min(others, key=center_count) if len(others) > 1 else power_name
    return threat, target


def _fill_template(template: Dict[str, Any], threat: str, target: str) -> Dict[str, Any]:
    filled = json.loads(json.dumps(template))
    message = filled.get("message_frame", {})
    ask = message.get("ask", "")
    if "{ASK_SUPPORT}" in ask:
        ask = ask.replace("{ASK_SUPPORT}", "support a key move or agree a DMZ")
    if "{THREAT}" in ask:
        ask = ask.replace("{THREAT}", threat)
    if "{TARGET}" in ask:
        ask = ask.replace("{TARGET}", target)
    message["ask"] = ask

    claim = message.get("claim", "")
    claim = claim.replace("{THREAT}", threat).replace("{TARGET}", target).replace("{ZONE}", "a key border zone")
    message["claim"] = claim

    evidence = message.get("evidence", "")
    evidence = evidence.replace("{THREAT}", threat).replace("{TARGET}", target).replace("{ZONE}", "a key border zone")
    message["evidence"] = evidence

    filled["message_frame"] = message
    return filled


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


def _get_text(node: Any) -> str:
    if isinstance(node, dict):
        if isinstance(node.get("value"), str):
            return node["value"]
        if isinstance(node.get("text"), str):
            return node["text"]
    if isinstance(node, str):
        return node
    return ""


def _is_ending(encounter: Dict[str, Any]) -> bool:
    if not encounter:
        return True
    options = encounter.get("options", []) or []
    if not options:
        return True
    # If all options have no reactions/consequence, treat as terminal
    for opt in options:
        reactions = opt.get("reactions", []) or []
        if reactions and reactions[0].get("consequence_id"):
            return False
    return True


def _extract_choice_id(raw: str, valid_ids: List[str]) -> Optional[str]:
    if not raw:
        return None
    raw = raw.strip()
    # direct match
    if raw in valid_ids:
        return raw
    # look for id in text
    for cid in valid_ids:
        if cid in raw:
            return cid
    # try JSON
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            cid = data.get("choice_id") or data.get("id")
            if isinstance(cid, str) and cid in valid_ids:
                return cid
    except Exception:
        pass
    return None


def _play_storyworld_path(path: Path, max_steps: int = 3) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    encounters = data.get("encounters", []) or []
    spools = data.get("spools", []) or []

    first_encounter_id = None
    for spool in spools:
        if spool.get("starts_active") and spool.get("encounters"):
            first_encounter_id = spool["encounters"][0]
            break
    if not first_encounter_id and encounters:
        first_encounter_id = encounters[0].get("id")

    history = []
    current_id = first_encounter_id
    steps = 0
    visited = set()

    while current_id and steps < max_steps and current_id not in visited:
        visited.add(current_id)
        encounter = next((e for e in encounters if e.get("id") == current_id), None)
        if not encounter:
            break
        options = encounter.get("options", []) or []
        chosen = options[0] if options else None
        choice_text = _get_text(chosen.get("text_script", {})) if chosen else ""
        reaction = None
        next_id = None
        if chosen and chosen.get("reactions"):
            reaction = chosen["reactions"][0]
            next_id = reaction.get("consequence_id")

        history.append(
            {
                "encounter_id": encounter.get("id"),
                "encounter_title": encounter.get("title", ""),
                "choice": choice_text,
                "next_encounter_id": next_id,
            }
        )
        current_id = next_id
        steps += 1

    return {
        "storyworld_id": data.get("IFID") or data.get("storyworld_title") or data.get("storyworld_title", ""),
        "steps": len(history),
        "history": history,
    }


async def _play_storyworld_with_model(path: Path, model_client, max_steps: int = 12) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    encounters = data.get("encounters", []) or []
    spools = data.get("spools", []) or []

    first_encounter_id = None
    for spool in spools:
        if spool.get("starts_active") and spool.get("encounters"):
            first_encounter_id = spool["encounters"][0]
            break
    if not first_encounter_id and encounters:
        first_encounter_id = encounters[0].get("id")

    history = []
    current_id = first_encounter_id
    steps = 0
    visited = set()

    while current_id and steps < max_steps and current_id not in visited:
        visited.add(current_id)
        encounter = next((e for e in encounters if e.get("id") == current_id), None)
        if not encounter:
            break

        options = encounter.get("options", []) or []
        option_ids = [o.get("id") for o in options if isinstance(o.get("id"), str)]
        option_texts = []
        for opt in options:
            option_texts.append(
                {
                    "id": opt.get("id"),
                    "text": _get_text(opt.get("text_script", {})),
                }
            )

        prompt = {
            "task": "Choose exactly one option id.",
            "encounter_title": encounter.get("title", ""),
            "encounter_text": _get_text(encounter.get("text_script", {})),
            "options": option_texts,
            "output_schema": {"choice_id": "opt_id"},
        }
        try:
            try:
                raw = await model_client.generate_response(json.dumps(prompt, ensure_ascii=False), temperature=0.2)
            except TypeError:
                raw = await model_client.generate_response(json.dumps(prompt, ensure_ascii=False))
        except Exception:
            raw = ""

        choice_id = _extract_choice_id(raw, option_ids) if option_ids else None
        if not choice_id and options:
            choice_id = options[0].get("id")

        chosen = next((o for o in options if o.get("id") == choice_id), None)
        choice_text = _get_text(chosen.get("text_script", {})) if chosen else ""
        reaction = None
        next_id = None
        if chosen and chosen.get("reactions"):
            reaction = chosen["reactions"][0]
            next_id = reaction.get("consequence_id")

        history.append(
            {
                "encounter_id": encounter.get("id"),
                "encounter_title": encounter.get("title", ""),
                "choice_id": choice_id,
                "choice_text": choice_text,
                "next_encounter_id": next_id,
                "raw_model_output": raw[:500],
            }
        )

        if _is_ending(encounter) or not next_id:
            break

        current_id = next_id
        steps += 1

    return {
        "storyworld_id": data.get("IFID") or data.get("storyworld_title") or data.get("storyworld_title", ""),
        "steps": len(history),
        "history": history,
    }


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

    phase = getattr(game, "current_short_phase", "")
    bank_dir = _storyworld_bank_dir()
    bank_only = os.getenv("STORYWORLD_BANK_ONLY", "0").strip() == "1"
    playback_enabled = os.getenv("STORYWORLD_PLAYBACK", "0").strip() == "1"
    play_mode = os.getenv("STORYWORLD_PLAY_MODE", "heuristic").strip().lower()
    max_steps = int(os.getenv("STORYWORLD_PLAY_MAX_STEPS", "12"))

    world_path = storyworld_path or _default_storyworld_path()
    if not world_path.exists() and not bank_dir.exists():
        return None

    try:
        data = load_storyworld(world_path) if world_path.exists() else {}
    except Exception:
        return None

    env = DiplomacyStoryworldEnv(data, seed=seed, log_path=None)
    state = env.reset(seed=seed)

    chosen, mapping = _pick_storyworld_agents(active_powers, power_name)
    threat, target = _choose_threat_and_target(chosen, board_state, power_name)

    if bank_only:
        template = _select_template(power_name, phase, bank_dir)
        if not template:
            return None
        filled = _fill_template(template, threat, target)
        data = {
            "storyworld_id": filled.get("id", "template"),
            "target_power": target,
            "forecast": filled.get("forecast", {}),
            "confidence": filled.get("forecast", {}).get("probabilities", {}).get("target_attacks", 0.6),
            "reasoning": filled.get("intent", ""),
            "message_frame": filled.get("message_frame", {}),
        }
        if playback_enabled and template.get("source"):
            try:
                if play_mode == "model":
                    playback = await _play_storyworld_with_model(Path(template["source"]), model_client, max_steps=max_steps)
                else:
                    playback = _play_storyworld_path(Path(template["source"]), max_steps=3)
                data["playback"] = playback
            except Exception:
                pass
        raw = json.dumps(data)
    else:
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
