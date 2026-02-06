#!/usr/bin/env python3
"""
Enforce richer p/p2 desirability and effect coverage for Diplomacy *_p storyworlds.

For each reaction in power_* storyworlds, this script now ensures:
1) legacy fields remain populated for compatibility (`desirability_script`, `after_effects`)
2) editor AST fields are populated (`inclination_ast`, `desirability_ast`) with p and p2 terms
3) editor effect rows are populated (`effects`) using SweepWeave TS `Effect` shapes
4) authored Trust/Threat depth is at least 2 so perceived fields are editable in UI
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


COOPERATIVE_KEYWORDS = (
    "ally",
    "alliance",
    "coalition",
    "support",
    "cooperate",
    "cooperation",
    "trust",
    "pact",
    "peace",
    "joint",
    "coordinate",
    "concede",
    "concession",
)

AGGRESSIVE_KEYWORDS = (
    "betray",
    "backstab",
    "punish",
    "revenge",
    "threat",
    "attack",
    "war",
    "strike",
    "defect",
    "treach",
    "collapse",
    "ultimatum",
    "coerc",
    "blame",
)

AST_NODE_TYPES = {
    "Constant",
    "BNumberProperty",
    "ArithmeticNegation",
    "Proximity",
    "Average",
    "Blend",
    "Maximum",
    "Minimum",
    "Nudge",
    "IfThen",
}

DEFAULT_GLOBS = (
    r"ai_diplomacy/storyworld_sources/*_p.json",
    r"ai_diplomacy/storyworld_bank_extracted/*_p.json",
    r"ai_diplomacy/storyworld_bank_focus_1915/*.json",
    r"ai_diplomacy/storyworld_bank_focus_1915/full_storyworlds/*.json",
)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def ast_constant(value: float) -> Dict[str, Any]:
    return {"type": "Constant", "value": round(clamp01(float(value)), 4)}


def ast_bnumber_property(
    character_id: str,
    property_id: str,
    perceived_1: str = "",
    perceived_2: str = "",
) -> Dict[str, Any]:
    return {
        "type": "BNumberProperty",
        "characterId": character_id,
        "propertyId": property_id,
        "perceivedCharacterId": perceived_1,
        "perceivedCharacterId2": perceived_2,
    }


def walk(node: Any) -> Iterable[Any]:
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from walk(value)


def reaction_text(reaction: Dict[str, Any]) -> str:
    text_script = reaction.get("text_script")
    if isinstance(text_script, dict):
        for key in ("value", "text"):
            value = text_script.get(key)
            if isinstance(value, str):
                return value
    if isinstance(text_script, str):
        return text_script
    return ""


def classify_reaction(reaction: Dict[str, Any]) -> str:
    rid = str(reaction.get("id", "")).lower()
    cid = str(reaction.get("consequence_id", "")).lower()
    text = reaction_text(reaction).lower()
    haystack = f"{rid} {cid} {text}"

    if any(keyword in haystack for keyword in AGGRESSIVE_KEYWORDS):
        return "aggressive"
    if any(keyword in haystack for keyword in COOPERATIVE_KEYWORDS):
        return "cooperative"
    return "neutral"


def coeff_profile(kind: str) -> Dict[str, float]:
    if kind == "cooperative":
        return {
            "des_p": 0.28,
            "des_p2": 0.18,
            "des_threat": -0.16,
            "eff_p": 0.08,
            "eff_p2": 0.05,
            "eff_threat": -0.05,
        }
    if kind == "aggressive":
        return {
            "des_p": -0.30,
            "des_p2": -0.20,
            "des_threat": 0.20,
            "eff_p": -0.09,
            "eff_p2": -0.06,
            "eff_threat": 0.06,
        }
    return {
        "des_p": 0.08,
        "des_p2": 0.05,
        "des_threat": -0.02,
        "eff_p": 0.03,
        "eff_p2": 0.02,
        "eff_threat": 0.01,
    }


def const_ptr(value: float) -> Dict[str, Any]:
    return {
        "pointer_type": "Bounded Number Constant",
        "script_element_type": "Pointer",
        "value": value,
    }


def prop_ptr(character: str, keyring: List[str], coefficient: float = 1.0) -> Dict[str, Any]:
    return {
        "pointer_type": "Bounded Number Property",
        "script_element_type": "Pointer",
        "character": character,
        "keyring": keyring,
        "coefficient": coefficient,
    }


def add_expr(base_ptr: Dict[str, Any], delta: float) -> Dict[str, Any]:
    return {
        "script_element_type": "Bounded Number Operator",
        "operator_type": "Addition",
        "operands": [base_ptr, const_ptr(delta)],
    }


def normalize_desirability(script: Any) -> Dict[str, Any]:
    if isinstance(script, dict):
        return script
    if isinstance(script, bool):
        return const_ptr(1.0 if script else 0.0)
    if isinstance(script, (int, float)):
        return const_ptr(float(script))
    return const_ptr(0.0)


def choose_property_id(data: Dict[str, Any], preferred_tokens: Tuple[str, ...], fallback: Optional[str]) -> Optional[str]:
    authored = data.get("authored_properties", []) or []
    candidates = [str(prop.get("id", "")) for prop in authored if isinstance(prop, dict) and prop.get("id")]

    for token in preferred_tokens:
        for cid in candidates:
            if token in cid.lower():
                return cid
    return fallback


def collect_keyrings(node: Any) -> List[List[str]]:
    out: List[List[str]] = []
    for item in walk(node):
        if not isinstance(item, dict):
            continue
        keyring = item.get("keyring")
        if isinstance(keyring, list) and keyring:
            out.append([str(part) for part in keyring])
    return out


def has_exact_keyring(node: Any, keyring: List[str]) -> bool:
    return any(kr == keyring for kr in collect_keyrings(node))


def has_set_effect(after_effects: List[Dict[str, Any]], character: str, keyring: List[str]) -> bool:
    for effect in after_effects:
        if not isinstance(effect, dict):
            continue
        if effect.get("effect_type") != "Set":
            continue
        setter = effect.get("Set", {})
        if isinstance(setter, dict) and setter.get("character") == character and setter.get("keyring") == keyring:
            return True
    return False


def add_set_effect(after_effects: List[Dict[str, Any]], character: str, keyring: List[str], delta: float) -> bool:
    if has_set_effect(after_effects, character, keyring):
        return False
    base = prop_ptr(character=character, keyring=keyring, coefficient=1.0)
    after_effects.append(
        {
            "effect_type": "Set",
            "Set": base,
            "to": add_expr(base, delta),
        }
    )
    return True


def infer_target_and_witness(
    reaction: Dict[str, Any],
    focal: str,
    default_target: str,
    default_witness: str,
    trust_prop: str,
    chars: List[str],
) -> Tuple[str, str]:
    keyrings = collect_keyrings(reaction)

    for keyring in keyrings:
        if len(keyring) >= 3 and keyring[0] == trust_prop:
            target = keyring[1]
            witness = keyring[2]
            if target in chars and witness in chars and target != focal and witness != target:
                return target, witness

    for keyring in keyrings:
        if len(keyring) >= 2 and keyring[0] == trust_prop:
            target = keyring[1]
            if target in chars and target != focal:
                witness = next((c for c in chars if c not in {focal, target}), default_witness)
                return target, witness

    return default_target, default_witness


def ensure_property_depth(data: Dict[str, Any], prop_id: Optional[str], min_depth: int = 2) -> bool:
    if not prop_id:
        return False
    changed = False
    for prop in data.get("authored_properties", []) or []:
        if not isinstance(prop, dict):
            continue
        if str(prop.get("id")) != prop_id:
            continue
        current = prop.get("depth", 0)
        try:
            current_depth = int(current)
        except (TypeError, ValueError):
            current_depth = 0
        if current_depth < min_depth:
            prop["depth"] = min_depth
            changed = True
    return changed


def node_constant_value(node: Any) -> Optional[float]:
    if not isinstance(node, dict):
        return None
    if node.get("type") != "Constant":
        return None
    value = node.get("value")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def legacy_to_ast(
    node: Any,
    fallback_character: str,
    fallback_property: str,
) -> Dict[str, Any]:
    if isinstance(node, dict) and node.get("type") in AST_NODE_TYPES:
        node_type = node.get("type")
        if node_type == "Constant":
            return ast_constant(float(node.get("value", 0.0)))
        if node_type == "BNumberProperty":
            return ast_bnumber_property(
                character_id=str(node.get("characterId", fallback_character)),
                property_id=str(node.get("propertyId", fallback_property)),
                perceived_1=str(node.get("perceivedCharacterId", "")),
                perceived_2=str(node.get("perceivedCharacterId2", "")),
            )
        if node_type == "ArithmeticNegation":
            return {"type": "ArithmeticNegation", "child": legacy_to_ast(node.get("child"), fallback_character, fallback_property)}
        if node_type in {"Proximity", "Average", "Maximum", "Minimum"}:
            return {
                "type": node_type,
                "left": legacy_to_ast(node.get("left"), fallback_character, fallback_property),
                "right": legacy_to_ast(node.get("right"), fallback_character, fallback_property),
            }
        if node_type == "Blend":
            return {
                "type": "Blend",
                "left": legacy_to_ast(node.get("left"), fallback_character, fallback_property),
                "right": legacy_to_ast(node.get("right"), fallback_character, fallback_property),
                "weight": legacy_to_ast(node.get("weight"), fallback_character, fallback_property),
            }
        if node_type == "Nudge":
            return {
                "type": "Nudge",
                "base": legacy_to_ast(node.get("base"), fallback_character, fallback_property),
                "nudge": legacy_to_ast(node.get("nudge"), fallback_character, fallback_property),
            }
        if node_type == "IfThen":
            return {
                "type": "IfThen",
                "condition": node.get("condition", {"type": "Constant", "value": False}),
                "thenNode": legacy_to_ast(node.get("thenNode"), fallback_character, fallback_property),
                "elseNode": legacy_to_ast(node.get("elseNode"), fallback_character, fallback_property),
            }
        return ast_constant(0.0)

    if isinstance(node, bool):
        return ast_constant(1.0 if node else 0.0)
    if isinstance(node, (int, float)):
        return ast_constant(float(node))
    if not isinstance(node, dict):
        return ast_constant(0.0)

    pointer_type = str(node.get("pointer_type", ""))
    script_element_type = str(node.get("script_element_type", ""))

    if script_element_type == "Pointer" and pointer_type == "Bounded Number Constant":
        value = node.get("value", node.get("coefficient", 0.0))
        if isinstance(value, (int, float)):
            return ast_constant(float(value))
        return ast_constant(0.0)

    if script_element_type == "Pointer" and pointer_type == "Bounded Number Property":
        keyring = node.get("keyring") if isinstance(node.get("keyring"), list) else []
        keyring = [str(part) for part in keyring]
        property_id = keyring[0] if keyring else fallback_property
        perceived_1 = keyring[1] if len(keyring) > 1 else ""
        perceived_2 = keyring[2] if len(keyring) > 2 else ""
        return ast_bnumber_property(
            character_id=str(node.get("character", fallback_character)),
            property_id=property_id,
            perceived_1=perceived_1,
            perceived_2=perceived_2,
        )

    if script_element_type == "Bounded Number Operator":
        operator_type = str(node.get("operator_type", "")).lower()
        operands_raw = node.get("operands") if isinstance(node.get("operands"), list) else []
        operands = [legacy_to_ast(op, fallback_character, fallback_property) for op in operands_raw]
        if not operands:
            return ast_constant(0.0)

        if operator_type in {"addition", "bounded sum"}:
            if len(operands) == 1:
                return operands[0]
            left = operands[0]
            right = operands[1]
            right_const = node_constant_value(right)
            left_const = node_constant_value(left)
            if right_const is not None:
                if right_const >= 0:
                    return {"type": "Nudge", "base": left, "nudge": ast_constant(abs(right_const))}
                return {"type": "Blend", "left": left, "right": ast_constant(0.0), "weight": ast_constant(abs(right_const))}
            if left_const is not None:
                if left_const >= 0:
                    return {"type": "Nudge", "base": right, "nudge": ast_constant(abs(left_const))}
                return {"type": "Blend", "left": right, "right": ast_constant(0.0), "weight": ast_constant(abs(left_const))}
            merged = {"type": "Average", "left": left, "right": right}
            for extra in operands[2:]:
                merged = {"type": "Average", "left": merged, "right": extra}
            return merged

        if operator_type in {"average", "arithmetic mean"}:
            if len(operands) == 1:
                return operands[0]
            return {"type": "Average", "left": operands[0], "right": operands[1]}

        if operator_type in {"maximum"}:
            if len(operands) == 1:
                return operands[0]
            return {"type": "Maximum", "left": operands[0], "right": operands[1]}

        if operator_type in {"minimum"}:
            if len(operands) == 1:
                return operands[0]
            return {"type": "Minimum", "left": operands[0], "right": operands[1]}

        if operator_type in {"proximity", "proximity to"}:
            if len(operands) == 1:
                return operands[0]
            return {"type": "Proximity", "left": operands[0], "right": operands[1]}

        if operator_type in {"blend"}:
            if len(operands) == 1:
                return operands[0]
            if len(operands) == 2:
                return {"type": "Blend", "left": operands[0], "right": operands[1], "weight": ast_constant(0.5)}
            return {"type": "Blend", "left": operands[0], "right": operands[1], "weight": operands[2]}

        if operator_type in {"nudge"}:
            if len(operands) == 1:
                return operands[0]
            return {"type": "Nudge", "base": operands[0], "nudge": operands[1]}

        if operator_type in {"arithmetic negation", "negation"}:
            return {"type": "ArithmeticNegation", "child": operands[0]}

    if "to" in node:
        return legacy_to_ast(node.get("to"), fallback_character, fallback_property)

    return ast_constant(0.0)


def build_desirability_ast(
    focal: str,
    target: str,
    witness: str,
    trust_prop: str,
    threat_prop: Optional[str],
    kind: str,
) -> Dict[str, Any]:
    trust_p = ast_bnumber_property(focal, trust_prop, target, "")
    trust_p2 = ast_bnumber_property(focal, trust_prop, target, witness)
    trust_mean = {"type": "Average", "left": trust_p, "right": trust_p2}
    trust_alignment = {"type": "Proximity", "left": trust_p, "right": trust_p2}
    trust_logic = {
        "type": "Blend",
        "left": trust_mean,
        "right": trust_alignment,
        "weight": ast_constant(0.35),
    }

    if not threat_prop:
        return trust_logic

    threat_p = ast_bnumber_property(focal, threat_prop, target, "")
    safety = {"type": "ArithmeticNegation", "child": threat_p}

    if kind == "cooperative":
        # Cooperative choices value high trust, coherent beliefs, and low threat.
        return {
            "type": "Blend",
            "left": trust_logic,
            "right": safety,
            "weight": ast_constant(0.28),
        }

    if kind == "aggressive":
        # Aggressive choices model suspicion + threat salience + belief volatility.
        suspicion = {
            "type": "Blend",
            "left": {"type": "ArithmeticNegation", "child": trust_mean},
            "right": threat_p,
            "weight": ast_constant(0.62),
        }
        volatility = {"type": "ArithmeticNegation", "child": trust_alignment}
        return {
            "type": "Blend",
            "left": suspicion,
            "right": volatility,
            "weight": ast_constant(0.36),
        }

    balanced = {
        "type": "Blend",
        "left": trust_logic,
        "right": safety,
        "weight": ast_constant(0.45),
    }
    return {
        "type": "Blend",
        "left": balanced,
        "right": trust_alignment,
        "weight": ast_constant(0.2),
    }


def ast_from_keyring(character: str, keyring: List[str]) -> Dict[str, Any]:
    property_id = keyring[0]
    perceived_1 = keyring[1] if len(keyring) > 1 else ""
    perceived_2 = keyring[2] if len(keyring) > 2 else ""
    return ast_bnumber_property(
        character_id=character,
        property_id=property_id,
        perceived_1=perceived_1,
        perceived_2=perceived_2,
    )


def extract_delta_from_set_effect(entry: Dict[str, Any]) -> float:
    to_expr = entry.get("to")
    if not isinstance(to_expr, dict):
        return 0.0
    pointer_type = str(to_expr.get("pointer_type", ""))
    if pointer_type == "Bounded Number Constant":
        value = to_expr.get("value", to_expr.get("coefficient", 0.0))
        if isinstance(value, (int, float)):
            return float(value)
        return 0.0

    script_element_type = str(to_expr.get("script_element_type", ""))
    operator_type = str(to_expr.get("operator_type", "")).lower()
    if script_element_type != "Bounded Number Operator" or operator_type not in {"addition", "bounded sum"}:
        return 0.0

    delta = 0.0
    for operand in to_expr.get("operands", []) or []:
        if not isinstance(operand, dict):
            continue
        if str(operand.get("pointer_type", "")) != "Bounded Number Constant":
            continue
        value = operand.get("value", operand.get("coefficient", 0.0))
        if isinstance(value, (int, float)):
            delta += float(value)
    return delta


def build_effect_signals(
    focal: str,
    target: str,
    witness: str,
    trust_prop: str,
    threat_prop: Optional[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    trust_p = ast_bnumber_property(focal, trust_prop, target, "")
    trust_p2 = ast_bnumber_property(focal, trust_prop, target, witness)
    trust_mean = {"type": "Average", "left": trust_p, "right": trust_p2}
    trust_alignment = {"type": "Proximity", "left": trust_p, "right": trust_p2}
    trust_signal: Dict[str, Any] = {
        "type": "Blend",
        "left": trust_mean,
        "right": trust_alignment,
        "weight": ast_constant(0.35),
    }

    if threat_prop:
        threat_p = ast_bnumber_property(focal, threat_prop, target, "")
        threat_signal: Dict[str, Any] = {
            "type": "Blend",
            "left": threat_p,
            "right": {"type": "ArithmeticNegation", "child": trust_alignment},
            "weight": ast_constant(0.45),
        }
    else:
        threat_signal = {"type": "ArithmeticNegation", "child": trust_signal}
    return trust_signal, threat_signal


def select_effect_signal(
    property_id: str,
    kind: str,
    trust_signal: Dict[str, Any],
    threat_signal: Dict[str, Any],
) -> Dict[str, Any]:
    prop_lower = property_id.lower()
    is_threat = "threat" in prop_lower

    if kind == "aggressive":
        return threat_signal if is_threat else {"type": "ArithmeticNegation", "child": trust_signal}
    if kind == "cooperative":
        return {"type": "ArithmeticNegation", "child": threat_signal} if is_threat else trust_signal

    if is_threat:
        return {
            "type": "Blend",
            "left": threat_signal,
            "right": {"type": "ArithmeticNegation", "child": trust_signal},
            "weight": ast_constant(0.5),
        }
    return {
        "type": "Blend",
        "left": trust_signal,
        "right": {"type": "ArithmeticNegation", "child": threat_signal},
        "weight": ast_constant(0.3),
    }


def build_effect_value_ast(
    base: Dict[str, Any],
    signal: Dict[str, Any],
    delta: float,
    kind: str,
) -> Dict[str, Any]:
    magnitude = max(0.03, min(0.4, abs(delta)))
    if delta < -1e-6:
        # Negative deltas are rendered as explicit reversals.
        reversal = {
            "type": "Blend",
            "left": {"type": "ArithmeticNegation", "child": base},
            "right": {"type": "ArithmeticNegation", "child": signal},
            "weight": ast_constant(0.4),
        }
        return {
            "type": "Blend",
            "left": base,
            "right": reversal,
            "weight": ast_constant(magnitude),
        }

    if kind == "cooperative":
        blended = {
            "type": "Blend",
            "left": base,
            "right": signal,
            "weight": ast_constant(min(0.65, magnitude + 0.15)),
        }
        return {
            "type": "Nudge",
            "base": blended,
            "nudge": ast_constant(min(0.6, magnitude + 0.05)),
        }

    if kind == "aggressive":
        return {
            "type": "Blend",
            "left": base,
            "right": signal,
            "weight": ast_constant(min(0.75, magnitude + 0.2)),
        }

    return {
        "type": "Blend",
        "left": base,
        "right": signal,
        "weight": ast_constant(min(0.55, magnitude + 0.1)),
    }


def ast_has_p2(node: Any) -> bool:
    if isinstance(node, dict):
        if node.get("type") == "BNumberProperty" and str(node.get("perceivedCharacterId2", "")):
            return True
        return any(ast_has_p2(value) for value in node.values())
    if isinstance(node, list):
        return any(ast_has_p2(value) for value in node)
    return False


def convert_after_effects_to_effects(
    reaction: Dict[str, Any],
    focal: str,
    target: str,
    witness: str,
    trust_prop: str,
    threat_prop: Optional[str],
    kind: str,
    profile: Dict[str, float],
) -> List[Dict[str, Any]]:
    effects: List[Dict[str, Any]] = []
    seen: set = set()
    after_effects = reaction.get("after_effects")
    if not isinstance(after_effects, list):
        after_effects = []

    trust_signal, threat_signal = build_effect_signals(
        focal=focal,
        target=target,
        witness=witness,
        trust_prop=trust_prop,
        threat_prop=threat_prop,
    )

    for entry in after_effects:
        if not isinstance(entry, dict):
            continue
        if entry.get("effect_type") != "Set":
            continue
        setter = entry.get("Set", {})
        if not isinstance(setter, dict):
            continue
        keyring = setter.get("keyring") if isinstance(setter.get("keyring"), list) else []
        keyring = [str(part) for part in keyring]
        if not keyring:
            continue
        character = str(setter.get("character", "")).strip()
        if not character:
            continue
        property_id = keyring[0]
        base = ast_from_keyring(character, keyring)
        delta = extract_delta_from_set_effect(entry)
        signal = select_effect_signal(
            property_id=property_id,
            kind=kind,
            trust_signal=trust_signal,
            threat_signal=threat_signal,
        )
        value_ast = build_effect_value_ast(
            base=base,
            signal=signal,
            delta=delta,
            kind=kind,
        )
        fingerprint = (character, property_id, json.dumps(value_ast, sort_keys=True))
        if fingerprint in seen:
            continue
        effects.append(
            {
                "type": "SetBNumberProperty",
                "characterId": character,
                "propertyId": property_id,
                "value": value_ast,
            }
        )
        seen.add(fingerprint)

    if not effects:
        fallback_base = ast_bnumber_property(focal, trust_prop, target, "")
        fallback_signal = select_effect_signal(
            property_id=trust_prop,
            kind=kind,
            trust_signal=trust_signal,
            threat_signal=threat_signal,
        )
        fallback_value = build_effect_value_ast(
            base=fallback_base,
            signal=fallback_signal,
            delta=float(profile["eff_p"]),
            kind=kind,
        )
        effects.append(
            {
                "type": "SetBNumberProperty",
                "characterId": focal,
                "propertyId": trust_prop,
                "value": fallback_value,
            }
        )

    if not any(ast_has_p2(effect.get("value")) for effect in effects):
        p2_base = ast_bnumber_property(focal, trust_prop, target, witness)
        p2_signal = select_effect_signal(
            property_id=trust_prop,
            kind=kind,
            trust_signal=trust_signal,
            threat_signal=threat_signal,
        )
        p2_value = build_effect_value_ast(
            base=p2_base,
            signal=p2_signal,
            delta=float(profile["eff_p2"]),
            kind=kind,
        )
        effects.append(
            {
                "type": "SetBNumberProperty",
                "characterId": focal,
                "propertyId": trust_prop,
                "value": p2_value,
            }
        )

    return effects


def update_reaction(
    reaction: Dict[str, Any],
    focal: str,
    target: str,
    witness: str,
    trust_prop: str,
    threat_prop: Optional[str],
) -> Tuple[bool, int, int]:
    kind = classify_reaction(reaction)
    profile = coeff_profile(kind)
    changed = False
    added_des_terms = 0
    added_effects = 0

    desirability = normalize_desirability(reaction.get("desirability_script"))
    new_terms: List[Dict[str, Any]] = []

    trust_p_keyring = [trust_prop, target]
    trust_p2_keyring = [trust_prop, target, witness]

    if not has_exact_keyring(desirability, trust_p_keyring):
        new_terms.append(prop_ptr(focal, trust_p_keyring, profile["des_p"]))
    if not has_exact_keyring(desirability, trust_p2_keyring):
        new_terms.append(prop_ptr(focal, trust_p2_keyring, profile["des_p2"]))
    if threat_prop:
        threat_keyring = [threat_prop, target]
        if not has_exact_keyring(desirability, threat_keyring):
            new_terms.append(prop_ptr(focal, threat_keyring, profile["des_threat"]))

    if new_terms:
        reaction["desirability_script"] = {
            "script_element_type": "Bounded Number Operator",
            "operator_type": "Addition",
            "operands": [desirability] + new_terms,
        }
        changed = True
        added_des_terms = len(new_terms)

    after_effects = reaction.setdefault("after_effects", [])
    if not isinstance(after_effects, list):
        after_effects = []
        reaction["after_effects"] = after_effects
        changed = True

    if add_set_effect(after_effects, focal, trust_p_keyring, profile["eff_p"]):
        changed = True
        added_effects += 1
    if add_set_effect(after_effects, focal, trust_p2_keyring, profile["eff_p2"]):
        changed = True
        added_effects += 1
    if threat_prop:
        threat_keyring = [threat_prop, target]
        if add_set_effect(after_effects, focal, threat_keyring, profile["eff_threat"]):
            changed = True
            added_effects += 1

    ast = build_desirability_ast(
        focal=focal,
        target=target,
        witness=witness,
        trust_prop=trust_prop,
        threat_prop=threat_prop,
        kind=kind,
    )
    if reaction.get("inclination_ast") != ast:
        reaction["inclination_ast"] = ast
        changed = True
    if reaction.get("desirability_ast") != ast:
        reaction["desirability_ast"] = ast
        changed = True

    effects = convert_after_effects_to_effects(
        reaction=reaction,
        focal=focal,
        target=target,
        witness=witness,
        trust_prop=trust_prop,
        threat_prop=threat_prop,
        kind=kind,
        profile=profile,
    )
    if reaction.get("effects") != effects:
        reaction["effects"] = effects
        changed = True

    return changed, added_des_terms, added_effects


def is_diplomacy_power_cast(data: Dict[str, Any]) -> bool:
    chars = [c.get("id") for c in data.get("characters", []) if isinstance(c, dict) and c.get("id")]
    return bool(chars) and all(str(char).startswith("power_") for char in chars)


def augment_file(path: Path, dry_run: bool = False) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not is_diplomacy_power_cast(data):
        return {"file": str(path), "skipped": True, "reason": "non_diplomacy_power_cast"}

    chars = [c.get("id") for c in data.get("characters", []) if isinstance(c, dict) and c.get("id")]
    if len(chars) < 2:
        return {"file": str(path), "skipped": True, "reason": "insufficient_characters"}

    focal = chars[0]
    default_target = chars[1]
    default_witness = next((c for c in chars if c not in {focal, default_target}), default_target)
    trust_prop = choose_property_id(data, ("trust", "honest", "cooper", "commit"), "Trust")
    threat_prop = choose_property_id(data, ("threat", "hostile", "danger"), None)
    if not trust_prop:
        trust_prop = "Trust"

    original = copy.deepcopy(data)
    total_reactions = 0
    touched_reactions = 0
    added_des_terms = 0
    added_effects = 0
    depth_bumps = 0

    if ensure_property_depth(data, trust_prop, min_depth=2):
        depth_bumps += 1
    if ensure_property_depth(data, threat_prop, min_depth=2):
        depth_bumps += 1

    for encounter in data.get("encounters", []):
        for option in encounter.get("options", []) or []:
            for reaction in option.get("reactions", []) or []:
                if not isinstance(reaction, dict):
                    continue
                total_reactions += 1

                target, witness = infer_target_and_witness(
                    reaction=reaction,
                    focal=focal,
                    default_target=default_target,
                    default_witness=default_witness,
                    trust_prop=trust_prop,
                    chars=chars,
                )

                changed, des_count, effect_count = update_reaction(
                    reaction=reaction,
                    focal=focal,
                    target=target,
                    witness=witness,
                    trust_prop=trust_prop,
                    threat_prop=threat_prop,
                )
                if changed:
                    touched_reactions += 1
                added_des_terms += des_count
                added_effects += effect_count

    changed = data != original
    if changed and not dry_run:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "file": str(path),
        "skipped": False,
        "changed": changed,
        "total_reactions": total_reactions,
        "touched_reactions": touched_reactions,
        "added_des_terms": added_des_terms,
        "added_effects": added_effects,
        "depth_bumps": depth_bumps,
        "focal": focal,
        "trust_prop": trust_prop,
        "threat_prop": threat_prop,
    }


def collect_files(patterns: List[str]) -> List[Path]:
    seen = set()
    out: List[Path] = []
    for pattern in patterns:
        for raw in glob.glob(pattern):
            p = Path(raw).resolve()
            if p in seen or not p.is_file():
                continue
            seen.add(p)
            out.append(p)
    return sorted(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enforce p/p2 desirability+effects across diplomacy *_p storyworld reactions")
    parser.add_argument("--glob", action="append", default=[], help="Glob pattern to include (repeatable)")
    parser.add_argument("--dry-run", action="store_true", help="Analyze without writing")
    args = parser.parse_args()

    patterns = args.glob or list(DEFAULT_GLOBS)
    files = collect_files(patterns)
    results = [augment_file(path, dry_run=args.dry_run) for path in files]

    changed = [r for r in results if not r.get("skipped") and r.get("changed")]
    touched = [r for r in results if not r.get("skipped")]
    payload = {
        "files_considered": len(files),
        "files_touched": len(touched),
        "files_changed": len(changed),
        "dry_run": args.dry_run,
        "results": results,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
