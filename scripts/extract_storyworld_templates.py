import json
import re
from pathlib import Path

SRC_DIR = Path(r"C:\projects\GPTStoryworld\storyworlds")
OUT_DIR = Path(r"C:\projects\AI_Diplomacy\ai_diplomacy\storyworld_bank_extracted")
OUT_DIR.mkdir(parents=True, exist_ok=True)

files = [
    SRC_DIR / "england_to_france_honest.json",
    SRC_DIR / "france_germany_machiavellian_extended.json",
    SRC_DIR / "france_to_germany_machiavellian.json",
    SRC_DIR / "russia_to_austria_grudger.json",
]

archetype_map = {
    "honest": "honest_signal",
    "machiavellian": "machiavellian",
    "grudger": "grudger",
}

def get_text(node, fallback=""):
    if isinstance(node, dict):
        if node.get("value"):
            return node.get("value")
        if node.get("text"):
            return node.get("text")
    if isinstance(node, str):
        return node
    return fallback

for path in files:
    if not path.exists():
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    name = path.stem
    archetype = "persuasion"
    for key, val in archetype_map.items():
        if key in name:
            archetype = val
            break

    about = get_text(data.get("about_text", {}), "")
    encounters = data.get("encounters", []) or []
    first_enc = encounters[0] if encounters else {}
    enc_text = get_text(first_enc.get("text_script", {}), "")
    options = first_enc.get("options", []) or []
    first_opt = options[0] if options else {}
    opt_text = get_text(first_opt.get("text_script", {}), "")

    claim = about or enc_text or "A forecast indicates a near-term shift in intent."
    evidence = enc_text or "Recent moves and silence suggest imminent pressure."
    ask = opt_text or "Coordinate a concrete support or DMZ this phase."

    template = {
        "id": name,
        "title": data.get("storyworld_title", name),
        "archetype": archetype,
        "intent": about[:240] if about else "Persuasive alignment using forecast evidence.",
        "forecast": {
            "question": "Will the named rival move aggressively next phase?",
            "likely_outcome": "aggression",
            "probabilities": {"aggression": 0.6, "restraint": 0.4},
        },
        "message_frame": {
            "claim": claim.strip(),
            "evidence": evidence.strip(),
            "ask": ask.strip(),
        },
        "source": str(path),
    }

    out_path = OUT_DIR / f"{name}.json"
    out_path.write_text(json.dumps(template, indent=2, ensure_ascii=True), encoding="utf-8")

print(f"Wrote {len(list(OUT_DIR.glob('*.json')))} templates to {OUT_DIR}")
