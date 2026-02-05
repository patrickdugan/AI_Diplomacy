import os
import json
import re
import logging
import ast  # For literal_eval in JSON fallback parsing
import time
from datetime import datetime
import subprocess
import shutil
import aiohttp  # For direct HTTP requests to Responses API
from pathlib import Path

from typing import List, Dict, Optional, Tuple, NamedTuple
from dotenv import load_dotenv

# Use Async versions of clients
from openai import AsyncOpenAI
from openai import AsyncOpenAI as AsyncDeepSeekOpenAI  # Alias for clarity
from anthropic import AsyncAnthropic
import asyncio
import requests
from enum import StrEnum

try:
    import google.generativeai as genai
except Exception:
    genai = None
try:
    from together import AsyncTogether
    from together.error import APIError as TogetherAPIError  # For specific error handling
except Exception:
    AsyncTogether = None
    TogetherAPIError = Exception

from config import config
from .game_history import GameHistory
from .utils import load_prompt, run_llm_and_log, log_llm_response, log_llm_response_async, generate_random_seed, get_prompt_path

# Import DiplomacyAgent for type hinting if needed, but avoid circular import if possible
from .prompt_constructor import construct_order_generation_prompt, build_context_prompt
# Moved formatter imports to avoid circular import - imported locally where needed
from .storyworld_adapter import generate_storyworld_forecast

# set logger back to just info
logger = logging.getLogger("client")
logger.setLevel(logging.DEBUG)  # Keep debug for now during async changes
# Note: BasicConfig might conflict if already configured in lm_game. Keep client-specific for now.
# logging.basicConfig(level=logging.DEBUG) # Might be redundant if lm_game configures root

load_dotenv()

##############################################################################
# Oracle (Codex) helpers
##############################################################################
_ORACLE_CLIENT_CACHE: Dict[Tuple[str, str], "OpenAIClient"] = {}
_ORACLE_CALL_GUARD: Dict[Tuple[str, str, str], bool] = {}


def _parse_power_list(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [p.strip().upper() for p in raw.split(",") if p.strip()]


def _safe_json_load(raw: str) -> Optional[object]:
    try:
        return json.loads(raw)
    except Exception:
        try:
            return ast.literal_eval(raw)
        except Exception:
            return None


def _extract_json_candidate(raw: str) -> Optional[str]:
    if not raw:
        return None
    fence = "```"
    idx = raw.find(fence)
    while idx != -1:
        lang_end = raw.find("\n", idx + len(fence))
        if lang_end == -1:
            break
        lang = raw[idx + len(fence) : lang_end].strip().lower()
        end = raw.find(fence, lang_end + 1)
        if end == -1:
            break
        content = raw[lang_end + 1 : end].strip()
        if lang in ("json", "jsonl") and content:
            return content
        idx = raw.find(fence, end + len(fence))
    # fallback: try to grab a simple object block
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start : end + 1]
    return None


def _get_oracle_client() -> Optional["OpenAIClient"]:
    model = os.environ.get("CODEX_ORACLE_MODEL")
    if not model:
        return None
    base_url = os.environ.get("CODEX_ORACLE_BASE_URL")
    api_key = os.environ.get("CODEX_ORACLE_API_KEY")
    # Build a model spec with optional inline base/key, so load_model_client can pick the right client
    model_spec = model
    if base_url:
        model_spec = f"{model_spec}@{base_url}"
    if api_key:
        model_spec = f"{model_spec}#{api_key}"

    cache_key = (model_spec, base_url or "")
    client = _ORACLE_CLIENT_CACHE.get(cache_key)
    if client is None:
        client = load_model_client(model_spec, prompts_dir=None)
        max_tokens = os.environ.get("CODEX_ORACLE_MAX_OUTPUT_TOKENS")
        if max_tokens and max_tokens.isdigit():
            client.max_tokens = int(max_tokens)
        _ORACLE_CLIENT_CACHE[cache_key] = client
    return client


def _oracle_call_allowed(power_name: str, phase: str, mode: str) -> bool:
    key = (power_name, phase, mode)
    if _ORACLE_CALL_GUARD.get(key):
        return False
    _ORACLE_CALL_GUARD[key] = True
    return True


def _oracle_log_path(log_file_path: str) -> Optional[Path]:
    if not log_file_path:
        return None
    try:
        return Path(log_file_path).resolve().parent / "codex_oracle_reasoning.jsonl"
    except Exception:
        return None


def _build_oracle_prompt(
    *,
    mode: str,
    power_name: str,
    phase: str,
    base_prompt: str,
    messages_to_power: Optional[List[Dict[str, str]]] = None,
    storyworld_artifact: Optional[Dict[str, object]] = None,
) -> str:
    header = (
        "You are a negotiation reasoning oracle. Output JSON only. "
        "Provide explicit reasoning steps in a structured, machine-readable form. "
        "If STORYWORLD_FORECAST_ARTIFACT is provided, you MUST include a "
        "`storyworld_implications` field and reference it in `reasoning_trace`."
    )
    schema = {
        "situation_assessment": {
            "key_risks": ["..."],
            "key_opportunities": ["..."],
            "most_important_unknowns": ["..."],
        },
        "storyworld_implications": "If storyworld artifact provided, summarize its implications in 1-3 sentences.",
        "recommended_belief_updates": [
            {"power": "AUSTRIA", "threat_delta": 0.1, "trust_delta": -0.2, "because": "..."}
        ],
        "candidate_orders": [
            {"orders": ["F EDI-NTH"], "pros": ["..."], "cons": ["..."], "risk": 0.35}
        ],
        "candidate_messages": [
            {"to": "FRANCE", "intent": "reassure", "text": "...", "expected_effect": "...", "risk": "..."}
        ],
        "reasoning_trace": [
            {"step": 1, "claim": "...", "evidence": "message_id:X / board_fact:Y"}
        ],
        "recommendation": {"pick_orders_index": 0, "send_messages_indices": [0]},
    }
    if mode == "reply":
        payload = {
            "mode": "reply_oracle",
            "power": power_name,
            "phase": phase,
            "messages_to_power": messages_to_power or [],
        }
        if storyworld_artifact:
            payload["storyworld_forecast_artifact"] = storyworld_artifact
        return f"{header}\n\nINPUT:\n{json.dumps(payload, ensure_ascii=True)}\n\nJSON_SCHEMA:\n{json.dumps(schema, ensure_ascii=True)}"
    payload = {"mode": "plan_oracle", "power": power_name, "phase": phase, "context_prompt": base_prompt}
    if storyworld_artifact:
        payload["storyworld_forecast_artifact"] = storyworld_artifact
    return f"{header}\n\nINPUT:\n{json.dumps(payload, ensure_ascii=True)}\n\nJSON_SCHEMA:\n{json.dumps(schema, ensure_ascii=True)}"


async def _run_oracle_call(
    *,
    power_name: str,
    phase: str,
    mode: str,
    oracle_prompt: str,
    log_file_path: str,
) -> Tuple[str, Optional[object]]:
    if os.environ.get("CODEX_ORACLE_CLI", "").strip().lower() in {"1", "true", "yes"}:
        return await _run_oracle_cli_call(
            power_name=power_name,
            phase=phase,
            mode=mode,
            oracle_prompt=oracle_prompt,
            log_file_path=log_file_path,
        )
    oracle_client = _get_oracle_client()
    if oracle_client is None:
        return "", None

    raw_oracle = ""
    parsed: Optional[object] = None
    error: Optional[str] = None
    try:
        raw_oracle = await run_llm_and_log(
            client=oracle_client,
            prompt=oracle_prompt,
            power_name=power_name,
            phase=phase,
            response_type=f"codex_oracle_{mode}",
        )
        parsed = _safe_json_load(raw_oracle)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        logger.warning(
            f"[oracle] Failed for {power_name} in {phase} ({mode}): {error}"
        )

    log_path = _oracle_log_path(log_file_path)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "power": power_name,
            "phase": phase,
            "mode": mode,
            "model": oracle_client.model_name,
            "oracle_input": oracle_prompt,
            "oracle_output_raw": raw_oracle,
            "oracle_output_parsed": parsed if isinstance(parsed, (dict, list)) else None,
            "error": error,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=True) + "\n")

    return raw_oracle, parsed


async def _run_oracle_cli_call(
    *,
    power_name: str,
    phase: str,
    mode: str,
    oracle_prompt: str,
    log_file_path: str,
) -> Tuple[str, Optional[object]]:
    raw_oracle = ""
    parsed: Optional[object] = None
    error: Optional[str] = None
    log_path = _oracle_log_path(log_file_path)
    last_message_path = None

    try:
        cli_path = os.environ.get(
            "CODEX_ORACLE_CLI_PATH",
            r"C:\projects\node_modules\@openai\codex\vendor\x86_64-pc-windows-msvc\codex\codex.exe",
        )
        resolved = shutil.which(cli_path) if not os.path.isabs(cli_path) else cli_path
        if not resolved or not os.path.exists(resolved):
            raise FileNotFoundError(f"Codex CLI not found: {cli_path}")

        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            last_message_path = log_path.parent / f"codex_oracle_last_message_{power_name}_{phase}_{mode}.txt"
        else:
            last_message_path = Path.cwd() / f"codex_oracle_last_message_{power_name}_{phase}_{mode}.txt"

        timeout_seconds = int(os.environ.get("CODEX_ORACLE_CLI_TIMEOUT", "120"))
        sandbox = os.environ.get("CODEX_ORACLE_CLI_SANDBOX", "").strip()

        cmd = [resolved, "exec"]
        if sandbox:
            cmd.extend(["--sandbox", sandbox])
        if last_message_path:
            cmd.extend(["--output-last-message", str(last_message_path)])
        cmd.append(oracle_prompt)

        def _run():
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )

        result = await asyncio.to_thread(_run)
        if result.returncode != 0:
            raise RuntimeError(f"Codex CLI exited {result.returncode}: {result.stderr[:200]}")

        if last_message_path and last_message_path.exists():
            raw_oracle = last_message_path.read_text(encoding="utf-8", errors="ignore")
        else:
            raw_oracle = result.stdout or ""

        candidate = _extract_json_candidate(raw_oracle)
        if candidate:
            parsed = _safe_json_load(candidate)
        else:
            parsed = _safe_json_load(raw_oracle)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        logger.warning(f"[oracle-cli] Failed for {power_name} in {phase} ({mode}): {error}")

    if log_path:
        entry = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "power": power_name,
            "phase": phase,
            "mode": mode,
            "model": "codex-cli",
            "transport": "cli",
            "oracle_input": oracle_prompt,
            "oracle_output_raw": raw_oracle,
            "oracle_output_parsed": parsed if isinstance(parsed, (dict, list)) else None,
            "oracle_output_path": str(last_message_path) if last_message_path else None,
            "error": error,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=True) + "\n")

    return raw_oracle, parsed

##############################################################################
# 1) Base Interface
##############################################################################
class BaseModelClient:
    """
    Base interface for any LLM client we want to plug in.
    Each must provide:
      - generate_response(prompt: str) -> str
      - get_orders(board_state, power_name, possible_orders) -> List[str]
      - get_conversation_reply(power_name, conversation_so_far, game_phase) -> str
    """

    def __init__(self, model_name: str, prompts_dir: Optional[str] = None):
        self.model_name = model_name
        self.prompts_dir = prompts_dir
        logger.info(f"[{model_name}] BaseModelClient initialized with prompts_dir: {prompts_dir}")
        # Load a default initially, can be overwritten by set_system_prompt
        self.system_prompt = load_prompt("system_prompt.txt", prompts_dir=self.prompts_dir)
        self.max_tokens = 16000  # default unless overridden

    def set_system_prompt(self, content: str):
        """Allows updating the system prompt after initialization."""
        self.system_prompt = content
        logger.info(f"[{self.model_name}] System prompt updated.")

    async def generate_response(self, prompt: str, temperature: float = 0.0, inject_random_seed: bool = True) -> str:
        """
        Returns a raw string from the LLM.
        Subclasses override this.
        """
        raise NotImplementedError("Subclasses must implement generate_response().")

    # build_context_prompt and build_prompt (now construct_order_generation_prompt)
    # have been moved to prompt_constructor.py

    async def get_orders(
        self,
        game,
        board_state,
        power_name: str,
        possible_orders: Dict[str, List[str]],
        conversation_text: str,  # This is GameHistory
        model_error_stats: dict,
        log_file_path: str,
        phase: str,
        agent_goals: Optional[List[str]] = None,
        agent_relationships: Optional[Dict[str, str]] = None,
        agent_private_diary_str: Optional[str] = None,  # Added
    ) -> List[str]:
        """
        1) Builds the prompt with conversation context if available
        2) Calls LLM
        3) Parses JSON block
        """
        # The 'conversation_text' parameter was GameHistory. Renaming for clarity.
        game_history_obj = conversation_text

        prompt = construct_order_generation_prompt(
            system_prompt=self.system_prompt,
            game=game,
            board_state=board_state,
            power_name=power_name,
            possible_orders=possible_orders,
            game_history=game_history_obj,  # Pass GameHistory object
            agent_goals=agent_goals,
            agent_relationships=agent_relationships,
            agent_private_diary_str=agent_private_diary_str,
            prompts_dir=self.prompts_dir,
        )

        raw_response = ""
        # Initialize success status. Will be updated based on outcome.
        success_status = "Failure: Initialized"
        parsed_orders_for_return = self.fallback_orders(possible_orders)  # Default to fallback

        try:
            # Call LLM using the logging wrapper
            raw_response = await run_llm_and_log(
                client=self,
                prompt=prompt,
                power_name=power_name,
                phase=phase,
                response_type="order",  # Context for run_llm_and_log's own error logging
                temperature=0,
            )
            logger.debug(f"[{self.model_name}] Raw LLM response for {power_name} orders:\n{raw_response}")

            # Conditionally format the response based on USE_UNFORMATTED_PROMPTS
            if config.USE_UNFORMATTED_PROMPTS:
                # Local import to avoid circular dependency
                from .formatter import format_with_gemini_flash, FORMAT_ORDERS

                # Format the natural language response into structured format
                formatted_response = await format_with_gemini_flash(
                    raw_response, FORMAT_ORDERS, power_name=power_name, phase=phase, log_file_path=log_file_path
                )
            else:
                # Use the raw response directly (already formatted)
                formatted_response = raw_response

            # Attempt to parse the final "orders" from the formatted response
            move_list = self._extract_moves(formatted_response, power_name)
            if isinstance(move_list, list):
                cleaned_moves = []
                for move in move_list:
                    if isinstance(move, str):
                        move = move.strip()
                        if move:
                            cleaned_moves.append(move)
                    else:
                        logger.debug(
                            f"[{self.model_name}] Dropping non-string move for {power_name}: {move}"
                        )
                move_list = cleaned_moves

            if not move_list:
                logger.warning(f"[{self.model_name}] Could not extract moves for {power_name}. Using fallback.")
                if model_error_stats is not None and self.model_name in model_error_stats:
                    model_error_stats[self.model_name].setdefault("order_decoding_errors", 0)
                    model_error_stats[self.model_name]["order_decoding_errors"] += 1
                success_status = "Failure: No moves extracted"
                # Fallback is already set to parsed_orders_for_return
            else:
                # Validate or fallback
                validated_moves, invalid_moves_list = self._validate_orders(move_list, possible_orders)
                logger.debug(f"[{self.model_name}] Validated moves for {power_name}: {validated_moves}")
                parsed_orders_for_return = validated_moves
                if invalid_moves_list:
                    # Truncate if too many invalid moves to keep log readable
                    max_invalid_to_log = 5
                    display_invalid_moves = invalid_moves_list[:max_invalid_to_log]
                    omitted_count = len(invalid_moves_list) - len(display_invalid_moves)

                    invalid_moves_str = ", ".join(display_invalid_moves)
                    if omitted_count > 0:
                        invalid_moves_str += f", ... ({omitted_count} more)"

                    success_status = f"Failure: Invalid LLM Moves ({len(invalid_moves_list)}): {invalid_moves_str}"
                    # If some moves were validated despite others being invalid, it's still not a full 'Success'
                    # because the LLM didn't provide a fully usable set of orders without intervention/fallbacks.
                    # The fallback_orders logic within _validate_orders might fill in missing pieces,
                    # but the key is that the LLM *proposed* invalid moves.
                    if not validated_moves:  # All LLM moves were invalid
                        logger.warning(f"[{power_name}] All LLM-proposed moves were invalid. Using fallbacks. Invalid: {invalid_moves_list}")
                    else:
                        logger.info(f"[{power_name}] Some LLM-proposed moves were invalid. Using fallbacks/validated. Invalid: {invalid_moves_list}")
                else:
                    success_status = "Success"

        except Exception as e:
            logger.error(f"[{self.model_name}] LLM error for {power_name} in get_orders: {e}", exc_info=True)
            success_status = f"Failure: Exception ({type(e).__name__})"
            # Fallback is already set to parsed_orders_for_return
        finally:
            # Log the attempt regardless of outcome
            if log_file_path:  # Only log if a path is provided
                await log_llm_response_async(
                    log_file_path=log_file_path,
                    model_name=self.model_name,
                    power_name=power_name,
                    phase=phase,
                    response_type="order_generation",  # Specific type for CSV logging
                    raw_input_prompt=prompt,  # Renamed from 'prompt' to match log_llm_response arg
                    raw_response=raw_response,
                    success=success_status,
                    # token_usage and cost can be added later if available and if log_llm_response supports them
                )
        return parsed_orders_for_return

    def _extract_moves(self, raw_response: str, power_name: str) -> Optional[List[str]]:
        """
        Attempt multiple parse strategies to find JSON array of moves.

        1. Regex for PARSABLE OUTPUT lines.
        2. If that fails, also look for fenced code blocks with { ... }.
        3. Attempt bracket-based fallback if needed.

        Returns a list of move strings or None if everything fails.
        """
        # 1) Regex for "PARSABLE OUTPUT:{...}"
        pattern = r"PARSABLE OUTPUT:\s*(\{[\s\S]*\})"
        matches = re.search(pattern, raw_response, re.DOTALL)

        if not matches:
            # Some LLMs might not put the colon or might have triple backtick fences.
            logger.debug(f"[{self.model_name}] Regex parse #1 failed for {power_name}. Trying alternative patterns.")

            # 1b) Check for inline JSON after "PARSABLE OUTPUT"
            pattern_alt = r"PARSABLE OUTPUT\s*\{(.*?)\}\s*$"
            matches = re.search(pattern_alt, raw_response, re.DOTALL)

        if not matches:
            # 1c) Check for **PARSABLE OUTPUT:** pattern (with asterisks)
            logger.debug(f"[{self.model_name}] Regex parse #2 failed for {power_name}. Trying asterisk-wrapped pattern.")
            pattern_asterisk = r"\*\*PARSABLE OUTPUT:\*\*\s*(\{[\s\S]*?\})"
            matches = re.search(pattern_asterisk, raw_response, re.DOTALL)

        if not matches:
            logger.debug(f"[{self.model_name}] Regex parse #3 failed for {power_name}. Trying triple-backtick code fences.")

        # 2) If still no match, check for triple-backtick code fences containing JSON
        if not matches:
            code_fence_pattern = r"```json\n(.*?)\n```"
            matches = re.search(code_fence_pattern, raw_response, re.DOTALL)
            if matches:
                logger.debug(f"[{self.model_name}] Found triple-backtick JSON block for {power_name}.")

        # 2b) Also try plain ``` code fences without json marker
        if not matches:
            code_fence_plain = r"```\n(.*?)\n```"
            matches = re.search(code_fence_plain, raw_response, re.DOTALL)
            if matches:
                logger.debug(f"[{self.model_name}] Found plain triple-backtick block for {power_name}.")

        # 2c) Try to find bare JSON object anywhere in the response
        if not matches:
            logger.debug(f"[{self.model_name}] No explicit markers found for {power_name}. Looking for bare JSON.")
            # Look for a JSON object that contains "orders" key
            bare_json_pattern = r'(\{[^{}]*"orders"\s*:\s*\[[^\]]*\][^{}]*\})'
            matches = re.search(bare_json_pattern, raw_response, re.DOTALL)
            if matches:
                logger.debug(f"[{self.model_name}] Found bare JSON object with 'orders' key for {power_name}.")

        # 3) Attempt to parse JSON if we found anything
        json_text = None
        if matches:
            # Add braces back around the captured group if needed
            captured = matches.group(1).strip()
            if captured.startswith(r"{{"):
                json_text = captured[1:-1]
            elif captured.startswith(r"{"):
                json_text = captured
            else:
                json_text = "{%s}" % captured

            json_text = json_text.strip()

        if not json_text:
            logger.debug(f"[{self.model_name}] No JSON text found in LLM response for {power_name}.")
            return None

        # 3a) Try JSON loading
        try:
            data = json.loads(json_text)
            return data.get("orders", None)
        except json.JSONDecodeError as e:
            logger.warning(f"[{self.model_name}] JSON decode failed for {power_name}: {e}. Trying to fix common issues.")

            # Try to fix common JSON issues
            try:
                # Remove trailing commas
                fixed_json = re.sub(r",\s*([\}\]])", r"\1", json_text)
                # Fix single quotes to double quotes
                fixed_json = fixed_json.replace("'", '"')
                # Try parsing again
                data = json.loads(fixed_json)
                logger.info(f"[{self.model_name}] Successfully parsed JSON after fixes for {power_name}")
                return data.get("orders", None)
            except json.JSONDecodeError:
                logger.warning(f"[{self.model_name}] JSON decode still failed after fixes for {power_name}. Trying to remove inline comments.")

                # Try to remove inline comments (// style)
                try:
                    # Remove // comments from each line
                    lines = json_text.split("\n")
                    cleaned_lines = []
                    for line in lines:
                        # Find // that's not inside quotes
                        comment_pos = -1
                        in_quotes = False
                        escape_next = False
                        for i, char in enumerate(line):
                            if escape_next:
                                escape_next = False
                                continue
                            if char == "\\":
                                escape_next = True
                                continue
                            if char == '"' and not escape_next:
                                in_quotes = not in_quotes
                            if not in_quotes and line[i : i + 2] == "//":
                                comment_pos = i
                                break

                        if comment_pos >= 0:
                            # Remove comment but keep any trailing comma
                            cleaned_line = line[:comment_pos].rstrip()
                        else:
                            cleaned_line = line
                        cleaned_lines.append(cleaned_line)

                    comment_free_json = "\n".join(cleaned_lines)
                    # Also remove trailing commas after comment removal
                    comment_free_json = re.sub(r",\s*([\}\]])", r"\1", comment_free_json)

                    data = json.loads(comment_free_json)
                    logger.info(f"[{self.model_name}] Successfully parsed JSON after removing inline comments for {power_name}")
                    return data.get("orders", None)
                except json.JSONDecodeError:
                    logger.warning(f"[{self.model_name}] JSON decode still failed after removing comments for {power_name}. Trying bracket fallback.")

        # 3b) Attempt bracket fallback: we look for the substring after "orders"
        #     E.g. "orders: ['A BUD H']" and parse it. This is risky but can help with minor JSON format errors.
        #     We only do this if we see something like "orders": ...
        bracket_pattern = r'["\']orders["\']\s*:\s*\[([^\]]*)\]'
        bracket_match = re.search(bracket_pattern, json_text, re.DOTALL)
        if bracket_match:
            try:
                raw_list_str = "[" + bracket_match.group(1).strip() + "]"
                moves = ast.literal_eval(raw_list_str)
                if isinstance(moves, list):
                    return moves
            except Exception as e2:
                logger.warning(f"[{self.model_name}] Bracket fallback parse also failed for {power_name}: {e2}")

        # If all attempts failed
        return None

    def _validate_orders(self, moves: List[str], possible_orders: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:  # MODIFIED RETURN TYPE
        """
        Filter out invalid moves, fill missing with HOLD, else fallback.
        Returns a tuple: (validated_moves, invalid_moves_found)
        """
        logger.debug(f"[{self.model_name}] Proposed LLM moves: {moves}")
        validated = []
        invalid_moves_found = []  # ADDED: To collect invalid moves
        used_locs = set()

        if not isinstance(moves, list):
            logger.debug(f"[{self.model_name}] Moves not a list, fallback.")
            # Return fallback and empty list for invalid_moves_found as no specific LLM moves were processed
            return self.fallback_orders(possible_orders), []

        for move_str in moves:
            # Check if it's in possible orders
            if any(move_str in loc_orders for loc_orders in possible_orders.values()):
                validated.append(move_str)
                parts = move_str.split()
                if len(parts) >= 2:
                    used_locs.add(parts[1][:3])
            else:
                logger.debug(f"[{self.model_name}] Invalid move from LLM: {move_str}")
                invalid_moves_found.append(move_str)  # ADDED: Collect invalid move

        # Fill missing with hold
        for loc, orders_list in possible_orders.items():
            if loc not in used_locs and orders_list:
                hold_candidates = [o for o in orders_list if o.endswith("H")]
                validated.append(hold_candidates[0] if hold_candidates else orders_list[0])

        if not validated and not invalid_moves_found:  # Only if LLM provided no valid moves and no invalid moves (e.g. empty list from LLM)
            logger.warning(f"[{self.model_name}] No valid LLM moves provided and no invalid ones to report. Using fallback.")
            return self.fallback_orders(possible_orders), []
        elif not validated and invalid_moves_found:  # All LLM moves were invalid
            logger.warning(
                f"[{self.model_name}] All LLM moves invalid ({len(invalid_moves_found)} found), using fallback. Invalid: {invalid_moves_found}"
            )
            # We return empty list for validated, but the invalid_moves_found list is populated
            return self.fallback_orders(possible_orders), invalid_moves_found

        # If we have some validated moves, return them along with any invalid ones found
        return validated, invalid_moves_found

    def fallback_orders(self, possible_orders: Dict[str, List[str]]) -> List[str]:
        """
        Just picks HOLD if possible, else first option.
        """
        fallback = []
        for loc, orders_list in possible_orders.items():
            if orders_list:
                holds = [o for o in orders_list if o.endswith("H")]
                fallback.append(holds[0] if holds else orders_list[0])
        return fallback

    def build_planning_prompt(
        self,
        game,
        board_state,
        power_name: str,
        possible_orders: Dict[str, List[str]],
        game_history: GameHistory,
        # game_phase: str, # Not used directly by build_context_prompt
        # log_file_path: str, # Not used directly by build_context_prompt
        agent_goals: Optional[List[str]] = None,
        agent_relationships: Optional[Dict[str, str]] = None,
        agent_private_diary_str: Optional[str] = None,  # Added
    ) -> str:
        instructions = load_prompt("planning_instructions.txt", prompts_dir=self.prompts_dir)

        context = self.build_context_prompt(
            game,
            board_state,
            power_name,
            possible_orders,
            game_history,
            agent_goals=agent_goals,
            agent_relationships=agent_relationships,
            agent_private_diary=agent_private_diary_str,  # Pass diary string
            prompts_dir=self.prompts_dir,
        )

        return context + "\n\n" + instructions

    def build_conversation_prompt(
        self,
        game,
        board_state,
        power_name: str,
        possible_orders: Dict[str, List[str]],
        game_history: GameHistory,
        # game_phase: str, # Not used directly by build_context_prompt
        # log_file_path: str, # Not used directly by build_context_prompt
        agent_goals: Optional[List[str]] = None,
        agent_relationships: Optional[Dict[str, str]] = None,
        agent_private_diary_str: Optional[str] = None,  # Added
    ) -> str:
        # MINIMAL CHANGE: Just change to load unformatted version conditionally
        # Check if country-specific prompts are enabled
        if config.COUNTRY_SPECIFIC_PROMPTS:
            # Try to load country-specific version first, but fall back safely
            country_specific_name = f"conversation_instructions_{power_name.lower()}.txt"
            country_specific_path = (
                Path(self.prompts_dir) / get_prompt_path(country_specific_name)
                if self.prompts_dir is not None
                else Path(__file__).resolve().parent / "prompts" / get_prompt_path(country_specific_name)
            )
            if country_specific_path.exists():
                instructions = load_prompt(get_prompt_path(country_specific_name), prompts_dir=self.prompts_dir)
            else:
                instructions = load_prompt(get_prompt_path("conversation_instructions.txt"), prompts_dir=self.prompts_dir)
        else:
            # Load generic conversation instructions
            instructions = load_prompt(get_prompt_path("conversation_instructions.txt"), prompts_dir=self.prompts_dir)

        # KEEP ORIGINAL: Use build_context_prompt as before
        context = build_context_prompt(
            game,
            board_state,
            power_name,
            possible_orders,
            game_history,
            agent_goals=agent_goals,
            agent_relationships=agent_relationships,
            agent_private_diary=agent_private_diary_str,  # Pass diary string
            prompts_dir=self.prompts_dir,
        )

        # KEEP ORIGINAL: Get recent messages targeting this power to prioritize responses
        recent_messages_to_power = game_history.get_recent_messages_to_power(power_name, limit=3)

        # KEEP ORIGINAL: Debug logging to verify messages
        logger.info(f"[{power_name}] Found {len(recent_messages_to_power)} high priority messages to respond to")
        if recent_messages_to_power:
            for i, msg in enumerate(recent_messages_to_power):
                logger.info(f"[{power_name}] Priority message {i + 1}: From {msg['sender']} in {msg['phase']}: {msg['content'][:50]}...")

        # KEEP ORIGINAL: Add a section for unanswered messages
        unanswered_messages = "\n\nRECENT MESSAGES REQUIRING YOUR ATTENTION:\n"
        if recent_messages_to_power:
            for msg in recent_messages_to_power:
                unanswered_messages += f"\nFrom {msg['sender']} in {msg['phase']}: {msg['content']}\n"
        else:
            unanswered_messages += "\nNo urgent messages requiring direct responses.\n"

        final_prompt = context + unanswered_messages + "\n\n" + instructions
        final_prompt = (
            final_prompt.replace("AUSTRIA", "Austria")
            .replace("ENGLAND", "England")
            .replace("FRANCE", "France")
            .replace("GERMANY", "Germany")
            .replace("ITALY", "Italy")
            .replace("RUSSIA", "Russia")
            .replace("TURKEY", "Turkey")
        )
        return final_prompt

    async def get_planning_reply(  # Renamed from get_plan to avoid conflict with get_plan in agent.py
        self,
        game,
        board_state,
        power_name: str,
        possible_orders: Dict[str, List[str]],
        game_history: GameHistory,
        game_phase: str,  # Used for logging
        log_file_path: str,  # Used for logging
        agent_goals: Optional[List[str]] = None,
        agent_relationships: Optional[Dict[str, str]] = None,
        agent_private_diary_str: Optional[str] = None,  # Added
    ) -> str:
        prompt = self.build_planning_prompt(
            game,
            board_state,
            power_name,
            possible_orders,
            game_history,
            # game_phase, # Not passed to build_planning_prompt directly
            # log_file_path, # Not passed to build_planning_prompt directly
            agent_goals=agent_goals,
            agent_relationships=agent_relationships,
            agent_private_diary_str=agent_private_diary_str,  # Pass diary string
        )

        # Call LLM using the logging wrapper
        raw_response = await run_llm_and_log(
            client=self,
            prompt=prompt,
            power_name=power_name,
            phase=game_phase,  # Use game_phase for logging
            response_type="plan_reply",  # Changed from 'plan' to avoid confusion
        )
        logger.debug(f"[{self.model_name}] Raw LLM response for {power_name} planning reply:\n{raw_response}")
        return raw_response

    async def get_conversation_reply(
        self,
        game,
        board_state,
        power_name: str,
        possible_orders: Dict[str, List[str]],
        game_history: GameHistory,
        game_phase: str,
        log_file_path: str,
        active_powers: Optional[List[str]] = None,
        agent_goals: Optional[List[str]] = None,
        agent_relationships: Optional[Dict[str, str]] = None,
        agent_private_diary_str: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Generates a negotiation message, considering agent state.
        """
        raw_input_prompt = ""  # Initialize for finally block
        raw_response = ""  # Initialize for finally block
        success_status = "Failure: Initialized"  # Default status
        messages_to_return = []  # Initialize to ensure it's defined

        try:
            raw_input_prompt = self.build_conversation_prompt(
                game,
                board_state,
                power_name,
                possible_orders,
                game_history,
                agent_goals=agent_goals,
                agent_relationships=agent_relationships,
                agent_private_diary_str=agent_private_diary_str,
            )

            logger.debug(f"[{self.model_name}] Conversation prompt for {power_name}:\n{raw_input_prompt}")

            # Optional Codex oracle consultation
            oracle_plan_powers = set(_parse_power_list(os.environ.get("CODEX_ORACLE_PLAN_POWERS")))
            oracle_reply_powers = set(_parse_power_list(os.environ.get("CODEX_ORACLE_REPLY_POWERS")))
            power_key = power_name.upper()
            oracle_mode = None

            if power_key in oracle_plan_powers:
                oracle_mode = "plan"
            elif power_key in oracle_reply_powers:
                recent_msgs = game_history.get_recent_messages_to_power(power_name, limit=3)
                recent_msgs = [m for m in recent_msgs if m.get("phase") == game_phase]
                if recent_msgs:
                    oracle_mode = "reply"

            if oracle_mode and _oracle_call_allowed(power_key, game_phase, oracle_mode):
                oracle_prompt = _build_oracle_prompt(
                    mode=oracle_mode,
                    power_name=power_name,
                    phase=game_phase,
                    base_prompt=raw_input_prompt,
                    messages_to_power=recent_msgs if oracle_mode == "reply" else None,
                )
                oracle_raw, oracle_parsed = await _run_oracle_call(
                    power_name=power_name,
                    phase=game_phase,
                    mode=oracle_mode,
                    oracle_prompt=oracle_prompt,
                    log_file_path=log_file_path,
                )

                oracle_insert = None
                if isinstance(oracle_parsed, dict):
                    oracle_insert = json.dumps(
                        {
                            "situation_assessment": oracle_parsed.get("situation_assessment"),
                            "recommended_belief_updates": oracle_parsed.get("recommended_belief_updates"),
                            "candidate_messages": oracle_parsed.get("candidate_messages"),
                            "recommendation": oracle_parsed.get("recommendation"),
                            "reasoning_trace": oracle_parsed.get("reasoning_trace"),
                        },
                        ensure_ascii=True,
                    )
                if oracle_insert:
                    raw_input_prompt = f"{raw_input_prompt}\n\n# ORACLE CONSULTATION\n{oracle_insert}"

                log_path = _oracle_log_path(log_file_path)
                if log_path:
                    entry = {
                        "ts": datetime.utcnow().isoformat() + "Z",
                        "power": power_name,
                        "phase": game_phase,
                        "mode": oracle_mode,
                        "model": _get_oracle_client().model_name if _get_oracle_client() else None,
                        "oracle_used_in_prompt": bool(oracle_insert),
                    }
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(entry, ensure_ascii=True) + "\n")

            raw_response = await run_llm_and_log(
                client=self,
                prompt=raw_input_prompt,
                power_name=power_name,
                phase=game_phase,
                response_type="negotiation",  # For run_llm_and_log's internal context
            )
            logger.debug(f"[{self.model_name}] Raw LLM response for {power_name}:\n{raw_response}")

            # Conditionally format the response based on USE_UNFORMATTED_PROMPTS
            if config.USE_UNFORMATTED_PROMPTS:
                # Local import to avoid circular dependency
                from .formatter import format_with_gemini_flash, FORMAT_CONVERSATION

                # Format the natural language response into structured JSON
                formatted_response = await format_with_gemini_flash(
                    raw_response, FORMAT_CONVERSATION, power_name=power_name, phase=game_phase, log_file_path=log_file_path
                )
            else:
                # Use the raw response directly (already formatted)
                formatted_response = raw_response

            parsed_messages = []
            json_blocks = []
            json_decode_error_occurred = False

            # For formatted response, we expect a clean JSON array
            try:
                data = json.loads(formatted_response)
                if isinstance(data, list):
                    parsed_messages = data
                    json_blocks = [json.dumps(item) for item in data if isinstance(item, dict)]
                else:
                    logger.warning(f"[{self.model_name}] Formatted response is not a list")
            except json.JSONDecodeError:
                logger.warning(f"[{self.model_name}] Failed to parse formatted response as JSON, falling back to regex")
                # Fall back to original parsing logic using formatted_response
                raw_response = formatted_response

            # Original parsing logic as fallback
            if not parsed_messages:
                # Attempt to find blocks enclosed in {{...}}
                double_brace_blocks = re.findall(r"\{\{(.*?)\}\}", raw_response, re.DOTALL)
                if double_brace_blocks:
                    # If {{...}} blocks are found, assume each is a self-contained JSON object
                    json_blocks.extend(["{" + block.strip() + "}" for block in double_brace_blocks])
                else:
                    # If no {{...}} blocks, look for ```json ... ``` markdown blocks
                    code_block_match = re.search(r"```json\n(.*?)\n```", raw_response, re.DOTALL)
                    if code_block_match:
                        potential_json_array_or_objects = code_block_match.group(1).strip()
                        # Try to parse as a list of objects or a single object
                        try:
                            data = json.loads(potential_json_array_or_objects)
                            if isinstance(data, list):
                                json_blocks = [json.dumps(item) for item in data if isinstance(item, dict)]
                            elif isinstance(data, dict):
                                json_blocks = [json.dumps(data)]
                        except json.JSONDecodeError:
                            # If parsing the whole block fails, fall back to regex for individual objects
                            json_blocks = re.findall(r"\{.*?\}", potential_json_array_or_objects, re.DOTALL)
                    else:
                        # If no markdown block, fall back to regex for any JSON object in the response
                        json_blocks = re.findall(r"\{.*?\}", raw_response, re.DOTALL)

            # Process json_blocks if we have them from fallback parsing
            if not parsed_messages and json_blocks:
                for block_index, block in enumerate(json_blocks):
                    try:
                        cleaned_block = block.strip()
                        # Attempt to fix common JSON issues like trailing commas before parsing
                        cleaned_block = re.sub(r",\s*([\}\]])", r"\1", cleaned_block)
                        parsed_message = json.loads(cleaned_block)
                        parsed_messages.append(parsed_message)
                    except json.JSONDecodeError as e:
                        logger.warning(f"[{self.model_name}] Failed to parse JSON block {block_index} for {power_name}: {e}")
                        json_decode_error_occurred = True

            if not parsed_messages:
                logger.warning(f"[{self.model_name}] No valid messages found in response for {power_name}")
                success_status = "Success: No messages found"
                # messages_to_return remains empty
            else:
                # Validate parsed messages
                validated_messages = []
                for msg in parsed_messages:
                    if isinstance(msg, dict) and "message_type" in msg and "content" in msg:
                        if msg["message_type"] == "private" and "recipient" not in msg:
                            logger.warning(f"[{self.model_name}] Private message missing recipient for {power_name}")
                            continue
                        validated_messages.append(msg)
                    else:
                        logger.warning(f"[{self.model_name}] Invalid message structure for {power_name}")
                parsed_messages = validated_messages

            # Set final status and return value
            if parsed_messages:
                success_status = "Success: Messages extracted"
                messages_to_return = parsed_messages
            else:
                success_status = "Success: No valid messages"
                messages_to_return = []

            logger.debug(f"[{self.model_name}] Validated conversation replies for {power_name}: {messages_to_return}")
            # return messages_to_return # Return will happen in finally block or after

        except Exception as e:
            logger.error(f"[{self.model_name}] Error in get_conversation_reply for {power_name}: {e}", exc_info=True)
            success_status = f"Failure: Exception ({type(e).__name__})"
            messages_to_return = []  # Ensure empty list on general exception
        finally:
            if log_file_path:
                await log_llm_response_async(
                    log_file_path=log_file_path,
                    model_name=self.model_name,
                    power_name=power_name,
                    phase=game_phase,
                    response_type="negotiation_message",
                    raw_input_prompt=raw_input_prompt,
                    raw_response=raw_response,
                    success=success_status,
                )
            return messages_to_return

    async def get_plan(  # This is the original get_plan, now distinct from get_planning_reply
        self,
        game,
        board_state,
        power_name: str,
        # possible_orders: Dict[str, List[str]], # Not typically needed for high-level plan
        game_history: GameHistory,
        log_file_path: str,
        agent_goals: Optional[List[str]] = None,
        agent_relationships: Optional[Dict[str, str]] = None,
        agent_private_diary_str: Optional[str] = None,  # Added
    ) -> str:
        """
        Generates a strategic plan for the given power based on the current state.
        This method is called by the agent's generate_plan method.
        """
        logger.info(f"Client generating strategic plan for {power_name}...")

        planning_instructions = load_prompt("planning_instructions.txt", prompts_dir=self.prompts_dir)
        if not planning_instructions:
            logger.error("Could not load planning_instructions.txt! Cannot generate plan.")
            return "Error: Planning instructions not found."

        # For planning, possible_orders might be less critical for the context,
        # but build_context_prompt expects it. We can pass an empty dict or calculate it.
        # For simplicity, let's pass empty if not strictly needed by context for planning.
        possible_orders_for_context = {}  # game.get_all_possible_orders() if needed by context

        context_prompt = self.build_context_prompt(
            game,
            board_state,
            power_name,
            possible_orders_for_context,
            game_history,
            agent_goals=agent_goals,
            agent_relationships=agent_relationships,
            agent_private_diary=agent_private_diary_str,  # Pass diary string
            prompts_dir=self.prompts_dir,
        )

        full_prompt = f"{context_prompt}\n\n{planning_instructions}"
        if self.system_prompt:
            full_prompt = f"{self.system_prompt}\n\n{full_prompt}"

        raw_plan_response = ""
        success_status = "Failure: Initialized"
        plan_to_return = f"Error: Plan generation failed for {power_name} (initial state)"

        try:
            # Use run_llm_and_log for the actual LLM call
            raw_plan_response = await run_llm_and_log(
                client=self,  # Pass self (the client instance)
                prompt=full_prompt,
                power_name=power_name,
                phase=game.current_short_phase,
                response_type="plan_generation",  # More specific type for run_llm_and_log context
            )
            logger.debug(f"[{self.model_name}] Raw LLM response for {power_name} plan generation:\n{raw_plan_response}")
            # No parsing needed for the plan, return the raw string
            plan_to_return = raw_plan_response.strip()
            success_status = "Success"
        except Exception as e:
            logger.error(f"Failed to generate plan for {power_name}: {e}", exc_info=True)
            success_status = f"Failure: Exception ({type(e).__name__})"
            plan_to_return = f"Error: Failed to generate plan for {power_name} due to exception: {e}"
        finally:
            if log_file_path:  # Only log if a path is provided
                await log_llm_response_async(
                    log_file_path=log_file_path,
                    model_name=self.model_name,
                    power_name=power_name,
                    phase=game.current_short_phase if game else "UnknownPhase",
                    response_type="plan_generation",  # Specific type for CSV logging
                    raw_input_prompt=full_prompt,  # Renamed from 'full_prompt' to match log_llm_response arg
                    raw_response=raw_plan_response,
                    success=success_status,
                    # token_usage and cost can be added later
                )
        return plan_to_return


##############################################################################
# 2) Concrete Implementations
##############################################################################

class OpenAIClient(BaseModelClient):
    """Async client for OpenAI-compatible chat-completion endpoints."""

    def __init__(
        self,
        model_name: str,
        prompts_dir: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        super().__init__(model_name, prompts_dir=prompts_dir)

        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY missing and no inline key provided")

        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def generate_response(
        self,
        prompt: str,
        temperature: float = 0.0,
        inject_random_seed: bool = True,
    ) -> str:
        try:
            system_prompt_content = f"{generate_random_seed()}\n\n{self.system_prompt}" if inject_random_seed else self.system_prompt
            prompt_with_cta = f"{prompt}\n\nPROVIDE YOUR RESPONSE BELOW:"

            # Determine which parameter to use based on model
            completion_params = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt_content},
                    {"role": "user", "content": prompt_with_cta},
                ],
            }
            
            # Handle model-specific parameters.
            # Many newer OpenAI models require max_completion_tokens (not max_tokens).
            model_id = self.model_name.lower()
            uses_max_completion_tokens = (
                model_id.startswith("gpt-5")
                or model_id.startswith("gpt-4.1")
                or model_id.startswith("o3")
                or model_id.startswith("o4")
                or model_id.startswith("nectarine")
            )
            # Some models only allow the default temperature (1.0).
            temperature_default_only = model_id.startswith("gpt-5") or model_id in ["o4-mini", "o3-mini", "o3"]
            
            if uses_max_completion_tokens:
                completion_params["max_completion_tokens"] = self.max_tokens
            else:
                completion_params["max_tokens"] = self.max_tokens
            if temperature_default_only:
                completion_params["temperature"] = 1.0
            else:
                completion_params["temperature"] = temperature
            
            response = await self.client.chat.completions.create(**completion_params)

            if (
                not response
                or not response.choices
                or not response.choices[0].message
                or not response.choices[0].message.content
            ):
                raise ValueError(f"[{self.model_name}] LLM returned an empty or invalid response.")

            return response.choices[0].message.content.strip()

        except json.JSONDecodeError as json_err:
            logger.error(f"[{self.model_name}] JSON decode error: {json_err}")
            raise
        except Exception as e:
            extra = ""
            try:
                from openai import OpenAIError  # runtime import avoids circulars
                if isinstance(e, OpenAIError):
                    status = getattr(e, "status_code", None)
                    resp  = getattr(e, "response", None)
                    if status:
                        extra += f" (status {status})"
                    if resp is not None:
                        try:
                            body = resp.json() if hasattr(resp, "json") else resp
                        except Exception:
                            body = str(resp)
                        body_str = (
                            json.dumps(body) if isinstance(body, (dict, list)) else str(body)
                        )
                        if len(body_str) > 3_000:
                            body_str = body_str[:3_000] + "…[truncated]"
                        extra += f" – body: {body_str}"
            except Exception:
                # best‑effort only; never mask original error
                pass

            logger.error(f"[{self.model_name}] OpenAI client error: {e}{extra}", exc_info=True)
            raise


class ClaudeClient(BaseModelClient):
    """
    For 'claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022', etc.
    """

    def __init__(self, model_name: str, prompts_dir: Optional[str] = None):
        super().__init__(model_name, prompts_dir=prompts_dir)
        self.client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    async def generate_response(self, prompt: str, temperature: float = 0.0, inject_random_seed: bool = True) -> str:
        # Updated Claude messages format
        try:
            system_prompt_content = self.system_prompt
            if inject_random_seed:
                random_seed = generate_random_seed()
                system_prompt_content = f"{random_seed}\n\n{self.system_prompt}"

            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                system=system_prompt_content,  # system is now a top-level parameter
                messages=[{"role": "user", "content": prompt + "\n\nPROVIDE YOUR RESPONSE BELOW:"}],
                temperature=temperature,
            )
            if not response.content or not response.content[0].text:
                raise ValueError(f"[{self.model_name}] LLM returned an empty or invalid response.")
            return response.content[0].text.strip()
        except json.JSONDecodeError as json_err:
            logger.error(f"[{self.model_name}] JSON decoding failed in generate_response: {json_err}")
            raise
        except Exception as e:
            extra = ""
            try:
                import anthropic
                if isinstance(e, anthropic.errors.APIStatusError):
                    extra += f" (status {e.status_code})"
                    body = getattr(e, "response_json", None)
                    if body:
                        body_str = json.dumps(body)
                        if len(body_str) > 3_000:
                            body_str = body_str[:3_000] + "…[truncated]"
                        extra += f" – body: {body_str}"
            except Exception:
                pass

            logger.error(f"[{self.model_name}] Claude client error: {e}{extra}", exc_info=True)
            raise


class GeminiClient(BaseModelClient):
    """
    For 'gemini-1.5-flash' or other Google Generative AI models.
    """

    def __init__(self, model_name: str, prompts_dir: Optional[str] = None):
        super().__init__(model_name, prompts_dir=prompts_dir)
        # Configure and get the model (corrected initialization)
        if genai is None:
            raise ImportError("google-generativeai is not installed. Install it or avoid Gemini models.")
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model_name)
        logger.debug(f"[{self.model_name}] Initialized Gemini client (genai.GenerativeModel)")

    async def generate_response(self, prompt: str, temperature: float = 0.0, inject_random_seed: bool = True) -> str:
        system_prompt_content = self.system_prompt
        if inject_random_seed:
            random_seed = generate_random_seed()
            system_prompt_content = f"{random_seed}\n\n{self.system_prompt}"

        full_prompt = system_prompt_content + prompt + "\n\nPROVIDE YOUR RESPONSE BELOW:"

        try:
            generation_config = genai.types.GenerationConfig(temperature=temperature, max_output_tokens=self.max_tokens)
            response = await self.client.generate_content_async(
                contents=full_prompt,
                generation_config=generation_config,
            )

            if not response or not response.text:
                raise ValueError(f"[{self.model_name}] LLM returned an empty or invalid response.")
            return response.text.strip()
        except Exception as e:
            # Gemini’s sdk wraps grpc errors; include full message
            msg = str(e)
            if len(msg) > 3_000:
                msg = msg[:3_000] + "…[truncated]"
            logger.error(f"[{self.model_name}] Gemini client error: {msg}", exc_info=True)
            raise


class DeepSeekClient(BaseModelClient):
    """
    For DeepSeek R1 'deepseek-reasoner'
    """

    def __init__(self, model_name: str, prompts_dir: Optional[str] = None):
        super().__init__(model_name, prompts_dir=prompts_dir)
        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        self.client = AsyncDeepSeekOpenAI(api_key=self.api_key, base_url="https://api.deepseek.com/")

    async def generate_response(self, prompt: str, temperature: float = 0.0, inject_random_seed: bool = True) -> str:
        try:
            # Append the call to action to the user's prompt
            prompt_with_cta = prompt + "\n\nPROVIDE YOUR RESPONSE BELOW:"

            system_prompt_content = self.system_prompt
            if inject_random_seed:
                random_seed = generate_random_seed()
                system_prompt_content = f"{random_seed}\n\n{self.system_prompt}"

            # Determine which parameter to use based on model
            completion_params = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt_content},
                    {"role": "user", "content": prompt_with_cta},
                ],
                "stream": False,
                "temperature": temperature,
            }
            
            # Use max_completion_tokens for o4-mini, o3-mini models and nectarine models
            if self.model_name in ["o4-mini", "o3-mini"] or self.model_name.startswith("nectarine"):
                completion_params["max_completion_tokens"] = self.max_tokens
            else:
                completion_params["max_tokens"] = self.max_tokens
            
            response = await self.client.chat.completions.create(**completion_params)

            logger.debug(f"[{self.model_name}] Raw DeepSeek response:\n{response}")

            if not response or not response.choices or not response.choices[0].message.content:
                raise ValueError(f"[{self.model_name}] LLM returned an empty or invalid response.")

            content = response.choices[0].message.content.strip()
            return content

        except Exception as e:
            extra = ""
            try:
                from openai import OpenAIError
                if isinstance(e, OpenAIError):
                    status = getattr(e, "status_code", None)
                    if status:
                        extra += f" (status {status})"
                    resp = getattr(e, "response", None)
                    if resp is not None:
                        try:
                            body = resp.json() if hasattr(resp, "json") else resp
                        except Exception:
                            body = str(resp)
                        body_str = (
                            json.dumps(body) if isinstance(body, (dict, list)) else str(body)
                        )
                        if len(body_str) > 3_000:
                            body_str = body_str[:3_000] + "…[truncated]"
                        extra += f" – body: {body_str}"
            except Exception:
                pass

            logger.error(f"[{self.model_name}] DeepSeek client error: {e}{extra}", exc_info=True)
            raise


class OpenAIResponsesClient(BaseModelClient):
    """
    For OpenAI o3-pro model using the new Responses API endpoint.
    This client makes direct HTTP requests to the v1/responses endpoint.
    """

    def __init__(self, model_name: str, prompts_dir: Optional[str] = None, api_key: Optional[str] = None, reasoning_effort: Optional[str] = None):
        super().__init__(model_name, prompts_dir=prompts_dir)
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        self.base_url = "https://api.openai.com/v1/responses"
        self._session = None  # Lazy initialization for connection pooling
        self.reasoning_effort = reasoning_effort  # For models that support reasoning effort
        logger.info(f"[{self.model_name}] Initialized OpenAI Responses API client with reasoning_effort={reasoning_effort}")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session for connection pooling."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def generate_response(self, prompt: str, temperature: float = 0.0, inject_random_seed: bool = True) -> str:
        response_data = None
        try:
            # The Responses API uses a different format than chat completions
            # Combine system prompt and user prompt into a single input
            system_prompt_content = self.system_prompt
            if inject_random_seed:
                random_seed = generate_random_seed()
                system_prompt_content = f"{random_seed}\n\n{self.system_prompt}"

            full_prompt = f"{system_prompt_content}\n\n{prompt}\n\nPROVIDE YOUR RESPONSE BELOW:"

            # Prepare the request payload
            payload = {
                "model": self.model_name,
                "input": full_prompt,
            }

            # The Responses API uses max_output_tokens for all models
            payload["max_output_tokens"] = self.max_tokens
            
            # Only add temperature for models that support it
            models_without_temp = [
                "o3",
                "o4-mini",
                "gpt-5-reasoning-alpha-2025-07-19",
                "nectarine-alpha-2025-07-25",
                "nectarine-alpha-new-reasoning-effort-2025-07-25",
            ]
            if not (self.model_name.startswith("gpt-5") or self.model_name in models_without_temp):
                payload["temperature"] = temperature

            # Add reasoning effort for models that support it
            reasoning_models = [
                'gpt-5-reasoning-alpha-2025-07-19',
                'o4-mini',
                'nectarine-alpha-2025-07-25',
                'o4-mini-alpha-2025-07-11',
                'nectarine-alpha-new-reasoning-effort-2025-07-25',
            ]
            if self.model_name.startswith("gpt-5"):
                payload["reasoning"] = {"effort": self.reasoning_effort or "minimal"}
            elif self.reasoning_effort and self.model_name in reasoning_models:
                payload["reasoning"] = {"effort": self.reasoning_effort}

            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

            # Make the API call. Close the session each time to avoid unclosed-session warnings.
            session = await self._get_session()
            try:
                async with session.post(self.base_url, json=payload, headers=headers) as response:
                    response.raise_for_status()  # Will raise for non-2xx responses
                    response_data = await response.json()
            finally:
                await session.close()
                self._session = None

            # Extract the text from the nested response structure
            try:
                if response_data is None:
                    raise ValueError(f"[{self.model_name}] No response data received.")

                if "output_text" in response_data and isinstance(response_data["output_text"], str):
                    return response_data["output_text"].strip()

                outputs = response_data.get("output", [])
                if not outputs:
                    logger.error(f"[{self.model_name}] Response structure: {json.dumps(response_data, indent=2)}")
                    raise ValueError(f"[{self.model_name}] Unexpected output structure: empty 'output' list.")

                # Some models return output_text directly in output items.
                for output_item in outputs:
                    if isinstance(output_item, dict):
                        if output_item.get("type") in ("output_text", "text") and isinstance(output_item.get("text"), str):
                            return output_item["text"].strip()
                        if isinstance(output_item.get("content"), str):
                            return output_item["content"].strip()

                message_output = next((o for o in outputs if o.get("type") == "message"), None)
                if not message_output:
                    logger.error(f"[{self.model_name}] Response structure: {json.dumps(response_data, indent=2)}")
                    raise ValueError(f"[{self.model_name}] No 'message' output item found.")

                content_list = message_output.get("content", [])
                if not content_list:
                    raise ValueError(f"[{self.model_name}] Empty 'content' list in message output.")

                text_content = ""
                for content_item in content_list:
                    if content_item.get("type") in ("output_text", "text"):
                        text_content = content_item.get("text", "")
                        if text_content:
                            break

                if not text_content:
                    raise ValueError(f"[{self.model_name}] No text content found in message output.")

                return text_content.strip()

            except (KeyError, IndexError, TypeError) as e:
                # Wrap parsing error in a more informative exception
                raise ValueError(f"[{self.model_name}] Error parsing response structure: {e}") from e

        except aiohttp.ClientError as e:
            logger.error(f"[{self.model_name}] HTTP client error in generate_response: {e}")
            raise
        except Exception as e:
            logger.error(f"[{self.model_name}] Unexpected error in generate_response: {e}")
            raise


class OpenRouterClient(BaseModelClient):
    """
    For OpenRouter models, with default being 'openrouter/quasar-alpha'
    """

    def __init__(self, model_name: str = "openrouter/quasar-alpha", prompts_dir: Optional[str] = None):
        # Allow specifying just the model identifier or the full path
        if not model_name.startswith("openrouter/") and "/" not in model_name:
            model_name = f"openrouter/{model_name}"
        if model_name.startswith("openrouter-"):
            model_name = model_name.replace("openrouter-", "")

        super().__init__(model_name, prompts_dir=prompts_dir)
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")

        self.client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.api_key)

        logger.debug(f"[{self.model_name}] Initialized OpenRouter client")

    async def generate_response(self, prompt: str, temperature: float = 0.0, inject_random_seed: bool = True) -> str:
        """Generate a response using OpenRouter with robust error handling."""
        try:
            # Append the call to action to the user's prompt
            prompt_with_cta = prompt + "\n\nPROVIDE YOUR RESPONSE BELOW:"

            system_prompt_content = self.system_prompt
            if inject_random_seed:
                random_seed = generate_random_seed()
                system_prompt_content = f"{random_seed}\n\n{self.system_prompt}"

            # Prepare standard OpenAI-compatible request
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": system_prompt_content}, {"role": "user", "content": prompt_with_cta}],
                max_tokens=self.max_tokens,
                temperature=temperature,
            )

            if not response.choices or not response.choices[0].message.content:
                raise ValueError(f"[{self.model_name}] LLM returned an empty or invalid response.")

            content = response.choices[0].message.content.strip()
            return content

        except Exception as e:
            extra = ""
            try:
                from openai import OpenAIError
                if isinstance(e, OpenAIError):
                    status = getattr(e, "status_code", None)
                    if status:
                        extra += f" (status {status})"
                    resp = getattr(e, "response", None)
                    if resp is not None:
                        try:
                            body = resp.json() if hasattr(resp, "json") else resp
                        except Exception:
                            body = str(resp)
                        body_str = (
                            json.dumps(body) if isinstance(body, (dict, list)) else str(body)
                        )
                        if len(body_str) > 3_000:
                            body_str = body_str[:3_000] + "…[truncated]"
                        extra += f" – body: {body_str}"
            except Exception:
                pass

            logger.error(f"[{self.model_name}] OpenRouter client error: {e}{extra}", exc_info=True)
            raise


##############################################################################
# TogetherAI Client
##############################################################################
class TogetherAIClient(BaseModelClient):
    """
    Client for Together AI models.
    Model names should be passed without the 'together-' prefix.
    """

    def __init__(self, model_name: str, prompts_dir: Optional[str] = None):
        super().__init__(model_name, prompts_dir=prompts_dir)  # model_name here is the actual Together AI model identifier
        if AsyncTogether is None:
            raise ImportError("together is not installed. Install it or avoid TogetherAI models.")
        self.api_key = os.environ.get("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY environment variable is required for TogetherAIClient")

        # The model_name passed to super() is used for logging and identification.
        # The actual model name for the API call is self.model_name (from super class).
        self.client = AsyncTogether(api_key=self.api_key)
        logger.info(f"[{self.model_name}] Initialized TogetherAI client for model: {self.model_name}")

    async def generate_response(self, prompt: str) -> str:
        """
        Generates a response from the Together AI model.
        """
        logger.debug(f"[{self.model_name}] Generating response with prompt (first 100 chars): {prompt[:100]}...")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            # Ensure the model name used here is the one intended for the API,
            # which is self.model_name as set by BaseModelClient.__init__
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                # Consider adding max_tokens, temperature, etc. as needed
                # max_tokens=2048, # Example
            )

            if not response.choices or not response.choices[0].message or response.choices[0].message.content is None:
                raise ValueError(f"[{self.model_name}] LLM returned an empty or invalid response.")

            content = response.choices[0].message.content
            return content.strip()
        except TogetherAPIError as e:
            body = getattr(e, "body", None) or str(e)
            if len(body) > 3_000:
                body = body[:3_000] + "…[truncated]"
            logger.error(f"[{self.model_name}] TogetherAI API error: {body}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"[{self.model_name}] Unexpected error in TogetherAIClient: {e}", exc_info=True)
            raise


##############################################################################
# RequestsOpenAIClient – sync requests, wrapped async (original + api_key)
##############################################################################


class RequestsOpenAIClient(BaseModelClient):
    """
    Synchronous `requests`-based client for any OpenAI-compatible API.
    Wrapped in `asyncio.to_thread` so call-sites remain async.
    """

    def __init__(
        self,
        model_name: str,
        prompts_dir: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        super().__init__(model_name, prompts_dir=prompts_dir)

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY missing and no inline key provided")

        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")

        self.endpoint = f"{self.base_url}/chat/completions"

    # ---------------- internal blocking helper ---------------- #
    def _post_sync(self, payload: dict) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        r = requests.post(self.endpoint, headers=headers, json=payload, timeout=600)

        if r.status_code >= 400:
            # try to surface the real OpenAI error message
            body_excerpt = r.text.strip()
            # don’t blow the logs with megabytes of prompt echo
            if len(body_excerpt) > 3_000:
                body_excerpt = body_excerpt[:3_000] + "…[truncated]"
            raise requests.HTTPError(
                f"{r.status_code} {r.reason} – OpenAI response body:\n{body_excerpt}",
                response=r,
            )

        return r.json()


    # ---------------- public async API ---------------- #
    async def generate_response(
        self,
        prompt: str,
        temperature: float = 0.0,
        inject_random_seed: bool = True,
    ) -> str:
        system_prompt_content = f"{generate_random_seed()}\n\n{self.system_prompt}" if inject_random_seed else self.system_prompt

        if self.model_name == "qwen/qwen3-235b-a22b":
            system_prompt_content += "\n/no_think"

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt_content},
                {"role": "user", "content": f"{prompt}\n\nPROVIDE YOUR RESPONSE BELOW:"},
            ],
            "temperature": temperature,
        }
        
        # Use max_completion_tokens for newer OpenAI models
        model_id = self.model_name.lower()
        if (
            model_id.startswith("gpt-5")
            or model_id.startswith("gpt-4.1")
            or model_id.startswith("o3")
            or model_id.startswith("o4")
            or model_id.startswith("nectarine")
        ):
            payload["max_completion_tokens"] = self.max_tokens
        else:
            payload["max_tokens"] = self.max_tokens

        # gpt-5 models only support the default temperature (1.0)
        if model_id.startswith("gpt-5"):
            payload["temperature"] = 1.0

        #if self.model_name == "qwen/qwen3-235b-a22b" and self.base_url == "https://openrouter.ai/api/v1":
        #    payload["provider"] = {
        #        "order": ["Cerebras"],     # fast qwen-2-35B
        #        "allow_fallbacks": False,
        #    }

        if model_id in ["o3", "o4-mini"]:
            del payload["temperature"]
            if "max_tokens" in payload:
                del payload["max_tokens"]
            payload["max_completion_tokens"] = self.max_tokens

        loop = asyncio.get_running_loop()
        try:
            data = await loop.run_in_executor(None, self._post_sync, payload)
            if not data.get("choices") or not data["choices"][0].get("message") or not data["choices"][0]["message"].get("content"):
                raise ValueError(f"[{self.model_name}] LLM returned an empty or invalid response.")
            content = data["choices"][0]["message"]["content"].strip()
            if '<think>' in content and '</think>' in content:
                content = content[content.rfind('</think>') + len('</think>'):]
            return content
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"[{self.model_name}] Bad response format: {e}", exc_info=True)
            raise
        except requests.RequestException as e:
            # bubble up the richer message we attached in _post_sync
            logger.error(f"[{self.model_name}] HTTP error while calling OpenAI: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"[{self.model_name}] Unexpected error: {e}", exc_info=True)
            raise


##############################################################################
# Codex Web Hook Client
##############################################################################


class CodexWebClient(BaseModelClient):
    """
    Calls a local/web hook (Codex web login token) instead of an API key.
    Expects an HTTP JSON endpoint defined by CODEX_WEBHOOK_URL.
    """

    def __init__(self, model_name: str = "codex-web", prompts_dir: Optional[str] = None, endpoint: Optional[str] = None, token: Optional[str] = None):
        super().__init__(model_name, prompts_dir=prompts_dir)
        self.endpoint = (endpoint or os.environ.get("CODEX_WEBHOOK_URL") or "").strip()
        self.token = (token or os.environ.get("CODEX_WEBHOOK_TOKEN") or "").strip()
        self.timeout = int(os.environ.get("CODEX_WEBHOOK_TIMEOUT", "180"))
        if not self.endpoint:
            raise ValueError("CODEX_WEBHOOK_URL is required for CodexWebClient")

    async def generate_response(self, prompt: str, temperature: float = 0.0, inject_random_seed: bool = True) -> str:
        system_prompt_content = f"{generate_random_seed()}\n\n{self.system_prompt}" if inject_random_seed else self.system_prompt

        payload = {
            "model": self.model_name,
            "system": system_prompt_content,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        async with aiohttp.ClientSession() as session:
            async with session.post(self.endpoint, json=payload, headers=headers, timeout=self.timeout) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    body_excerpt = text[:3000] + ("...[truncated]" if len(text) > 3000 else "")
                    raise ValueError(f"[{self.model_name}] Codex webhook error {resp.status}: {body_excerpt}")

                # Try JSON first
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    return text.strip()

                # Common response shapes
                if isinstance(data, dict):
                    if "content" in data and isinstance(data["content"], str):
                        return data["content"].strip()
                    if "text" in data and isinstance(data["text"], str):
                        return data["text"].strip()
                    if "response" in data and isinstance(data["response"], str):
                        return data["response"].strip()
                    if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                        choice = data["choices"][0]
                        if isinstance(choice, dict):
                            message = choice.get("message", {})
                            if isinstance(message, dict) and isinstance(message.get("content"), str):
                                return message["content"].strip()
                            if isinstance(choice.get("text"), str):
                                return choice["text"].strip()

                # Fallback: return stringified JSON
                return json.dumps(data).strip()

##############################################################################
# Silent Client (No-op baseline)
##############################################################################


class SilentClient(BaseModelClient):
    """
    A no-op client that emits no negotiation messages and only HOLD orders.
    Useful for keeping a 3-way experiment inside the 7-player engine.
    """

    def __init__(self, model_name: str = "silent", prompts_dir: Optional[str] = None):
        super().__init__(model_name, prompts_dir=prompts_dir)

    async def generate_response(self, prompt: str, temperature: float = 0.0, inject_random_seed: bool = True) -> str:
        return "{}"

    async def get_orders(
        self,
        game,
        board_state,
        power_name: str,
        possible_orders: Dict[str, List[str]],
        conversation_text: str,
        model_error_stats: dict,
        log_file_path: str,
        phase: str,
        agent_goals: Optional[List[str]] = None,
        agent_relationships: Optional[Dict[str, str]] = None,
        agent_private_diary_str: Optional[str] = None,
    ) -> List[str]:
        return self.fallback_orders(possible_orders)

    async def get_conversation_reply(
        self,
        game,
        board_state,
        power_name: str,
        possible_orders: Dict[str, List[str]],
        game_history: GameHistory,
        game_phase: str,
        log_file_path: str,
        active_powers: Optional[List[str]] = None,
        agent_goals: Optional[List[str]] = None,
        agent_relationships: Optional[Dict[str, str]] = None,
        agent_private_diary_str: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        return []


##############################################################################
# Storyworld Persuasion Wrapper
##############################################################################


class StoryworldPersuasionClient(BaseModelClient):
    """
    Wraps a base model client and injects a storyworld forecast artifact
    into negotiation prompts (orders unaffected).
    """

    def __init__(
        self,
        base_client: BaseModelClient,
        prompts_dir: Optional[str] = None,
        storyworld_client: Optional[BaseModelClient] = None,
        storyworld_path: Optional[str] = None,
    ):
        super().__init__(f"storyworld+{base_client.model_name}", prompts_dir=prompts_dir)
        self.base_client = base_client
        self.storyworld_client = storyworld_client or base_client
        self.storyworld_path = Path(storyworld_path) if storyworld_path else None

    def set_system_prompt(self, content: str):
        self.base_client.set_system_prompt(content)
        self.system_prompt = content

    async def generate_response(self, prompt: str, temperature: float = 0.0, inject_random_seed: bool = True) -> str:
        return await self.base_client.generate_response(prompt, temperature=temperature, inject_random_seed=inject_random_seed)

    async def get_orders(
        self,
        game,
        board_state,
        power_name: str,
        possible_orders: Dict[str, List[str]],
        conversation_text: str,
        model_error_stats: dict,
        log_file_path: str,
        phase: str,
        agent_goals: Optional[List[str]] = None,
        agent_relationships: Optional[Dict[str, str]] = None,
        agent_private_diary_str: Optional[str] = None,
    ) -> List[str]:
        return await self.base_client.get_orders(
            game,
            board_state,
            power_name,
            possible_orders,
            conversation_text,
            model_error_stats,
            log_file_path,
            phase,
            agent_goals=agent_goals,
            agent_relationships=agent_relationships,
            agent_private_diary_str=agent_private_diary_str,
        )

    async def get_conversation_reply(
        self,
        game,
        board_state,
        power_name: str,
        possible_orders: Dict[str, List[str]],
        game_history: GameHistory,
        game_phase: str,
        log_file_path: str,
        active_powers: Optional[List[str]] = None,
        agent_goals: Optional[List[str]] = None,
        agent_relationships: Optional[Dict[str, str]] = None,
        agent_private_diary_str: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        raw_input_prompt = ""
        raw_response = ""
        success_status = "Failure: Initialized"
        messages_to_return: List[Dict[str, str]] = []
        forecast = None
        storyworld_log_path = None

        try:
            raw_input_prompt = self.base_client.build_conversation_prompt(
                game,
                board_state,
                power_name,
                possible_orders,
                game_history,
                agent_goals=agent_goals,
                agent_relationships=agent_relationships,
                agent_private_diary_str=agent_private_diary_str,
            )

            log_dir = Path(log_file_path).parent if log_file_path else None
            forecast_log = (log_dir / "storyworld_forecasts.jsonl") if log_dir else None
            storyworld_log_path = (log_dir / "storyworld_impact.jsonl") if log_dir else None

            forecast = await generate_storyworld_forecast(
                power_name=power_name,
                board_state=board_state,
                game=game,
                active_powers=active_powers or [],
                model_client=self.storyworld_client,
                log_path=forecast_log,
                storyworld_path=self.storyworld_path,
            )

            if forecast:
                artifact = {
                    "storyworld_id": forecast.storyworld_id,
                    "mapped_agents": forecast.mapped_agents,
                    "target_power": forecast.target_power,
                    "forecast": forecast.forecast,
                    "confidence": forecast.confidence,
                    "reasoning": forecast.reasoning,
                }
                raw_input_prompt = (
                    raw_input_prompt
                    + "\n\nSTORYWORLD_FORECAST_ARTIFACT (Use this as evidence for persuasion):\n"
                    + json.dumps(artifact, ensure_ascii=False, indent=2)
                )

            # Optional Codex oracle consultation (storyworld-aware)
            oracle_plan_powers = set(_parse_power_list(os.environ.get("CODEX_ORACLE_PLAN_POWERS")))
            oracle_reply_powers = set(_parse_power_list(os.environ.get("CODEX_ORACLE_REPLY_POWERS")))
            power_key = power_name.upper()
            oracle_mode = None
            recent_msgs = None

            if power_key in oracle_plan_powers:
                oracle_mode = "plan"
            elif power_key in oracle_reply_powers:
                recent_msgs = game_history.get_recent_messages_to_power(power_name, limit=3)
                recent_msgs = [m for m in recent_msgs if m.get("phase") == game_phase]
                if recent_msgs:
                    oracle_mode = "reply"

            if oracle_mode and _oracle_call_allowed(power_key, game_phase, oracle_mode):
                oracle_prompt = _build_oracle_prompt(
                    mode=oracle_mode,
                    power_name=power_name,
                    phase=game_phase,
                    base_prompt=raw_input_prompt,
                    messages_to_power=recent_msgs if oracle_mode == "reply" else None,
                    storyworld_artifact=artifact if forecast else None,
                )
                oracle_raw, oracle_parsed = await _run_oracle_call(
                    power_name=power_name,
                    phase=game_phase,
                    mode=oracle_mode,
                    oracle_prompt=oracle_prompt,
                    log_file_path=log_file_path,
                )

                oracle_insert = None
                if isinstance(oracle_parsed, dict):
                    oracle_insert = json.dumps(
                        {
                            "situation_assessment": oracle_parsed.get("situation_assessment"),
                            "storyworld_implications": oracle_parsed.get("storyworld_implications"),
                            "recommended_belief_updates": oracle_parsed.get("recommended_belief_updates"),
                            "candidate_messages": oracle_parsed.get("candidate_messages"),
                            "recommendation": oracle_parsed.get("recommendation"),
                            "reasoning_trace": oracle_parsed.get("reasoning_trace"),
                        },
                        ensure_ascii=True,
                    )
                if oracle_insert:
                    raw_input_prompt = f"{raw_input_prompt}\n\n# ORACLE CONSULTATION\n{oracle_insert}"

                log_path = _oracle_log_path(log_file_path)
                if log_path:
                    entry = {
                        "ts": datetime.utcnow().isoformat() + "Z",
                        "power": power_name,
                        "phase": game_phase,
                        "mode": oracle_mode,
                        "model": _get_oracle_client().model_name if _get_oracle_client() else None,
                        "oracle_used_in_prompt": bool(oracle_insert),
                        "storyworld_id": forecast.storyworld_id if forecast else None,
                    }
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(entry, ensure_ascii=True) + "\n")

            raw_response = await run_llm_and_log(
                client=self.base_client,
                prompt=raw_input_prompt,
                power_name=power_name,
                phase=game_phase,
                response_type="negotiation",
            )

            # Conditionally format the response based on USE_UNFORMATTED_PROMPTS
            if config.USE_UNFORMATTED_PROMPTS:
                from .formatter import format_with_gemini_flash, FORMAT_CONVERSATION

                formatted_response = await format_with_gemini_flash(
                    raw_response, FORMAT_CONVERSATION, power_name=power_name, phase=game_phase, log_file_path=log_file_path
                )
            else:
                formatted_response = raw_response

            parsed_messages = []
            json_blocks = []

            try:
                data = json.loads(formatted_response)
                if isinstance(data, list):
                    parsed_messages = data
                    json_blocks = [json.dumps(item) for item in data if isinstance(item, dict)]
            except json.JSONDecodeError:
                raw_response = formatted_response

            if not parsed_messages:
                double_brace_blocks = re.findall(r"\{\{(.*?)\}\}", raw_response, re.DOTALL)
                if double_brace_blocks:
                    json_blocks.extend(["{" + block.strip() + "}" for block in double_brace_blocks])
                else:
                    code_block_match = re.search(r"```json\n(.*?)\n```", raw_response, re.DOTALL)
                    if code_block_match:
                        potential = code_block_match.group(1).strip()
                        try:
                            data = json.loads(potential)
                            if isinstance(data, list):
                                json_blocks = [json.dumps(item) for item in data if isinstance(item, dict)]
                            elif isinstance(data, dict):
                                json_blocks = [json.dumps(data)]
                        except json.JSONDecodeError:
                            json_blocks = re.findall(r"\{.*?\}", potential, re.DOTALL)
                    else:
                        json_blocks = re.findall(r"\{.*?\}", raw_response, re.DOTALL)

            if not parsed_messages and json_blocks:
                for block in json_blocks:
                    try:
                        cleaned = re.sub(r",\s*([\}\]])", r"\1", block.strip())
                        parsed_message = json.loads(cleaned)
                        parsed_messages.append(parsed_message)
                    except json.JSONDecodeError:
                        pass

            if parsed_messages:
                validated = []
                for msg in parsed_messages:
                    if isinstance(msg, dict) and "message_type" in msg and "content" in msg:
                        if msg["message_type"] == "private" and "recipient" not in msg:
                            continue
                        validated.append(msg)
                parsed_messages = validated

            if parsed_messages:
                success_status = "Success: Messages extracted"
                messages_to_return = parsed_messages
            else:
                success_status = "Success: No valid messages"
                messages_to_return = []

        except Exception as e:
            logger.error(f"[{self.model_name}] Error in storyworld conversation for {power_name}: {e}", exc_info=True)
            success_status = f"Failure: Exception ({type(e).__name__})"
            messages_to_return = []
        finally:
            # Log storyworld impact heuristics (best-effort)
            if forecast and storyworld_log_path:
                try:
                    contents = " ".join(
                        m.get("content", "")
                        for m in messages_to_return
                        if isinstance(m, dict)
                    ).lower()
                    matches = []
                    for term in [
                        "forecast",
                        "probab",
                        "likelihood",
                        "odds",
                        str(forecast.storyworld_id).lower(),
                        str(forecast.target_power).lower(),
                    ]:
                        if term and term in contents:
                            matches.append(term)
                    impact_flag = "explicit" if matches else "implicit"
                    recipients = [
                        m.get("recipient", "")
                        for m in messages_to_return
                        if isinstance(m, dict) and m.get("message_type") == "private"
                    ]
                    event = {
                        "ts": time.time(),
                        "phase": game_phase,
                        "power": power_name,
                        "storyworld_id": forecast.storyworld_id,
                        "target_power": forecast.target_power,
                        "confidence": forecast.confidence,
                        "forecast": forecast.forecast,
                        "reasoning": forecast.reasoning,
                        "message_count": len(messages_to_return),
                        "recipients": recipients,
                        "matches": matches,
                        "impact_flag": impact_flag,
                    }
                    storyworld_log_path.parent.mkdir(parents=True, exist_ok=True)
                    with storyworld_log_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(event, ensure_ascii=False) + "\n")
                except Exception:
                    pass
            if log_file_path:
                await log_llm_response_async(
                    log_file_path=log_file_path,
                    model_name=self.model_name,
                    power_name=power_name,
                    phase=game_phase,
                    response_type="negotiation_message",
                    raw_input_prompt=raw_input_prompt,
                    raw_response=raw_response,
                    success=success_status,
                )
            return messages_to_return

##############################################################################
# 3) Factory to Load Model Client
##############################################################################
class ModelSpec(NamedTuple):
    prefix: Optional[str]  # 'openai', 'requests', …
    model: str  # 'gpt-4o'
    base: Optional[str]  # 'https://proxy.foo'
    key: Optional[str]  # 'sk-…' (may be None)


def _parse_model_spec(raw: str) -> ModelSpec:
    """
    Splits once on '#' (API key) and once on '@' (base URL).  A leading
    '<prefix>:' is optional.  Nothing else is interpreted.
    """
    raw = raw.strip()

    pre_hash, _, key_part = raw.partition("#")
    pre_at, _, base_part = pre_hash.partition("@")

    maybe_pref, sep, model_part = pre_at.partition(":")
    if sep:  # explicit prefix was present
        prefix, model = maybe_pref.lower(), model_part
    else:
        prefix, model = None, maybe_pref

    return ModelSpec(prefix, model, base_part or None, key_part or None)

class Prefix(StrEnum):
    OPENAI            = "openai"
    OPENAI_REQUESTS   = "openai-requests"
    OPENAI_RESPONSES  = "openai-responses"
    ANTHROPIC         = "anthropic"
    GEMINI            = "gemini"
    DEEPSEEK          = "deepseek"
    OPENROUTER        = "openrouter"
    TOGETHER          = "together"
    STORYWORLD        = "storyworld"
    SILENT            = "silent"
    CODEXWEB          = "codexweb"

def load_model_client(model_id: str, prompts_dir: Optional[str] = None) -> BaseModelClient:
    """
    Recognises strings like
        gpt-4o
        anthropic:claude-3.7-sonnet
        openai:llama-3-2-3b@https://localhost:8000#myapikey
        gpt-5-reasoning-alpha-2025-07-19:minimal
    and returns the appropriate client.

    • If a prefix is omitted the function falls back to the original
      heuristic mapping exactly as before.
    • If an inline API-key ('#…') is present it overrides environment vars.
    • For reasoning models, effort can be specified with :minimal, :medium, or :high
    """
    # Extract reasoning effort if present (before general parsing)
    reasoning_effort = None
    actual_model_id = model_id
    
    # Check if this is a reasoning model with effort specified
    reasoning_models = ['gpt-5-reasoning-alpha-2025-07-19', 'o4-mini', 'nectarine-alpha-2025-07-25', 'nectarine-alpha-new-reasoning-effort-2025-07-25']
    for model in reasoning_models:
        if model_id.startswith(model + ':'):
            parts = model_id.split(':', 1)
            effort_part = parts[1]
            # Check if the effort part is valid before treating it as effort
            # (it could be a prefix like "openai:")
            if effort_part.lower() in ['minimal', 'medium', 'high']:
                actual_model_id = parts[0]
                reasoning_effort = effort_part.lower()
                break
    
    spec = _parse_model_spec(actual_model_id)
    logger.info(f"[load_model_client] Loading client for model_id='{model_id}', parsed spec: prefix={spec.prefix}, model={spec.model}, reasoning_effort={reasoning_effort}")

    # Inline key overrides env; otherwise fall back as usual *per client*
    inline_key = spec.key

    # ------------------------------------------------------------------ #
    # 1. Explicit prefix path                                           #
    # ------------------------------------------------------------------ #
    if spec.prefix:
        try:
            pref = Prefix(spec.prefix.lower())
        except ValueError as exc:
            raise ValueError(
                f"[load_model_client] unknown prefix '{spec.prefix}'. "
                "Allowed prefixes: openai, openai-requests, openai-responses, "
                "anthropic, gemini, deepseek, openrouter, together."
            ) from exc

        match pref:
            case Prefix.OPENAI:
                return OpenAIClient(
                    model_name=spec.model,
                    prompts_dir=prompts_dir,
                    base_url=spec.base,
                    api_key=inline_key,
                )
            case Prefix.OPENAI_REQUESTS:
                return RequestsOpenAIClient(
                    model_name=spec.model,
                    prompts_dir=prompts_dir,
                    base_url=spec.base,
                    api_key=inline_key,
                )
            case Prefix.OPENAI_RESPONSES:
                return OpenAIResponsesClient(spec.model, prompts_dir, api_key=inline_key, reasoning_effort=reasoning_effort)
            case Prefix.ANTHROPIC:
                return ClaudeClient(spec.model, prompts_dir)
            case Prefix.GEMINI:
                return GeminiClient(spec.model, prompts_dir)
            case Prefix.DEEPSEEK:
                return DeepSeekClient(spec.model, prompts_dir)
            case Prefix.OPENROUTER:
                return OpenRouterClient(spec.model, prompts_dir)
            case Prefix.TOGETHER:
                return TogetherAIClient(spec.model, prompts_dir)
            case Prefix.SILENT:
                return SilentClient(spec.model or "silent", prompts_dir=prompts_dir)
            case Prefix.STORYWORLD:
                base = load_model_client(spec.model, prompts_dir=prompts_dir)
                return StoryworldPersuasionClient(base, prompts_dir=prompts_dir)
            case Prefix.CODEXWEB:
                return CodexWebClient(spec.model or "codex-web", prompts_dir=prompts_dir)

    # ------------------------------------------------------------------ #
    # 2. Heuristic fallback path (identical to the original behaviour)   #
    # ------------------------------------------------------------------ #
    lower_id = spec.model.lower()
    logger.info(f"[load_model_client] Heuristic path: checking model='{spec.model}', lower_id='{lower_id}'")

    # Check if this is a reasoning model that should use Responses API
    reasoning_models_requiring_responses = ['gpt-5-reasoning-alpha-2025-07-19', 'o4-mini', 'nectarine-alpha-2025-07-25', 'nectarine-alpha-new-reasoning-effort-2025-07-25']
    if spec.model in reasoning_models_requiring_responses:
        logger.info(f"[load_model_client] Selected OpenAIResponsesClient for reasoning model '{spec.model}'")
        return OpenAIResponsesClient(spec.model, prompts_dir, api_key=inline_key, reasoning_effort=reasoning_effort)

    if lower_id.startswith("gpt-5"):
        logger.info(f"[load_model_client] Selected OpenAIResponsesClient for '{spec.model}'")
        return OpenAIResponsesClient(spec.model, prompts_dir, api_key=inline_key, reasoning_effort=reasoning_effort)

    if lower_id == "o3-pro":
        logger.info(f"[load_model_client] Selected OpenAIResponsesClient for '{spec.model}'")
        return OpenAIResponsesClient(spec.model, prompts_dir, api_key=inline_key)

    if spec.model.startswith("together-"):
        # e.g. "together-mixtral-8x7b"
        logger.info(f"[load_model_client] Selected TogetherAIClient for '{spec.model}'")
        return TogetherAIClient(spec.model.split("together-", 1)[1], prompts_dir)

    if "openrouter" in lower_id:
        logger.info(f"[load_model_client] Selected OpenRouterClient for '{spec.model}'")
        return OpenRouterClient(spec.model, prompts_dir)

    if "claude" in lower_id:
        logger.info(f"[load_model_client] Selected ClaudeClient for '{spec.model}'")
        return ClaudeClient(spec.model, prompts_dir)

    if "gemini" in lower_id:
        logger.info(f"[load_model_client] Selected GeminiClient for '{spec.model}'")
        return GeminiClient(spec.model, prompts_dir)

    if "deepseek" in lower_id:
        logger.info(f"[load_model_client] Selected DeepSeekClient for '{spec.model}'")
        return DeepSeekClient(spec.model, prompts_dir)

    if lower_id.startswith("silent"):
        logger.info(f"[load_model_client] Selected SilentClient for '{spec.model}'")
        return SilentClient(spec.model, prompts_dir=prompts_dir)

    if lower_id.startswith("codexweb"):
        logger.info(f"[load_model_client] Selected CodexWebClient for '{spec.model}'")
        return CodexWebClient(spec.model, prompts_dir=prompts_dir)

    # Default: OpenAI-compatible async client
    logger.info(f"[load_model_client] No specific match found, using default OpenAIClient for '{spec.model}'")
    return OpenAIClient(
        model_name=spec.model,
        prompts_dir=prompts_dir,
        base_url=spec.base,
        api_key=inline_key,
    )


##############################################################################
# 1) Add a method to filter visible messages (near top-level or in BaseModelClient)
##############################################################################
def get_visible_messages_for_power(conversation_messages, power_name):
    """
    Returns a chronological subset of conversation_messages that power_name can legitimately see.
    """
    visible = []
    for msg in conversation_messages:
        # GLOBAL might be 'ALL' or 'GLOBAL' depending on your usage
        if msg["recipient"] == "ALL" or msg["recipient"] == "GLOBAL" or msg["sender"] == power_name or msg["recipient"] == power_name:
            visible.append(msg)
    return visible  # already in chronological order if appended that way
