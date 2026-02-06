import re
from typing import Any


_BEARER_RE = re.compile(r"(?i)(\bBearer\s+)([A-Za-z0-9._\-]{8,})")
_KEY_ASSIGN_RE = re.compile(
    r"""(?ix)
    (
        \b(?:\$env:)?[A-Za-z_][A-Za-z0-9_]*(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD)[A-Za-z0-9_]*\b
        \s*[:=]\s*
        ["']?
    )
    ([A-Za-z0-9._\-]{8,})
    (["']?)
    """
)
_MODEL_SPEC_KEY_RE = re.compile(
    r"""(?x)
    (
        (?:[A-Za-z0-9._-]+:)?
        [A-Za-z0-9._/-]+
        (?:@[A-Za-z0-9:/._-]+)?
        \#
    )
    ([A-Za-z0-9._\-]{8,})
    """
)
_OPENAI_KEY_RE = re.compile(r"\b(sk-[A-Za-z0-9_-]{16,})\b")
_ANTHROPIC_KEY_RE = re.compile(r"\b(sk-ant-[A-Za-z0-9_-]{16,})\b")
_GOOGLE_KEY_RE = re.compile(r"\b(AIza[0-9A-Za-z_-]{20,})\b")

_SENSITIVE_KEY_MARKERS = (
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "authorization",
    "auth_token",
    "access_token",
    "refresh_token",
)


def truncate_secret(secret: str, keep: int = 4) -> str:
    if secret is None:
        return secret
    if not isinstance(secret, str):
        secret = str(secret)
    if len(secret) <= keep * 2:
        return secret
    return f"{secret[:keep]}...{secret[-keep:]}"


def _sub_group2(match: re.Match) -> str:
    return f"{match.group(1)}{truncate_secret(match.group(2))}"


def _sub_group2_group3(match: re.Match) -> str:
    return f"{match.group(1)}{truncate_secret(match.group(2))}{match.group(3)}"


def redact_text(text: str) -> str:
    if text is None:
        return text
    if not isinstance(text, str):
        text = str(text)

    redacted = text
    redacted = _BEARER_RE.sub(_sub_group2, redacted)
    redacted = _KEY_ASSIGN_RE.sub(_sub_group2_group3, redacted)
    redacted = _MODEL_SPEC_KEY_RE.sub(_sub_group2, redacted)
    redacted = _OPENAI_KEY_RE.sub(lambda m: truncate_secret(m.group(1)), redacted)
    redacted = _ANTHROPIC_KEY_RE.sub(lambda m: truncate_secret(m.group(1)), redacted)
    redacted = _GOOGLE_KEY_RE.sub(lambda m: truncate_secret(m.group(1)), redacted)
    return redacted


def _is_sensitive_key(key: str) -> bool:
    key_l = key.lower()
    return any(marker in key_l for marker in _SENSITIVE_KEY_MARKERS)


def redact_data(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if isinstance(k, str) and _is_sensitive_key(k) and isinstance(v, str):
                out[k] = truncate_secret(v)
            else:
                out[k] = redact_data(v)
        return out
    if isinstance(value, list):
        return [redact_data(v) for v in value]
    if isinstance(value, tuple):
        return tuple(redact_data(v) for v in value)
    if isinstance(value, str):
        return redact_text(value)
    return value

