import hashlib
import json
import re
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

SENSITIVE = re.compile(
    r"password|passwd|secret|token|session|cookie|authorization|api.?key|credential",
    re.I,
)
METER_FIELDS = {
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "image_tokens",
    "cached_tokens",
    "reasoning_tokens",
    "prompt_cache_hit_tokens",
    "prompt_cache_miss_tokens",
}
METER_CONTAINERS = {"prompt_tokens_details", "completion_tokens_details"}


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def digest(value):
    return hashlib.sha256(
        (value if isinstance(value, bytes) else canonical(value).encode("utf-8"))
    ).hexdigest()


def redact_url(value):
    try:
        p = urlsplit(value)
        host = p.hostname or ""
        if ":" in host:
            host = "[" + host + "]"
        if p.port:
            host += ":" + str(p.port)
        # Keep names only, including innocently named query parameters.
        query = urlencode(
            [
                (key, "[REDACTED]")
                for key, _ in parse_qsl(p.query, keep_blank_values=True)
            ]
        )
        path = re.sub(r"(?<=/)bot\d+:[A-Za-z0-9_-]+", "bot[REDACTED]", p.path)
        path = re.sub(r"(?<=/)[A-Za-z0-9_-]{32,}(?=/|$)", "[REDACTED]", path)
        return urlunsplit((p.scheme, host, path, query, ""))
    except ValueError:
        return "[INVALID_URL]"


def sanitize(value):
    if isinstance(value, dict):
        return {
            k: (
                v
                if k in METER_FIELDS
                and isinstance(v, (int, float))
                and not isinstance(v, bool)
                else sanitize(v)
                if k in METER_CONTAINERS and isinstance(v, dict)
                else "[REDACTED]"
                if SENSITIVE.search(k)
                else sanitize(v)
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, str):
        value = re.sub(r'https?://[^\s<>"\']+', lambda m: redact_url(m.group()), value)
        value = re.sub(
            r"(?i)\b(?:Bearer\s+\S+|sk-[A-Za-z0-9_-]{8,})", "[REDACTED]", value
        )
        value = re.sub(
            r'(?i)((?:password|token|secret|session|api[_-]?key)\s*[:=]\s*)[^\s,;<>"\']+',
            r"\1[REDACTED]",
            value,
        )
        # A literal @ is necessary; avoid quadratic failed searches on long
        # base64/data-URI text. Same result as the old regex when @ is absent.
        if "@" in value:
            value = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[EMAIL]", value)
        return value
    return value
