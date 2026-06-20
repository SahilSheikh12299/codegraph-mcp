import json
import re
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urljoin


_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_MODEL = "qwen2.5:1.5b"


class OllamaError(RuntimeError):
    pass


def _normalize_one_sentence(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    # Remove common wrappers / markdown.
    t = re.sub(r"^```.*?$", "", t, flags=re.MULTILINE).strip()
    t = t.strip("`").strip()
    # Collapse whitespace.
    t = re.sub(r"\s+", " ", t).strip()
    # Keep only the first sentence-ish span (simple heuristic).
    # This prevents small models from emitting extra lines.
    m = re.match(r"^(.+?[.!?])(\s|$)", t)
    if m:
        t = m.group(1).strip()
    return t


def generate_intent_docstring(
    *,
    function_name: str,
    snippet: str,
    model: str = _DEFAULT_MODEL,
    base_url: str = _DEFAULT_BASE_URL,
    timeout_s: float = 20.0,
) -> str:
    """Generate a short, single-sentence intent docstring for retrieval metadata."""
    name = (function_name or "").strip()
    code = (snippet or "").strip()
    if not name or not code:
        return ""

    prompt = (
        "Write EXACTLY one sentence (12-25 words) describing what this Python function does.\n"
        "Rules: no preamble, no markdown, no quotes, do not start with 'This function'.\n"
        "Output only the sentence.\n\n"
        f"Function: {name}\n"
        "Code:\n"
        f"{code}\n"
    )

    url = urljoin(base_url.rstrip("/") + "/", "api/generate")
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1},
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError) as e:
        raise OllamaError(f"Ollama request failed: {e}") from e

    try:
        data = json.loads(body)
    except json.JSONDecodeError as e:
        raise OllamaError(f"Invalid Ollama JSON response: {e}") from e

    text = _normalize_one_sentence(str(data.get("response") or ""))
    # Basic validation: 12-25 words as requested (soft-fail by returning normalized).
    words = [w for w in re.split(r"\s+", text) if w]
    if len(words) < 6:
        return ""
    return text


def unload_model(
    model: str = _DEFAULT_MODEL,
    base_url: str = _DEFAULT_BASE_URL,
    timeout_s: float = 10.0,
) -> None:
    """Unload model from Ollama process memory (keep_alive=0). Best-effort."""
    url = urljoin(base_url.rstrip("/") + "/", "api/generate")
    payload: dict[str, Any] = {
        "model": model,
        "prompt": "",
        "stream": False,
        "keep_alive": 0,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s):
            pass
    except Exception:
        return

