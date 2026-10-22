"""Run progressive live Gemma requests to isolate investigation-payload failures.

This test calls Google's hosted endpoint directly. It does not call the local API
and does not print GEMMA_API_KEY. The request construction uses the provider's
real prompt builder for stage E.
"""
import json
import logging
import os
import ssl
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import certifi

from app.agent_schemas import AgentDecision
from app.providers.gemma import (
    DEFAULT_API_BASE_URL,
    DEFAULT_MODEL_NAME,
    GemmaProvider,
    SYSTEM_INSTRUCTION,
)
from app.simulator import Vehicle
from app.tools.registry import tool_descriptions


def load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("\"'")
        os.environ.setdefault(key.strip(), value)


def redact(text: str, key: str, limit: int = 2400) -> str:
    if key:
        text = text.replace(key, "[REDACTED]")
    import re
    text = re.sub(r"(?i)(x-goog-api-key|authorization)\s*[:=]\s*[^\s,;]+",
                  r"\1: [REDACTED]", text)
    text = re.sub(r'(?i)("?(?:api[_-]?key|x-goog-api-key|authorization)"?\s*:\s*)"?[^",}\s]+"?',
                  r'\1"[REDACTED]"', text)
    return text if len(text) <= limit else text[:limit] + f"… [truncated; total_chars={len(text)}]"


def response_text(body: bytes, key: str) -> str:
    text = body.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(text)
        for candidate in parsed.get("candidates", []):
            content = candidate.get("content", {})
            if "parts" in content:
                content["parts"] = [part for part in content["parts"] if not part.get("thought")]
        text = json.dumps(parsed, ensure_ascii=False, indent=2)
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    return redact(text, key)


def send(payload: dict, provider: GemmaProvider) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(provider.endpoint, data=body, method="POST", headers={
        "Content-Type": "application/json",
        "Accept": "application/json",
        "x-goog-api-key": provider.api_key,
    })
    started = time.perf_counter()
    status: int | str
    try:
        with urlopen(request, timeout=provider.timeout,
                     context=ssl.create_default_context(cafile=certifi.where())) as response:
            status = response.status
            response_body = response.read()
    except HTTPError as exc:
        status = exc.code
        response_body = exc.read()
    except (URLError, TimeoutError, OSError) as exc:
        status = "unavailable"
        response_body = f"{type(exc).__name__}: {exc}".encode()
    elapsed = time.perf_counter() - started
    print(f"HTTP status: {status}")
    print(f"Elapsed seconds: {elapsed:.2f}")
    print(f"Response size bytes: {len(response_body)}")
    print(f"Response body: {response_text(response_body, provider.api_key)}")

    if isinstance(status, int) and 200 <= status < 300:
        try:
            response_json = json.loads(response_body)
            candidate_text = GemmaProvider._decision_text(response_json)
            try:
                AgentDecision.model_validate_json(candidate_text)
                parseable = True
            except Exception:
                parseable = False
            print(f"Provider decision parseable: {parseable}")
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            print("Provider decision parseable: False")
    return {"status": status, "elapsed": elapsed, "response_bytes": len(response_body)}


def main() -> int:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.environ["GEMMA_DIAGNOSTICS"] = "1"
    model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
    provider = GemmaProvider(model_name=model_name, timeout=float(os.getenv("MODEL_TIMEOUT", "60")))
    base_url = os.getenv("GEMMA_API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")
    if provider.model_name != "gemma-4-31b-it":
        raise SystemExit("This diagnostic is pinned to MODEL_NAME=gemma-4-31b-it.")
    provider.endpoint = f"{base_url}/models/{quote(provider.model_name, safe='')}:generateContent"

    vehicle = Vehicle(fault="misfire")
    telemetry = vehicle.telemetry()
    initial_observation = {
        "id": "obs_001", "kind": "initial_telemetry", "tool": "observe_vehicle", "data": telemetry,
    }
    tools = tool_descriptions()
    empty_context = {
        "initial_telemetry": {}, "observations": [], "hypotheses": [], "previous_actions": [],
        "step": 1, "remaining_action_budget": 8,
    }
    telemetry_context = {
        **empty_context,
        "initial_telemetry": telemetry,
        "observations": [initial_observation],
    }
    tool_context = {**telemetry_context}

    basic_prompt = 'Return JSON: {"action":"finish"}'
    telemetry_prompt = "Return one compact JSON diagnostic decision using only this initial telemetry: " + json.dumps(
        telemetry, separators=(",", ":"), ensure_ascii=False)
    tools_prompt = "Initial telemetry: " + json.dumps(telemetry, separators=(",", ":"), ensure_ascii=False)
    tools_prompt += "\nAvailable diagnostic tools: " + json.dumps(tools, separators=(",", ":"), ensure_ascii=False)
    compact_system_payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "contents": [{"role": "user", "parts": [{"text": basic_prompt}]}],
    }

    # A->D progressively add input content; E is the exact payload built by the provider.
    stages = [
        ("A — simple JSON", {"contents": [{"parts": [{"text": basic_prompt}]}]}),
        ("B — current system instructions, no telemetry", compact_system_payload),
        ("C — instructions plus initial telemetry", {
            "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
            "contents": [{"role": "user", "parts": [{"text": telemetry_prompt}]}],
        }),
        ("D — instructions, telemetry, and tool definitions", {
            "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
            "contents": [{"role": "user", "parts": [{"text": tools_prompt}]}],
        }),
        ("E — exact application investigation payload", GemmaProvider.build_payload(tool_context, tools)),
    ]
    failures = []
    for label, payload in stages:
        print(f"\n=== {label} ===")
        result = send(payload, provider)
        if not isinstance(result["status"], int) or not 200 <= result["status"] < 300:
            failures.append((label, result["status"]))
    print("\nProgressive summary:")
    print(json.dumps({"first_failed_stage": failures[0][0] if failures else None,
                      "failed_stages": failures}, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
