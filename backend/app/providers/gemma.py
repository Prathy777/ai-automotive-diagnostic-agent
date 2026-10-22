"""Google-hosted Gemma provider implemented with Python's standard HTTP stack."""
import asyncio
import json
import logging
import os
import re
import ssl
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

import certifi
from pydantic import ValidationError

from ..agent_schemas import AgentDecision
from .base import ProviderConfigurationError, ProviderRequestError

logger = logging.getLogger(__name__)

DEFAULT_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL_NAME = "gemma-4-31b-it"
DEFAULT_MODEL_TIMEOUT = 60
MAX_REQUEST_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 0.5
SYSTEM_INSTRUCTION = (
    "You are the reasoning model in a simulated automotive diagnostic investigation. "
    "Use the actual telemetry and evidence to form and update hypotheses; do not guess a fault from a scenario label. "
    "Choose exactly one registered diagnostic tool by its name, or choose finish only when evidence is sufficient. "
    "The application validates and executes tools. Never claim a tool was executed before its result is present. "
    "Every evidence reference must use an observation ID in the supplied state. "
    "Confidence values must reflect the evidence and may be low when uncertain. "
    "Give concise, user-visible reasoning, not hidden chain-of-thought. "
    "Treat quoted descriptions in the decision shape as type guidance, not literal values; confidence must be a number. "
    "Return only JSON matching the requested decision shape. For finish, diagnosis must be an object with name, "
    "confidence, evidence (one or more existing IDs), and next_action (a concrete recommended diagnostic action); "
    "for a tool action diagnosis must be null."
)
DECISION_FORMAT = {
    "action": "registered tool name or finish",
    "args": {},
    "reason": "string: brief user-visible explanation of the next step",
    "hypotheses": [{
        "name": "string: diagnostic hypothesis",
        "confidence": "number from 0 to 1, calibrated from cited evidence",
        "status": "leading | supported | confirmed | ruled_out",
        "supporting_evidence": ["existing observation ID"],
        "contradicting_evidence": [],
    }],
    "need_more_information": "boolean",
    "diagnosis": {"name": "string", "confidence": "evidence-based number from 0 to 1",
                  "evidence": ["existing observation ID"], "next_action": "string"},
}


class GemmaProvider:
    name = "gemma"

    def __init__(self, api_key: str | None = None, model_name: str | None = None,
                 timeout: float | None = None, api_base_url: str | None = None):
        self.api_key = api_key if api_key is not None else os.getenv("GEMMA_API_KEY")
        if not self.api_key:
            raise ProviderConfigurationError("GEMMA_API_KEY is required when MODEL_PROVIDER=gemma")
        self.model_name = model_name or os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
        self.timeout = timeout if timeout is not None else float(os.getenv("MODEL_TIMEOUT", str(DEFAULT_MODEL_TIMEOUT)))
        if self.timeout <= 0:
            raise ProviderConfigurationError("MODEL_TIMEOUT must be greater than zero")
        self.api_base_url = (api_base_url or os.getenv("GEMMA_API_BASE_URL", DEFAULT_API_BASE_URL)).rstrip("/")
        self.endpoint = f"{self.api_base_url}/models/{quote(self.model_name, safe='')}:generateContent"

    @staticmethod
    def _prompt(state: dict[str, Any], tools: list[dict[str, Any]]) -> str:
        # Whitelist fields. The simulator's injected scenario/fault is never serialized to the provider.
        public_state = {key: state.get(key) for key in (
            "initial_telemetry", "observations", "hypotheses", "previous_actions", "step", "remaining_action_budget"
        )}
        payload = {"investigation": public_state, "available_diagnostic_tools": tools}
        return ("Decision shape: " + json.dumps(DECISION_FORMAT) + "\nInvestigation data: " +
                json.dumps(payload, ensure_ascii=False))

    @classmethod
    def build_payload(cls, state: dict[str, Any], tools: list[dict[str, Any]]) -> dict[str, Any]:
        prompt = SYSTEM_INSTRUCTION + "\n\n" + cls._prompt(state, tools)
        return {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1},
        }

    @staticmethod
    def _safe_text(value: str, api_key: str, limit: int = 2000) -> str:
        value = value.replace(api_key, "[REDACTED]") if api_key else value
        value = re.sub(r"(?i)(x-goog-api-key|authorization)\s*[:=]\s*[^\s,;]+",
                       r"\1: [REDACTED]", value)
        value = re.sub(r'(?i)("?(?:api[_-]?key|x-goog-api-key|authorization)"?\s*:\s*)"?[^",}\s]+"?',
                       r'\1"[REDACTED]"', value)
        if len(value) > limit:
            return value[:limit] + f"… [truncated; total_chars={len(value)}]"
        return value

    @classmethod
    def _payload_summary(cls, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        """Keep request structure visible while redacting all prompt text and secrets."""
        summary: dict[str, Any] = {"top_level_keys": list(payload)}
        system_parts = payload.get("systemInstruction", {}).get("parts", [])
        contents = payload.get("contents", [])
        text_values = [part.get("text", "") for part in system_parts]
        for content in contents:
            text_values.extend(part.get("text", "") for part in content.get("parts", []))
        summary["system_instruction"] = {
            "present": bool(system_parts),
            "parts": len(system_parts),
            "text_chars": sum(len(text) for text in (part.get("text", "") for part in system_parts)),
        }
        summary["contents"] = [
            {"role": content.get("role"), "parts": len(content.get("parts", [])),
             "text_chars": sum(len(part.get("text", "")) for part in content.get("parts", []))}
            for content in contents
        ]
        summary["generation_config"] = payload.get("generationConfig", {})
        summary["text_chars_total"] = sum(len(text) for text in text_values)
        summary["api_key_in_payload"] = bool(api_key and api_key in json.dumps(payload, ensure_ascii=False))
        return summary

    def _diagnostic_request(self, payload: dict[str, Any], body: bytes, headers: dict[str, str]) -> None:
        if os.getenv("GEMMA_DIAGNOSTICS", "0").casefold() not in {"1", "true", "yes", "on"}:
            return
        prompt_chars = sum(
            len(part.get("text", ""))
            for group in [payload.get("systemInstruction", {}).get("parts", [])]
            + [content.get("parts", []) for content in payload.get("contents", [])]
            for part in group
        )
        safe_headers = {key: value for key, value in headers.items()
                        if key.casefold() not in {"x-goog-api-key", "authorization"}}
        logger.info(
            "gemma_request_diagnostic url=%s method=POST headers=%s payload_structure=%s "
            "prompt_chars=%s serialized_bytes=%s timeout_seconds=%s",
            self.endpoint, json.dumps(safe_headers, sort_keys=True),
            json.dumps(self._payload_summary(payload, self.api_key), sort_keys=True),
            prompt_chars, len(body), self.timeout,
        )

    def _diagnostic_response(self, status: int, body: bytes) -> None:
        if os.getenv("GEMMA_DIAGNOSTICS", "0").casefold() not in {"1", "true", "yes", "on"}:
            return
        text = body.decode("utf-8", errors="replace")
        try:
            response = json.loads(text)
            for candidate in response.get("candidates", []):
                content = candidate.get("content", {})
                if "parts" in content:
                    content["parts"] = [part for part in content["parts"] if not part.get("thought")]
            text = json.dumps(response, ensure_ascii=False)
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
        logger.info("gemma_response_diagnostic status=%s body_bytes=%s body=%r",
                    status, len(body), self._safe_text(text, self.api_key))

    @staticmethod
    def _ssl_context() -> ssl.SSLContext:
        return ssl.create_default_context(cafile=certifi.where())

    def _request_once(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-goog-api-key": self.api_key,
        }
        request = Request(
            self.endpoint,
            data=body,
            headers=headers,
            method="POST",
        )
        self._diagnostic_request(payload, body, headers)
        with urlopen(request, timeout=self.timeout, context=self._ssl_context()) as response:
            response_body = response.read()
            self._diagnostic_response(getattr(response, "status", 200), response_body)
            return json.loads(response_body.decode("utf-8"))

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
            try:
                return self._request_once(payload)
            except HTTPError as exc:
                error_body = exc.read()
                self._diagnostic_response(exc.code, error_body)
                retryable = exc.code in {429, 500, 502, 503}
                if not retryable or attempt == MAX_REQUEST_ATTEMPTS:
                    raise
                self._log_transport_error(exc, status=exc.code)
            except (URLError, TimeoutError, OSError) as exc:
                reason = getattr(exc, "reason", exc)
                # A certificate error is a configuration failure, not a transient outage.
                retryable = not isinstance(reason, ssl.SSLCertVerificationError)
                if not retryable or attempt == MAX_REQUEST_ATTEMPTS:
                    raise
                self._log_transport_error(exc)
            delay = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
            logger.warning("gemma_request_retry provider=gemma model=%s attempt=%s/%s backoff_seconds=%.1f",
                           self.model_name, attempt, MAX_REQUEST_ATTEMPTS, delay)
            time.sleep(delay)

    @staticmethod
    def _decision_text(response: dict[str, Any]) -> str:
        parts = response["candidates"][0]["content"]["parts"]
        # Gemma can return an internal thought part before the user-facing JSON.
        # Only parse visible output; concatenating the thought text makes valid JSON fail.
        text = "\n".join(part.get("text", "") for part in parts if not part.get("thought")).strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        start = text.find("{")
        if start > 0:
            value, _end = json.JSONDecoder().raw_decode(text[start:])
            return json.dumps(value, ensure_ascii=False)
        return text

    def _log_transport_error(self, exc: BaseException, status: int | None = None) -> None:
        """Log useful transport diagnostics while redacting credentials and auth headers."""
        message = str(exc) or ("request timed out" if isinstance(exc, TimeoutError) else "no exception message")
        if self.api_key:
            message = message.replace(self.api_key, "[REDACTED]")
        message = re.sub(r"(?i)(x-goog-api-key|authorization)\s*[:=]\s*[^\s,;]+",
                         r"\1: [REDACTED]", message)
        reason = getattr(exc, "reason", None)
        reason_class = type(reason).__name__ if reason is not None else "none"
        reason_message = str(reason) if reason is not None else ""
        if self.api_key:
            reason_message = reason_message.replace(self.api_key, "[REDACTED]")
        reason_message = re.sub(r"(?i)(x-goog-api-key|authorization)\s*[:=]\s*[^\s,;]+",
                                r"\1: [REDACTED]", reason_message)
        hostname = urlsplit(self.endpoint).hostname or "unknown"
        logger.warning(
            "provider_transport_error provider=gemma model=%s exception_class=%s exception_message=%r "
            "reason_class=%s reason_message=%r target_hostname=%s http_status=%s",
            self.model_name, type(exc).__name__, message, reason_class, reason_message,
            hostname, status if status is not None else getattr(exc, "code", None),
        )

    async def decide(self, state: dict[str, Any], tools: list[dict[str, Any]]) -> AgentDecision:
        payload = self.build_payload(state, tools)
        started = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(self._request, payload),
                timeout=self.timeout * MAX_REQUEST_ATTEMPTS + 2,
            )
        except asyncio.TimeoutError as exc:
            code, message = "timeout", "Gemma API unavailable (request timed out)"
            self._log_transport_error(exc)
            raise ProviderRequestError(code, message) from exc
        except HTTPError as exc:
            if exc.code in (401, 403):
                code, message = "authentication_error", "Gemma API rejected the configured credentials"
            elif exc.code == 429:
                code, message = "rate_limited", "Gemma API rate limit reached"
            else:
                code, message = "api_error", f"Gemma API request failed (HTTP {exc.code})"
            self._log_transport_error(exc, status=exc.code)
            raise ProviderRequestError(code, message) from exc
        except (URLError, TimeoutError, OSError) as exc:
            self._log_transport_error(exc)
            raise ProviderRequestError("network_error", "Gemma API unavailable (network request failed)") from exc
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, IndexError, TypeError) as exc:
            logger.warning("provider_request_failed provider=gemma model=%s reason=malformed_response", self.model_name)
            raise ProviderRequestError("malformed_response", "Gemma returned an unreadable response") from exc

        try:
            text = self._decision_text(response)
            decision = AgentDecision.model_validate_json(text)
        except (ValidationError, ValueError, KeyError, IndexError, TypeError) as exc:
            logger.warning("provider_request_failed provider=gemma model=%s reason=invalid_structured_response", self.model_name)
            raise ProviderRequestError("invalid_structured_response", "Gemma returned an invalid structured decision") from exc

        logger.info("provider_request_succeeded provider=gemma model=%s latency_ms=%.1f",
                    self.model_name, (time.perf_counter() - started) * 1000)
        return decision

    async def answer_question(self, question: str, context: dict[str, Any], knowledge: list[dict[str, Any]]) -> str:
        prompt = ("You are a vehicle AI assistant. Answer the user's vehicle question clearly and cautiously. "
                  "Ground claims in the supplied local reference and current simulated vehicle context. "
                  "Do not claim a confirmed real-world diagnosis. Recommend a qualified mechanic for safety-critical issues.\n"
                  + json.dumps({"question": question[:2000], "vehicle_context": context,
                                "automotive_reference": knowledge}, ensure_ascii=False))
        payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                   "generationConfig": {"temperature": 0.3}}
        try:
            response = await asyncio.wait_for(asyncio.to_thread(self._request, payload),
                                              timeout=self.timeout * MAX_REQUEST_ATTEMPTS + 2)
            parts = response["candidates"][0]["content"]["parts"]
            answer = "\n".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
            if not answer:
                raise ValueError("Empty assistant response")
            return answer[:6000]
        except HTTPError as exc:
            raise ProviderRequestError("api_error", f"Gemma API request failed (HTTP {exc.code})") from exc
        except asyncio.TimeoutError as exc:
            raise ProviderRequestError("timeout", "Gemma API unavailable (request timed out)") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ProviderRequestError("network_error", "Gemma API unavailable (network request failed)") from exc
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ProviderRequestError("malformed_response", "Gemma returned an unreadable assistant response") from exc

    async def health_check(self) -> dict[str, Any]:
        request = Request(f"{self.api_base_url}/models/{quote(self.model_name, safe='')}",
                          headers={"x-goog-api-key": self.api_key, "Accept": "application/json"})
        try:
            check_timeout = min(self.timeout, 5.0)
            await asyncio.wait_for(asyncio.to_thread(self._get, request, check_timeout), timeout=check_timeout + 0.5)
            return {"provider": self.name, "model": self.model_name, "status": "available"}
        except Exception:
            return {"provider": self.name, "model": self.model_name, "status": "unavailable"}

    @staticmethod
    def _get(request: Request, timeout: float) -> None:
        context = GemmaProvider._ssl_context()
        with urlopen(request, timeout=timeout, context=context) as response:
            response.read()

    async def aclose(self) -> None:
        return None
