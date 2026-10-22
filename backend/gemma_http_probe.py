"""One-request, standard-library probe for the hosted Gemma GenerateContent endpoint."""
import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


def _redact(text: str, api_key: str) -> str:
    if api_key:
        text = text.replace(api_key, "[REDACTED]")
    text = re.sub(r'(?i)("?(?:api[_-]?key|x-goog-api-key|authorization)"?\s*:\s*)"?[^",}\s]+"?',
                  r'\1"[REDACTED]"', text)
    text = re.sub(r"(?i)(x-goog-api-key|authorization)\s*[:=]\s*[^\s,;]+",
                  r"\1: [REDACTED]", text)
    return text


def main() -> int:
    api_key = os.getenv("GEMMA_API_KEY", "")
    if not api_key:
        print("HTTP status: not sent")
        print("Response body: GEMMA_API_KEY environment variable is required")
        return 2

    model = os.getenv("MODEL_NAME", "gemma-4-31b-it")
    base_url = os.getenv("GEMMA_API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
    url = f"{base_url}/models/{quote(model, safe='')}:generateContent"
    payload = {"contents": [{"parts": [{"text": "Return a simple JSON diagnostic decision."}]}]}
    request = Request(url, data=json.dumps(payload).encode("utf-8"), method="POST", headers={
        "Content-Type": "application/json",
        "Accept": "application/json",
        "x-goog-api-key": api_key,
    })

    try:
        with urlopen(request, timeout=float(os.getenv("MODEL_TIMEOUT", "60"))) as response:
            status = response.status
            body = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        status = exc.code
        body = exc.read().decode("utf-8", errors="replace")
    except (URLError, TimeoutError, OSError) as exc:
        status = "unavailable"
        body = f"{type(exc).__name__}: {exc}"

    print(f"HTTP status: {status}")
    print(f"Response body: {_redact(body, api_key)}")
    return 0 if isinstance(status, int) and 200 <= status < 300 else 1


if __name__ == "__main__":
    raise SystemExit(main())
