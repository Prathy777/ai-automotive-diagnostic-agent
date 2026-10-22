"""Dependency-light local API server for the static dashboard."""
import asyncio
import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from app.agent import investigate
from app.knowledge import KNOWLEDGE, search_knowledge
from app.providers.base import ProviderConfigurationError, ProviderRequestError
from app.providers.factory import create_provider, provider_health, provider_settings
from app.simulator import Vehicle

SCENARIOS = {"normal", "misfire", "overheating", "alternator_failure", "wheel_imbalance", "fuel_air_mixture"}
SESSIONS = {}
FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
DEMO = lambda: bool(provider_settings()["demo_mode"])


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status, value):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "http://localhost:5173")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "http://localhost:5173")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        config = {**asyncio.run(provider_health()), "model_timeout": int(os.getenv("MODEL_TIMEOUT", "60")),
                  "scenarios": sorted(SCENARIOS)}
        if path == "/api/health":
            self.send_json(200, {"status": "ok", **config, "simulator_status": "ready"})
        elif path == "/api/config": self.send_json(200, config)
        elif path == "/api/vehicle": self.send_json(200, {"vehicle_id": "SIM-001", "name": "Digital Twin Vehicle", "type": "simulated passenger car", "environment": "simulation"})
        elif path == "/api/telemetry": self.send_json(200, {"vehicle_id": "SIM-001", "telemetry": Vehicle().telemetry(), "environment": "simulation"})
        elif path == "/api/knowledge":
            query = parse_qs(parsed.query).get("q", parse_qs(parsed.query).get("code", [""]))[0]
            self.send_json(200, {"items": search_knowledge(query) if query else KNOWLEDGE})
        elif path.startswith("/api/investigations/"):
            result = SESSIONS.get(path.rsplit("/", 1)[-1])
            self.send_json(200, result) if result else self.send_json(404, {"detail": "Investigation not found"})
        else:
            relative = "index.html" if path == "/" else path.lstrip("/")
            target = (FRONTEND / relative).resolve()
            if not target.is_relative_to(FRONTEND.resolve()) or not target.is_file():
                return self.send_json(404, {"detail": "Not found"})
            body = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        except (json.JSONDecodeError, ValueError):
            return self.send_json(400, {"detail": "Invalid JSON"})
        if path == "/api/investigations":
            scenario, steps = request.get("scenario"), request.get("max_steps", 8)
            if scenario not in SCENARIOS or isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 8:
                return self.send_json(422, {"detail": "Invalid scenario or max_steps (allowed: 1–8)"})
            result = asyncio.run(investigate(scenario, steps))
            SESSIONS[result["session_id"]] = result
            return self.send_json(200, result)
        if path == "/api/chat":
            message = str(request.get("message", "")).strip()[:2000]
            if not message: return self.send_json(422, {"detail": "message is required"})
            session_id = request.get("session_id")
            session = SESSIONS.get(session_id or "")
            knowledge = search_knowledge(message)
            try:
                model = create_provider("demo" if DEMO() else None)
                context = ({key: session.get(key) for key in ("observations", "hypotheses", "diagnosis", "confidence", "next_action")}
                           if session else {"vehicle": Vehicle().telemetry()})
                try: answer = asyncio.run(model.answer_question(message, context, knowledge))
                finally: asyncio.run(model.aclose())
                provider = model.name
            except (ProviderConfigurationError, ProviderRequestError) as exc:
                return self.send_json(503, {"detail": str(exc), "code": getattr(exc, "code", "provider_configuration_error")})
            return self.send_json(200, {"answer": answer, "provider": provider, "session_id": session_id,
                                        "knowledge_references": [item["id"] for item in knowledge],
                                        "suggest_investigation": any(w in message.lower() for w in ("investigate", "diagnose", "find the cause"))})
        self.send_json(404, {"detail": "Not found"})

    def log_message(self, format, *args):
        print("%s - %s" % (self.address_string(), format % args), flush=True)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8001"))
    print(f"Automind API listening on http://127.0.0.1:{port} (demo_mode={DEMO()})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
