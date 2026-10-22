# Vehicle AI Diagnostic Agent

A simulation-first diagnostic agent that observes vehicle telemetry, selects registered tests, updates hypotheses, and presents an evidence-linked result. **Google-hosted Gemma is the only production LLM.** A deterministic Demo Mode keeps the complete investigation usable without an API key or network access.

> This system operates on simulated vehicle data and provides informational output only. It is not a substitute for professional inspection or manufacturer service procedures.

## What it does

A typical vehicle chatbot lists possible causes. This application runs a bounded investigation: Gemma proposes one next action, the Agent Harness validates it, and the Tool Registry executes the selected diagnostic against a deterministic vehicle digital twin. The observation returns to Gemma for its next decision. The harness—not the model—owns the session, safety limits, evidence, and final validation.

```text
Vehicle Assistant ─────┐
                       ├─ FastAPI ─ Agent Harness ─ GemmaProvider ─ Google Generative API
Diagnostic Agent ──────┘                  │               └──── DemoProvider (explicit only)
                                          ▼
                                    Tool Registry
                                          ▼
                                  Vehicle Simulator
                                          ▼
                              Observations / Hypotheses
                                          ▼
                                   Evidence / Diagnosis

Separate developer analysis: repository ─ Graphify ─ graph.html / graph.json / GRAPH_REPORT.md
```

## Components

- **Agent Harness:** maintains investigation state and action history, validates every structured response, enforces the eight-step limit, rejects unknown/repeated tools and invalid evidence, and prevents premature conclusions.
- **GemmaProvider:** calls `gemma-4-31b-it` over Google’s GenerateContent REST API using Python `urllib`, `certifi`, and the `x-goog-api-key` header. It filters thought parts, parses JSON/fenced JSON, validates decisions with Pydantic, applies bounded retries to transient errors, and never silently changes providers.
- **DemoProvider:** explicit deterministic decision source that uses the same harness, simulator, real diagnostic tools, observations, hypotheses, and evidence. The UI labels it **Demo Simulation Mode**; it never claims to be Gemma.
- **Vehicle Simulator:** deterministic scenarios for normal operation, misfire, overheating, alternator failure, wheel imbalance, and fuel/air mixture faults. Hidden scenario labels are not serialized into provider prompts.
- **Tool Registry:** allowlisted telemetry inspection and diagnostic tests. Model output cannot execute Python, shell commands, or arbitrary code.
- **Vehicle Assistant:** Gemma-backed, vehicle-focused answers use the local DTC/symptom/maintenance reference layer and current telemetry or session context. Demo answers are clearly labeled.
- **Graphify:** local repository/code knowledge graph for developers only. It is not imported by or required by the application and receives no vehicle telemetry.

## Local setup

Requires Python 3.12.

```bash
[ -f .env ] || cp .env.example .env
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Run the deterministic demo

No API key or hosted model is needed:

```bash
cd backend
MODEL_PROVIDER=demo DEMO_MODE=true python run_local.py
```

Open <http://localhost:8001>. The default misfire path observes telemetry, calls `inspect_recent_telemetry`, runs `run_2000_rpm_test` and `run_mixture_test`, then returns a result with evidence references and a safe recommended next action.

### Run with hosted Gemma

Set the following in the root `.env`:

```dotenv
MODEL_PROVIDER=gemma
MODEL_NAME=gemma-4-31b-it
GEMMA_API_KEY=your_google_generative_language_api_key
MODEL_TIMEOUT=60
DEMO_MODE=false
```

Load the variables and start the FastAPI application:

```bash
cd backend
set -a
source ../.env
set +a
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

Optional checks, run from `backend/` with `.env` loaded:

```bash
python -m app.smoke_gemma
python gemma_http_probe.py
```

The smoke test makes a real decision request. The HTTP probe reports status and a redacted response; neither prints the API key. Google may return transient errors, so the application retries only a bounded number of times and reports a clear failure. It does not fall back to Demo Mode. Select Demo explicitly to run offline.

### Environment variables

| Variable | Purpose |
| --- | --- |
| `MODEL_PROVIDER` | `gemma` (default) or `demo` |
| `MODEL_NAME` | Hosted Gemma model; default `gemma-4-31b-it` |
| `GEMMA_API_KEY` | Private Google Generative Language API key |
| `GEMMA_API_BASE_URL` | API base; default `https://generativelanguage.googleapis.com/v1beta` |
| `MODEL_TIMEOUT` | Per-request timeout in seconds; default `60` |
| `DEMO_MODE` | Explicit deterministic demo override (`true`/`false`) |
| `PORT` | Server port; Docker defaults to `8000`, local helper to `8001` |

Keep `.env` private. The API key is only sent to Google in the `x-goog-api-key` request header; it is not returned by health/config endpoints or sent to the frontend.

## API

- `GET /api/health` — application, simulator, provider, model, and verified provider status; no secrets.
- `GET /api/config` — provider status and available scenarios.
- `GET /api/vehicle`, `GET /api/telemetry`
- `GET /api/knowledge?q=...` or `?code=P0301`
- `POST /api/investigations` — for example `{"scenario":"misfire","max_steps":8}`.
- `GET /api/investigations/{session_id}`
- `POST /api/chat` — for example `{"message":"What does P0301 mean?","session_id":null}`.

Investigation state is held in memory and is cleared when the application restarts.

## Tests

```bash
cd backend
python -m unittest discover -s tests -v
python -m compileall -q app run_local.py gemma_http_probe.py gemma_progressive_test.py gemma_investigation_repro.py
node --check ../frontend/src/main.js
```

The test suite mocks the Gemma HTTP boundary and does not need a live key. It covers request construction, prompt privacy, thought filtering, JSON parsing, malformed responses, retries and errors, harness safety, evidence validation, simulator tools, and Demo Mode.

## CPU hosting / DigitalOcean

The application needs no GPU, CUDA, model weights, inference server, or Google SDK. Gemma runs remotely at Google; the DigitalOcean host only runs the CPU-compatible application container.

On an Ubuntu CPU Droplet with Docker Engine and Compose installed:

1. Clone this repository and create `.env` using the values above.
2. Permit only the required inbound ports in the host and DigitalOcean firewalls. Put HTTPS behind a TLS reverse proxy before public use.
3. Start and inspect the app:

```bash
docker compose up --build -d
docker compose ps
curl -fsS http://127.0.0.1:8000/api/health
docker compose logs -f automind
```

`Dockerfile` and `docker-compose.yml` run FastAPI, the harness, simulator, tools, assistant, and static dashboard. No DigitalOcean resources were created or deployed during this code task. Keep credentials out of Git and rotate any key that has been exposed.

## Repository knowledge graph (separate developer tool)

Graphify is a repository-analysis tool, not a production dependency. It is never called by the harness and never receives telemetry. The generated `graphify-out/graph.html`, `graphify-out/graph.json`, and `graphify-out/GRAPH_REPORT.md` describe this codebase. Regenerate them locally after source changes with:

```bash
graphify . --code-only --no-cluster
graphify cluster-only . --no-label
```

Code-only extraction avoids sending source or documentation to an LLM; `--no-label` keeps community naming local and deterministic. Cache and transient analysis metadata are ignored.

## Safety boundary

All vehicle values and test results are simulated. The agent does not connect to CAN/OBD hardware, issue repair commands, or validate a real fault. Treat outputs as educational and seek qualified inspection for real vehicle symptoms, especially safety-critical ones.
