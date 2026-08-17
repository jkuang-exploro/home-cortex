# Home Cortex API

This package provides a FastAPI RAG service over SurrealDB. The API calls
Ollama, dispatches the model's allowlisted graph tools, and returns the final
grounded answer.

## Endpoints

- `GET /health` checks SurrealDB.
- `POST /admin/ingest` imports `/app/data/nodes/*.json` and
  `/app/data/edges/*.json`. Re-running it updates nodes and relationships by
  stable record ID without creating duplicates.
- `POST /v1/retrieve` returns the graph context used for a question.
- `POST /v1/chat` runs the default steward agent for backward compatibility.
- `POST /agent/steward/chat` invokes the named household steward directly.
- `GET /v1/models` advertises `The Butler` to OpenAI-compatible clients.
- `POST /v1/chat/completions` provides an OpenAI-compatible chat endpoint backed
  by the agent loop. It supports ordinary JSON responses and token-streamed SSE
  responses for clients such as Open WebUI.

For `stream: true`, Cortex consumes Ollama's async response stream on every
agent step. Tool-selection responses stay internal; chunks from the final
answer are forwarded immediately as OpenAI-compatible SSE events. If the client
disconnects, cancellation closes the active agent and Ollama streams and records
a privacy-safe `stream_cancelled` log.

## First run

From `docker/cortex` on the server:

Copy `.env.example` to `.env`, then set `SURREAL_PASS` and `OLLAMA_MODEL` in
that file. `OLLAMA_MODEL` is the single deployment setting used by the Cortex
API when selecting an Ollama model.

```sh
docker compose up -d --build
curl http://localhost:8001/health
curl -X POST http://localhost:8001/admin/ingest
curl -X POST http://localhost:8001/v1/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query":"Fort Cerritos"}'
curl -X POST http://localhost:8001/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Who lives at Fort Cerritos?"}'
curl -X POST http://localhost:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"The Butler","stream":false,"messages":[{"role":"user","content":"Who resides at Fort Cerritos?"}]}'
```

Open the interactive API documentation at
`http://192.168.68.59:8001/docs`.

## Open WebUI

The Compose stack configures two model paths:

- Direct Ollama at `http://ollama:11434` for model debugging.
- The OpenAI-compatible Cortex API at `http://cortex-api:8000/v1` for grounded
  graph answers.

Select `The Butler` in Open WebUI to use the steward and its SurrealDB tools.
The agent refers to itself as `the butler` in English and `老管家` in Chinese.
Selecting a raw Ollama model bypasses Cortex. Open WebUI persists its
connection settings, so an existing deployment may require adding the Cortex
connection once in the administrator connection settings with API key
`unused`.

## Named agents

Home Cortex is the shared platform; named agents are role-specific interfaces
on top of it. The `steward` definition lives under
`home_cortex/agents/steward` and owns its display name, prompt, model preference,
settings, and tool allowlist. The generic loop remains in `agent.py`.

The steward's model is `OLLAMA_MODEL` when its `config.yaml` model name is null.
A future specialized agent can select a different model and tools without
changing the shared runtime. For example, a future `accountant` directory can
define `账房` and finance-only tools; those tools will not be granted to
`steward` automatically.

## Observability

Every HTTP response includes a server-generated `X-Request-ID`. Error responses
use the same JSON envelope for validation errors, API errors, and unexpected
failures:

```json
{
  "error": {
    "code": "internal_server_error",
    "message": "An unexpected server error occurred",
    "request_id": "96f149cf430442d48fb6010899cde986"
  }
}
```

The agent logs each model step, tool name, success status, record count,
execution time, and final stop reason. Stop reasons are `answer`, `step_limit`,
`tool_error`, or `timeout`. Logs intentionally omit prompts, tool arguments,
tool results, and private record fields. View them with:

```sh
docker compose logs -f cortex-api
```
