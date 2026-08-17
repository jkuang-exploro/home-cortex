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
- `POST /v1/chat` runs a bounded Ollama tool-calling loop over the graph.
- `GET /v1/models` advertises the `home-cortex` virtual model.
- `POST /v1/chat/completions` provides a non-streaming OpenAI-compatible chat
  endpoint backed by the agent loop.

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
  -d '{"model":"home-cortex","stream":false,"messages":[{"role":"user","content":"Who resides at Fort Cerritos?"}]}'
```

Open the interactive API documentation at
`http://192.168.68.59:8001/docs`.

## Open WebUI

The Compose stack configures two model paths:

- Direct Ollama at `http://ollama:11434` for model debugging.
- The OpenAI-compatible Cortex API at `http://cortex-api:8000/v1` for grounded
  graph answers.

Select `home-cortex` in Open WebUI to use the Cortex agent and its SurrealDB
tools. Selecting a raw Ollama model bypasses Cortex. Open WebUI persists its
connection settings, so an existing deployment may require adding the Cortex
connection once in the administrator connection settings with API key
`unused`.
