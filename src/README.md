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
```

Open the interactive API documentation at
`http://192.168.68.59:8001/docs`.
