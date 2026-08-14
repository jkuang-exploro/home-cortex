# Home Cortex API

This package provides a minimal FastAPI service between Open WebUI, SurrealDB,
and Ollama.

## Endpoints

- `GET /health` checks SurrealDB and Ollama.
- `POST /admin/ingest` imports `/app/data/nodes/*.json` and
  `/app/data/edges/*.json`. Re-running it updates nodes and replaces matching
  relation pairs.
- `POST /v1/retrieve` returns the graph context used for a question.
- `GET /v1/models` and `POST /v1/chat/completions` form the OpenAI-compatible
  interface used by Open WebUI.

## First run

From `docker/cortex` on the server:

```sh
docker compose up -d --build
curl http://localhost:8001/health
curl -X POST http://localhost:8001/admin/ingest
curl -X POST http://localhost:8001/v1/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query":"Who resides at Fort Cerritos?"}'
```

Open the interactive API documentation at
`http://192.168.68.59:8001/docs`.

