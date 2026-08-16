# Home Cortex API

This package provides a FastAPI RAG service over SurrealDB. Ollama stays a
separate LAN endpoint; the model calls this API when it needs home facts.

## Endpoints

- `GET /health` checks SurrealDB.
- `POST /admin/ingest` imports `/app/data/nodes/*.json` and
  `/app/data/edges/*.json`. Re-running it updates nodes and relationships by
  stable record ID without creating duplicates.
- `POST /v1/retrieve` returns the graph context used for a question.

## First run

From `docker/cortex` on the server:

```sh
docker compose up -d --build
curl http://localhost:8001/health
curl -X POST http://localhost:8001/admin/ingest
curl -X POST http://localhost:8001/v1/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query":"Fort Cerritos"}'
```

Open the interactive API documentation at
`http://192.168.68.59:8001/docs`.
