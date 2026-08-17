Create DB
docker compose --env-file .env -f docker-compose.yml up -d surrealdb

Start FastAPI:
SURREAL_URL=ws://localhost:8000 \
uv run --env-file docker/cortex/.env \
uvicorn home_cortex.api:app \
  --host 127.0.0.1 \
  --port 8001 \
  --reload

Deployment

update cortex-api
-- docker compose build --no-cache cortex-api
-- docker compose up -d --force-recreate --no-deps cortex-api
-- curl -sS -X POST http://localhost:8001/admin/ingest | jq

Connect Open WebUI to Cortex

The Compose configuration keeps the direct Ollama connection for debugging and
adds Cortex as an OpenAI-compatible connection. Open WebUI reaches Cortex over
the internal Compose network at `http://cortex-api:8000/v1`; port `8001` is the
host-side mapping and must not be used between containers.

For a new Open WebUI data volume, the connection is seeded by these environment
variables:

```yaml
ENABLE_OPENAI_API: "true"
OPENAI_API_BASE_URLS: http://cortex-api:8000/v1
OPENAI_API_KEYS: unused
```

Open WebUI persists connection settings. If the existing volume already has an
OpenAI configuration, sign in as an administrator and add or update an
OpenAI-compatible connection with:

```text
Base URL: http://cortex-api:8000/v1
API key: unused
```

Recreate Open WebUI without deleting its data volume:

```sh
docker compose up -d --force-recreate --no-deps open-webui
```

Select `The Butler` in the model picker. Selecting `qwen3:8b` or another raw
Ollama model bypasses Cortex, SurrealDB retrieval, and the agent tools.

Verify the Cortex model endpoint from the Compose network:

```sh
docker compose exec open-webui python -c \
  'import requests; print(requests.get("http://cortex-api:8000/v1/models").json())'
```

Then ask `Who resides at Fort Cerritos?` using `The Butler` and compare the
answer with:

```sh
curl -sS -X POST http://localhost:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"The Butler","stream":false,"messages":[{"role":"user","content":"Who resides at Fort Cerritos?"}]}' | jq
```

Open WebUI normally requests streaming responses. Verify its request shape with:

```sh
curl -N -sS -X POST http://localhost:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"The Butler","stream":true,"messages":[{"role":"user","content":"Who resides at Fort Cerritos?"}]}'
```

Cortex keeps tool-selection responses internal and forwards each final-answer
chunk from Ollama as an OpenAI-compatible SSE event. The stream ends with a
chunk whose `finish_reason` is `stop`, followed by `data: [DONE]`.
