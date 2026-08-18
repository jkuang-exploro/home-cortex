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
ENABLE_FORWARD_USER_INFO_HEADERS: "true"
OPENAI_API_BASE_URLS: http://cortex-api:8000/v1
OPENAI_API_KEYS: ${CORTEX_API_KEY}
```

Set a shared API key and map the email used by your Open WebUI account in
`docker/cortex/.env`:

```dotenv
CORTEX_API_KEY=replace-with-a-long-random-secret
CORTEX_IDENTITY_MAP={"email:your-login@example.com":"person:jian_kuang"}
```

You can instead use the immutable Open WebUI user ID as the map key:

```dotenv
CORTEX_IDENTITY_MAP={"id:open-webui-user-uuid":"person:jian_kuang"}
```

For each chat request, Cortex resolves the mapped Person before calling the
model and supplies its stored `name` and optional `address_as` as trusted
identity context. The mapped Person must therefore already exist in SurrealDB;
run `/admin/ingest` after changing person data.

Open WebUI persists connection settings. If the existing volume already has an
OpenAI configuration, sign in as an administrator and add or update an
OpenAI-compatible connection with:

```text
Base URL: http://cortex-api:8000/v1
API key: the value of CORTEX_API_KEY
```

Recreate Open WebUI without deleting its data volume:

```sh
docker compose up -d --force-recreate --no-deps open-webui
```

Select `老管家` in the model picker. Selecting `qwen3:8b` or another raw
Ollama model bypasses Cortex, SurrealDB retrieval, and the agent tools.

Verify the Cortex model endpoint from the Compose network:

```sh
docker compose exec open-webui python -c \
  'import os, requests; print(requests.get("http://cortex-api:8000/v1/models", headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEYS"]}).json())'
```

Then ask `Where do I live?` using `老管家` and compare the
answer with:

```sh
curl -sS -X POST http://localhost:8001/v1/chat/completions \
  -H 'Authorization: Bearer replace-with-a-long-random-secret' \
  -H 'X-OpenWebUI-User-Email: your-login@example.com' \
  -H 'Content-Type: application/json' \
  -d '{"model":"老管家","stream":false,"messages":[{"role":"user","content":"Where do I live?"}]}' | jq
```

Open WebUI normally requests streaming responses. Verify its request shape with:

```sh
curl -N -sS -X POST http://localhost:8001/v1/chat/completions \
  -H 'Authorization: Bearer replace-with-a-long-random-secret' \
  -H 'X-OpenWebUI-User-Email: your-login@example.com' \
  -H 'Content-Type: application/json' \
  -d '{"model":"老管家","stream":true,"messages":[{"role":"user","content":"Where do I live?"}]}'
```

Cortex keeps tool-selection responses internal and forwards each final-answer
chunk from Ollama as an OpenAI-compatible SSE event. The stream ends with a
chunk whose `finish_reason` is `stop`, followed by `data: [DONE]`.
