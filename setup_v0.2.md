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
-- curl -sS -X POST http://localhost:8001/admin/ingest \
  -H 'Authorization: Bearer replace-with-a-long-random-secret' | jq

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

Add a contextual `household_role` to the Person's edge to the home. Do not add
`is_guest` or `person_type: guest` to the Person itself. For example:

```json
{
  "from": "person:jian_kuang",
  "to": "location:fort_cerritos",
  "residence_type": "primary",
  "household_role": "owner"
}
```

After editing an edge role, re-ingest it:

```sh
curl -sS -X POST http://localhost:8001/admin/ingest \
  -H 'Authorization: Bearer replace-with-a-long-random-secret' | jq
```

Verify the steward's deterministic Chinese greeting:

```sh
curl -sS -X POST http://localhost:8001/agent/steward/conversations \
  -H 'Authorization: Bearer replace-with-a-long-random-secret' \
  -H 'X-OpenWebUI-User-Email: your-login@example.com' \
  -H 'Content-Type: application/json' \
  -d '{"language":"zh"}' | jq
```

An owner mapped to Jian should receive:

```text
先生，您回来了。老管家在此，今日有什么需要吩咐？
```

The OpenAI-compatible endpoint also detects a first turn as one user message
with no previous assistant message. Its first answer includes the deterministic
greeting; later requests carrying chat history do not repeat it.

Open WebUI persists connection settings. If the existing volume already has an
OpenAI configuration, sign in as an administrator and add or update an
OpenAI-compatible connection with:

```text
Base URL: http://cortex-api:8000/v1
API key: the value of CORTEX_API_KEY
```

Recreate Open WebUI without deleting its data volume:

```sh
docker compose build open-webui
docker compose up -d --force-recreate --no-deps open-webui
```

Select `老管家` in the model picker. Selecting `qwen3:8b` or another raw
Ollama model bypasses Cortex, SurrealDB retrieval, and the agent tools.

The Compose file builds a small customization on top of the pinned Open WebUI
v0.9.5 image. On a blank new chat, selecting `老管家` calls Cortex's steward
conversation endpoint and inserts the returned greeting as the first assistant
message. No user prompt is required. The greeting is saved in normal Open WebUI
chat history and is included in the first later request, so Cortex does not
greet twice.

The browser authenticates only to Open WebUI. A same-origin Open WebUI backend
route forwards the verified user's immutable ID and email to Cortex and adds
`CORTEX_API_KEY` on the server. The Cortex API key is never sent to browser
JavaScript.

Compose sets `CORTEX_GREETING_LANGUAGE: zh`, so the proactive initial greeting
is always Chinese even when Open WebUI itself is displayed in English. Later
answers continue to follow the language of the user's request.

After changing the Open WebUI customization, rebuild it explicitly:

```sh
docker compose build --no-cache open-webui
docker compose up -d --force-recreate --no-deps open-webui
docker compose logs --tail=100 open-webui
```

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
