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

Connect the household GUI to Cortex

Compose publishes the GUI at port 3000 through an independent nginx proxy.
The GUI container serves static files; Cortex owns sessions, transcripts, and
answers. Port `8001` remains the host mapping for curl.

Set a shared API key and map the email used to sign in in
`docker/cortex/.env`:

```dotenv
CORTEX_API_KEY=replace-with-a-long-random-secret
CORTEX_IDENTITY_MAP={"email:your-login@example.com":"person:jian_kuang"}
```

You can instead use a stable user id as the map key:

```dotenv
CORTEX_IDENTITY_MAP={"id:household-user-uuid":"person:jian_kuang"}
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
  "to": "address:fort_cerritos",
  "residence_type": "primary",
  "household_role": "owner"
}
```

After editing an edge role, re-ingest it:

```sh
curl -sS -X POST http://localhost:8001/admin/ingest -H "Authorization: Bearer ${CORTEX_API_KEY}" | jq
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

Open http://localhost:3000 and sign in with the mapped email plus
`CORTEX_API_KEY`. The GUI stores a session cookie, not the key. Select `老管家`
for the steward or a bare Ollama/OpenRouter model to test the underlying LLM.

Rebuild the GUI after frontend changes:

```sh
docker compose build home-gui
docker compose up -d --force-recreate --no-deps home-gui proxy
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

Streaming:

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

Checking user access
curl -sS -X POST \
  "http://localhost:3000/api/v1/users $user_id/update" -H "Authorization: Bearer $OPEN_WEBUI_TOKEN" -H 'Content-Type: application/json' -d '{"role":"user"}' jq '{id,email,role}'

Granting Butler Access
User_ID='xxxxxx'

jq -n \
  --arg user_id "$User_ID" \
  '{
    id: "老管家",
    name: "老管家",
    access_grants: [
      {
        principal_type: "user",
        principal_id: $user_id,
        permission: "read"
      }
    ]
  }' \
| curl --fail-with-body -sS -X POST \
    'http://localhost:3000/api/v1/models/model/access/update' \
    -H "Authorization: Bearer $OPEN_WEBUI_TOKEN" \
    -H 'Content-Type: application/json' \
    --data-binary @- \
| jq '{id, name, is_active, access_grants}'