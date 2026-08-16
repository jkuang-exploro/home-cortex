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
S