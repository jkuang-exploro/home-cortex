FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
COPY scripts ./scripts
COPY schemas ./schemas
COPY benchmarks ./benchmarks
RUN pip install --no-cache-dir .

CMD ["uvicorn", "home_cortex.api:app", "--host", "0.0.0.0", "--port", "8000"]
