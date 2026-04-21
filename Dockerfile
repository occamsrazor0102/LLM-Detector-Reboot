FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY beet ./beet
COPY configs ./configs

RUN pip install --no-cache-dir -e ".[api]"

RUN mkdir -p /data/vault /data/monitoring /app/models

ENV BEET_CONFIG=/app/configs/production.yaml
EXPOSE 8000

CMD ["uvicorn", "beet.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
