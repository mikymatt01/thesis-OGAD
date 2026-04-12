FROM --platform=linux/amd64 python:3.11-slim
ARG POETRY_VERSION=2.2.1
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

WORKDIR /app/src/d3pm
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-ansi --no-root && rm -rf "$POETRY_CACHE_DIR"

COPY . .

CMD ["poetry", "run", "python", "ogad_model_finetune.py"]
