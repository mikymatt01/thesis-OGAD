FROM python:3.10-slim
ARG POETRY_VERSION=2.2.1
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

WORKDIR /app/src/sigl
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-ansi --no-root --no-interaction --only main
RUN rm -rf "$POETRY_CACHE_DIR"

RUN poetry run python -m pip install --no-cache-dir \
    torch_sparse -f https://data.pyg.org/whl/torch-1.13.1+cpu.html \
    torch_scatter -f https://data.pyg.org/whl/torch-1.13.1+cpu.html

COPY . .

CMD ["poetry", "run", "python", "Main.py"]
