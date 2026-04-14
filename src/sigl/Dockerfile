FROM python:3.10-slim
ARG POETRY_VERSION=2.2.1
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

WORKDIR /app/src/sigl

COPY pyproject.toml poetry.lock* ./

RUN poetry config virtualenvs.create false

RUN poetry install --no-ansi --no-root --only main

RUN pip install torch==1.13.1 torchvision==0.14.1 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.1+cpu.html
RUN pip install torch-sparse -f https://data.pyg.org/whl/torch-1.13.1+cpu.html

COPY . .

CMD ["python", "Main.py"]