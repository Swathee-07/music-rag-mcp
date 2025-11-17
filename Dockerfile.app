# syntax=docker/dockerfile:1.7-labs

############################
# Stage 1: Base system (shared)
############################
FROM python:3.11-slim-bookworm AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /install

RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

############################
# Stage 2: Dependency install (cached)
############################
FROM base AS deps
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip wheel setuptools && \
    pip install --prefix /install -r requirements.txt --prefer-binary

############################
# Stage 3: Parallel “fetch” stage (independent)
############################
FROM python:3.11-slim-bookworm AS fetch
WORKDIR /tmp/fetch
RUN --mount=type=cache,target=/tmp/cache \
    echo "Preparing UI assets..." && \
    sleep 3 && \
    echo "Assets ready"

############################
# Stage 4: Final runtime
############################
FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY --from=deps /install /usr/local
COPY --from=fetch /tmp/fetch /assets
COPY . .

EXPOSE 8501
CMD ["streamlit","run","streamlit_app.py","--server.port=8501","--server.address=0.0.0.0"]
