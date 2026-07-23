# Polymarket Bot Arena — production image (arena + dashboard share this image).
#
# Build:
#   docker compose build
# Run:
#   docker compose up -d
# See docs/docker.md for local and VPS deployment.

FROM python:3.12-slim-bookworm

LABEL org.opencontainers.image.title="polymarket-bot-arena" \
      org.opencontainers.image.description="Polymarket BTC 5-min bot arena + dashboard" \
      org.opencontainers.image.source="https://github.com/senseirandystl/polymarket-bot-arena"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Persist DB / logs / secrets on a volume (see docker-compose.yml)
    ARENA_DB_PATH=/data/bot_arena.db \
    ARENA_LOG_DIR=/data/logs \
    ARENA_CREDENTIALS_FILE=/data/secrets/credentials.enc \
    ARENA_CREDENTIALS_KEY_FILE=/data/secrets/arena_fernet.key \
    # When the dashboard is a sibling compose service, arena must not spawn a second one
    ARENA_NO_DASHBOARD=1 \
    DASHBOARD_HOST=0.0.0.0 \
    DASHBOARD_PORT=8501

# ca-certificates for HTTPS (CLOB / Gamma / Binance); tzdata for America/New_York day boundaries
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for better layer caching
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install tzdata

# Application source (keep .dockerignore tight so this stays small)
COPY . .

# Non-root runtime user; data volume is re-chowned at entry if needed
RUN useradd --create-home --uid 1000 --shell /bin/bash arena \
    && mkdir -p /data/logs /data/secrets \
    && chown -R arena:arena /app /data

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Entrypoint runs as root only long enough to chown the data volume, then
# drops to user "arena" (uid 1000) via runuser. See docker/entrypoint.sh.
USER root

VOLUME ["/data"]

EXPOSE 8501

ENTRYPOINT ["/entrypoint.sh"]
# Default command is the arena; compose overrides for the dashboard service.
CMD ["python", "arena.py"]
