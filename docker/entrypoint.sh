#!/bin/sh
# Prepare the /data volume then drop privileges to the arena user when started as root.
set -eu

DB_PATH="${ARENA_DB_PATH:-/data/bot_arena.db}"
LOG_DIR="${ARENA_LOG_DIR:-/data/logs}"
CRED_FILE="${ARENA_CREDENTIALS_FILE:-/data/secrets/credentials.enc}"
KEY_FILE="${ARENA_CREDENTIALS_KEY_FILE:-/data/secrets/arena_fernet.key}"

mkdir -p "$LOG_DIR" \
         "$(dirname "$DB_PATH")" \
         "$(dirname "$CRED_FILE")" \
         "$(dirname "$KEY_FILE")"

if [ "$(id -u)" = "0" ]; then
    # Bind mounts often arrive root-owned; fix ownership once per start.
    chown -R arena:arena /data 2>/dev/null || true
    exec runuser -u arena -- "$@"
fi

exec "$@"
