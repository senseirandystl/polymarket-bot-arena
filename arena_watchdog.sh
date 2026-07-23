#!/bin/bash
# Watchdog for the arena process.
# Restarts the arena (via launchd) if its log file hasn't advanced in 5 minutes
# — a proxy for "the process is hung / dead". Wire it to a cron or launchd
# StartInterval job; it's a single-shot check, safe to run every minute.
#
# Paths resolve RELATIVE TO THIS SCRIPT (it lives at the repo root), so a fresh
# clone works with no edits — the previous hardcoded /Users/ben/... paths were
# dead on any other host (the exact bug class CLAUDE.md warns about). Or probe
# the dashboard's unauthenticated liveness endpoint instead:
#   curl -s localhost:8501/healthz | jq .arena_log_stale

REPO_ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
LOG="$REPO_ROOT/logs/arena.log"
WATCHDOG_LOG="$REPO_ROOT/logs/arena_watchdog.log"
STALE_SECONDS="${ARENA_LOG_STALE_SEC:-300}"  # 5 minutes; matches /healthz default

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1" >> "$WATCHDOG_LOG"
}

if [ ! -f "$LOG" ]; then
    log "WARN: $LOG not found, skipping check"
    exit 0
fi

# Get seconds since last modification
if [[ "$OSTYPE" == "darwin"* ]]; then
    LAST_MOD=$(stat -f %m "$LOG")
else
    LAST_MOD=$(stat -c %Y "$LOG")
fi
NOW=$(date +%s)
AGE=$((NOW - LAST_MOD))

if [ "$AGE" -gt "$STALE_SECONDS" ]; then
    log "RESTART: arena.log is ${AGE}s stale (threshold: ${STALE_SECONDS}s), restarting arena"
    launchctl kickstart -k "gui/$(id -u)/com.polymarket.botarena" >> "$WATCHDOG_LOG" 2>&1
    log "Restart command sent"
else
    log "OK: arena.log updated ${AGE}s ago"
fi
