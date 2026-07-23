# Docker deployment — Polymarket Bot Arena

> Parent overview: [README.md](../README.md) · path to live:
> [README § Path to live](../README.md#path-to-live-trading)

Run the **arena** (trading loop) and **dashboard** (FastAPI UI) as two containers that share one data volume. Designed for reliable 24/7 operation on a laptop or a VPS.

| Path | Purpose |
|------|---------|
| `Dockerfile` | Shared image (Python 3.12 + deps) |
| `docker-compose.yml` | `arena` + `dashboard` services, healthchecks, volumes |
| `.env.example` | Documented environment variables |
| `docker/entrypoint.sh` | Ensures `/data` subdirs exist on first boot |

**Alternatives (still supported):**

- **macOS launchd** — `com.polymarket.botarena.plist` + `com.polymarket.dashboard.plist` (see `CLAUDE.md` → *launchd Services*)
- **Linux systemd** — `deploy/systemd/*.service` (bare metal, no Docker)
- **Terminal** — `./bin/arena` for ad-hoc interactive runs

---

## Architecture

```
┌─────────────────┐     shared volume      ┌──────────────────┐
│  pba-arena      │◄──── /data ───────────►│  pba-dashboard   │
│  python arena.py│   bot_arena.db         │  dashboard/server│
│  (no host port) │   logs/                │  :8501 → host    │
└─────────────────┘   secrets/             └──────────────────┘
```

- **Paper mode** needs **no API keys** — market data is public.
- **Live mode** needs Polymarket CLOB credentials (dashboard **Settings** tab, or pre-seed the secrets volume).
- Arena sets `ARENA_NO_DASHBOARD=1` so it does **not** spawn a second dashboard inside its container.
- SQLite uses **WAL** so the dashboard can read while the arena writes.

Data layout on the host (`PBA_DATA_DIR`, default `./data`):

```
data/
  bot_arena.db              # SQLite (trades, bots, settings)
  bot_arena.db-wal          # WAL sidecar (created at runtime)
  bot_arena.db-shm
  logs/
    arena.log
    dashboard.log           # only if something writes here; uvicorn logs go to docker logs
    KILL_SWITCH             # optional risk kill-switch file
  secrets/
    credentials.enc         # Fernet-encrypted API keys
    arena_fernet.key        # machine key (0600) — backup with credentials.enc
```

---

## Local deployment

### Prerequisites

- [Docker Engine](https://docs.docker.com/engine/install/) + Compose v2 (`docker compose version`)
- ~1 GB free disk; outbound HTTPS to Polymarket + Binance

### First run

```bash
cd /path/to/pba

# 1. Env file (change the dashboard password)
cp .env.example .env
# edit DASHBOARD_PASS=...

# 2. Build + start both services
docker compose up -d --build

# 3. Watch logs
docker compose logs -f arena

# 4. Open the dashboard
open http://127.0.0.1:8501   # or browse to that URL
# Login: DASHBOARD_USER / DASHBOARD_PASS from .env
```

Non-interactive startup (no TTY) **resumes** the previous DB slate, or seeds the **default 8-bot** roster on a fresh DB — same behavior as launchd.

### Common commands

```bash
docker compose ps                    # status + health
docker compose logs -f dashboard     # dashboard logs
docker compose restart arena         # restart trading only
docker compose down                  # stop (keeps ./data)
docker compose down -v               # stop; does NOT delete bind-mounted ./data
docker compose pull                  # n/a for local build
docker compose up -d --build         # rebuild after code changes
```

### Healthchecks

| Service | Probe | Healthy when |
|---------|-------|--------------|
| `arena` | `arena.log` mtime &lt; `ARENA_LOG_STALE_SEC` (default 300s) | Process is writing logs |
| `dashboard` | `GET http://127.0.0.1:8501/healthz` | Uvicorn answers (unauthenticated) |

```bash
docker inspect --format='{{.State.Health.Status}}' pba-arena
docker inspect --format='{{.State.Health.Status}}' pba-dashboard
curl -s http://127.0.0.1:8501/healthz
# → {"status":"ok","arena_log_age_sec":...,"arena_log_stale":false,...}
```

If `arena_log_stale` is true for several minutes, restart: `docker compose restart arena`.

### Import an existing DB

```bash
# Stop first so WAL is quiet
docker compose down

# Copy your host DB into the data dir
mkdir -p data
cp /path/to/bot_arena.db data/bot_arena.db
# optional: also copy .credentials.enc → data/secrets/credentials.enc
#           and arena_fernet.key → data/secrets/arena_fernet.key

docker compose up -d
```

### Live trading (optional)

1. Start in paper (default). Confirm dashboard + bots look healthy.
2. In **Settings**, enter Polymarket L2 API key / secret / passphrase.
3. Toggle individual bots to **live** only after paper is stable.
4. Secrets persist under `data/secrets/` — **back them up** and never commit them.

The arena still **refuses** `--mode live` without an interactive TTY. Use the dashboard mode toggle instead.

---

## Cloud / VPS deployment

Target: a small Linux VPS (1–2 vCPU, 1–2 GB RAM is enough for paper). Ubuntu 22.04/24.04 examples below.

### 1. Install Docker

```bash
# Official convenience script (or use your distro packages)
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
# log out/in so the group applies
```

### 2. Clone and configure

```bash
sudo mkdir -p /opt/pba
sudo chown "$USER":"$USER" /opt/pba
cd /opt/pba
git clone https://github.com/senseirandystl/polymarket-bot-arena.git .
# or rsync your private tree

cp .env.example .env
chmod 600 .env
```

Edit `.env` on the server:

```bash
DASHBOARD_USER=admin
DASHBOARD_PASS=<long-random-password>
# Keep dashboard off the public interface unless you terminate TLS elsewhere
DASHBOARD_BIND=127.0.0.1
DASHBOARD_PORT=8501
PBA_DATA_DIR=/opt/pba/data
TZ=America/New_York
```

### 3. Start and enable restart

```bash
cd /opt/pba
docker compose up -d --build
docker compose ps
```

`restart: unless-stopped` brings containers back after reboot and crashes. No extra systemd unit is required when using Docker.

Optional: log rotation is already limited via `json-file` `max-size` / `max-file` in compose.

### 4. Expose the dashboard safely (recommended: reverse proxy)

Do **not** set `DASHBOARD_BIND=0.0.0.0` on a public IP without TLS + a strong password.

**Option A — SSH tunnel (simplest ops):**

```bash
# from your laptop
ssh -L 8501:127.0.0.1:8501 user@your-vps
# then open http://127.0.0.1:8501 locally
```

**Option B — Caddy / nginx with HTTPS** (terminate TLS on the host, proxy to 127.0.0.1:8501):

Example Caddy snippet:

```
dashboard.example.com {
    reverse_proxy 127.0.0.1:8501
}
```

HTTP Basic auth remains on the app; you can add another layer at the proxy if you want.

**Option C — firewall only:**

```bash
sudo ufw allow OpenSSH
sudo ufw allow 443/tcp    # if using a reverse proxy
sudo ufw enable
# do not open 8501 publicly if bound to 127.0.0.1
```

### 5. Updates

```bash
cd /opt/pba
git pull
docker compose up -d --build
# data volume is preserved
```

### 6. Backups

Back up the entire data directory (DB + secrets + logs):

```bash
# while running is usually OK (SQLite WAL); for a cold copy:
docker compose stop arena dashboard
tar -czf "pba-backup-$(date -u +%Y%m%dT%H%M%SZ).tgz" -C /opt/pba data
docker compose start arena dashboard
```

Restore:

```bash
docker compose down
tar -xzf pba-backup-....tgz -C /opt/pba
docker compose up -d
```

### 7. Resource notes

- Outbound HTTPS (443) required to: `clob.polymarket.com`, `gamma-api.polymarket.com`, `data-api.polymarket.com`, Binance WS/REST.
- No inbound ports required for trading itself.
- Disk: SQLite + logs grow over time; prune old logs under `data/logs/` if needed.

---

## Environment reference

| Variable | Default | Used by | Notes |
|----------|---------|---------|-------|
| `DASHBOARD_USER` | `admin` | dashboard | HTTP Basic user |
| `DASHBOARD_PASS` | `Thor` | dashboard | **Change before public expose** |
| `DASHBOARD_BIND` | `127.0.0.1` | compose port publish | Host interface |
| `DASHBOARD_PORT` | `8501` | compose + app | Container always listens on 8501 |
| `PBA_DATA_DIR` | `./data` | compose volumes | Host path → `/data` |
| `ARENA_DB_PATH` | `/data/bot_arena.db` | both | Set in compose |
| `ARENA_LOG_DIR` | `/data/logs` | both | Set in compose |
| `ARENA_CREDENTIALS_FILE` | `/data/secrets/credentials.enc` | both | Encrypted API keys |
| `ARENA_CREDENTIALS_KEY_FILE` | `/data/secrets/arena_fernet.key` | both | Fernet key |
| `ARENA_NO_DASHBOARD` | `1` | arena | Prevent nested dashboard |
| `ARENA_PAPER_BANKROLL` | (config) | arena | Optional override |
| `ARENA_KELLY_FRACTION` | (config) | arena | Optional override |
| `ARENA_LOG_JSON` | off | arena | Structured JSON logs |
| `ARENA_LOG_STALE_SEC` | `300` | healthchecks | Log age / `/healthz` |
| `TZ` | `America/New_York` | both | ET day boundaries |

Path overrides also work outside Docker (launchd/systemd/terminal) if you export the same `ARENA_*` variables.

---

## Troubleshooting

| Symptom | What to check |
|---------|----------------|
| Dashboard 401 | `DASHBOARD_USER` / `DASHBOARD_PASS` in `.env`; recreate: `docker compose up -d` |
| Empty / fresh bots after upgrade | Wrong `PBA_DATA_DIR` or wiped `./data` |
| `database is locked` | Both services must mount the **same** `/data`; avoid two hosts on one NFS SQLite |
| Arena unhealthy | `docker compose logs arena`; network to Polymarket/Binance; `data/logs/arena.log` |
| Credentials lost after rebuild | Secrets must live under `PBA_DATA_DIR/secrets/`, not only in the image |
| Permission errors on `/data` | Host dir owned by UID 1000 (container user `arena`), or world-writable data dir |

```bash
# Force UID ownership on the host (Linux)
sudo chown -R 1000:1000 /opt/pba/data
```

---

## Bare-metal alternatives

### macOS launchd

Keep using the repo plists (no Docker). See `CLAUDE.md` → *launchd Services*.

### Linux systemd (no Docker)

Unit files live in `deploy/systemd/`. They assume a venv at `/opt/pba/.venv` and the repo at `/opt/pba` — edit paths, then:

```bash
cd /opt/pba
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

sudo cp deploy/systemd/polymarket-botarena.service /etc/systemd/system/
sudo cp deploy/systemd/polymarket-dashboard.service /etc/systemd/system/
# edit User=, WorkingDirectory=, paths as needed
sudo systemctl daemon-reload
sudo systemctl enable --now polymarket-dashboard polymarket-botarena
sudo systemctl status polymarket-botarena polymarket-dashboard
```

Prefer **either** Docker **or** systemd for a given host — not both against the same DB.
