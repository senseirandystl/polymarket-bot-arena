"""Encrypted credentials store for Simmer + Polymarket API keys.

Stores an encrypted `<repo>/.credentials.enc` blob (gitignored) backed by a
machine-local Fernet key (~/.config/polymarket/arena_fernet.key, NOT in the
repo, 0600 perms). Lets the arena start, the dashboard render, and bots idle
gracefully when credentials are missing — defaults to a "config needed" state
rather than a hard crash.

Threat model: accidental `git push` of the repo, snapshots/backups of the
repo directory, unattended laptops. The encrypted blob is useless without
the machine-local key; the key is 0600 on the user's home disk.

Public API: callers should `import credentials_store` and use:
    get_credential(key, default=None)     -> str | None
    set_credentials(updates: dict)        -> None    (atomic, encrypted write)
    credentials_status()                  -> list    (for Settings tab + warnings)
    is_credential_configured(key)         -> bool

On first run, auto-migrates any legacy plaintext files at
`~/.config/simmer/{credentials,bot_keys}.json` and
`~/.config/polymarket/credentials.json`, then renames them to `.bak` so the
plaintext versions can never silently drift away from the encrypted store.
"""

import json
import logging
import os
import tempfile
from pathlib import Path

try:
    from cryptography.fernet import Fernet, InvalidToken
except ImportError as _crypto_err:
    raise ImportError(
        "credentials_store.py requires the 'cryptography' package for Fernet "
        "symmetric encryption of credentials. The Polymarket Bot Arena runs "
        "inside the project-local venv at .venv/bin/python3 (the launchd plists "
        "com.polymarket.botarena.plist and com.polymarket.dashboard.plist already "
        "point at this interpreter). If you ran `python3` manually, do one of:\n"
        "  - .venv/bin/python3 arena.py\n"
        "  - source .venv/bin/activate && python3 arena.py\n"
        "  - pip install 'cryptography>=42.0'   # into your current interpreter\n"
        f"\nOriginal error: {_crypto_err}"
    ) from _crypto_err


logger = logging.getLogger("credentials_store")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Encrypted blob: in the repo by default, gitignored. Override with
# ARENA_CREDENTIALS_FILE for Docker (persist under a data volume).
CREDENTIALS_FILE = Path(
    os.environ.get("ARENA_CREDENTIALS_FILE")
    or (Path(__file__).parent / ".credentials.enc")
)

# Fernet key: outside the repo by default, 0600 perms. Generated on first write.
# Override with ARENA_CREDENTIALS_KEY_FILE so containers can mount a secrets dir.
CREDENTIALS_KEY_FILE = Path(
    os.environ.get("ARENA_CREDENTIALS_KEY_FILE")
    or (Path.home() / ".config/polymarket/arena_fernet.key")
)

# Legacy plaintext locations we auto-migrate from (if present) on first run.
# After migration these are renamed to `.bak` so the plaintext copy can never
# silently drift away from the encrypted store.
LEGACY_SOURCES = [
    Path.home() / ".config/polymarket/credentials.json",
]

# Human-readable labels for the dashboard Settings tab + warning banner.
CREDENTIAL_LABELS = {
    "polymarket_api_key": (
        "Polymarket API key",
        "Live trading only — Polymarket CLOB L2 auth.",
    ),
    "polymarket_api_secret": (
        "Polymarket API secret",
        "Live trading only — base64-encoded HMAC secret used to sign L2 requests.",
    ),
    "polymarket_api_passphrase": (
        "Polymarket API passphrase",
        "Live trading only — passphrase bound to the L2 API key.",
    ),
    "polymarket_signer_address": (
        "Polymarket signer address",
        "Live trading only — your proxy / funder wallet address (0x…).",
    ),
    "polymarket_private_key": (
        "Polymarket private key",
        "Live trading only — signs on-chain order transactions. Never logged.",
    ),
    # Optional production alerts (arena/alerts.py) — all optional
    "alert_telegram_bot_token": (
        "Telegram bot token",
        "Alerts — from @BotFather. Used with chat id below.",
    ),
    "alert_telegram_chat_id": (
        "Telegram chat id",
        "Alerts — numeric chat/channel id that receives messages.",
    ),
    "alert_discord_webhook": (
        "Discord webhook URL",
        "Alerts — channel Integrations → Webhooks URL.",
    ),
    "alert_smtp_host": (
        "SMTP host",
        "Email alerts — e.g. smtp.gmail.com",
    ),
    "alert_smtp_port": (
        "SMTP port",
        "Email alerts — 587 (STARTTLS) or 465 (SSL).",
    ),
    "alert_smtp_user": (
        "SMTP username",
        "Email alerts — login user (often the from address).",
    ),
    "alert_smtp_password": (
        "SMTP password",
        "Email alerts — app password / SMTP secret. Never logged.",
    ),
    "alert_smtp_from": (
        "SMTP from address",
        "Email alerts — From: header (defaults to SMTP user).",
    ),
    "alert_smtp_to": (
        "SMTP to address(es)",
        "Email alerts — comma-separated recipients.",
    ),
}


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class CredentialsStore:
    """In-memory cache of the encrypted credentials file with mtime hot-reload.

    Reads are O(1) once warm: if the file mtime hasn't changed since the
    last load, the in-memory cache is returned. Writes are atomic
    (tempfile + os.replace) so concurrent readers can never observe a
    partially-written file.
    """

    def __init__(self) -> None:
        self._cache: dict = {}
        self._mtime: float = 0.0
        # Don't generate the key on __init__ — wait until first save.
        # A fresh clone that just *reads* (no writes) shouldn't litter ~/.config.

    # ----- key management --------------------------------------------------

    def _load_or_create_key(self) -> bytes:
        """Read the Fernet key from disk, generating & chmod-0600'ing on first run."""
        if CREDENTIALS_KEY_FILE.exists():
            with open(CREDENTIALS_KEY_FILE, "rb") as f:
                key = f.read()
            if not key:
                raise RuntimeError(f"Empty Fernet key at {CREDENTIALS_KEY_FILE}")
            return key

        # First write ever — generate a fresh 32-byte URL-safe key.
        CREDENTIALS_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
        key = Fernet.generate_key()
        fd: int | None
        fd, tmp_path = tempfile.mkstemp(
            dir=str(CREDENTIALS_KEY_FILE.parent), prefix=".arena_fernet."
        )
        try:
            os.write(fd, key)
            os.fchmod(fd, 0o600)
            os.close(fd)
            fd = None
            os.replace(tmp_path, CREDENTIALS_KEY_FILE)
            logger.info(
                f"Generated new Fernet key at {CREDENTIALS_KEY_FILE} (0600 perms)."
            )
        except Exception:
            if fd is not None:
                try:
                    os.close(fd)
                except Exception:
                    pass
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            raise
        return key

    # ----- I/O -------------------------------------------------------------

    def _maybe_reload(self) -> None:
        """Reload from disk if the file changed (mtime) or cache is empty."""
        if not CREDENTIALS_FILE.exists():
            self._cache = {}
            self._mtime = 0.0
            return
        try:
            mtime = CREDENTIALS_FILE.stat().st_mtime
        except OSError:
            return
        if mtime == self._mtime and self._cache is not None and mtime > 0:
            return
        try:
            with open(CREDENTIALS_FILE, "rb") as f:
                token = f.read()
            plain = self._fernet_only_for_read().decrypt(token)
            data = json.loads(plain.decode("utf-8"))
            self._cache = data if isinstance(data, dict) else {}
            self._mtime = mtime
        except (InvalidToken, FileNotFoundError, json.JSONDecodeError, OSError) as e:
            logger.warning(
                f"Could not decrypt credentials store ({type(e).__name__}: {e}); "
                f"treating as empty. Re-enter credentials via Settings tab."
            )
            self._cache = {}
            self._mtime = mtime  # suppress repeated warnings until the file changes

    def _fernet_only_for_read(self) -> Fernet:
        """Helper used by _maybe_reload — same as _load_or_create_key but
        named distinctly so the read/write code paths are obvious."""
        return Fernet(self._load_or_create_key())

    # ----- public methods ---------------------------------------------------

    def get(self, key: str, default=None):
        """Return the decrypted value for `key`, or `default` if not configured."""
        self._maybe_reload()
        return self._cache.get(key, default)

    def all_configured(self) -> dict:
        """Return the entire decrypted dict (for read-only inspection). Never raises."""
        self._maybe_reload()
        return dict(self._cache)

    def save(self, updates: dict) -> None:
        """Merge `updates` into the store and encrypt+write atomically.

        Empty strings / None values are stripped so a partially-filled form
        doesn't accidentally wipe out previously-saved fields.
        """
        self._maybe_reload()
        merged = dict(self._cache)
        for k, v in updates.items():
            if v in (None, ""):
                merged.pop(k, None)
            else:
                merged[k] = v
        self._cache = merged

        try:
            cipher = Fernet(self._load_or_create_key())
            plaintext = json.dumps(self._cache).encode("utf-8")
            token = cipher.encrypt(plaintext)
        except Exception as e:
            logger.error(f"Failed to encrypt credentials store: {e}")
            raise

        CREDENTIALS_FILE.parent.mkdir(parents=True, exist_ok=True)
        fd: int | None
        fd, tmp_path = tempfile.mkstemp(
            dir=str(CREDENTIALS_FILE.parent), prefix=".credentials."
        )
        try:
            os.write(fd, token)
            os.fchmod(fd, 0o600)
            os.close(fd)
            fd = None
            os.replace(tmp_path, CREDENTIALS_FILE)
            try:
                self._mtime = CREDENTIALS_FILE.stat().st_mtime
            except OSError:
                self._mtime = 0.0
            logger.info(
                f"Wrote encrypted credentials store ({len(self._cache)} keys, 0600 perms)."
            )
        except Exception:
            if fd is not None:
                try:
                    os.close(fd)
                except Exception:
                    pass
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            raise

    def delete_all(self) -> None:
        """Erase the encrypted blob (does NOT touch the Fernet key — same key can
        encrypt new values later). For testing / nuking the store from disk."""
        try:
            CREDENTIALS_FILE.unlink()
        except FileNotFoundError:
            pass
        except OSError as e:
            logger.warning(f"Could not delete {CREDENTIALS_FILE}: {e}")
        self._cache = {}
        self._mtime = 0.0

    def status(self) -> list:
        """Return [{key, label, description, configured}, ...] for Settings UI."""
        self._maybe_reload()
        result = []
        for key, (label, description) in CREDENTIAL_LABELS.items():
            val = self._cache.get(key)
            configured = bool(val) and val not in ("", None)
            result.append({
                "key": key,
                "label": label,
                "description": description,
                "configured": configured,
            })
        return result


# ---------------------------------------------------------------------------
# Legacy migration
# ---------------------------------------------------------------------------


def _migrate_polymarket_bundle(target: dict) -> bool:
    """Translate legacy `~/.config/polymarket/credentials.json` into flat fields."""
    legacy = Path.home() / ".config/polymarket/credentials.json"
    if not legacy.exists():
        return False
    try:
        with open(legacy, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    mapping = {
        "api_key":        "polymarket_api_key",
        "api_secret":     "polymarket_api_secret",
        "api_passphrase": "polymarket_api_passphrase",
        "signer_address": "polymarket_signer_address",
        "private_key":    "polymarket_private_key",
    }
    moved = False
    for from_k, to_k in mapping.items():
        v = data.get(from_k)
        if v and not target.get(to_k):
            target[to_k] = v
            moved = True
    return moved


def _migrate_legacy(store: CredentialsStore) -> bool:
    """If .credentials.enc is missing but a legacy plaintext file exists, import.

    After migration, each legacy file is renamed to `.bak` so the plaintext
    copy can never silently drift away from the encrypted store.
    """
    if CREDENTIALS_FILE.exists():
        return False

    migrated: dict = {}

    # Polymarket L2 bundle (flattened into separate store fields)
    _migrate_polymarket_bundle(migrated)

    if not migrated:
        return False

    # Write the encrypted blob first; only rename legacy once that succeeds.
    try:
        store.save(migrated)
    except Exception as e:
        logger.error(
            f"Could not auto-encrypt legacy credentials ({e}); "
            f"legacy files left in place."
        )
        return False

    for path in LEGACY_SOURCES:
        if not path.exists():
            continue
        backup = path.with_name(path.name + ".bak")
        try:
            path.rename(backup)
            logger.info(f"Renamed legacy {path} → {backup}")
        except OSError as e:
            logger.warning(
                f"Could not rename legacy {path} → .bak ({e}); "
                f"manual cleanup recommended."
            )

    logger.info(
        f"Auto-migrated {len(migrated)} credential key(s) from legacy plaintext "
        f"files to encrypted store. Legacy files renamed to .bak."
    )
    return True


# ---------------------------------------------------------------------------
# Module-level singleton + auto-migration on import
# ---------------------------------------------------------------------------

_store = CredentialsStore()
_migrate_legacy(_store)


def get_credential(key: str, default=None):
    """Decrypted value for `key`, or `default` if not configured."""
    return _store.get(key, default)


def set_credentials(updates: dict) -> None:
    """Merge `updates` into the store, encrypt, write atomically."""
    _store.save(updates)


def credentials_status() -> list:
    """List of {key, label, description, configured} for the Settings tab."""
    return _store.status()


def is_credential_configured(key: str) -> bool:
    """True iff `key` is present and non-empty in the store."""
    val = _store.get(key)
    return bool(val) and val != ""
