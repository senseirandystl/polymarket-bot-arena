"""Security regression tests: credentials never leak into logs, stdout, or
plaintext on disk, and credential-adjacent modules stay print()-free.
"""

import logging
import re
from pathlib import Path

import pytest

import credentials_store as cs

REPO_ROOT = Path(__file__).resolve().parents[2]

SECRET = {
    "polymarket_api_key": "TEST-API-KEY-a1b2c3",
    "polymarket_api_secret": "TEST-SECRET-hmac-d4e5f6==",
    "polymarket_api_passphrase": "TEST-PASSPHRASE-g7h8",
    "polymarket_private_key": "0xdeadbeefcafe0123456789abcdef0123456789abcdef0123456789abcdef0123",
}


@pytest.fixture()
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "CREDENTIALS_FILE", tmp_path / ".credentials.enc")
    monkeypatch.setattr(cs, "CREDENTIALS_KEY_FILE", tmp_path / "fernet.key")
    monkeypatch.setattr(cs, "LEGACY_SOURCES", [])
    return cs.CredentialsStore()


def test_save_and_load_never_log_secret_values(isolated_store, caplog, capsys):
    """The full save→reload→get path must not emit any secret value to the
    log stream (any level) or to stdout/stderr."""
    with caplog.at_level(logging.DEBUG):
        isolated_store.save(SECRET)
        fresh = cs.CredentialsStore()
        for key, value in SECRET.items():
            assert fresh.get(key) == value

    all_log_text = "\n".join(r.getMessage() for r in caplog.records)
    captured = capsys.readouterr()
    for value in SECRET.values():
        assert value not in all_log_text, "secret value leaked into logging"
        assert value not in captured.out + captured.err, "secret printed"


def test_store_file_is_encrypted_at_rest(isolated_store):
    isolated_store.save(SECRET)
    raw = cs.CREDENTIALS_FILE.read_bytes()
    for value in SECRET.values():
        assert value.encode() not in raw, "plaintext secret on disk"


def test_key_file_has_0600_perms(isolated_store):
    isolated_store.save(SECRET)
    mode = cs.CREDENTIALS_KEY_FILE.stat().st_mode & 0o777
    assert mode == 0o600


def test_status_masks_values(isolated_store):
    """The dashboard status listing must never carry the secret values."""
    isolated_store.save(SECRET)
    flat = repr(isolated_store.status())
    for value in SECRET.values():
        assert value not in flat


# --- static source scan ----------------------------------------------------

CREDENTIAL_ADJACENT = [
    "credentials_store.py",
    "polymarket_client.py",
    "venues/live.py",
]

# logger/f-string interpolation of a secret-named variable, e.g.
# logger.info(f"... {private_key}") / print(api_secret)
_LEAK_PATTERN = re.compile(
    r"(logger\.\w+|print)\([^)]*\{[^}]*(private_key|api_secret|passphrase|api_key)\b"
)


@pytest.mark.parametrize("relpath", CREDENTIAL_ADJACENT)
def test_no_print_statements_in_credential_modules(relpath):
    src = (REPO_ROOT / relpath).read_text()
    for lineno, line in enumerate(src.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert not re.match(r"^\s*print\(", line), (
            f"{relpath}:{lineno} uses print() — route through logging instead")


@pytest.mark.parametrize("relpath", CREDENTIAL_ADJACENT)
def test_no_secret_interpolation_into_logs(relpath):
    src = (REPO_ROOT / relpath).read_text()
    for lineno, line in enumerate(src.splitlines(), 1):
        assert not _LEAK_PATTERN.search(line), (
            f"{relpath}:{lineno} interpolates a secret-named variable into "
            f"a log/print call: {line.strip()}")


def test_rate_limit_awareness_configured():
    """429 must be in the transient retry set with exponential backoff bounds."""
    import config
    assert 429 in config.HTTP_RETRY_STATUSES
    assert config.HTTP_BACKOFF_BASE > 0
    assert config.HTTP_BACKOFF_CAP >= config.HTTP_BACKOFF_BASE
    assert config.HTTP_MAX_RETRIES >= 1
