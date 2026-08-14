"""Telegram command bot — auth, dispatch, rate limit, control safety.

The command poller accepts INBOUND messages, so its auth boundary is the
security-critical part: the bot token is a bearer credential and anyone who
finds the bot can DM it. Every test here that touches ``handle_update`` is
guarding that boundary or the destructive-command confirmations.
"""

import time

import pytest

from arena import telegram_commands as tc


@pytest.fixture()
def bot(monkeypatch):
    """Command bot wired to a fake chat, capturing replies instead of HTTP."""
    sent = []
    state = {}
    monkeypatch.setattr(tc, "_chat_id", lambda: "555")
    monkeypatch.setattr(tc, "_token", lambda: "tok")
    monkeypatch.setattr(tc, "_send", lambda chat, text: sent.append((chat, text)))
    monkeypatch.setattr(tc.db, "get_arena_state", lambda k, d=None: state.get(k, d))
    monkeypatch.setattr(tc.db, "set_arena_state", lambda k, v: state.__setitem__(k, v))
    monkeypatch.setattr(tc.config, "TELEGRAM_COMMANDS_ENABLED", True)
    monkeypatch.setattr(tc.config, "TELEGRAM_COMMANDS_CONTROL_ENABLED", True)
    monkeypatch.setattr(tc.config, "TELEGRAM_COMMANDS_RATE_LIMIT_SEC", 0.0)
    tc._last_command_at.clear()
    tc._pending_confirm.clear()
    tc._unauth_logged.clear()
    tc._offset_memo[0] = 0  # process-global high-water mark; reset per test
    # The registry is module-level; snapshot it so a test that stubs a command
    # can't leak that stub into the next test's /help output.
    original = dict(tc._REGISTRY)
    yield {"sent": sent, "state": state}
    tc._REGISTRY.clear()
    tc._REGISTRY.update(original)


def _update(text, chat_id="555", update_id=1, from_id=None):
    cid = int(chat_id) if str(chat_id).lstrip("-").isdigit() else chat_id
    return {
        "update_id": update_id,
        "message": {
            "message_id": 10,
            "chat": {"id": cid},
            "from": {"id": from_id if from_id is not None else cid},
            "date": int(time.time()),
            "text": text,
        },
    }


# ---------------------------------------------------------------------------
# Auth boundary
# ---------------------------------------------------------------------------

def test_foreign_chat_is_ignored_silently(bot):
    """A DM from any chat other than the configured one gets NO reply.

    Replying would confirm the bot exists and is live to a stranger; the
    allowlist is the only thing standing between a leaked bot username and
    someone typing /kill.
    """
    handled = tc.handle_update(_update("/status", chat_id="999"))
    assert handled is False
    assert bot["sent"] == []


def test_configured_chat_is_served(bot, monkeypatch):
    monkeypatch.setattr(tc, "cmd_status", lambda args: "ok-status")
    tc._REGISTRY["status"] = tc._Command("status", tc.cmd_status, "", False)
    assert tc.handle_update(_update("/status")) is True
    assert bot["sent"] and "ok-status" in bot["sent"][0][1]


def test_non_message_update_is_ignored(bot):
    assert tc.handle_update({"update_id": 2, "edited_message": {}}) is False
    assert bot["sent"] == []


def test_other_group_member_cannot_command(bot, monkeypatch):
    """If the alert target is a GROUP, only the allowlisted sender may command.

    Otherwise every current and future member of the group — anyone with the
    invite link — inherits /kill and /retire.
    """
    monkeypatch.setattr(tc.config, "TELEGRAM_COMMANDS_ALLOWED_USER_IDS", ())
    assert tc.handle_update(_update("/status", from_id=777)) is False
    assert bot["sent"] == []


def test_missing_sender_is_rejected(bot):
    """An auth check must not be conditional on the presence of the field it
    authorizes — Message.from is optional in the Bot API."""
    u = _update("/status")
    u["message"].pop("from")
    assert tc.handle_update(u) is False
    assert bot["sent"] == []


def test_channel_sender_chat_is_rejected(bot):
    """sender_chat = anonymous group admin / channel post: no user identity
    to allowlist, so it can never be the operator."""
    u = _update("/status")
    u["message"]["sender_chat"] = {"id": -100123}
    assert tc.handle_update(u) is False
    assert bot["sent"] == []


def test_explicitly_allowlisted_sender_is_served(bot, monkeypatch):
    monkeypatch.setattr(tc.config, "TELEGRAM_COMMANDS_ALLOWED_USER_IDS", (777,))
    tc._REGISTRY["ping"] = tc._Command("ping", lambda args: "pong", "", False)
    assert tc.handle_update(_update("/ping", from_id=777)) is True
    assert bot["sent"][-1][1] == "pong"


def test_control_commands_blocked_when_disabled(bot, monkeypatch):
    monkeypatch.setattr(tc.config, "TELEGRAM_COMMANDS_CONTROL_ENABLED", False)
    tc.handle_update(_update("/kill"))
    assert "disabled" in bot["sent"][-1][1].lower()


def test_read_only_commands_work_when_control_disabled(bot, monkeypatch):
    monkeypatch.setattr(tc.config, "TELEGRAM_COMMANDS_CONTROL_ENABLED", False)
    tc._REGISTRY["ping"] = tc._Command("ping", lambda args: "pong", "", False)
    tc.handle_update(_update("/ping"))
    assert bot["sent"][-1][1] == "pong"


# ---------------------------------------------------------------------------
# Parsing / dispatch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("/hour", ("hour", [])),
    ("/hour  ", ("hour", [])),
    ("/retire momentum-v1", ("retire", ["momentum-v1"])),
    ("/deploy phantom hybrid", ("deploy", ["phantom", "hybrid"])),
    ("/status@pba_arena_bot", ("status", [])),   # group-mention suffix
    ("HOUR", ("hour", [])),                       # bare + case-insensitive
])
def test_parse_command(text, expected):
    assert tc.parse_command(text) == expected


def test_parse_non_command_returns_none():
    assert tc.parse_command("just chatting") is None
    assert tc.parse_command("") is None


def test_bare_control_words_are_not_commands():
    """The alert chat is also a conversation. 'Kill the losses' must not
    halt trading; only an explicit /kill is a command."""
    for text in ("kill the losses", "Kill switch?", "pause all",
                 "resume momentum-v1", "deploy phantom", "retire momentum-v1"):
        assert tc.parse_command(text) is None


def test_unknown_command_replies_with_hint(bot):
    tc.handle_update(_update("/nonsense"))
    assert "unknown" in bot["sent"][-1][1].lower()


def test_handler_exception_replies_instead_of_crashing(bot):
    def boom(args):
        raise RuntimeError("db exploded")
    tc._REGISTRY["boom"] = tc._Command("boom", boom, "", False)
    try:
        assert tc.handle_update(_update("/boom")) is True
        assert "error" in bot["sent"][-1][1].lower()
    finally:
        tc._REGISTRY.pop("boom", None)


def test_rate_limit_suppresses_rapid_repeat(bot, monkeypatch):
    monkeypatch.setattr(tc.config, "TELEGRAM_COMMANDS_RATE_LIMIT_SEC", 60.0)
    calls = []
    tc._REGISTRY["ping"] = tc._Command(
        "ping", lambda args: calls.append(1) or "pong", "", False)
    try:
        tc.handle_update(_update("/ping", update_id=1))
        tc.handle_update(_update("/ping", update_id=2))
        assert len(calls) == 1
        assert "too fast" in bot["sent"][-1][1].lower()
    finally:
        tc._REGISTRY.pop("ping", None)


# ---------------------------------------------------------------------------
# Offset persistence — a restart must not replay old commands
# ---------------------------------------------------------------------------

def test_offset_persists_and_advances(bot):
    tc._save_offset(41)
    assert tc._load_offset() == 41
    tc.process_updates([_update("/help", update_id=77)])
    assert tc._load_offset() == 78  # next offset = last seen + 1


def test_offset_advances_before_handler_runs(bot):
    """Ack-then-execute: a handler that dies must not leave the update
    re-deliverable, or Telegram replays the control command forever."""
    seen = {}

    def explode(args):
        seen["offset_during"] = tc._load_offset()
        raise RuntimeError("boom")

    tc._REGISTRY["boom"] = tc._Command("boom", explode, "", False)
    tc._save_offset(0)
    tc.process_updates([_update("/boom", update_id=5)])
    assert seen["offset_during"] == 6
    assert tc._load_offset() == 6


def test_offset_survives_db_write_failure(bot, monkeypatch):
    """A locked SQLite (routine — the arena writes the same WAL DB every
    second) must not make the batch re-execute in a tight loop."""
    def fail(k, v):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(tc.db, "set_arena_state", fail)
    calls = []
    tc._REGISTRY["ping"] = tc._Command(
        "ping", lambda args: calls.append(1) or "pong", "", False)
    tc.process_updates([_update("/ping", update_id=9)])
    tc.process_updates([_update("/ping", update_id=9)])
    assert len(calls) == 1  # in-memory offset held the line


def test_stale_updates_are_skipped(bot):
    """Telegram redelivers until acked; a lower update_id must not re-run."""
    calls = []
    tc._REGISTRY["ping"] = tc._Command(
        "ping", lambda args: calls.append(1) or "pong", "", False)
    try:
        tc._save_offset(100)
        tc.process_updates([_update("/ping", update_id=50)])
        assert calls == []
    finally:
        tc._REGISTRY.pop("ping", None)


# ---------------------------------------------------------------------------
# Staleness — the replay guard that does not depend on drain bookkeeping
# ---------------------------------------------------------------------------

def test_stale_control_command_is_refused(bot, monkeypatch):
    """A /kill queued while the laptop slept must not fire hours later.

    The startup drain only covers a process RESTART; a sleeping Mac keeps the
    dashboard alive, so the backlog arrives mid-loop. Message age closes the
    whole replay class rather than one door into it.
    """
    monkeypatch.setattr(tc.config, "TELEGRAM_COMMANDS_MAX_AGE_SEC", 300)
    armed = {}
    _fake_risk(monkeypatch, armed)
    u = _update("/kill")
    u["message"]["date"] = int(time.time()) - 6 * 3600
    tc.handle_update(u)
    assert armed == {}
    assert "too old" in bot["sent"][-1][1].lower()


def test_stale_report_command_still_answers(bot, monkeypatch):
    """Reports are side-effect free — answering a late /hour is fine."""
    monkeypatch.setattr(tc.config, "TELEGRAM_COMMANDS_MAX_AGE_SEC", 300)
    tc._REGISTRY["ping"] = tc._Command("ping", lambda args: "pong", "", False)
    u = _update("/ping")
    u["message"]["date"] = int(time.time()) - 6 * 3600
    tc.handle_update(u)
    assert bot["sent"][-1][1] == "pong"


def test_control_command_without_a_date_is_refused(bot, monkeypatch):
    armed = {}
    _fake_risk(monkeypatch, armed)
    u = _update("/kill")
    u["message"].pop("date", None)
    tc.handle_update(u)
    assert armed == {}


def test_fresh_control_command_passes(bot, monkeypatch):
    armed = {}
    _fake_risk(monkeypatch, armed)
    tc.handle_update(_update("/kill"))
    assert armed["armed"] is True


# ---------------------------------------------------------------------------
# Startup backlog drain — the 24h replay guard
# ---------------------------------------------------------------------------

def test_backlog_is_acked_not_executed(bot, monkeypatch):
    calls = []
    tc._REGISTRY["ping"] = tc._Command(
        "ping", lambda args: calls.append(1) or "pong", "", False)
    pages = [[_update("/ping", update_id=3)], []]
    monkeypatch.setattr(tc, "_get_updates",
                        lambda offset, timeout, limit=100: pages.pop(0))
    assert tc._drain_backlog() is True
    assert calls == []
    assert tc._load_offset() == 4


def test_backlog_drain_pages_past_the_first_100(bot, monkeypatch):
    """getUpdates returns at most 100 per call — acking one page would let
    update 101 (a real /kill) fall through into the executing loop."""
    page1 = [_update("x", update_id=i) for i in range(1, 101)]
    page2 = [_update("x", update_id=101)]
    pages = [page1, page2, []]
    monkeypatch.setattr(tc, "_get_updates",
                        lambda offset, timeout, limit=100: pages.pop(0))
    assert tc._drain_backlog() is True
    assert tc._load_offset() == 102


def test_backlog_drain_reports_failure(bot, monkeypatch):
    """A failed drain must be distinguishable from an empty one — otherwise
    the poller starts anyway and executes yesterday's backlog."""
    monkeypatch.setattr(tc, "_get_updates",
                        lambda offset, timeout, limit=100: None)
    assert tc._drain_backlog() is False


def test_backlog_drain_page_cap_is_failure(bot, monkeypatch):
    """Hitting the page cap with the queue still full must NOT look like a
    successful drain — leftover updates would then execute."""
    monkeypatch.setattr(
        tc, "_get_updates",
        lambda offset, timeout, limit=100: [
            _update("x", update_id=offset + i) for i in range(100)
        ])
    assert tc._drain_backlog() is False


def test_start_poller_is_inert_under_pytest(bot, monkeypatch):
    """TestClient(dashboard.app) runs the FastAPI lifespan. If that started
    the live poller, a unit test would drain the real Telegram queue (and
    409 with the running dashboard)."""
    monkeypatch.setattr(tc.config, "TELEGRAM_COMMANDS_ENABLED", True)
    assert tc.start_poller() is False
    assert tc._poller_thread is None or not tc._poller_thread.is_alive()


def test_transport_failure_returns_none_not_empty(bot, monkeypatch):
    class Boom:
        def __call__(self, *a, **k):
            raise OSError("network down")
    monkeypatch.setattr(tc.urllib.request, "urlopen", Boom())
    assert tc._get_updates(0, 0) is None


def test_api_not_ok_returns_none(bot, monkeypatch):
    """401 (revoked token) / 409 (webhook set) / 429 must back off, not spin."""
    import io

    class Resp(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(
        tc.urllib.request, "urlopen",
        lambda *a, **k: Resp(b'{"ok":false,"error_code":409}'))
    assert tc._get_updates(0, 0) is None


# ---------------------------------------------------------------------------
# Control commands
# ---------------------------------------------------------------------------

def test_kill_arms_switch(bot, monkeypatch):
    armed = {}
    monkeypatch.setattr(
        tc, "_risk_engine",
        lambda: type("R", (), {
            "set_kill_switch": staticmethod(
                lambda a, reason="", source="": armed.update(
                    {"armed": a, "source": source}) or {"kill_switch": a}),
        }),
    )
    tc.handle_update(_update("/kill"))
    assert armed["armed"] is True
    assert armed["source"] == "telegram"
    assert "armed" in bot["sent"][-1][1].lower()


def _fake_risk(monkeypatch, armed, *, kill_source="telegram"):
    monkeypatch.setattr(
        tc, "_risk_engine",
        lambda: type("R", (), {
            "set_kill_switch": staticmethod(
                lambda a, reason="", source="": armed.update({"armed": a})
                or {"kill_switch": a}),
            "load_state": staticmethod(
                lambda: {"kill_switch": True, "kill_source": kill_source,
                         "kill_reason": "test"}),
        }),
    )


def test_unkill_requires_confirmation(bot, monkeypatch):
    """Clearing a halt is the one command that can restart losses — and the
    risk engine arms this switch automatically on daily-loss / drawdown."""
    armed = {}
    _fake_risk(monkeypatch, armed)
    tc.handle_update(_update("/unkill", update_id=1))
    assert armed == {}
    assert "confirm" in bot["sent"][-1][1].lower()

    tc.handle_update(_update("/unkill confirm", update_id=2))
    assert armed["armed"] is False


def test_unkill_warns_when_halt_was_automatic(bot, monkeypatch):
    """A risk-engine halt cleared from a phone should say what it is
    overriding, not silently resume trading."""
    armed = {}
    _fake_risk(monkeypatch, armed, kill_source="risk_engine")
    tc.handle_update(_update("/unkill", update_id=1))
    body = bot["sent"][-1][1].lower()
    assert "risk_engine" in body or "automatic" in body


def _stub_retire(monkeypatch, retired, paused=None):
    paused = [] if paused is None else paused
    monkeypatch.setattr(tc.db, "get_active_bots",
                        lambda: [{"bot_name": "momentum-v1",
                                  "strategy_type": "momentum"}])
    monkeypatch.setattr(tc.db, "retire_bot", lambda n: retired.append(n))
    monkeypatch.setattr(
        tc, "_risk_engine",
        lambda: type("R", (), {
            "pause_bot": staticmethod(
                lambda n, reason="": paused.append(n) or {"status": "paused"}),
        }),
    )
    return paused


def test_retire_requires_confirmation(bot, monkeypatch):
    """Retiring is irreversible for a running bot — never on one message."""
    retired = []
    paused = _stub_retire(monkeypatch, retired)

    tc.handle_update(_update("/retire momentum-v1", update_id=1))
    assert retired == []
    assert paused == []
    assert "confirm" in bot["sent"][-1][1].lower()

    tc.handle_update(_update("/retire momentum-v1 confirm", update_id=2))
    assert retired == ["momentum-v1"]
    assert paused == ["momentum-v1"]


def test_retire_confirm_without_a_prompt_is_refused(bot, monkeypatch):
    """The confirm step must be a real two-message handshake. If a lone
    '/retire x confirm' works, then one replayed update retires a live bot.
    """
    retired = []
    _stub_retire(monkeypatch, retired)
    tc.handle_update(_update("/retire momentum-v1 confirm"))
    assert retired == []
    assert "confirm" in bot["sent"][-1][1].lower()


def test_repeated_confirm_alone_never_completes_the_handshake(bot, monkeypatch):
    """The refusal must not arm the gate, or one message delivered twice
    satisfies the handshake and the operator never sees what they destroyed.
    """
    retired = []
    _stub_retire(monkeypatch, retired)
    for i in range(3):
        tc.handle_update(_update("/retire momentum-v1 confirm", update_id=i + 1))
    assert retired == []


def test_retire_confirmation_expires(bot, monkeypatch):
    retired = []
    _stub_retire(monkeypatch, retired)
    tc.handle_update(_update("/retire momentum-v1", update_id=1))
    for key in list(tc._pending_confirm):
        tc._pending_confirm[key] = time.time() - 1  # age it past the TTL
    tc.handle_update(_update("/retire momentum-v1 confirm", update_id=2))
    assert retired == []
    assert "expired" in bot["sent"][-1][1].lower()


def test_confirm_followup_is_not_rate_limited(bot, monkeypatch):
    """The handshake is two messages of the SAME command seconds apart — the
    rate limiter must not eat the confirm and strand the operator."""
    monkeypatch.setattr(tc.config, "TELEGRAM_COMMANDS_RATE_LIMIT_SEC", 60.0)
    retired = []
    _stub_retire(monkeypatch, retired)
    tc.handle_update(_update("/retire momentum-v1", update_id=1))
    tc.handle_update(_update("/retire momentum-v1 confirm", update_id=2))
    assert retired == ["momentum-v1"]


def test_retire_rejects_unknown_bot(bot, monkeypatch):
    retired = []
    _stub_retire(monkeypatch, retired)
    tc.handle_update(_update("/retire ghost-v9 confirm"))
    assert retired == []
    assert "not active" in bot["sent"][-1][1].lower()


def test_pause_single_bot(bot, monkeypatch):
    paused = []
    monkeypatch.setattr(tc.db, "get_active_bots",
                        lambda: [{"bot_name": "momentum-v1",
                                  "strategy_type": "momentum"}])
    monkeypatch.setattr(
        tc, "_risk_engine",
        lambda: type("R", (), {
            "pause_bot": staticmethod(
                lambda n, reason="": paused.append(n) or {"status": "paused"}),
        }),
    )
    tc.handle_update(_update("/pause momentum-v1"))
    assert paused == ["momentum-v1"]


def test_pause_all_pauses_every_active_bot(bot, monkeypatch):
    paused = []
    monkeypatch.setattr(tc.db, "get_active_bots", lambda: [
        {"bot_name": "momentum-v1", "strategy_type": "momentum"},
        {"bot_name": "sniper-v3", "strategy_type": "sniper"},
    ])
    monkeypatch.setattr(
        tc, "_risk_engine",
        lambda: type("R", (), {
            "pause_bot": staticmethod(
                lambda n, reason="": paused.append(n) or {"status": "paused"}),
        }),
    )
    tc.handle_update(_update("/pause all"))
    assert sorted(paused) == ["momentum-v1", "sniper-v3"]


def test_resume_single_bot_requires_confirmation(bot, monkeypatch):
    """resume_bot sets manual_resume (trades past automatic drawdown)."""
    resumed = []
    monkeypatch.setattr(tc.db, "get_active_bots", lambda: [
        {"bot_name": "momentum-v1", "strategy_type": "momentum"},
    ])
    monkeypatch.setattr(
        tc, "_risk_engine",
        lambda: type("R", (), {
            "resume_bot": staticmethod(
                lambda n: resumed.append(n) or {"status": "ok"}),
        }),
    )
    tc.handle_update(_update("/resume momentum-v1", update_id=1))
    assert resumed == []
    assert "confirm" in bot["sent"][-1][1].lower()
    tc.handle_update(_update("/resume momentum-v1 confirm", update_id=2))
    assert resumed == ["momentum-v1"]


def test_resume_all_requires_confirmation(bot, monkeypatch):
    """resume_bot sets manual_resume, which force-allows trading PAST the
    automatic drawdown limits — doing that to the whole slate in one word is
    the highest-consequence command in the module."""
    resumed = []
    monkeypatch.setattr(tc.db, "get_active_bots", lambda: [
        {"bot_name": "momentum-v1", "strategy_type": "momentum"},
        {"bot_name": "sniper-v3", "strategy_type": "sniper"},
    ])
    monkeypatch.setattr(
        tc, "_risk_engine",
        lambda: type("R", (), {
            "resume_bot": staticmethod(
                lambda n: resumed.append(n) or {"status": "ok"}),
        }),
    )
    tc.handle_update(_update("/resume all", update_id=1))
    assert resumed == []
    tc.handle_update(_update("/resume all confirm", update_id=2))
    assert sorted(resumed) == ["momentum-v1", "sniper-v3"]


def test_deploy_queues_strategy(bot, monkeypatch):
    monkeypatch.setattr(tc, "_strategy_catalog",
                        lambda: [{"strategy_type": "phantom"},
                                 {"strategy_type": "sniper"}])
    monkeypatch.setattr(tc.db, "get_active_bots",
                        lambda: [{"bot_name": "sniper-v3",
                                  "strategy_type": "sniper"}])
    tc.handle_update(_update("/deploy phantom"))
    import json
    queued = json.loads(bot["state"]["pending_bot_deploys"])
    assert queued["strategies"] == ["phantom"]
    assert queued["source"] == "telegram"


def test_deploy_rejects_unknown_and_active(bot, monkeypatch):
    monkeypatch.setattr(tc, "_strategy_catalog",
                        lambda: [{"strategy_type": "phantom"},
                                 {"strategy_type": "sniper"}])
    monkeypatch.setattr(tc.db, "get_active_bots",
                        lambda: [{"bot_name": "sniper-v3",
                                  "strategy_type": "sniper"}])
    tc.handle_update(_update("/deploy nosuch sniper"))
    body = bot["sent"][-1][1].lower()
    assert "unknown" in body and "already active" in body
    assert "pending_bot_deploys" not in bot["state"]


def test_deploy_without_args_lists_catalog(bot, monkeypatch):
    monkeypatch.setattr(tc, "_strategy_catalog",
                        lambda: [{"strategy_type": "phantom"}])
    monkeypatch.setattr(tc.db, "get_active_bots", lambda: [])
    tc.handle_update(_update("/deploy"))
    assert "phantom" in bot["sent"][-1][1]


# ---------------------------------------------------------------------------
# Reporting commands
# ---------------------------------------------------------------------------

def test_hour_report_uses_one_hour_window(bot, monkeypatch):
    seen = {}

    def fake_stats(hours=1.0):
        seen["hours"] = hours
        return {"hour_pnl": 1.5, "hour_n": 4, "hour_wins": 3, "hour_losses": 1,
                "hour_wr": 0.75, "open": 2, "bots": [
                    {"bot": "momentum-v1", "n": 4, "wins": 3, "losses": 1,
                     "pnl": 1.5, "wr": 0.75}]}

    monkeypatch.setattr(tc, "_window_trade_stats", fake_stats)
    tc.handle_update(_update("/hour"))
    assert seen["hours"] == 1.0
    body = bot["sent"][-1][1]
    assert "momentum-v1" in body and "+1.50" in body


def test_day_and_week_windows(bot, monkeypatch):
    seen = []
    monkeypatch.setattr(tc, "_window_trade_stats",
                        lambda hours=1.0: seen.append(hours) or {
                            "hour_pnl": 0, "hour_n": 0, "hour_wins": 0,
                            "hour_losses": 0, "hour_wr": 0, "open": 0,
                            "bots": []})
    tc.handle_update(_update("/day", update_id=1))
    tc.handle_update(_update("/week", update_id=2))
    assert seen == [24.0, 168.0]


def test_hour_accepts_custom_window(bot, monkeypatch):
    seen = []
    monkeypatch.setattr(tc, "_window_trade_stats",
                        lambda hours=1.0: seen.append(hours) or {
                            "hour_pnl": 0, "hour_n": 0, "hour_wins": 0,
                            "hour_losses": 0, "hour_wr": 0, "open": 0,
                            "bots": []})
    tc.handle_update(_update("/hour 6"))
    assert seen == [6.0]


def test_report_with_no_trades_says_so(bot, monkeypatch):
    monkeypatch.setattr(tc, "_window_trade_stats", lambda hours=1.0: {
        "hour_pnl": 0.0, "hour_n": 0, "hour_wins": 0, "hour_losses": 0,
        "hour_wr": 0.0, "open": 0, "bots": []})
    tc.handle_update(_update("/hour"))
    assert "no resolved trades" in bot["sent"][-1][1].lower()


def test_help_lists_commands_and_marks_control(bot):
    tc.handle_update(_update("/help"))
    body = bot["sent"][-1][1]
    for name in ("/hour", "/day", "/status", "/bots", "/kill", "/pause",
                 "/deploy", "/retire"):
        assert name in body


def test_long_reply_is_truncated(bot, monkeypatch):
    """Telegram hard-limits ~4096 chars; we must truncate, not 400."""
    tc._REGISTRY["big"] = tc._Command("big", lambda args: "x" * 9000, "", False)
    try:
        monkeypatch.setattr(tc, "_send", tc._send)  # keep fixture's capture
        assert len(tc._clip("x" * 9000)) <= 4000
    finally:
        tc._REGISTRY.pop("big", None)
