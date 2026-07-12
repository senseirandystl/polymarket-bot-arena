"""One-time backfill: re-resolve trades that expired at $0 but actually settled.

Some trades were marked ``outcome='expired', pnl=0`` by the 1h stale-sweep even
though their market had in fact resolved on Simmer (the resolver missed them —
e.g. the arena was down, or resolution was routed through an invalid per-bot
key before that bug was fixed in arena/resolver.py). This script re-checks every
expired trade against Simmer and, for any whose market now reports a concrete
Up/Down outcome, rewrites the real win/loss + P&L and feeds the result into the
learning system — exactly as the live resolver would have.

Idempotent: trades whose market never resolved (genuinely stuck ``active``
markets) are left as ``expired``. Safe to re-run.

Run:  .venv/bin/python3 scripts/backfill_expired_resolutions.py
Add   --dry-run  to preview without writing.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import db  # noqa: E402
from arena.resolver import TradeResolver  # noqa: E402


def _candidate_keys() -> list:
    keys: list = []
    default = config.get_credential("simmer_api_key")
    raw = config.get_credential("simmer_bot_keys")
    bot_keys = {}
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                bot_keys = parsed
        except (json.JSONDecodeError, TypeError):
            pass
    for k in [default, *bot_keys.values()]:
        if k and k not in keys:
            keys.append(k)
    return keys


def main(dry_run: bool = False) -> None:
    resolver = TradeResolver()
    candidates = _candidate_keys()
    if not candidates:
        print("No Simmer API key configured — nothing to do.")
        return

    with db.get_conn() as conn:
        expired = conn.execute(
            "SELECT id, market_id, bot_name, side, amount, shares_bought, "
            "trade_features, reasoning FROM trades WHERE outcome='expired'"
        ).fetchall()
    print(f"Found {len(expired)} expired trades to re-check.")
    if not expired:
        return

    # Pick the first candidate key that authenticates.
    headers = None
    probe_mid = expired[0]["market_id"]
    for key in candidates:
        h = {"Authorization": f"Bearer {key}"}
        state, _ = resolver._fetch_market_outcome(h, probe_mid)
        if state != "auth_error":
            headers = h
            print(f"Using key {key[:10]}… (probe state: {state})")
            break
    if headers is None:
        print("Every candidate key was rejected (401/403). Fix Simmer keys first.")
        return

    recovered = wins = losses = 0
    total_pnl = 0.0
    for tr in expired:
        state, outcome = resolver._fetch_market_outcome(headers, tr["market_id"])
        if state != "resolved":
            continue
        shares = tr["shares_bought"] or 0
        if shares <= 0:
            continue
        won = (outcome is True) if tr["side"] == "yes" else (outcome is False)
        pnl = (shares - tr["amount"]) if won else -tr["amount"]
        wins += 1 if won else 0
        losses += 0 if won else 1
        total_pnl += pnl
        recovered += 1
        print(
            f"  trade {tr['id']:>4} [{tr['bot_name']}] {tr['side']} "
            f"{str(tr['market_id'])[:8]} -> {'WIN' if won else 'LOSS'} "
            f"pnl={pnl:+.2f}"
        )
        if not dry_run:
            # Reuse the resolver's settle path so learning is recorded too.
            resolver._settle_trade(tr, outcome)

    verb = "Would recover" if dry_run else "Recovered"
    print(
        f"\n{verb} {recovered} trades ({wins}W / {losses}L), "
        f"net P&L {total_pnl:+.2f}. "
        f"{len(expired) - recovered} remain genuinely unsettled (still 'expired')."
    )


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
