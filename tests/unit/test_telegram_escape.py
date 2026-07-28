"""Telegram Markdown escape — prevent intermittent 400 parse-entity failures."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from arena.alerts import _telegram_escape_md


def test_escapes_underscores_and_stars():
    s = _telegram_escape_md("meanrev-v1 low_vol_range *bold*")
    assert "\\_" in s
    assert "\\*" in s
    assert "meanrev-v1" in s.replace("\\", "")


def test_empty_safe():
    assert _telegram_escape_md("") == ""
    assert _telegram_escape_md(None) == ""
