"""Dashboard-editable regime settings (arena_state + config defaults)."""

import db
from arena import regime_settings as rs


def test_bool_defaults_from_config(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    db.init_db()
    rs.invalidate_cache()
    # continuous_blend defaults False in config
    assert rs.get_bool("continuous_blend") is False
    assert rs.get_bool("use_relative") is True


def test_set_bool_persists(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    db.init_db()
    rs.invalidate_cache()
    assert rs.set_bool("continuous_blend", True) is True
    rs.invalidate_cache()
    assert rs.get_bool("continuous_blend") is True
    assert rs.set_bool("continuous_blend", False) is False
    rs.invalidate_cache()
    assert rs.get_bool("continuous_blend") is False


def test_adapt_primary(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    db.init_db()
    rs.invalidate_cache()
    assert rs.set_adapt_primary("throttle") == "throttle"
    rs.invalidate_cache()
    assert rs.get_adapt_primary() == "throttle"
    assert rs.set_adapt_primary("style") == "style"


def test_unknown_flag_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    db.init_db()
    try:
        rs.set_bool("not_a_real_flag", True)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_snapshot_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    db.init_db()
    rs.invalidate_cache()
    snap = rs.snapshot()
    assert "flags" in snap and "continuous_blend" in snap["flags"]
    assert "labels" in snap and "blurb" in snap
    assert snap["adapt_primary"] in ("style", "throttle")
