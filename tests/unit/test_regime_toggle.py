import config
import db


def test_config_defaults_present():
    assert config.REGIME_CONDITIONING_ENABLED is True
    assert config.REGIME_MIN_SAMPLES == 60
    assert 0.0 < config.REGIME_ALLOC_MIN_WEIGHT < config.REGIME_ALLOC_MAX_TILT < 1.0


def test_toggle_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "tg.db")
    db.init_db()
    assert db.get_regime_conditioning() is True   # default from config
    db.set_regime_conditioning(False)
    assert db.get_regime_conditioning() is False
