from arena.portfolio import apply_regime_tilt


def test_tilt_bounded_and_floored():
    scores = {"a": 1.0, "b": 1.0, "c": 1.0}
    edges = {"a": 5.0, "b": -5.0}          # c absent -> neutral
    out = apply_regime_tilt(scores, edges, max_tilt=0.25, min_weight=0.05)
    assert out["a"] > out["c"] > out["b"]  # winner up, loser down, neutral middle
    assert min(out.values()) >= 0.05 * max(out.values())  # explore floor kept relative
    # tilt magnitude bounded
    assert out["a"] <= scores["a"] * (1 + 0.25) + 1e-9
    assert out["b"] >= scores["b"] * (1 - 0.25) - 1e-9


def test_no_edges_is_identity():
    scores = {"a": 1.0, "b": 2.0}
    assert apply_regime_tilt(scores, None, max_tilt=0.25, min_weight=0.05) == scores


def test_does_not_mutate_input():
    scores = {"a": 1.0, "b": 1.0}
    edges = {"a": 3.0, "b": -3.0}
    snapshot = dict(scores)
    apply_regime_tilt(scores, edges, max_tilt=0.25, min_weight=0.05)
    assert scores == snapshot  # input untouched (immutability)
