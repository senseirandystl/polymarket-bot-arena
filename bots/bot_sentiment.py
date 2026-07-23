"""Bot 3: Sentiment-based strategy using X/social signals."""

from bots.base_bot import BaseBot

DEFAULT_PARAMS = {
    # Polymarket in-market sentiment weights (see analyze()). score = PM
    # YES-price momentum * pm_weight + executed flow (CVD) * cvd_weight.
    "pm_weight": 3.0,
    "cvd_weight": 0.5,
    "deadband": 0.05,        # |score| below this = neutral (hold)
    "position_size_pct": 0.04,
    "min_confidence": 0.55,
}


class SentimentBot(BaseBot):
    def __init__(self, name="sentiment-v1", params=None, generation=0, lineage=None):
        super().__init__(
            name=name,
            strategy_type="sentiment",
            params=params or DEFAULT_PARAMS.copy(),
            generation=generation,
            lineage=lineage,
        )

    def analyze(self, market: dict, signals: dict) -> dict:
        """Polymarket in-market sentiment: how *this market's* traders are
        positioning, from PM YES-price momentum + executed flow (CVD).

        Repurposed 2026-07-15: the original X/social feed never existed
        post-Simmer, so this bot always held and was a base-signal clone. Its
        distinct thesis is now book sentiment (in-market repricing + aggressor
        flow), which leads/lags BTC spot and differs from the momentum bot's
        BTC-spot trend read.
        """
        from signals.lab import SignalView
        from bots.base_bot import strategy_decision
        sv = SignalView.of(signals)
        pm = sv.pm_momentum    # PM YES price momentum
        cvd = sv.cvd           # executed buy-sell flow, [-1,1]

        pm_w = self.strategy_params.get("pm_weight", 3.0)
        cvd_w = self.strategy_params.get("cvd_weight", 0.5)
        score = pm * pm_w + cvd * cvd_w                       # >0 bullish YES, <0 bearish

        contributing = {"pm_momentum": pm, "cvd": cvd, "score": score}
        deadband = self.strategy_params.get("deadband", 0.05)
        if abs(score) <= deadband:
            return strategy_decision(
                "hold", signals=contributing,
                reasoning=f"Neutral market sentiment: score={score:+.3f}")

        import config
        amount = config.get_max_position() * self.strategy_params["position_size_pct"]
        side = "yes" if score > 0 else "no"
        confidence = min(0.95, 0.35 + abs(score))
        return strategy_decision(
            "buy", side,
            edge=min(0.10, (abs(score) - deadband) * 0.10),
            confidence=confidence,
            reasoning=f"Market sentiment {side}: pm={pm:+.3f} cvd={cvd:+.3f} score={score:+.3f}",
            signals=contributing,
            suggested_amount=amount,
        )
