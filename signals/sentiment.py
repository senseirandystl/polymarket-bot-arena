"""Crypto news/social sentiment for BTC/SOL.

Scoring is two-tier:

1. **LLM scorer** — if a local Ollama server is reachable
   (``OLLAMA_URL``, default http://localhost:11434) each headline is scored
   0..1 by a small local model. Availability is probed lazily and re-checked
   every ``OLLAMA_REPROBE_SEC`` so a stopped/started Ollama is picked up
   without a restart. All scoring happens on the feed's background thread —
   never on the trading hot path.
2. **Keyword fallback** — the original bull/bear keyword ratio, used when
   Ollama is absent, times out, or returns garbage.

The sentiment dict feeds SentimentBot's analyze() thesis only (the ``strat``
lane); it carries no direct lane weight of its own.
"""

import os
import time
import threading
import logging
from collections import deque

logger = logging.getLogger(__name__)

# Simple keyword-based sentiment (fallback tier)
BULLISH_KEYWORDS = [
    "bull", "moon", "pump", "breakout", "ath", "buy", "long", "rocket",
    "surge", "rally", "green", "bullish", "up only", "send it", "wagmi",
]
BEARISH_KEYWORDS = [
    "bear", "dump", "crash", "sell", "short", "rug", "red", "bearish",
    "down", "collapse", "plunge", "rekt", "ngmi", "capitulate",
]

# Known crypto influencers (can be expanded)
INFLUENCERS = [
    "elonmusk", "vitalikbuterin", "caborossi", "cz_binance",
    "aaborossi", "solanalegend", "cryptowizardd",
]

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_SENTIMENT_MODEL", "llama3.2")
OLLAMA_TIMEOUT = 8.0
OLLAMA_REPROBE_SEC = 600.0

_PROMPT = (
    "Rate the sentiment of this crypto headline for the asset's short-term "
    "price. Reply with ONLY a number from 0.0 (very bearish) to 1.0 (very "
    "bullish); 0.5 is neutral.\nHeadline: {text}\nScore:"
)


class OllamaScorer:
    """Lazy-probed local-LLM headline scorer; None result = use fallback."""

    def __init__(self):
        self._available: bool | None = None
        self._probed_at = 0.0
        self._lock = threading.Lock()

    def _probe(self) -> bool:
        try:
            import requests
            resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
            ok = resp.status_code == 200
        except Exception:
            ok = False
        if ok != self._available:
            logger.info(f"Ollama sentiment scorer {'available' if ok else 'unavailable'}")
        return ok

    def available(self) -> bool:
        with self._lock:
            now = time.time()
            if self._available is None or (now - self._probed_at) > OLLAMA_REPROBE_SEC:
                self._available = self._probe()
                self._probed_at = now
            return self._available

    def score(self, text: str) -> float | None:
        """0..1 sentiment, or None on any failure (caller falls back)."""
        if not self.available():
            return None
        try:
            import requests
            resp = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": OLLAMA_MODEL,
                      "prompt": _PROMPT.format(text=text[:300]),
                      "stream": False,
                      "options": {"temperature": 0.0, "num_predict": 8}},
                timeout=OLLAMA_TIMEOUT)
            if resp.status_code != 200:
                return None
            raw = (resp.json().get("response") or "").strip()
            # Accept "0.7", "0.7 (bullish)", "Score: 0.7" — first float wins.
            for token in raw.replace(":", " ").split():
                try:
                    val = float(token)
                except ValueError:
                    continue
                if 0.0 <= val <= 1.0:
                    return val
            return None
        except Exception as e:
            logger.debug(f"Ollama scoring failed: {e}")
            # One failure marks it unavailable until the next re-probe —
            # don't pay an 8s timeout per headline while the server is down.
            with self._lock:
                self._available = False
                self._probed_at = time.time()
            return None


class SentimentFeed:
    def __init__(self, window_minutes=5, max_posts=500):
        self.posts = {"btc": deque(maxlen=max_posts), "sol": deque(maxlen=max_posts)}
        self.sentiment_history = {"btc": deque(maxlen=60), "sol": deque(maxlen=60)}
        self._running = False
        self._thread = None
        self.window_minutes = window_minutes
        self._llm = OllamaScorer()

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Sentiment feed started")

    def stop(self):
        self._running = False

    def _keyword_score(self, text: str) -> float:
        text_lower = text.lower()
        bull = sum(1 for kw in BULLISH_KEYWORDS if kw in text_lower)
        bear = sum(1 for kw in BEARISH_KEYWORDS if kw in text_lower)
        total = bull + bear
        return 0.5 if total == 0 else bull / total

    def _score_post(self, text: str, author: str = "") -> tuple:
        """Score a single post. Returns (score 0-1, is_influencer, source)."""
        llm_score = self._llm.score(text)
        if llm_score is not None:
            score, source = llm_score, "llm"
        else:
            score, source = self._keyword_score(text), "keyword"
        is_influencer = any(inf in author.lower() for inf in INFLUENCERS)
        return score, is_influencer, source

    def _run(self):
        """Poll for sentiment data on the background thread."""
        while self._running:
            try:
                self._fetch_sentiment()
            except Exception as e:
                logger.error(f"Sentiment fetch error: {e}")
            time.sleep(60)  # Check every minute

    def _fetch_sentiment(self):
        """Fetch recent crypto sentiment from available sources."""
        try:
            import requests

            # Option 1: CryptoPanic API (free tier)
            try:
                resp = requests.get(
                    "https://cryptopanic.com/api/free/v1/posts/",
                    params={"auth_token": "free", "currencies": "BTC,SOL", "kind": "news"},
                    timeout=10
                )
                if resp.status_code == 200:
                    data = resp.json()
                    seen = {p["text"] for sym in self.posts for p in self.posts[sym]}
                    for post in data.get("results", [])[:20]:
                        title = post.get("title", "")
                        if not title or title in seen:
                            continue  # don't re-score (LLM calls cost seconds)
                        score, is_inf, source = self._score_post(title)
                        symbol = "btc" if "btc" in title.lower() or "bitcoin" in title.lower() else "sol"
                        self.posts[symbol].append({
                            "text": title,
                            "score": score,
                            "is_influencer": is_inf,
                            "source": source,
                            "time": time.time(),
                        })
                    return
            except Exception:
                pass

            logger.debug("No sentiment source available this cycle")

        except Exception as e:
            logger.debug(f"Sentiment source error: {e}")

    def get_signals(self, symbol: str) -> dict:
        """Get current sentiment signals for a symbol."""
        sym = symbol.lower()
        if sym not in self.posts:
            return {}

        now = time.time()
        window_sec = self.window_minutes * 60
        recent = [p for p in self.posts[sym] if now - p["time"] < window_sec]

        if not recent:
            return {}

        scores = [p["score"] for p in recent]
        inf_scores = [p["score"] for p in recent if p["is_influencer"]]
        llm_count = sum(1 for p in recent if p.get("source") == "llm")

        avg_score = sum(scores) / len(scores) if scores else 0.5
        avg_inf_score = sum(inf_scores) / len(inf_scores) if inf_scores else 0.5

        # Calculate momentum (change in sentiment)
        prev_scores = list(self.sentiment_history.get(sym, []))
        momentum = 0
        if prev_scores:
            momentum = avg_score - (sum(prev_scores) / len(prev_scores))

        self.sentiment_history[sym].append(avg_score)

        return {
            "sentiment": {
                "score": avg_score,
                "influencer_score": avg_inf_score,
                "post_count": len(recent),
                "llm_scored": llm_count,
                "momentum": momentum,
            }
        }


_feed = None


def get_feed() -> SentimentFeed:
    global _feed
    if _feed is None:
        _feed = SentimentFeed()
    return _feed
