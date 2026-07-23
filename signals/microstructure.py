"""Order-book microstructure features (pure, deterministic).

Inputs are the normalized book dicts produced by
``polymarket_markets.get_order_book`` (``bids``/``asks`` are ``(price, size)``
lists best-first, plus a ``valid`` flag). No network, no clocks, no state —
every function is a pure transform of the book(s) passed in.

Directional outputs (YES/Up-frame, bounded (-1, 1)):
- ``micro_obi_w``: distance-weighted order-book imbalance on the Up token —
  like the plain top-3 OBI but each level's size is decayed by its distance
  from the mid, so a wall 6c away no longer counts as much as size at the
  touch (the plain OBI's known blind spot).
- ``micro_cross``: cross-book bid-support imbalance — resting bid depth on
  the Up book vs the Down book. More real money willing to catch Up than
  Down = upward pressure.

Context outputs (non-directional):
- ``micro_spread``: Up-book spread as a fraction of price (raw, >= 0).
- ``micro_spread_score``: smooth 0..1 book-quality score (1 = tight book,
  ~0.5 at SPREAD_TYPICAL, → 0 on wide/gappy books).
- ``micro_depth``: total resting size within DEPTH_BAND of the Up mid (raw
  shares) — how much can actually be traded near the touch.

DIRECTIONAL OUTPUTS ARE CANDIDATES with no live weight: the plain OBI lane
measured INVERTED and is kill-switched (config.SIGNAL_WEIGHT_OBI = 0); these
variants must clear the offline/live validation bar before earning any weight.
Historical books are not archived, so these validate via live shadow
attribution (cand() reads in trade reasoning), not the offline backfill.
"""

import math

from signals.curves import sigmoid

OBI_LEVELS = 5              # book levels considered per side
OBI_DECAY = 0.02            # size at 2c from mid counts ~1/e of size at touch
DEPTH_BAND = 0.05           # depth counted within 5c of mid
SPREAD_TYPICAL = 0.03       # 3c spread on a ~50c token → spread_score ~0.5


def book_mid(book: dict) -> float:
    """Mid of best bid/ask (0.0 when the book is invalid or one-sided)."""
    if not book or not book.get("valid"):
        return 0.0
    bids, asks = book.get("bids") or [], book.get("asks") or []
    if not bids or not asks:
        return 0.0
    return (float(bids[0][0]) + float(asks[0][0])) / 2.0


def spread_pct(book: dict) -> float:
    """Best ask − best bid as a fraction of the mid (0.0 when unavailable)."""
    mid = book_mid(book)
    if mid <= 0:
        return 0.0
    spread = float(book["asks"][0][0]) - float(book["bids"][0][0])
    return max(0.0, spread / mid)


def weighted_imbalance(book: dict, levels: int = OBI_LEVELS,
                       decay: float = OBI_DECAY) -> float:
    """Distance-weighted OBI in [-1, 1]: +ve = bid-heavy (upward pressure)."""
    mid = book_mid(book)
    if mid <= 0:
        return 0.0
    bid_w = ask_w = 0.0
    for price, size in (book.get("bids") or [])[:levels]:
        bid_w += float(size) * math.exp(-abs(mid - float(price)) / decay)
    for price, size in (book.get("asks") or [])[:levels]:
        ask_w += float(size) * math.exp(-abs(float(price) - mid) / decay)
    total = bid_w + ask_w
    if total <= 0:
        return 0.0
    return max(-1.0, min(1.0, (bid_w - ask_w) / total))


def depth_within(book: dict, band: float = DEPTH_BAND) -> float:
    """Total resting size (both sides) within ``band`` of the mid, in shares."""
    mid = book_mid(book)
    if mid <= 0:
        return 0.0
    total = 0.0
    for price, size in (book.get("bids") or []):
        if mid - float(price) <= band:
            total += float(size)
    for price, size in (book.get("asks") or []):
        if float(price) - mid <= band:
            total += float(size)
    return total


def cross_book_pressure(yes_book: dict, no_book: dict,
                        band: float = DEPTH_BAND) -> float:
    """Bid-support imbalance between the Up and Down books, in [-1, 1].

    Compares resting BID depth near each book's own mid: more real money
    willing to catch the Up token than the Down token reads positive (YES
    pressure). 0.0 when either book is unusable.
    """
    def _bid_depth(book):
        mid = book_mid(book)
        if mid <= 0:
            return None
        return sum(float(size) for price, size in (book.get("bids") or [])
                   if mid - float(price) <= band)

    yes_d = _bid_depth(yes_book)
    no_d = _bid_depth(no_book)
    if yes_d is None or no_d is None or (yes_d + no_d) <= 0:
        return 0.0
    return max(-1.0, min(1.0, (yes_d - no_d) / (yes_d + no_d)))


def compute(yes_book: dict, no_book: dict | None = None) -> dict:
    """All microstructure features from the Up (and optionally Down) book."""
    spread = spread_pct(yes_book)
    return {
        "micro_obi_w": weighted_imbalance(yes_book),
        "micro_cross": (cross_book_pressure(yes_book, no_book)
                        if no_book else 0.0),
        "micro_spread": spread,
        # sigmoid falls as spread grows: 1 tight → 0 wide (steepness picked so
        # SPREAD_TYPICAL lands at 0.5).
        "micro_spread_score": sigmoid(-spread, center=-SPREAD_TYPICAL,
                                      steepness=100.0),
        "micro_depth": depth_within(yes_book),
    }
