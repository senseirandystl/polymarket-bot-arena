"""Desk-cycle knobs. Imported by desk/* via getattr(config, ...) fallbacks.

Copy these onto config.py (or export them as env) if you want them visible
in one place. desk.cycle already uses the same defaults when config omits them.
"""
DESK_FACTORY_MODE = False
DESK_CYCLE_ENABLED = True
DESK_CYCLE_INTERVAL_SEC = 300.0
DESK_MAX_OPEN_SPECS = 8
DESK_MAX_NEW_PER_TICK = 2
DESK_PROMOTE_MIN_TRADES = 100
DESK_PROMOTE_MIN_DAYS = 7
DESK_PROMOTE_TRADE_FLOOR = 30
DESK_AUTO_LIVE = False
DESK_LLM_PROVIDER = "none"
OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_MODEL = "llama3.1"
XAI_API_KEY = ""
XAI_MODEL = "grok-4"
CRYPTO_UNIVERSE_PHASE = 1
