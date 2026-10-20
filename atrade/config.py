"""Settings for the tool.

Every number here comes from the framework design document unless it is
marked "tool choice". Users can change these on the Settings tab; changes are
saved to data/settings.json.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "prices"
SENTIMENT_DIR = DATA_DIR / "sentiment"
SETTINGS_FILE = DATA_DIR / "settings.json"
WATCHLIST_FILE = DATA_DIR / "watchlist.json"

DEFAULTS: dict = {
    # ---- Data -------------------------------------------------------------
    "history_years": 20,              # framework asks for at least 15 years
    # ---- Framework 1: cleaning -------------------------------------------
    "clean_window": 60,               # trailing window N for the 3-sigma check (tool choice)
    "clean_sigma": 3.0,
    "bad_quote_move": 0.25,           # one-day move that is reversed next day = bad quote (tool choice)
    "max_missing_share": 0.05,        # drop a stock if any year has >5% missing days
    # ---- Framework 3: trend ----------------------------------------------
    "mac_short": 10, "mac_long": 50,
    "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
    "mom_w": 12, "mom_signal": 9,
    "breakout_ws": [20, 50],
    "sma_ws": [10, 50, 200],
    "trend_state_days": 100,
    "trend_state_ma": 50,
    "trend_up_share": 0.70, "trend_down_share": 0.30,
    "filter_type": "time",            # none | price | time | wholeday
    "filter_price_x": 0.01,
    "filter_time_k": 1,
    "alexander_x": 0.05,              # benchmark only
    # ---- Framework 4a: zones ---------------------------------------------
    "zone_w": 50, "zone_x": 0.03, "zone_min_points": 2, "zone_start_days": 504,
    # ---- Framework 4b: patterns (window w per pattern, Table 5.2) ---------
    "w_hs": 7, "w_triple": 15, "w_double": 15, "w_wedge": 7, "w_flag": 3, "w_pennant": 2,
    "w_round": 10, "round_bounds": 0.3, "round_min_width": 15, "round_fit": 0.9,
    "flag_prior_days": 60,            # 'thor' in the book's FLAGS() code
    # ---- Framework 4c: gauges --------------------------------------------
    "rsi_w": 14, "rsi_low": 30, "rsi_high": 70,
    "bb_w": 20, "bb_k": 2.0,
    "roc_w": 12, "roc_train": 250, "roc_tp_w": 12,
    # ---- Framework 5: volume ---------------------------------------------
    "rv_days": 50, "rv_full": 1.5, "rv_half": 1.0,
    "use_volume_weights": True,
    # ---- Framework 6: sentiment ------------------------------------------
    "sp2_extreme": 5.0, "sp2_lookback": 10,
    "sp2a_max_loss": 0.10, "sp2a_trail_giveback": 0.50, "sp2a_trail_delay": 9,
    "sp2b_hold": 30,
    "vix_osc": [50, 100],             # framework allows 21/50 or 50/100
    "vix_trigger_days": 5,
    "tsunami_level": 10.0, "tsunami_hold": 30,
    "fastvol_level": 5.0,
    # ---- Framework 7: rating ---------------------------------------------
    "w_trend": 0.35, "w_reversal": 0.40, "w_sentiment": 0.25,
    "buy_cutoff": 0.40, "sell_cutoff": -0.40, "veto_level": 0.50,
    "live_days_indicator": 10,
    "live_days_zone": 10,
    "confirm_days": 2,
    "wt_pattern": 1.0, "wt_support": 1.0, "wt_resistance": 0.5, "wt_gauge": 0.5,
    "wt_sent_buy": 1.0, "wt_sent_sell": 0.5,
    "only_passed_rules": False,       # use only rules that passed the back-test
    "use_tuned_weights": False,       # use the weights tuned on training data (results/tuned_weights.json)
    "boot_B": 500,                    # artificial histories per null model in "Run full validation"
    "dtw_query_days": 61, "dtw_tau": 15.0, "dtw_band": 0, "dtw_top": 10,
    # ---- Framework 8: trade management -----------------------------------
    "stop_days": 2, "stop_loss": 0.04, "default_hold": 10,
    # ---- Framework 9: validation -----------------------------------------
    "holding_periods": [1, 2, 5, 10, 20, 40],
    "judge_period": 10,
    "cost_one_way": 0.0005,           # 0.05% per one-way trade (Hsu and Kuan)
    "spread_one_way": 0.0005,         # bid-ask estimate, half spread (tool choice)
    "split_train": 0.60, "split_valid": 0.20,
    "alpha": 0.05,
}


def load_settings() -> dict:
    s = copy.deepcopy(DEFAULTS)
    try:
        if SETTINGS_FILE.exists():
            saved = json.loads(SETTINGS_FILE.read_text())
            for k, v in saved.items():
                if k in s:
                    s[k] = v
    except Exception:
        pass
    return s


def save_settings(s: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(s, indent=2))


def load_watchlist() -> list[str]:
    try:
        if WATCHLIST_FILE.exists():
            return [t.upper() for t in json.loads(WATCHLIST_FILE.read_text())]
    except Exception:
        pass
    return ["AAPL", "MSFT", "IBM", "WMT", "GE"]


def save_watchlist(tickers: list[str]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    WATCHLIST_FILE.write_text(json.dumps([t.upper() for t in tickers], indent=2))
