"""Sector and correlation analysis using sector ETFs as stand-ins for each
industry.

The eleven SPDR sector funds (XLK for technology, XLF for banks and finance,
and so on) track the S&P 500 split into its official sectors. We use them two
ways:

* **Which sector a stock belongs to** is decided from the data, not a hand-kept
  list: we see which sector fund the stock's daily moves most closely follow.
  This stays correct even when a company is reclassified or the universe
  changes.
* **Inter-sector** (how the sectors move relative to one another - which are
  leading, which are lagging, which rise and fall together) and **intra-sector**
  (how one stock compares with the others in its own sector) pictures, which
  feed both the indicators and a standalone read-out for the user.

Everything degrades gracefully: if the sector ETF price files are missing the
functions return ``None`` or empty results rather than failing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D

# The eleven S&P 500 sectors and the SPDR fund that tracks each one.
SECTOR_ETFS: dict[str, str] = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLC": "Communication Services",
}
MARKET_ETF = "SPY"


def _close(ticker: str, years: int = 25) -> pd.Series | None:
    """Closing prices for an ETF or stock, preferring the repo's saved files so
    the function works offline and in the research run."""
    repo = D.repo_prices(ticker)
    if repo is not None and len(repo):
        return repo["Close"].astype(float)
    try:
        df, _ = D.load_prices(ticker, years)
        return df["Close"].astype(float)
    except Exception:
        return None


def load_sector_closes(years: int = 25) -> dict[str, pd.Series]:
    """All available sector-ETF closing-price series, keyed by symbol."""
    out = {}
    for sym in SECTOR_ETFS:
        c = _close(sym, years)
        if c is not None and len(c) > 60:
            out[sym] = c
    return out


def _returns(c: pd.Series) -> pd.Series:
    return c.pct_change()


def rolling_corr(a: pd.Series, b: pd.Series, w: int) -> pd.Series:
    """Correlation of the daily returns of two price series over a trailing
    window of ``w`` days. A value near +1 means they move together, near 0 means
    unrelated, near -1 means opposite."""
    ra, rb = _returns(a), _returns(b)
    j = ra.index.intersection(rb.index)
    return ra.reindex(j).rolling(w).corr(rb.reindex(j))


def relative_strength(stock: pd.Series, bench: pd.Series, w: int = 63) -> pd.Series:
    """How much the stock out- or under-performed a benchmark over the last
    ``w`` days, as a simple return difference (stock return minus benchmark
    return). Positive means the stock is leading; negative means lagging."""
    j = stock.index.intersection(bench.index)
    s = stock.reindex(j)
    b = bench.reindex(j)
    return s.pct_change(w) - b.pct_change(w)


def assign_sector(ticker: str, stock_close: pd.Series | None = None, window: int | None = None,
                  sector_closes: dict | None = None) -> dict:
    """Decide a stock's sector from the data: the sector fund whose daily returns
    its own returns most closely match.

    ``window`` is how many of the most recent trading days to compare over;
    ``None`` (the default) uses the whole overlapping history. A long window is
    used on purpose, because a company's sector rarely changes and short windows
    are too noisy to pin it down (over a few months a stock can correlate as much
    with an unrelated sector as with its own).

    Returns ``{"etf", "name", "corr", "ranking"}`` where ``ranking`` lists every
    sector fund with its correlation, highest first; when the top two are close
    the match is genuinely ambiguous (common for mega-caps that move with the
    whole market). If the stock is itself a sector fund it matches itself.
    Returns ``etf=None`` when no sector data is available.
    """
    tkr = ticker.upper()
    if tkr in SECTOR_ETFS:
        return {"etf": tkr, "name": SECTOR_ETFS[tkr], "corr": 1.0,
                "ranking": [(tkr, 1.0)]}
    if stock_close is None:
        stock_close = _close(tkr)
    sect = sector_closes if sector_closes is not None else load_sector_closes()
    if stock_close is None or not sect:
        return {"etf": None, "name": None, "corr": np.nan, "ranking": []}
    rs = _returns(stock_close).dropna()
    if window is not None and len(rs) > window:
        rs = rs.iloc[-window:]
    scores = []
    for sym, c in sect.items():
        pair = pd.concat([rs.rename("a"), _returns(c).rename("b")], axis=1).dropna()
        if len(pair) < 30:
            continue
        cc = pair["a"].corr(pair["b"])
        if np.isfinite(cc):
            scores.append((sym, float(cc)))
    scores.sort(key=lambda x: x[1], reverse=True)
    if not scores:
        return {"etf": None, "name": None, "corr": np.nan, "ranking": []}
    best, corr = scores[0]
    return {"etf": best, "name": SECTOR_ETFS[best], "corr": corr, "ranking": scores}


def sector_momentum(sector_closes: dict | None = None, w: int = 63) -> pd.DataFrame:
    """Each sector fund's recent return and its distance (in %) above or below
    its own 50- and 200-day averages - a snapshot of which sectors are strong
    and which are weak right now. Sorted strongest first."""
    sect = sector_closes if sector_closes is not None else load_sector_closes()
    rows = []
    for sym, c in sect.items():
        if len(c) < 210:
            continue
        ret = c.pct_change(w).iloc[-1]
        ma50 = c.rolling(50).mean().iloc[-1]
        ma200 = c.rolling(200).mean().iloc[-1]
        px = c.iloc[-1]
        rows.append({"etf": sym, "sector": SECTOR_ETFS[sym], "return_w": float(ret),
                     "above_ma50": float(px / ma50 - 1) if ma50 else np.nan,
                     "above_ma200": float(px / ma200 - 1) if ma200 else np.nan})
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("return_w", ascending=False).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
    return df


def sector_corr_matrix(sector_closes: dict | None = None, w: int = 63) -> pd.DataFrame:
    """Correlation of daily returns between every pair of sector funds over the
    last ``w`` days. High correlation across the board means sectors are moving
    as one (little to diversify behind); low or negative values mean they are
    rotating."""
    sect = sector_closes if sector_closes is not None else load_sector_closes()
    if not sect:
        return pd.DataFrame()
    rets = pd.DataFrame({sym: _returns(c) for sym, c in sect.items()}).dropna()
    rets = rets.iloc[-w:] if len(rets) > w else rets
    if len(rets) < 20:
        return pd.DataFrame()
    return rets.corr()


def market_breadth(sector_closes: dict | None = None) -> dict:
    """A simple risk-on / risk-off reading: the share of sector funds trading
    above their own 50-day and 200-day averages. Near 1 means almost every
    sector is in an uptrend (broad strength); near 0 means broad weakness."""
    sect = sector_closes if sector_closes is not None else load_sector_closes()
    a50 = a200 = tot = 0
    for c in sect.values():
        if len(c) < 210:
            continue
        tot += 1
        px = c.iloc[-1]
        if px > c.rolling(50).mean().iloc[-1]:
            a50 += 1
        if px > c.rolling(200).mean().iloc[-1]:
            a200 += 1
    if tot == 0:
        return {"above_ma50": np.nan, "above_ma200": np.nan, "n": 0}
    return {"above_ma50": a50 / tot, "above_ma200": a200 / tot, "n": tot}


def correlation_report(ticker: str, stock_close: pd.Series | None = None,
                       years: int = 25) -> dict:
    """The full standalone correlation read-out for one stock: its data-chosen
    sector, how tightly it tracks that sector and the market now versus a year
    ago, whether it is leading or lagging both, which sectors are strong or weak,
    and how joined-up the sectors are. All context for the forward estimate."""
    tkr = ticker.upper()
    if stock_close is None:
        stock_close = _close(tkr, years)
    sect = load_sector_closes(years)
    spy = _close(MARKET_ETF, years)
    out: dict = {"ticker": tkr}
    asg = assign_sector(tkr, stock_close, sector_closes=sect)
    out["sector"] = asg
    if stock_close is None:
        return out
    etf_close = sect.get(asg["etf"]) if asg["etf"] else None

    def corr_now_then(bench):
        if bench is None:
            return None
        rc = rolling_corr(stock_close, bench, 63).dropna()
        if rc.empty:
            return None
        now = float(rc.iloc[-1])
        then = float(rc.iloc[-252]) if len(rc) > 252 else np.nan
        return {"now": now, "year_ago": then, "mean": float(rc.tail(252).mean())}

    out["corr_sector"] = corr_now_then(etf_close)
    out["corr_market"] = corr_now_then(spy)
    if etf_close is not None:
        rs = relative_strength(stock_close, etf_close, 63).dropna()
        out["rs_sector"] = float(rs.iloc[-1]) if len(rs) else np.nan
    if spy is not None:
        rs = relative_strength(stock_close, spy, 63).dropna()
        out["rs_market"] = float(rs.iloc[-1]) if len(rs) else np.nan
    out["sector_momentum"] = sector_momentum(sect)
    out["sector_corr_matrix"] = sector_corr_matrix(sect)
    out["market_breadth"] = market_breadth(sect)
    return out
