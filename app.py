"""Academic Trading - browser app.

Start it with run_windows.bat (or:  streamlit run app.py).
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from atrade import data as D
from atrade.analysis import analyze
from atrade.config import DEFAULTS, load_settings, load_watchlist, save_settings, save_watchlist
from atrade.validation import bernoulli_test

st.set_page_config(page_title="Academic Trading", page_icon="📈", layout="wide")

GREEN, RED, GREY, BLUE, AMBER = "#1a7f37", "#cf222e", "#6e7781", "#0969da", "#bf8700"
RCOL = {"BUY": GREEN, "SELL": RED, "HOLD": GREY}
GROUP_NAME = {"trend": "Trend", "reversal": "Reversal", "sentiment": "Sentiment"}

st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
.rating-card {border-radius: 14px; padding: 18px 22px; color: white; margin-bottom: 6px;}
.rating-word {font-size: 44px; font-weight: 800; letter-spacing: 1px; line-height: 1.1;}
.rating-sub {font-size: 15px; opacity: .95;}
.small-note {color: #6e7781; font-size: 13px;}
.check-ok {color: #1a7f37; font-weight: 600;} .check-no {color: #cf222e; font-weight: 600;}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ state ---
if "settings" not in st.session_state:
    st.session_state.settings = load_settings()
S = st.session_state.settings


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def cached_analyze(ticker: str, settings_json: str, day: str, refresh_token: int):
    return analyze(ticker, json.loads(settings_json), refresh=refresh_token > 0)


def run(ticker, refresh=False):
    tok = int(dt.datetime.now().timestamp()) if refresh else 0
    with st.spinner(f"Analysing {ticker} ..."):
        return cached_analyze(ticker, json.dumps(S, sort_keys=True), dt.date.today().isoformat(), tok)


def date_of(res, i):
    """Date of day number i; days after the last close are counted forward in trading days."""
    idx = res["df"].index
    if i is None:
        return None
    if 0 <= i < len(idx):
        return idx[i].date()
    return (idx[-1] + pd.tseries.offsets.BDay(i - (len(idx) - 1))).date()


def pct(x, digits=1):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:+.{digits}f}%"


# ---------------------------------------------------------------- sidebar ---
with st.sidebar:
    st.markdown("## 📈 Academic Trading")
    st.caption("Buy / Sell / Hold ratings from rules in *Technical Analysis for Algorithmic "
               "Pattern Recognition* and *Trading Against the Crowd*.")
    ticker = st.text_input("Stock ticker", value=st.session_state.get("ticker", "DEMO"),
                           help="For example AAPL or MSFT. Type DEMO to try simulated data.").upper().strip()
    c1, c2 = st.columns(2)
    go_btn = c1.button("Analyse", type="primary", width="stretch")
    refresh = c2.button("Re-download", width="stretch",
                        help="Ignore today's saved copy and download fresh prices.")
    st.caption("Prices: Yahoo Finance, adjusted for splits and dividends. "
               "Ratings use closing prices, so check after 4:00 pm New York time.")

if go_btn or refresh or "res" not in st.session_state or st.session_state.get("ticker") != ticker:
    if ticker:
        try:
            st.session_state.res = run(ticker, refresh=refresh)
            st.session_state.ticker = ticker
            st.session_state.err = None
        except Exception as e:
            st.session_state.err = f"Could not analyse {ticker}: {e}"
            st.session_state.res = None

res = st.session_state.get("res")
if st.session_state.get("err"):
    st.error(st.session_state.err)
if not res:
    st.info("Type a ticker in the box on the left and press **Analyse**.")
    st.stop()

df = res["df"]
n = len(df)
last_day = df.index[-1].date()

# --------------------------------------------------------------- header -----
h1, h2 = st.columns([3, 2])
h1.markdown(f"# {res['ticker']}")
h1.caption(f"Last close {df['Close'].iloc[-1]:,.2f} on {last_day:%a %d %b %Y} · {res['source']}")
ny = dt.datetime.now(ZoneInfo("America/New_York"))
if last_day == ny.date() and ny.hour < 16:
    h2.warning("Today's prices are not final until the market closes at 4:00 pm New York time.")
if not res["quality"]["passed"]:
    h2.error(f"Data check failed: {res['quality']['worst_year_missing']:.1%} of trading days were missing in "
             f"{res['quality']['worst_year']} (limit 5%). Treat this rating with extra caution.")

tabs = st.tabs(["Rating", "Chart", "Signals", "Back-test", "Watch list", "Options data",
                "Data quality", "Settings", "How it works"])

# =============================================================== RATING ====
with tabs[0]:
    rt = res["rating"]
    word = res["rating_now"]
    last = rt.iloc[-1]
    since = None
    r_arr = rt["rating"].to_numpy(); j = n - 1
    while j > 0 and r_arr[j - 1] == r_arr[-1]:
        j -= 1
    since = df.index[j].date()
    meaning = {
        "BUY": "Trend, reversal and sentiment evidence mostly point the same way: up.",
        "SELL": "Trend, reversal and sentiment evidence mostly point the same way: down.",
        "HOLD": "The evidence is weak or conflicting, so the tool does not take a side.",
    }[word]
    left, right = st.columns([2, 3])
    with left:
        st.markdown(f"""<div class="rating-card" style="background:{RCOL[word]}">
            <div class="rating-word">{word}</div>
            <div class="rating-sub">Since {since:%d %b %Y} · score {last['score']:+.2f}</div></div>""",
                    unsafe_allow_html=True)
        st.write(meaning)
        raw_word = {1: "BUY", -1: "SELL", 0: "HOLD"}[int(last["raw"])]
        if raw_word != word:
            st.info(f"Today's evidence alone says **{raw_word}**. The rating only changes after "
                    f"{S['confirm_days']} days in a row, to stop it flipping back and forth.")

    with right:
        state = int(res["state"][-1])
        state_txt = {1: "uptrend", -1: "downtrend", 0: "sideways"}[state]
        for g, label, desc in (("T", "Trend", f"Trend state: {state_txt}, plus live crossover and breakout votes"),
                               ("R", "Reversal", "Chart patterns, support/resistance bounces, RSI, Bollinger, ROC"),
                               ("S", "Sentiment", "Options-market fear and greed (contrarian)")):
            v = float(last[g])
            col = GREEN if v > 0 else RED if v < 0 else GREY
            w = abs(v) * 50
            bar = (f'<div style="background:#eaeef2;border-radius:6px;height:12px;position:relative">'
                   f'<div style="position:absolute;left:{50 if v >= 0 else 50 - w}%;width:{w}%;height:12px;'
                   f'background:{col};border-radius:6px"></div>'
                   f'<div style="position:absolute;left:50%;top:-3px;width:2px;height:18px;background:#57606a"></div></div>')
            st.markdown(f"**{label}** &nbsp; <span style='color:{col};font-weight:700'>{v:+.2f}</span> "
                        f"<span class='small-note'>· weight {S['w_' + {'T': 'trend', 'R': 'reversal', 'S': 'sentiment'}[g]]:.2f} · {desc}</span>{bar}",
                        unsafe_allow_html=True)
            st.write("")

        G = [last["T"], last["R"], last["S"]]
        def chk(ok, text):
            return f"<span class='{'check-ok' if ok else 'check-no'}'>{'✔' if ok else '✘'}</span> {text}"
        b = [chk(last["score"] >= S["buy_cutoff"], f"score ≥ {S['buy_cutoff']:+.2f}"),
             chk(sum(x > 0 for x in G) >= 2, "at least 2 of 3 groups positive"),
             chk(min(G) >= -S["veto_level"], f"no group below −{S['veto_level']:.2f}")]
        sl = [chk(last["score"] <= S["sell_cutoff"], f"score ≤ {S['sell_cutoff']:+.2f}"),
              chk(sum(x < 0 for x in G) >= 2, "at least 2 of 3 groups negative"),
              chk(max(G) <= S["veto_level"], f"no group above +{S['veto_level']:.2f}")]
        st.markdown("**Buy needs:** " + " &nbsp; ".join(b), unsafe_allow_html=True)
        st.markdown("**Sell needs:** " + " &nbsp; ".join(sl), unsafe_allow_html=True)

    plan = res["plan"]
    if plan:
        st.subheader("Trade plan (Framework 8)")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Enter", "Next open", help="Every signal is entered at the next day's opening price.")
        if plan.get("target") is not None:
            p2.metric("Price target", f"{plan['target']:,.2f}", pct(plan["target"] / df['Close'].iloc[-1] - 1))
        else:
            p2.metric("Price target", "None", help="No chart pattern is live, so the trade uses a fixed holding period.")
        p3.metric("Time limit", f"{plan.get('time_limit_date'):%d %b %Y}" if plan.get("time_limit_date") else "–")
        p4.metric("Stop-loss", f"{plan['stop_loss_pct']:.0%} loss",
                  help=f"Also close the trade if price closes back across the broken line for {plan['stop_days']} days in a row.")
        extra = f"Plan taken from: **{plan['source']}** ({plan['kind']})."
        if plan.get("line_today") is not None:
            extra += f" Broken line today: {plan['line_today']:,.2f} — close the trade if price closes back across it for {plan['stop_days']} days in a row."
        st.caption(extra + f" Approximate stop price from today's close: {plan['approx_stop_price']:,.2f}.")

    st.subheader("Why — signals counting today")
    rows = []
    for sg, contrib, note in res["votes"]:
        rows.append({"Group": GROUP_NAME.get(sg.group, sg.group), "Rule": sg.rule,
                     "Vote": "Buy" if sg.direction > 0 else "Sell", "Counts as": round(contrib, 2),
                     "Signal date": date_of(res, sg.idx),
                     "Counts until": date_of(res, sg.live_until) if sg.live_until is not None else "until its exit signal",
                     "Reason": sg.text + (f" ({note})" if note else "")})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    else:
        st.write("No signals are live today.")
    st.caption("'Counts as' is the vote after its weight and the volume check. A volume weight of 0 means "
               "the signal day's volume was below its 50-day average, so the signal is not confirmed.")

    passes = res["passes"]
    npass = int(passes["verdict"].str.startswith("Passed").sum())
    st.warning(f"**Evidence check:** {npass} of {len(passes)} rules beat buy-and-hold after costs in this stock's "
               f"final test period ({S['judge_period']}-day holding period). The books found that most rules do "
               f"not beat chance, so treat every rating as a research aid, not advice. See the Back-test tab.")

    st.subheader("Rating history (last 12 months)")
    k0 = max(0, n - 252)
    sub = rt.iloc[k0:]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=df.index[k0:], y=df["Close"].iloc[k0:], name="Close", line=dict(color="#24292f", width=1.4)), 1, 1)
    for val, col in ((1, GREEN), (-1, RED)):
        m = sub["rating"] == val
        fig.add_trace(go.Bar(x=sub.index[m], y=[1] * int(m.sum()), marker_color=col, name={1: "Buy", -1: "Sell"}[val],
                             width=86400000, showlegend=True), 2, 1)
    fig.add_trace(go.Scatter(x=sub.index, y=sub["score"], name="Score", line=dict(color=BLUE, width=1), yaxis="y3"), 2, 1)
    fig.update_yaxes(showticklabels=False, row=2, col=1, range=[-1, 1.05])
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), bargap=0, legend=dict(orientation="h"),
                      plot_bgcolor="white")
    st.plotly_chart(fig)

# ================================================================ CHART ====
with tabs[1]:
    cc = st.columns([2, 1, 1, 1, 1, 1])
    span = cc[0].radio("Period", ["6 months", "1 year", "2 years", "5 years", "All"], index=1, horizontal=True)
    show_zones = cc[1].checkbox("Zones", True)
    show_pat = cc[2].checkbox("Patterns", True)
    show_ind = cc[3].checkbox("Indicator signals", False)
    show_bb = cc[4].checkbox("Bollinger", False)
    show_tp = cc[5].checkbox("Turning points", False)
    days = {"6 months": 126, "1 year": 252, "2 years": 504, "5 years": 1260, "All": n}[span]
    a = max(0, n - days)
    x = df.index[a:]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.62, 0.18, 0.20], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=x, open=df["Open"].iloc[a:], high=df["High"].iloc[a:], low=df["Low"].iloc[a:],
                                 close=df["Close"].iloc[a:], name="Price", increasing_line_color=GREEN,
                                 decreasing_line_color=RED, showlegend=False), 1, 1)
    from atrade.indicators import sma
    c = df["Close"].to_numpy(float)
    for w, col in ((50, AMBER), (200, BLUE)):
        fig.add_trace(go.Scatter(x=x, y=sma(c, w)[a:], name=f"{w}-day average", line=dict(color=col, width=1.2)), 1, 1)
    if show_bb:
        bb = res["gauges"]["bb"]
        fig.add_trace(go.Scatter(x=x, y=bb["up"][a:], name="Bollinger upper", line=dict(color=GREY, width=0.8, dash="dot")), 1, 1)
        fig.add_trace(go.Scatter(x=x, y=bb["low"][a:], name="Bollinger lower", line=dict(color=GREY, width=0.8, dash="dot")), 1, 1)
    if show_zones:
        lo_v, hi_v = df["Low"].iloc[a:].min(), df["High"].iloc[a:].max()
        for (zlo, zhi, lvl, cnt) in res["zones_now"]:
            if zhi >= lo_v * 0.95 and zlo <= hi_v * 1.05:
                fig.add_hrect(y0=zlo, y1=zhi, fillcolor=BLUE, opacity=0.10, line_width=0, row=1, col=1)
                fig.add_annotation(x=x[-1], y=lvl, text=f"zone {lvl:,.2f} ({cnt} pts)", showarrow=False,
                                   xanchor="right", font=dict(size=10, color=BLUE), row=1, col=1)
    if show_tp:
        from atrade.turning_points import rolling_window
        tp = rolling_window(c, int(S["w_hs"]))
        tp = tp[tp["idx"] >= a]
        for k, sym, col in (("P", "triangle-down", RED), ("T", "triangle-up", GREEN)):
            q = tp[tp["kind"] == k]
            fig.add_trace(go.Scatter(x=df.index[q["idx"]], y=q["price"], mode="markers", name=f"{'Peaks' if k == 'P' else 'Troughs'} (w={S['w_hs']})",
                                     marker=dict(symbol=sym, size=7, color=col, opacity=0.6)), 1, 1)
    if show_pat:
        for sg in res["patterns"]:
            if sg.idx < a:
                continue
            col = GREEN if sg.direction > 0 else RED
            pts = [p for p in sg.points if p[2] != "prior"]
            fig.add_trace(go.Scatter(x=[df.index[p[0]] for p in pts], y=[p[1] for p in pts], mode="lines+markers",
                                     line=dict(color=col, width=2), marker=dict(size=6), name=sg.rule,
                                     hovertext=sg.text, showlegend=False), 1, 1)
            fig.add_trace(go.Scatter(x=[df.index[sg.idx]], y=[c[sg.idx]], mode="markers+text", text=[sg.rule],
                                     textposition="top center" if sg.direction > 0 else "bottom center",
                                     textfont=dict(size=10, color=col),
                                     marker=dict(symbol="star", size=13, color=col), showlegend=False,
                                     hovertext=f"{sg.text} Target {sg.target:,.2f}"), 1, 1)
            if sg.target is not None and sg.time_limit is not None:
                tl = min(sg.time_limit, n - 1)
                fig.add_trace(go.Scatter(x=[df.index[sg.idx], df.index[tl]], y=[sg.target, sg.target], mode="lines",
                                         line=dict(color=col, dash="dash", width=1), showlegend=False,
                                         hovertext="Price target until the time limit"), 1, 1)
    # zone bounces and (optionally) indicator signals
    marks = [sg for sg in res["signals"] if sg.idx >= a and (sg.kind == "zone" or (show_ind and sg.kind in ("crossover", "breakout", "gauge")))]
    for d, sym, col in ((1, "triangle-up", GREEN), (-1, "triangle-down", RED)):
        q = [sg for sg in marks if sg.direction == d]
        if q:
            fig.add_trace(go.Scatter(x=[df.index[sg.idx] for sg in q],
                                     y=[df["Low"].iloc[sg.idx] * 0.985 if d > 0 else df["High"].iloc[sg.idx] * 1.015 for sg in q],
                                     mode="markers", marker=dict(symbol=sym, size=9, color=col, line=dict(width=1, color="white")),
                                     name="Buy signals" if d > 0 else "Sell signals",
                                     hovertext=[f"{sg.rule}: {sg.text}" for sg in q]), 1, 1)
    rv = res["rv"]
    vcol = [GREEN if v >= S["rv_full"] else BLUE if v >= S["rv_half"] else "#afb8c1" for v in np.nan_to_num(rv[a:])]
    fig.add_trace(go.Bar(x=x, y=df["Volume"].iloc[a:], marker_color=vcol, name="Volume", showlegend=False), 2, 1)
    r = res["gauges"]["rsi"]
    fig.add_trace(go.Scatter(x=x, y=r[a:], name="RSI 14", line=dict(color="#8250df", width=1.2), showlegend=False), 3, 1)
    for lvl in (S["rsi_low"], S["rsi_high"]):
        fig.add_hline(y=lvl, line=dict(color=GREY, dash="dot", width=1), row=3, col=1)
    fig.update_layout(height=760, margin=dict(l=10, r=10, t=10, b=10), xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.02), plot_bgcolor="white", hovermode="x unified")
    fig.update_yaxes(title_text="Price", row=1, col=1); fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], gridcolor="#eaeef2",
                     range=[x[0] - pd.Timedelta(days=2), x[-1] + pd.Timedelta(days=5)])
    fig.update_yaxes(gridcolor="#eaeef2")
    st.plotly_chart(fig)
    st.caption("Volume bars: green = at least 1.5× the 50-day average (signal confirmed), blue = 1.0–1.5× (half weight), "
               "grey = below average (signal not confirmed). Shaded bands are today's support/resistance zones. "
               "Stars mark chart-pattern breaks; the dashed line is the price target until the time limit.")

# ============================================================== SIGNALS ====
with tabs[2]:
    st.write("Every signal the rules produced, newest first. Signals are dated on the day they became known; "
             "a trade would start at the next day's open.")
    f1, f2 = st.columns(2)
    groups = f1.multiselect("Groups", ["Trend", "Reversal", "Sentiment"], default=["Reversal", "Sentiment"])
    lookback = f2.slider("Show the last … trading days", 20, min(n, 2520), min(252, n), step=20)
    rows = []
    for sg in res["signals"]:
        if sg.idx < n - lookback or GROUP_NAME.get(sg.group) not in groups:
            continue
        tr = sg.extra.get("trade") or {}
        rows.append({"Date": date_of(res, sg.idx), "Group": GROUP_NAME[sg.group], "Rule": sg.rule,
                     "Signal": "Buy" if sg.direction > 0 else "Sell",
                     "Volume ×avg": round(float(res["rv"][sg.idx]), 2) if not np.isnan(res["rv"][sg.idx]) else None,
                     "Target": round(sg.target, 2) if sg.target is not None else None,
                     "Outcome": tr.get("outcome", ""), "Trade return": pct(tr.get("ret")) if "ret" in tr else "",
                     "Trend before": sg.extra.get("trend_before", ""), "Reason": sg.text})
    sdf = pd.DataFrame(rows)
    if not sdf.empty:
        sdf = sdf.sort_values("Date", ascending=False)
        st.dataframe(sdf, hide_index=True, height=520)
        st.download_button("Download as CSV", sdf.to_csv(index=False).encode(), f"{res['ticker']}_signals.csv")
    else:
        st.write("No signals in this period for the chosen groups.")

# ============================================================= BACKTEST ====
with tabs[3]:
    st.write("Framework 9 asks every rule to prove itself on data it has never seen. The history is split in time: "
             "the first 60% (**training**), the next 20% (**validation**) and the last 20% (**final test**).")
    hp = st.select_slider("Holding period (trading days)", options=[int(h) for h in S["holding_periods"]],
                          value=int(S["judge_period"]))
    R = res["results"]
    blk = st.radio("Block", ["Final test", "Validation", "Training", "Whole history"], horizontal=True)
    sub = R[(R["h"] == hp) & (R["block"] == blk)].copy()
    alpha = float(S["alpha"])
    def verdict(r):
        if r["n"] < 5:
            return "Too few signals"
        return "Beat buy-and-hold" if (r["p"] < alpha and r["mean_net"] > r["mean_market"]) else "Did not beat"
    sub["Result"] = sub.apply(verdict, axis=1)
    show = pd.DataFrame({
        "Rule": sub["rule"], "Signals": sub["n"], "Buys": sub["n_buy"], "Sells": sub["n_sell"],
        "Avg return after costs": sub["mean_net"].map(pct), "Buy-and-hold avg": sub["mean_market"].map(pct),
        "Hit rate": sub["hit_rate"].map(lambda v: "–" if np.isnan(v) else f"{v:.0%}"),
        "p-value": sub["p"].map(lambda v: "–" if np.isnan(v) else f"{v:.3f}"),
        "p (buys)": sub["p_buy"].map(lambda v: "–" if np.isnan(v) else f"{v:.3f}"),
        "p (sells)": sub["p_sell"].map(lambda v: "–" if np.isnan(v) else f"{v:.3f}"),
        "Result": sub["Result"]})
    st.dataframe(show, hide_index=True, height=560)
    with st.expander("How to read this table"):
        st.markdown(f"""
* **Avg return after costs** – the average result of one trade that enters at the next day's open and holds for
  {hp} days (or closes earlier on an opposite signal from the same rule). Costs of
  {2 * (S['cost_one_way'] + S['spread_one_way']):.2%} per round trip are taken off. Sell signals count as short sales, so a
  falling price is a gain.
* **Buy-and-hold avg** – the average {hp}-day return from simply owning the stock, measured the same way on every day.
* **Hit rate** – the share of signals where the price moved the predicted way.
* **p-value** – the chance of a result at least this good by luck alone (one-sided Welch t-test). Below
  {alpha:.2f} counts as "not luck". *p (buys)* compares buy signals with buy-and-hold; *p (sells)* compares
  sell signals with short-and-hold.
* Version 1 does **not yet** run the random-price bootstrap (Framework 9, step 3), which the book found removes even
  more "significant" results. Treat a pass here as necessary, not sufficient.
""")
    po = res["pattern_outcomes"]
    if not po.empty:
        st.subheader("Chart-pattern trade outcomes (Framework 8)")
        pb = po[po["block"] == blk]
        if pb.empty:
            st.write("No pattern trades closed in this block.")
        else:
            st.dataframe(pd.DataFrame({
                "Pattern": pb["rule"], "Trades": pb["trades"],
                "Target hit": pb["target hit"].map(lambda v: f"{v:.0%}"),
                "Time limit": pb["time limit"].map(lambda v: f"{v:.0%}"),
                "Stopped out": pb["stopped out"].map(lambda v: f"{v:.0%}"),
                "Avg return (before costs)": pb["avg return"].map(pct)}), hide_index=True)
            st.caption("In the book's tests (Tables 5.4–5.5) most pattern trades were stopped out, and similar shares "
                       "appeared in random price series.")

# =========================================================== WATCH LIST ====
with tabs[4]:
    wl_text = st.text_area("Tickers (one per line or separated by commas)", "\n".join(load_watchlist()), height=130)
    tick = [t.strip().upper() for t in wl_text.replace(",", "\n").splitlines() if t.strip()]
    b1, b2, b3 = st.columns(3)
    if b1.button("Save list"):
        save_watchlist(tick); st.success("Saved.")
    rate_all = b2.button("Rate all", type="primary")
    snap_all = b3.button("Save today's options data for all",
                         help="Builds each stock's own put/call and implied-volatility history. Run once a day after the close.")
    if snap_all:
        prog = st.progress(0.0)
        for i, t in enumerate(tick):
            try:
                row = D.snapshot_options(t)
                st.write(f"{t}: put/call {row['put_call_ratio']:.2f}, average implied volatility {row['avg_iv']:.1%} "
                         f"({D.snapshot_count(t)} days saved)")
            except Exception as e:
                st.write(f"{t}: {e}")
            prog.progress((i + 1) / len(tick))
    if rate_all:
        out, results = [], {}
        prog = st.progress(0.0)
        for i, t in enumerate(tick):
            try:
                r = run(t)
                results[t] = r
                last = r["rating"].iloc[-1]
                arr = r["rating"]["rating"].to_numpy(); j = len(arr) - 1
                while j > 0 and arr[j - 1] == arr[-1]:
                    j -= 1
                top = sorted(r["votes"], key=lambda v: -abs(v[1]))
                out.append({"Ticker": t, "Rating": r["rating_now"], "Since": r["df"].index[j].date(),
                            "Score": round(float(last["score"]), 2), "Trend": round(float(last["T"]), 2),
                            "Reversal": round(float(last["R"]), 2), "Sentiment": round(float(last["S"]), 2),
                            "Close": round(float(r["df"]["Close"].iloc[-1]), 2),
                            "Target": round(r["plan"]["target"], 2) if r["plan"] and r["plan"].get("target") else None,
                            "Main reason": top[0][0].rule if top and top[0][1] != 0 else ""})
            except Exception as e:
                out.append({"Ticker": t, "Rating": "error", "Main reason": str(e)[:80]})
            prog.progress((i + 1) / len(tick))
        st.session_state.wl = (out, results)
    if st.session_state.get("wl"):
        out, results = st.session_state.wl
        wdf = pd.DataFrame(out)
        st.dataframe(wdf.style.map(lambda v: f"color:{RCOL.get(v, 'inherit')};font-weight:700" if v in RCOL else "",
                                   subset=["Rating"]), hide_index=True)
        if len(results) >= 2:
            st.subheader("Does each rule work across these stocks? (Framework 9, step 5)")
            st.write("Each stock counts as one coin flip: a success if the rule's average return after costs beat "
                     f"buy-and-hold in that stock's final test period ({S['judge_period']}-day holding). "
                     "With no skill, about half the stocks would succeed by chance.")
            agg = {}
            for t, r in results.items():
                q = r["results"]; q = q[(q["block"] == "Final test") & (q["h"] == int(S["judge_period"])) & (q["n"] >= 1)]
                for _, row in q.iterrows():
                    a2 = agg.setdefault(row["rule"], [0, 0])
                    a2[1] += 1
                    a2[0] += int(row["mean_net"] > row["mean_market"])
            rows = []
            for rule, (x_, N_) in agg.items():
                bt = bernoulli_test(x_, N_)
                rows.append({"Rule": rule, "Stocks tested": N_, "Beat buy-and-hold": x_, "Method": bt["method"],
                             "z": None if np.isnan(bt["z"]) else round(bt["z"], 2), "p-value": round(bt["p"], 3)})
            st.dataframe(pd.DataFrame(rows).sort_values("p-value"), hide_index=True)
            st.caption("For fewer than 30 stocks the exact binomial probability is used instead of the z-score.")

# ========================================================= OPTIONS DATA ====
with tabs[5]:
    st.write("Framework 6 reads the mood of options traders and bets against extremes. Free sources do not "
             "provide a history of put/call ratios, so this tool builds one: save a snapshot every day after the "
             "close, or import a history file if you have one.")
    stt = pd.DataFrame(res["sentiment_status"])
    stt["Status"] = stt.apply(lambda r: "Active" if r["active"] else f"Waiting: {r['days']} of {r['needed']} days", axis=1)
    st.dataframe(stt[["system", "data", "Status"]].rename(columns={"system": "System", "data": "Data needed"}),
                 hide_index=True)
    o1, o2 = st.columns(2)
    with o1:
        st.subheader("Daily snapshot")
        st.write(f"Snapshots saved for {res['ticker']}: **{D.snapshot_count(res['ticker'])}** days.")
        if st.button(f"Save today's options data for {res['ticker']}"):
            try:
                row = D.snapshot_options(res["ticker"])
                st.success(f"Saved {row['date']}: put volume {row['put_volume']:,.0f}, call volume {row['call_volume']:,.0f}, "
                           f"put/call {row['put_call_ratio']:.2f}, average implied volatility {row['avg_iv']:.1%}.")
            except Exception as e:
                st.error(str(e))
        st.caption("Tip: the file collect_options_windows.bat saves snapshots for your whole watch list. "
                   "The README explains how to make Windows run it every weekday at 5 pm.")
        st.subheader("Equity-only put/call ratio")
        if st.button("Download Cboe's free archive"):
            try:
                st.success(D.try_download_cboe_equity_pc())
            except Exception as e:
                st.error(f"Download failed: {e}")
        st.caption("Cboe's free file stops in 2019. For later years, import a file from a data provider.")
    with o2:
        st.subheader("Import a history file")
        kind = st.selectbox("What is in the file?", [f"{res['ticker']} put/call ratio", f"{res['ticker']} implied volatility",
                                                     "Equity-only put/call ratio (whole market)"])
        up = st.file_uploader("CSV file with a date column and a value column", type=["csv"])
        if up is not None and st.button("Import"):
            try:
                sr = D.parse_series_csv(up.getvalue())
                key = ("equity_pc" if kind.startswith("Equity") else
                       f"{res['ticker']}_pc" if "put/call" in kind else f"{res['ticker']}_iv")
                cnt = D.save_import(key, sr)
                st.success(f"Imported {len(sr)} rows ({sr.index[0].date()} to {sr.index[-1].date()}); {cnt} days stored. "
                           "Press Re-download in the sidebar to re-run the analysis.")
            except Exception as e:
                st.error(str(e))
    ex = {k: v for k, v in res["extremes"].items() if v}
    if ex:
        st.subheader("How extreme is the mood right now? (10/250 oscillator vs. its record so far)")
        names = {"vix": "VIX", "equity_pc": "Equity put/call", "stock_pc": "Stock put/call", "stock_iv": "Stock implied volatility"}
        st.dataframe(pd.DataFrame([{"Series": names.get(k, k), "Oscillator": round(v["value"], 1),
                                    "Record high": round(v["record_high"], 1), "Record low": round(v["record_low"], 1),
                                    "Reading": "Extreme fear (buy setup)" if v["near_high"] else
                                    "Extreme greed (sell setup)" if v["near_low"] else "Not extreme"} for k, v in ex.items()]),
                     hide_index=True)

# ========================================================= DATA QUALITY ====
with tabs[6]:
    q = res["quality"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Trading days", f"{q['n_days']:,}", f"{q['first']} to {q['last']}", delta_color="off")
    m2.metric("Logical fixes", q["n_logical"])
    m3.metric("Days filled in", q["n_interpolated"])
    m4.metric("Extreme returns flagged", q["n_extreme"])
    st.write(("✔ Passed: " if q["passed"] else "✘ Failed: ") +
             f"the worst year ({q['worst_year']}) was missing {q['worst_year_missing']:.1%} of trading days (limit 5%).")
    st.caption("Extreme returns are kept in the prices (so chart levels stay true) and pulled back to the 3-standard-"
               "deviation limit only in the statistics.")
    log = pd.DataFrame(res["clean_log"])
    if not log.empty:
        st.dataframe(log, hide_index=True, height=420)
        st.download_button("Download cleaning log", log.to_csv(index=False).encode(), f"{res['ticker']}_cleaning_log.csv")

# ============================================================= SETTINGS ====
with tabs[7]:
    st.write("Starting values come from the framework. The books warn that tuning settings until the past looks "
             "good ('data snooping') makes results look better than they are, so change these sparingly.")
    with st.form("settings_form"):
        a1, a2, a3 = st.columns(3)
        with a1:
            st.markdown("**Rating engine (Framework 7)**")
            wt = st.number_input("Trend weight", 0.0, 1.0, float(S["w_trend"]), 0.05)
            wr = st.number_input("Reversal weight", 0.0, 1.0, float(S["w_reversal"]), 0.05)
            ws = st.number_input("Sentiment weight", 0.0, 1.0, float(S["w_sentiment"]), 0.05)
            cut = st.number_input("Buy/Sell cut-off (±)", 0.05, 1.0, float(S["buy_cutoff"]), 0.05)
            conf = st.number_input("Days a new rating must hold", 1, 10, int(S["confirm_days"]))
            onlyp = st.checkbox("Use only rules that passed the back-test on this stock", bool(S["only_passed_rules"]),
                                help="Framework 9's pass rule. Note: this uses the final-test results, so the rating's own back-test is no longer a clean test.")
        with a2:
            st.markdown("**Signals (Frameworks 3–5)**")
            ftype = st.selectbox("False-signal filter", ["time", "price", "wholeday", "none"],
                                 index=["time", "price", "wholeday", "none"].index(S["filter_type"]),
                                 format_func={"time": "Time: stay beyond the line k days", "price": "Price: move x% beyond the line",
                                              "wholeday": "Whole day: next day's range beyond the line", "none": "None"}.get)
            fk = st.number_input("Time filter k (days)", 1, 3, int(S["filter_time_k"]))
            fx = st.number_input("Price filter x (%)", 1.0, 3.0, float(S["filter_price_x"]) * 100, 0.5)
            live = st.number_input("Days an indicator signal stays live", 1, 40, int(S["live_days_indicator"]))
            usev = st.checkbox("Weight signals by volume", bool(S["use_volume_weights"]))
            zx = st.number_input("Zone bin width (%)", 1.0, 10.0, float(S["zone_x"]) * 100, 0.5)
            zw = st.number_input("Zone turning-point window w", 10, 100, int(S["zone_w"]))
        with a3:
            st.markdown("**Testing and data (Frameworks 8–9)**")
            jp = st.selectbox("Holding period used for pass/fail", [int(h) for h in S["holding_periods"]],
                              index=[int(h) for h in S["holding_periods"]].index(int(S["judge_period"])))
            cost = st.number_input("Commission per one-way trade (%)", 0.0, 1.0, float(S["cost_one_way"]) * 100, 0.01)
            spr = st.number_input("Half bid-ask spread per trade (%)", 0.0, 1.0, float(S["spread_one_way"]) * 100, 0.01)
            stop = st.number_input("Stop-loss (%)", 1.0, 15.0, float(S["stop_loss"]) * 100, 0.5)
            yrs = st.number_input("Years of history to download", 5, 40, int(S["history_years"]))
            vix = st.selectbox("VIX oscillator for Squeeze Play II", ["50/100", "21/50"],
                               index=0 if list(S["vix_osc"]) == [50, 100] else 1)
        saved = st.form_submit_button("Save and re-run", type="primary")
    if saved:
        S.update({"w_trend": wt, "w_reversal": wr, "w_sentiment": ws, "buy_cutoff": cut, "sell_cutoff": -cut,
                  "confirm_days": int(conf), "only_passed_rules": onlyp, "filter_type": ftype, "filter_time_k": int(fk),
                  "filter_price_x": fx / 100, "live_days_indicator": int(live), "use_volume_weights": usev,
                  "zone_x": zx / 100, "zone_w": int(zw), "judge_period": int(jp), "cost_one_way": cost / 100,
                  "spread_one_way": spr / 100, "stop_loss": stop / 100, "history_years": int(yrs),
                  "vix_osc": [50, 100] if vix == "50/100" else [21, 50]})
        save_settings(S)
        st.session_state.pop("res", None)
        st.rerun()
    if st.button("Reset all settings to the framework's starting values"):
        st.session_state.settings = dict(DEFAULTS)
        save_settings(st.session_state.settings)
        st.session_state.pop("res", None)
        st.rerun()

# ========================================================= HOW IT WORKS ====
with tabs[8]:
    st.markdown(Path(__file__).with_name("HOW_IT_WORKS.md").read_text(encoding="utf-8"))
