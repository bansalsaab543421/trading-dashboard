import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf

# ---------------- Page setup ----------------
st.set_page_config(page_title="Indian Stock Trading Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("### 🇮🇳 Pro Intraday Trading Dashboard")

col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    symbol = st.text_input("Enter NSE Stock Symbol:", "RELIANCE.NS")
with col2:
    interval = st.selectbox("Interval", ["15m", "1h", "1d"], index=2)
with col3:
    st.write("")
    if st.button("Refresh", type="primary"):
        st.rerun()

toggle_col1, toggle_col2 = st.columns(2)
with toggle_col1:
    show_rsi = st.checkbox("Show RSI panel", value=True)
with toggle_col2:
    show_macd = st.checkbox("Show MACD panel", value=True)


# ---------------- Indicators ----------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = (100 - (100 / (1 + rs))).fillna(50)

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    return df


def generate_signal(
    df: pd.DataFrame,
    rsi_overbought: float = 70,
    rsi_oversold: float = 30,
    atr_stop_mult: float = 1.5,
    atr_target_mult: float = 2.5,
) -> dict:
    """MACD-crossover entry confirmed by EMA trend and filtered by RSI,
    with ATR-based stop-loss/target. Fires BUY/SELL only on a fresh
    crossover, not on every single bar."""
    last = df.iloc[-1]
    prev = df.iloc[-2]

    trend = "UP" if last["EMA_20"] > last["EMA_50"] else "DOWN"

    action, reason = "HOLD", "No entry condition met"
    if pd.notna(prev["MACD"]) and pd.notna(prev["MACD_signal"]):
        macd_bull_cross = prev["MACD"] <= prev["MACD_signal"] and last["MACD"] > last["MACD_signal"]
        macd_bear_cross = prev["MACD"] >= prev["MACD_signal"] and last["MACD"] < last["MACD_signal"]

        if trend == "UP" and macd_bull_cross and last["RSI"] < rsi_overbought:
            action, reason = "BUY", "Uptrend + MACD bullish cross + RSI not overbought"
        elif trend == "DOWN" and macd_bear_cross and last["RSI"] > rsi_oversold:
            action, reason = "SELL", "Downtrend + MACD bearish cross + RSI not oversold"
    else:
        reason = "Warming up (not enough history yet)"

    ltp = float(last["Close"])
    atr_val = float(last["ATR"]) if pd.notna(last["ATR"]) else 0.0
    entry = sl = target = None

    if action == "BUY":
        entry = round(ltp, 2)
        sl = round(ltp - atr_stop_mult * atr_val, 2)
        target = round(ltp + atr_target_mult * atr_val, 2)
    elif action == "SELL":
        entry = round(ltp, 2)
        sl = round(ltp + atr_stop_mult * atr_val, 2)
        target = round(ltp - atr_target_mult * atr_val, 2)

    return {
        "trend": trend,
        "action": action,
        "reason": reason,
        "ltp": round(ltp, 2),
        "entry": entry,
        "sl": sl,
        "target": target,
        "rsi": round(float(last["RSI"]), 2) if pd.notna(last["RSI"]) else 50.0,
        "macd_hist": round(float(last["MACD_hist"]), 4) if pd.notna(last["MACD_hist"]) else 0.0,
    }


# ---------------- Main ----------------
try:
    period = "5d" if interval != "1d" else "6mo"
    df = yf.download(symbol, period=period, interval=interval)

    if df.empty:
        st.error("Invalid symbol or no data available.")
    else:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df = compute_indicators(df)

        if len(df) < 30:
            st.warning("Not enough candles yet for reliable indicators — try a longer period or the daily interval.")

        sig = generate_signal(df)

        prev_price = float(df["Close"].iloc[-2])
        chg = sig["ltp"] - prev_price
        chg_pct = (chg / prev_price) * 100 if prev_price else 0.0

        # ---- Single-line summary ----
        if sig["action"] in ("BUY", "SELL"):
            summary_line = (
                f"{sig['action']} {symbol} @ ₹{sig['entry']} | SL: ₹{sig['sl']} | "
                f"Target: ₹{sig['target']} | RSI: {sig['rsi']} | Trend: {sig['trend']}"
            )
        else:
            summary_line = (
                f"HOLD {symbol} — LTP ₹{sig['ltp']} | Trend: {sig['trend']} | "
                f"RSI: {sig['rsi']} | {sig['reason']}"
            )
        st.code(summary_line, language=None)

        # ---- Candlestick chart ----
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                increasing_line_color="#00d09c",
                increasing_fillcolor="#00d09c",
                decreasing_line_color="#eb5b3c",
                decreasing_fillcolor="#eb5b3c",
                name="Candles",
            )
        )
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_20"], line=dict(color="orange", width=1.5), name="EMA 20"))
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_50"], line=dict(color="purple", width=1.5), name="EMA 50"))

        # entry / SL / target lines + signal marker, drawn directly on the candles
        if sig["action"] in ("BUY", "SELL"):
            for price, color, label in [
                (sig["entry"], "#4f8cff", "Entry"),
                (sig["sl"], "#eb5b3c", "SL"),
                (sig["target"], "#00d09c", "Target"),
            ]:
                fig.add_hline(y=price, line_dash="dash", line_color=color, annotation_text=label, annotation_position="right")

            marker_color = "#00d09c" if sig["action"] == "BUY" else "#eb5b3c"
            marker_symbol = "triangle-up" if sig["action"] == "BUY" else "triangle-down"
            fig.add_trace(
                go.Scatter(
                    x=[df.index[-1]],
                    y=[sig["ltp"]],
                    mode="markers",
                    marker=dict(symbol=marker_symbol, size=16, color=marker_color, line=dict(width=1, color="white")),
                    name=sig["action"],
                )
            )

        fig.update_xaxes(
            rangebreaks=[dict(bounds=["sat", "mon"])] if interval == "1d" else [],
            showgrid=False,
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            margin=dict(l=10, r=10, t=10, b=10),
            height=450,
            xaxis_rangeslider_visible=False,
            yaxis=dict(showgrid=True, gridcolor="#31333F"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ---- RSI panel ----
        if show_rsi:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color="#4f8cff", width=1.5), name="RSI"))
            fig_rsi.add_hline(y=70, line_dash="dot", line_color="#eb5b3c")
            fig_rsi.add_hline(y=30, line_dash="dot", line_color="#00d09c")
            fig_rsi.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                margin=dict(l=10, r=10, t=10, b=10),
                height=150,
            )
            st.plotly_chart(fig_rsi, use_container_width=True)

        # ---- MACD panel ----
        if show_macd:
            fig_macd = go.Figure()
            colors = ["#00d09c" if v >= 0 else "#eb5b3c" for v in df["MACD_hist"].fillna(0)]
            fig_macd.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], marker_color=colors, name="Histogram"))
            fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD"], line=dict(color="#4f8cff", width=1.3), name="MACD"))
            fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], line=dict(color="orange", width=1.3), name="Signal"))
            fig_macd.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                margin=dict(l=10, r=10, t=10, b=10),
                height=150,
            )
            st.plotly_chart(fig_macd, use_container_width=True)

        # ---- Metric cards ----
        st.markdown("---")
        st.markdown("### 📊 Market Overview & Signal")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("LAST PRICE", f"₹{sig['ltp']:,.2f}", f"{chg:+.2f} ({chg_pct:+.2f}%)")
        m2.metric("RSI (14)", f"{sig['rsi']}")
        m3.metric("TREND", sig["trend"])
        m4.metric("SIGNAL", sig["action"])

        st.markdown("### 🎯 Actionable Levels")
        t1, t2, t3 = st.columns(3)
        t1.metric("ENTRY PRICE", f"₹{sig['entry']:,.2f}" if sig["entry"] else "--")
        t2.metric("TARGET PRICE", f"₹{sig['target']:,.2f}" if sig["target"] else "--")
        t3.metric("STOP LOSS", f"₹{sig['sl']:,.2f}" if sig["sl"] else "--")

        st.caption(sig["reason"])
        st.caption(
            "⚠️ Yahoo Finance data is a free feed and can be delayed / occasionally rate-limited — "
            "not a substitute for your broker's live terminal for actual execution timing. "
            "Educational tool, not financial advice. Signals are suggestions only — you decide and "
            "place any trade yourself through your broker."
        )

except Exception as e:
    st.warning("Please enter a valid stock symbol ending with .NS (e.g., RELIANCE.NS, TCS.NS)")
    st.exception(e)
