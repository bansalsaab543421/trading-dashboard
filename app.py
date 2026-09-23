import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf

# ============================================================
# Indian Stock Trading Dashboard - Enhanced Version
# ============================================================

st.set_page_config(
    page_title="Indian Stock Trading Dashboard",
    page_icon="📈",
    layout="wide",
)

# ---------------- Styling ----------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .signal-box {
        padding: 14px 18px;
        border-radius: 10px;
        border: 1px solid #31333F;
        background: #151922;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("## 🇮🇳 Pro Indian Stock Trading Dashboard")
st.caption("Technical-analysis dashboard with trend, momentum, volume, VWAP and ATR-based risk management.")

# ---------------- Inputs ----------------
col1, col2, col3, col4 = st.columns([3, 1.2, 1.2, 1])

with col1:
    symbol = st.text_input(
        "NSE Stock Symbol",
        "RELIANCE.NS",
        help="Examples: RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS",
    ).strip().upper()

with col2:
    interval = st.selectbox("Interval", ["15m", "1h", "1d"], index=0)

with col3:
    capital = st.number_input(
        "Trading Capital (₹)",
        min_value=1000.0,
        value=100000.0,
        step=5000.0,
    )

with col4:
    if st.button("🔄 Refresh", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

t1, t2, t3 = st.columns(3)

with t1:
    show_rsi = st.checkbox("Show RSI", value=True)

with t2:
    show_macd = st.checkbox("Show MACD", value=True)

with t3:
    show_volume = st.checkbox("Show Volume", value=True)


# ============================================================
# Indicator Functions
# ============================================================

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # EMAs
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = (100 - (100 / (1 + rs))).fillna(50)

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # ATR
    prev_close = df["Close"].shift(1)

    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["ATR"] = tr.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    # Volume average
    df["Volume_MA20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = (
        df["Volume"] / df["Volume_MA20"].replace(0, np.nan)
    )

    # VWAP
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3

    # For intraday data, calculate cumulative VWAP by trading day.
    if interval != "1d":
        session = pd.Series(df.index.date, index=df.index)
        cumulative_pv = (typical_price * df["Volume"]).groupby(session).cumsum()
        cumulative_volume = df["Volume"].groupby(session).cumsum()
        df["VWAP"] = cumulative_pv / cumulative_volume.replace(0, np.nan)
    else:
        cumulative_pv = (typical_price * df["Volume"]).cumsum()
        cumulative_volume = df["Volume"].cumsum()
        df["VWAP"] = cumulative_pv / cumulative_volume.replace(0, np.nan)

    # ADX / Directional Movement
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0.0,
        ),
        index=df.index,
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0.0,
        ),
        index=df.index,
    )

    atr = df["ATR"].replace(0, np.nan)

    plus_di = 100 * plus_dm.ewm(
        alpha=1 / 14,
        adjust=False,
    ).mean() / atr

    minus_di = 100 * minus_dm.ewm(
        alpha=1 / 14,
        adjust=False,
    ).mean() / atr

    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    df["ADX"] = dx.ewm(
        alpha=1 / 14,
        adjust=False,
    ).mean()

    # Simple Supertrend
    multiplier = 3.0
    hl2 = (df["High"] + df["Low"]) / 2

    basic_upper = hl2 + multiplier * df["ATR"]
    basic_lower = hl2 - multiplier * df["ATR"]

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    supertrend = pd.Series(index=df.index, dtype=float)

    for i in range(1, len(df)):
        if (
            basic_upper.iloc[i] < final_upper.iloc[i - 1]
            or df["Close"].iloc[i - 1] > final_upper.iloc[i - 1]
        ):
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        if (
            basic_lower.iloc[i] > final_lower.iloc[i - 1]
            or df["Close"].iloc[i - 1] < final_lower.iloc[i - 1]
        ):
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        if pd.isna(supertrend.iloc[i - 1]):
            supertrend.iloc[i] = final_upper.iloc[i]
        elif supertrend.iloc[i - 1] == final_upper.iloc[i - 1]:
            if df["Close"].iloc[i] <= final_upper.iloc[i]:
                supertrend.iloc[i] = final_upper.iloc[i]
            else:
                supertrend.iloc[i] = final_lower.iloc[i]
        else:
            if df["Close"].iloc[i] >= final_lower.iloc[i]:
                supertrend.iloc[i] = final_lower.iloc[i]
            else:
                supertrend.iloc[i] = final_upper.iloc[i]

    df["Supertrend"] = supertrend

    # Previous day high / low
    if interval != "1d":
        daily = df.resample("1D").agg(
            {"High": "max", "Low": "min"}
        ).dropna()

        daily["Prev_Day_High"] = daily["High"].shift(1)
        daily["Prev_Day_Low"] = daily["Low"].shift(1)

        prev_day = daily[["Prev_Day_High", "Prev_Day_Low"]].reindex(
            df.index.normalize(),
            method="ffill",
        )
        prev_day.index = df.index

        df["Prev_Day_High"] = prev_day["Prev_Day_High"].values
        df["Prev_Day_Low"] = prev_day["Prev_Day_Low"].values
    else:
        df["Prev_Day_High"] = df["High"].shift(1)
        df["Prev_Day_Low"] = df["Low"].shift(1)

    return df


# ============================================================
# Signal Engine
# ============================================================

def generate_signal(
    df: pd.DataFrame,
    atr_stop_mult: float = 1.5,
    atr_target_mult: float = 3.0,
) -> dict:

    if len(df) < 30:
        raise ValueError("Not enough historical candles.")

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Trend
    if last["EMA_20"] > last["EMA_50"]:
        trend = "UP"
    elif last["EMA_20"] < last["EMA_50"]:
        trend = "DOWN"
    else:
        trend = "SIDEWAYS"

    # MACD crossover
    bull_cross = (
        prev["MACD"] <= prev["MACD_signal"]
        and last["MACD"] > last["MACD_signal"]
    )

    bear_cross = (
        prev["MACD"] >= prev["MACD_signal"]
        and last["MACD"] < last["MACD_signal"]
    )

    # Conditions
    bull_conditions = {
        "EMA Trend": last["EMA_20"] > last["EMA_50"],
        "MACD": last["MACD"] > last["MACD_signal"],
        "Fresh MACD Cross": bull_cross,
        "RSI": 50 <= last["RSI"] < 70,
        "VWAP": last["Close"] > last["VWAP"],
        "Volume": last["Volume_Ratio"] >= 1.0,
        "ADX": last["ADX"] >= 20,
        "Supertrend": last["Close"] > last["Supertrend"],
    }

    bear_conditions = {
        "EMA Trend": last["EMA_20"] < last["EMA_50"],
        "MACD": last["MACD"] < last["MACD_signal"],
        "Fresh MACD Cross": bear_cross,
        "RSI": 30 < last["RSI"] <= 50,
        "VWAP": last["Close"] < last["VWAP"],
        "Volume": last["Volume_Ratio"] >= 1.0,
        "ADX": last["ADX"] >= 20,
        "Supertrend": last["Close"] < last["Supertrend"],
    }

    bull_score = sum(bull_conditions.values())
    bear_score = sum(bear_conditions.values())

    # Require multiple confirmations.
    if bull_score >= 6 and bull_conditions["Fresh MACD Cross"]:
        action = "BUY"
        score = bull_score
        conditions = bull_conditions
        reason = "Multiple bullish confirmations aligned."
    elif bear_score >= 6 and bear_conditions["Fresh MACD Cross"]:
        action = "SELL"
        score = bear_score
        conditions = bear_conditions
        reason = "Multiple bearish confirmations aligned."
    else:
        action = "HOLD"
        score = max(bull_score, bear_score)
        conditions = bull_conditions if bull_score >= bear_score else bear_conditions
        reason = "Waiting for stronger confirmation."

    ltp = float(last["Close"])
    atr = float(last["ATR"]) if pd.notna(last["ATR"]) else 0.0

    entry = sl = target = None

    if action == "BUY":
        entry = round(ltp, 2)
        sl = round(ltp - atr_stop_mult * atr, 2)
        target = round(ltp + atr_target_mult * atr, 2)

    elif action == "SELL":
        entry = round(ltp, 2)
        sl = round(ltp + atr_stop_mult * atr, 2)
        target = round(ltp - atr_target_mult * atr, 2)

    risk_per_share = (
        abs(entry - sl)
        if entry is not None and sl is not None
        else 0
    )

    reward_per_share = (
        abs(target - entry)
        if entry is not None and target is not None
        else 0
    )

    rr = (
        reward_per_share / risk_per_share
        if risk_per_share > 0
        else 0
    )

    confidence = round((score / 8) * 100, 1)

    return {
        "trend": trend,
        "action": action,
        "score": score,
        "confidence": confidence,
        "reason": reason,
        "conditions": conditions,
        "ltp": round(ltp, 2),
        "entry": entry,
        "sl": sl,
        "target": target,
        "risk_per_share": round(risk_per_share, 2),
        "reward_per_share": round(reward_per_share, 2),
        "rr": round(rr, 2),
        "rsi": round(float(last["RSI"]), 2),
        "adx": round(float(last["ADX"]), 2),
        "volume_ratio": round(float(last["Volume_Ratio"]), 2),
        "atr": round(atr, 2),
        "vwap": round(float(last["VWAP"]), 2),
    }


# ============================================================
# Data Download
# ============================================================

@st.cache_data(ttl=60, show_spinner=False)
def load_data(ticker: str, selected_interval: str) -> pd.DataFrame:
    period = "5d" if selected_interval != "1d" else "1y"

    data = yf.download(
        ticker,
        period=period,
        interval=selected_interval,
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        return data

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]
    data = data[[c for c in required if c in data.columns]].dropna(
        subset=["Open", "High", "Low", "Close"]
    )

    return data


# ============================================================
# Main
# ============================================================

try:
    if not symbol:
        st.warning("Please enter an NSE symbol.")
        st.stop()

    df = load_data(symbol, interval)

    if df.empty:
        st.error(
            "No data found. Please check the symbol, e.g. RELIANCE.NS or TCS.NS."
        )
        st.stop()

    df = compute_indicators(df)

    if len(df) < 30:
        st.warning(
            "Not enough candles for reliable analysis. "
            "Try Daily interval or another stock."
        )
        st.stop()

    sig = generate_signal(df)

    # Price change
    prev_price = float(df["Close"].iloc[-2])
    chg = sig["ltp"] - prev_price
    chg_pct = (chg / prev_price) * 100 if prev_price else 0.0

    # ========================================================
    # Signal Header
    # ========================================================

    signal_text = (
        f"{sig['action']} | Confidence: {sig['confidence']:.0f}% | "
        f"Score: {sig['score']}/8"
    )

    st.markdown(
        f"""
        <div class="signal-box">
        <h3>{signal_text}</h3>
        <b>{symbol}</b> | LTP ₹{sig['ltp']:,.2f} |
        Trend: <b>{sig['trend']}</b><br>
        {sig['reason']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ========================================================
    # Main Chart
    # ========================================================

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

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["EMA_20"],
            line=dict(color="orange", width=1.5),
            name="EMA 20",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["EMA_50"],
            line=dict(color="purple", width=1.5),
            name="EMA 50",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["VWAP"],
            line=dict(color="#4f8cff", width=1.4),
            name="VWAP",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Supertrend"],
            line=dict(color="#f5c542", width=1.2),
            name="Supertrend",
        )
    )

    # Previous day levels
    if interval != "1d":
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Prev_Day_High"],
                line=dict(color="#aaaaaa", width=1, dash="dot"),
                name="Prev Day High",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Prev_Day_Low"],
                line=dict(color="#777777", width=1, dash="dot"),
                name="Prev Day Low",
            )
        )

    # Entry / SL / Target
    if sig["action"] in ("BUY", "SELL"):
        for price, color, label in [
            (sig["entry"], "#4f8cff", "Entry"),
            (sig["sl"], "#eb5b3c", "Stop Loss"),
            (sig["target"], "#00d09c", "Target"),
        ]:
            fig.add_hline(
                y=price,
                line_dash="dash",
                line_color=color,
                annotation_text=label,
                annotation_position="right",
            )

        marker_color = (
            "#00d09c" if sig["action"] == "BUY" else "#eb5b3c"
        )

        marker_symbol = (
            "triangle-up" if sig["action"] == "BUY"
            else "triangle-down"
        )

        fig.add_trace(
            go.Scatter(
                x=[df.index[-1]],
                y=[sig["ltp"]],
                mode="markers",
                marker=dict(
                    symbol=marker_symbol,
                    size=17,
                    color=marker_color,
                    line=dict(width=1, color="white"),
                ),
                name=sig["action"],
            )
        )

    fig.update_xaxes(
        rangebreaks=(
            [dict(bounds=["sat", "mon"])]
            if interval == "1d"
            else []
        ),
        showgrid=False,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        margin=dict(l=10, r=10, t=20, b=10),
        height=500,
        xaxis_rangeslider_visible=False,
        yaxis=dict(
            showgrid=True,
            gridcolor="#31333F",
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ========================================================
    # Indicator Panels
    # ========================================================

    if show_volume:
        fig_volume = go.Figure()

        volume_colors = [
            "#00d09c" if c >= o else "#eb5b3c"
            for o, c in zip(df["Open"], df["Close"])
        ]

        fig_volume.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                marker_color=volume_colors,
                name="Volume",
            )
        )

        fig_volume.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Volume_MA20"],
                line=dict(color="orange", width=1.3),
                name="Volume MA20",
            )
        )

        fig_volume.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            margin=dict(l=10, r=10, t=10, b=10),
            height=170,
        )

        st.plotly_chart(fig_volume, use_container_width=True)

    if show_rsi:
        fig_rsi = go.Figure()

        fig_rsi.add_trace(
            go.Scatter(
                x=df.index,
                y=df["RSI"],
                line=dict(color="#4f8cff", width=1.5),
                name="RSI",
            )
        )

        fig_rsi.add_hline(
            y=70,
            line_dash="dot",
            line_color="#eb5b3c",
        )

        fig_rsi.add_hline(
            y=50,
            line_dash="dot",
            line_color="#777777",
        )

        fig_rsi.add_hline(
            y=30,
            line_dash="dot",
            line_color="#00d09c",
        )

        fig_rsi.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            margin=dict(l=10, r=10, t=10, b=10),
            height=170,
            yaxis=dict(range=[0, 100]),
        )

        st.plotly_chart(fig_rsi, use_container_width=True)

    if show_macd:
        fig_macd = go.Figure()

        hist_colors = [
            "#00d09c" if v >= 0 else "#eb5b3c"
            for v in df["MACD_hist"].fillna(0)
        ]

        fig_macd.add_trace(
            go.Bar(
                x=df.index,
                y=df["MACD_hist"],
                marker_color=hist_colors,
                name="Histogram",
            )
        )

        fig_macd.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD"],
                line=dict(color="#4f8cff", width=1.3),
                name="MACD",
            )
        )

        fig_macd.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD_signal"],
                line=dict(color="orange", width=1.3),
                name="Signal",
            )
        )

        fig_macd.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            margin=dict(l=10, r=10, t=10, b=10),
            height=170,
        )

        st.plotly_chart(fig_macd, use_container_width=True)

    # ========================================================
    # Market Overview
    # ========================================================

    st.markdown("---")
    st.markdown("### 📊 Market Overview")

    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric(
        "LAST PRICE",
        f"₹{sig['ltp']:,.2f}",
        f"{chg:+.2f} ({chg_pct:+.2f}%)",
    )

    m2.metric("RSI (14)", f"{sig['rsi']:.2f}")

    m3.metric("ADX (14)", f"{sig['adx']:.2f}")

    m4.metric("VWAP", f"₹{sig['vwap']:,.2f}")

    m5.metric(
        "VOLUME",
        f"{sig['volume_ratio']:.2f}x",
        "vs 20-candle average",
    )

    # ========================================================
    # Trade Setup
    # ========================================================

    st.markdown("### 🎯 Trade Setup")

    r1, r2, r3, r4 = st.columns(4)

    r1.metric(
        "ENTRY",
        f"₹{sig['entry']:,.2f}" if sig["entry"] else "--",
    )

    r2.metric(
        "TARGET",
        f"₹{sig['target']:,.2f}" if sig["target"] else "--",
    )

    r3.metric(
        "STOP LOSS",
        f"₹{sig['sl']:,.2f}" if sig["sl"] else "--",
    )

    r4.metric(
        "RISK : REWARD",
        f"1 : {sig['rr']:.2f}" if sig["rr"] else "--",
    )

    # ========================================================
    # Position Sizing
    # ========================================================

    st.markdown("### 💰 Risk & Position Sizing")

    if sig["action"] in ("BUY", "SELL") and sig["risk_per_share"] > 0:

        risk_pct = st.slider(
            "Maximum capital risk per trade (%)",
            min_value=0.25,
            max_value=5.0,
            value=1.0,
            step=0.25,
        )

        max_risk = capital * risk_pct / 100

        quantity_by_risk = int(
            max_risk / sig["risk_per_share"]
        )

        capital_required = quantity_by_risk * sig["entry"]

        p1, p2, p3 = st.columns(3)

        p1.metric(
            "Max Risk",
            f"₹{max_risk:,.0f}",
        )

        p2.metric(
            "Suggested Quantity",
            f"{quantity_by_risk:,}",
        )

        p3.metric(
            "Capital Required",
            f"₹{capital_required:,.0f}",
        )

        if capital_required > capital:
            st.warning(
                "Position size based on risk exceeds available trading capital. "
                "Reduce quantity or increase capital."
            )

    else:
        st.info(
            "Position sizing will appear when a confirmed BUY or SELL signal is generated."
        )

    # ========================================================
    # Signal Confirmation Matrix
    # ========================================================

    st.markdown("### 🔍 Signal Confirmation")

    confirmation_rows = []

    for name, passed in sig["conditions"].items():
        confirmation_rows.append(
            {
                "Indicator": name,
                "Status": "✅ Confirmed" if passed else "❌ Not Confirmed",
            }
        )

    confirmation_df = pd.DataFrame(confirmation_rows)

    st.dataframe(
        confirmation_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "⚠️ This dashboard is a technical-analysis/educational tool. "
        "Yahoo Finance data may be delayed or rate-limited and should not "
        "be treated as a broker-grade live execution feed. "
        "Always independently validate a trade before placing an order."
    )

except Exception as e:
    st.error(
        "Something went wrong while loading the stock data. "
        "Please check the NSE symbol and try again."
    )
    st.exception(e)
