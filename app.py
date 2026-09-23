import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yfinance as yf

# ============================================================
# Indian Stock Trading Dashboard - Single Screen Eye-Friendly
# ============================================================

st.set_page_config(
    page_title="Indian Stock Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------- Eye-Friendly CSS Styling ----------------
st.markdown(
    """
    <style>
    /* Global Background & Base Typography */
    .stApp {
        background-color: #f8f9fa;
        color: #1e293b;
    }
    
    /* Compact Header Space */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        max-width: 100%;
    }
    
    /* Card Container Styling */
    .side-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    
    .signal-header {
        border-radius: 6px;
        padding: 10px 14px;
        color: #ffffff;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .signal-buy { background-color: #059669; }
    .signal-sell { background-color: #dc2626; }
    .signal-hold { background-color: #d97706; }
    
    /* Clean Metric Table Formatting */
    .metric-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.88rem;
    }
    .metric-table td, .metric-table th {
        padding: 6px 8px;
        border-bottom: 1px solid #f1f5f9;
        text-align: left;
    }
    .metric-table th {
        color: #64748b;
        font-weight: 600;
    }
    
    /* Streamlit Default Adjustments */
    div[data-testid="stMetricValue"] {
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        color: #475569 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Indicator Functions
# ============================================================

def compute_indicators(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    df = df.copy()

    # EMAs
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

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

    df["ATR"] = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    # Volume average
    df["Volume_MA20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA20"].replace(0, np.nan)

    # VWAP
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    if interval != "1d":
        session = pd.Series(df.index.date, index=df.index)
        cumulative_pv = (typical_price * df["Volume"]).groupby(session).cumsum()
        cumulative_volume = df["Volume"].groupby(session).cumsum()
        df["VWAP"] = cumulative_pv / cumulative_volume.replace(0, np.nan)
    else:
        cumulative_pv = (typical_price * df["Volume"]).cumsum()
        cumulative_volume = df["Volume"].cumsum()
        df["VWAP"] = cumulative_pv / cumulative_volume.replace(0, np.nan)

    # ADX
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )

    atr = df["ATR"].replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["ADX"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    # Supertrend
    multiplier = 3.0
    hl2 = (df["High"] + df["Low"]) / 2
    basic_upper = hl2 + multiplier * df["ATR"]
    basic_lower = hl2 - multiplier * df["ATR"]

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    supertrend = pd.Series(index=df.index, dtype=float)

    for i in range(1, len(df)):
        if basic_upper.iloc[i] < final_upper.iloc[i - 1] or df["Close"].iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        if basic_lower.iloc[i] > final_lower.iloc[i - 1] or df["Close"].iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        if pd.isna(supertrend.iloc[i - 1]):
            supertrend.iloc[i] = final_upper.iloc[i]
        elif supertrend.iloc[i - 1] == final_upper.iloc[i - 1]:
            supertrend.iloc[i] = final_upper.iloc[i] if df["Close"].iloc[i] <= final_upper.iloc[i] else final_lower.iloc[i]
        else:
            supertrend.iloc[i] = final_lower.iloc[i] if df["Close"].iloc[i] >= final_lower.iloc[i] else final_upper.iloc[i]

    df["Supertrend"] = supertrend

    # Previous Day Levels
    if interval != "1d":
        daily = df.resample("1D").agg({"High": "max", "Low": "min"}).dropna()
        daily["Prev_Day_High"] = daily["High"].shift(1)
        daily["Prev_Day_Low"] = daily["Low"].shift(1)
        prev_day = daily[["Prev_Day_High", "Prev_Day_Low"]].reindex(
            df.index.normalize(), method="ffill"
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

def generate_signal(df: pd.DataFrame, atr_stop_mult: float = 1.5, atr_target_mult: float = 3.0) -> dict:
    if len(df) < 30:
        raise ValueError("Not enough historical candles.")

    last = df.iloc[-1]
    prev = df.iloc[-2]

    trend = "UP" if last["EMA_20"] > last["EMA_50"] else ("DOWN" if last["EMA_20"] < last["EMA_50"] else "SIDEWAYS")

    bull_cross = prev["MACD"] <= prev["MACD_signal"] and last["MACD"] > last["MACD_signal"]
    bear_cross = prev["MACD"] >= prev["MACD_signal"] and last["MACD"] < last["MACD_signal"]

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

    if bull_score >= 6 and bull_conditions["Fresh MACD Cross"]:
        action, score, conditions, reason = "BUY", bull_score, bull_conditions, "Multiple bullish confirmations aligned."
    elif bear_score >= 6 and bear_conditions["Fresh MACD Cross"]:
        action, score, conditions, reason = "SELL", bear_score, bear_conditions, "Multiple bearish confirmations aligned."
    else:
        action = "HOLD"
        score = max(bull_score, bear_score)
        conditions = bull_conditions if bull_score >= bear_score else bear_conditions
        reason = "Waiting for stronger confirmation."

    ltp = float(last["Close"])
    atr = float(last["ATR"]) if pd.notna(last["ATR"]) else 0.0

    entry = sl = target = None
    if action == "BUY":
        entry, sl, target = round(ltp, 2), round(ltp - atr_stop_mult * atr, 2), round(ltp + atr_target_mult * atr, 2)
    elif action == "SELL":
        entry, sl, target = round(ltp, 2), round(ltp + atr_stop_mult * atr, 2), round(ltp - atr_target_mult * atr, 2)

    risk_per_share = abs(entry - sl) if entry and sl else 0
    reward_per_share = abs(target - entry) if entry and target else 0
    rr = reward_per_share / risk_per_share if risk_per_share > 0 else 0
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
    data = yf.download(ticker, period=period, interval=selected_interval, auto_adjust=False, progress=False)

    if data.empty:
        return data

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]
    return data[[c for c in required if c in data.columns]].dropna(subset=["Open", "High", "Low", "Close"])


# ============================================================
# Main Layout Application
# ============================================================

try:
    # Top Control Bar (Compact Inline)
    c1, c2, c3, c4, c5, c6 = st.columns([2.5, 1.2, 1.5, 1, 1, 1])
    with c1:
        symbol = st.text_input("NSE Symbol", "RELIANCE.NS", label_visibility="collapsed").strip().upper()
    with c2:
        interval = st.selectbox("Interval", ["15m", "1h", "1d"], index=0, label_visibility="collapsed")
    with c3:
        capital = st.number_input("Capital (₹)", min_value=1000.0, value=100000.0, step=5000.0, label_visibility="collapsed")
    with c4:
        show_rsi = st.checkbox("RSI", value=True)
    with c5:
        show_macd = st.checkbox("MACD", value=True)
    with c6:
        if st.button("🔄 Refresh", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    df = load_data(symbol, interval)
    if df.empty:
        st.error("No data found. Check NSE symbol (e.g., RELIANCE.NS, TCS.NS).")
        st.stop()

    df = compute_indicators(df, interval)
    if len(df) < 30:
        st.warning("Not enough candles for analysis.")
        st.stop()

    sig = generate_signal(df)
    prev_price = float(df["Close"].iloc[-2])
    chg = sig["ltp"] - prev_price
    chg_pct = (chg / prev_price) * 100 if prev_price else 0.0

    # ------------------------------------------------------------
    # 2-Column Split Layout (Chart Left | Metrics Right)
    # ------------------------------------------------------------
    col_left, col_right = st.columns([7, 5])

    # ---------------- LEFT COLUMN: Subplot Charts ----------------
    with col_left:
        # Define Subplot rows dynamically based on visible indicators
        active_rows = 1
        if show_rsi: active_rows += 1
        if show_macd: active_rows += 1

        row_heights = [0.55] + ([0.225] * (active_rows - 1)) if active_rows > 1 else [1.0]

        fig = make_subplots(
            rows=active_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights,
        )

        # 1. Main Candlestick & Overlays
        fig.add_trace(
            go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                increasing_line_color="#059669", increasing_fillcolor="#10b981",
                decreasing_line_color="#dc2626", decreasing_fillcolor="#ef4444",
                name="Price",
            ),
            row=1, col=1,
        )

        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_20"], line=dict(color="#d97706", width=1.2), name="EMA 20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_50"], line=dict(color="#7c3aed", width=1.2), name="EMA 50"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"], line=dict(color="#2563eb", width=1.2), name="VWAP"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["Supertrend"], line=dict(color="#0284c7", width=1.2), name="Supertrend"), row=1, col=1)

        # Targets / SL Lines
        if sig["action"] in ("BUY", "SELL"):
            fig.add_hline(y=sig["entry"], line_dash="dash", line_color="#2563eb", annotation_text="Entry", row=1, col=1)
            fig.add_hline(y=sig["sl"], line_dash="dash", line_color="#dc2626", annotation_text="SL", row=1, col=1)
            fig.add_hline(y=sig["target"], line_dash="dash", line_color="#059669", annotation_text="Target", row=1, col=1)

        current_row = 2

        # 2. RSI Subplot
        if show_rsi:
            fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color="#2563eb", width=1.3), name="RSI"), row=current_row, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="#dc2626", row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="#059669", row=current_row, col=1)
            fig.update_yaxes(range=[0, 100], row=current_row, col=1)
            current_row += 1

        # 3. MACD Subplot
        if show_macd:
            hist_colors = ["#059669" if v >= 0 else "#dc2626" for v in df["MACD_hist"].fillna(0)]
            fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], marker_color=hist_colors, name="Hist"), row=current_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], line=dict(color="#2563eb", width=1.2), name="MACD"), row=current_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], line=dict(color="#d97706", width=1.2), name="Signal"), row=current_row, col=1)

        # Eye-Friendly Light Layout Formatting
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            margin=dict(l=10, r=10, t=10, b=10),
            height=620,  # Compact size fitting single screen
            showlegend=False,
            xaxis_rangeslider_visible=False,
        )
        fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
        fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")

        st.plotly_chart(fig, use_container_width=True)

    # ---------------- RIGHT COLUMN: Structured Horizontal Tables ----------------
    with col_right:
        # Signal Header Badge
        badge_class = "signal-buy" if sig['action'] == "BUY" else ("signal-sell" if sig['action'] == "SELL" else "signal-hold")
        st.markdown(
            f"""
            <div class="signal-header {badge_class}">
                <span style="font-size:1.1rem">{sig['action']}</span> | Conf: {sig['confidence']:.0f}% | Score: {sig['score']}/8
                <div style="font-size:0.8rem; font-weight:400; margin-top:2px">{sig['reason']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 1. Market Overview Table
        st.markdown("##### 📊 Market Overview")
        st.markdown(
            f"""
            <table class="metric-table">
                <tr>
                    <th>LTP</th>
                    <th>Change</th>
                    <th>RSI (14)</th>
                    <th>ADX (14)</th>
                    <th>VWAP</th>
                    <th>Vol Ratio</th>
                </tr>
                <tr>
                    <td><b>₹{sig['ltp']:,.2f}</b></td>
                    <td style="color:{'#059669' if chg>=0 else '#dc2626'}"><b>{chg:+.2f} ({chg_pct:+.2f}%)</b></td>
                    <td>{sig['rsi']:.1f}</td>
                    <td>{sig['adx']:.1f}</td>
                    <td>₹{sig['vwap']:,.1f}</td>
                    <td>{sig['volume_ratio']:.2f}x</td>
                </tr>
            </table>
            """,
            unsafe_allow_html=True,
        )

        # 2. Trade Setup Table
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("##### 🎯 Trade Setup")
        st.markdown(
            f"""
            <table class="metric-table">
                <tr>
                    <th>Entry</th>
                    <th>Target</th>
                    <th>Stop Loss</th>
                    <th>R : R</th>
                </tr>
                <tr>
                    <td><b>₹{sig['entry']:,.2f}</b> if sig['entry'] else "--"</td>
                    <td style="color:#059669"><b>₹{sig['target']:,.2f}</b> if sig['target'] else "--"</td>
                    <td style="color:#dc2626"><b>₹{sig['sl']:,.2f}</b> if sig['sl'] else "--"</td>
                    <td>1 : {sig['rr']:.2f}</td>
                </tr>
            </table>
            """,
            unsafe_allow_html=True,
        )

        # 3. Position Sizing
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("##### 💰 Risk & Position Sizing")
        if sig["action"] in ("BUY", "SELL") and sig["risk_per_share"] > 0:
            risk_pct = st.slider("Max Capital Risk (%)", 0.25, 5.0, 1.0, 0.25)
            max_risk = capital * risk_pct / 100
            qty = int(max_risk / sig["risk_per_share"])
            req_cap = qty * sig["entry"]

            st.markdown(
                f"""
                <table class="metric-table">
                    <tr>
                        <th>Max Risk</th>
                        <th>Suggested Qty</th>
                        <th>Capital Required</th>
                    </tr>
                    <tr>
                        <td>₹{max_risk:,.0f}</td>
                        <td><b>{qty:,}</b> shares</td>
                        <td>₹{req_cap:,.0f}</td>
                    </tr>
                </table>
                """,
                unsafe_allow_html=True,
            )
            if req_cap > capital:
                st.caption("⚠️ Position size exceeds total capital.")
        else:
            st.info("Sizing available on active BUY/SELL signals.")

        # 4. Signal Matrix
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("##### 🔍 Signal Matrix")
        matrix_items = [
            f"<span style='color:{'#059669' if v else '#dc2626'}'>{'✓' if v else '✗'} {k}</span>"
            for k, v in sig["conditions"].items()
        ]
        # Display 2 columns of indicators
        c_a, c_b = st.columns(2)
        with c_a:
            st.markdown("<br>".join(matrix_items[:4]), unsafe_allow_html=True)
        with c_b:
            st.markdown("<br>".join(matrix_items[4:]), unsafe_allow_html=True)

except Exception as e:
    st.error("Error loading dashboard data.")
    st.exception(e)
