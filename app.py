import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Intraday Technical Indicator Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: #070b14;
    color: #ffffff;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    max-width: 1500px;
}

[data-testid="stSidebar"] {
    background: #08101f;
    border-right: 1px solid #1e2b40;
}

.metric-card {
    background: #101722;
    border: 1px solid #263347;
    border-radius: 12px;
    padding: 18px;
    min-height: 115px;
}

.metric-title {
    color: #8fa3bd;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.metric-value {
    color: white;
    font-size: 25px;
    font-weight: 700;
    margin-top: 7px;
}

.metric-sub {
    font-size: 13px;
    margin-top: 5px;
}

.green {
    color: #00e676;
}

.red {
    color: #ff4d67;
}

.yellow {
    color: #ffc107;
}

.blue {
    color: #42a5f5;
}

.trade-box {
    background: #101722;
    border: 1px solid #263347;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 15px;
}

.trade-buy {
    background: linear-gradient(90deg, #063d2b, #075b3d);
    border: 1px solid #00d084;
    color: #00e676;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-size: 25px;
    font-weight: 700;
    margin-bottom: 15px;
}

.trade-sell {
    background: linear-gradient(90deg, #4b111d, #641522);
    border: 1px solid #ff4961;
    color: #ff4d67;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-size: 25px;
    font-weight: 700;
    margin-bottom: 15px;
}

.trade-hold {
    background: linear-gradient(90deg, #493800, #624b00);
    border: 1px solid #ffc107;
    color: #ffc107;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-size: 25px;
    font-weight: 700;
    margin-bottom: 15px;
}

.info-box {
    background: #101722;
    border: 1px solid #263347;
    border-radius: 12px;
    padding: 18px;
    margin-top: 10px;
}

.score-bar {
    height: 8px;
    border-radius: 10px;
    background: #243247;
    margin-top: 5px;
}

.small-text {
    color: #8fa3bd;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# FUNCTIONS
# =========================================================

@st.cache_data(ttl=30)
def download_data(symbol, period, interval):

    data = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column"
    )

    if data.empty:
        return data

    # Handle yfinance MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):

        # Take first level if possible
        data.columns = data.columns.get_level_values(0)

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    for col in required_columns:

        if col not in data.columns:
            raise ValueError(f"Missing column: {col}")

    data = data[required_columns].copy()

    data.dropna(inplace=True)

    return data


def calculate_indicators(data):

    df = data.copy()

    # -----------------------------
    # EMA
    # -----------------------------

    df["EMA_9"] = df["Close"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["EMA_21"] = df["Close"].ewm(
        span=21,
        adjust=False
    ).mean()

    # -----------------------------
    # RSI
    # -----------------------------

    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI"] = 100 - (100 / (1 + rs))

    # -----------------------------
    # MACD
    # -----------------------------

    ema12 = df["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = df["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_SIGNAL"] = df["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["MACD_HIST"] = (
        df["MACD"] -
        df["MACD_SIGNAL"]
    )

    # -----------------------------
    # ATR
    # -----------------------------

    previous_close = df["Close"].shift(1)

    tr1 = df["High"] - df["Low"]
    tr2 = abs(df["High"] - previous_close)
    tr3 = abs(df["Low"] - previous_close)

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["ATR"] = true_range.rolling(14).mean()

    # -----------------------------
    # Volume Average
    # -----------------------------

    df["Volume_SMA"] = df["Volume"].rolling(20).mean()

    # -----------------------------
    # VWAP
    # -----------------------------

    typical_price = (
        df["High"] +
        df["Low"] +
        df["Close"]
    ) / 3

    df["TPV"] = typical_price * df["Volume"]

    # Reset VWAP each trading day
    dates = df.index.date

    df["VWAP"] = (
        df["TPV"].groupby(dates).cumsum()
        /
        df["Volume"].groupby(dates).cumsum()
    )

    df.drop(columns=["TPV"], inplace=True)

    return df


# =========================================================
# SIGNAL ENGINE
# =========================================================

def generate_signal(df):

    latest = df.iloc[-1]
    previous = df.iloc[-2]

    score = 0

    breakdown = {}

    # =====================================================
    # EMA TREND - 25 POINTS
    # =====================================================

    if latest["EMA_9"] > latest["EMA_21"]:

        score += 25

        breakdown["EMA Trend"] = (
            25,
            "Bullish"
        )

    else:

        breakdown["EMA Trend"] = (
            0,
            "Bearish"
        )

    # =====================================================
    # VWAP - 20 POINTS
    # =====================================================

    if latest["Close"] > latest["VWAP"]:

        score += 20

        breakdown["VWAP"] = (
            20,
            "Above VWAP"
        )

    else:

        breakdown["VWAP"] = (
            0,
            "Below VWAP"
        )

    # =====================================================
    # RSI - 15 POINTS
    # =====================================================

    rsi = latest["RSI"]

    if 50 <= rsi <= 70:

        score += 15

        breakdown["RSI"] = (
            15,
            "Bullish momentum"
        )

    elif rsi > 70:

        score += 8

        breakdown["RSI"] = (
            8,
            "Overbought"
        )

    elif 30 <= rsi < 50:

        score += 5

        breakdown["RSI"] = (
            5,
            "Weak"
        )

    else:

        breakdown["RSI"] = (
            0,
            "Oversold / Weak"
        )

    # =====================================================
    # MACD - 15 POINTS
    # =====================================================

    if latest["MACD"] > latest["MACD_SIGNAL"]:

        score += 15

        breakdown["MACD"] = (
            15,
            "Bullish"
        )

    else:

        breakdown["MACD"] = (
            0,
            "Bearish"
        )

    # =====================================================
    # VOLUME - 15 POINTS
    # =====================================================

    if (
        latest["Volume_SMA"] > 0
        and latest["Volume"] > latest["Volume_SMA"]
    ):

        score += 15

        breakdown["Volume"] = (
            15,
            "Above Average"
        )

    else:

        score += 5

        breakdown["Volume"] = (
            5,
            "Below Average"
        )

    # =====================================================
    # PRICE ACTION - 10 POINTS
    # =====================================================

    if latest["Close"] > previous["Close"]:

        score += 10

        breakdown["Price Action"] = (
            10,
            "Positive"
        )

    else:

        breakdown["Price Action"] = (
            0,
            "Negative"
        )

    # =====================================================
    # SIGNAL
    # =====================================================

    if score >= 70:

        signal = "BUY"

    elif score <= 30:

        signal = "SELL"

    else:

        signal = "HOLD"

    return signal, score, breakdown


# =========================================================
# TRADE CALCULATION
# =========================================================

def calculate_trade_setup(
    signal,
    price,
    atr,
    capital,
    risk_percent
):

    if pd.isna(atr) or atr <= 0:

        return {
            "entry": None,
            "sl": None,
            "target1": None,
            "target2": None,
            "exit": None,
            "rr": None,
            "quantity": None
        }

    # Risk = 1.5 ATR
    risk_per_share = atr * 1.5

    capital_risk = capital * (
        risk_percent / 100
    )

    quantity = int(
        capital_risk / risk_per_share
    )

    # Don't allow position value above capital
    max_quantity = int(
        capital / price
    )

    quantity = min(
        quantity,
        max_quantity
    )

    if signal == "BUY":

        entry = price

        sl = entry - risk_per_share

        target1 = entry + (
            risk_per_share * 1
        )

        target2 = entry + (
            risk_per_share * 2
        )

        # Early exit / reversal level
        exit_price = entry - (
            atr * 0.75
        )

    elif signal == "SELL":

        entry = price

        sl = entry + risk_per_share

        target1 = entry - (
            risk_per_share * 1
        )

        target2 = entry - (
            risk_per_share * 2
        )

        exit_price = entry + (
            atr * 0.75
        )

    else:

        return {
            "entry": None,
            "sl": None,
            "target1": None,
            "target2": None,
            "exit": None,
            "rr": None,
            "quantity": None
        }

    rr = abs(
        target2 - entry
    ) / abs(
        entry - sl
    )

    return {
        "entry": entry,
        "sl": sl,
        "target1": target1,
        "target2": target2,
        "exit": exit_price,
        "rr": rr,
        "quantity": quantity
    }


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    "## ⚙️ Settings"
)

symbol = st.sidebar.text_input(
    "Stock Symbol",
    "RELIANCE.NS"
).upper().strip()

interval = st.sidebar.selectbox(
    "Timeframe",
    [
        "5m",
        "15m",
        "30m",
        "1h"
    ],
    index=1
)

period = st.sidebar.selectbox(
    "Period",
    [
        "1d",
        "5d",
        "1mo"
    ],
    index=1
)

capital = st.sidebar.number_input(
    "Trading Capital (₹)",
    min_value=1000.0,
    value=100000.0,
    step=5000.0
)

risk_percent = st.sidebar.number_input(
    "Risk / Trade (%)",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.1
)

st.sidebar.markdown("---")

analyze = st.sidebar.button(
    "🔍 ANALYZE",
    use_container_width=True
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    "### ⚡ Quick Symbols"
)

quick_symbols = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS"
]

for s in quick_symbols:

    st.sidebar.write(
        f"• {s}"
    )

st.sidebar.markdown("---")

st.sidebar.info(
    "Data source: Yahoo Finance\n\n"
    "This dashboard is for technical analysis and "
    "does not guarantee future returns."
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <h1 style='margin-bottom:0'>
    🤖 AI Intraday Technical Indicator Bot
    </h1>
    <p style='color:#8fa3bd'>
    Analyze • Technical Indicators • Risk Management • Trade Setup
    </p>
    """,
    unsafe_allow_html=True
)


# =========================================================
# ANALYSIS
# =========================================================

if analyze or "data" not in st.session_state:

    with st.spinner(
        "Fetching market data..."
    ):

        try:

            data = download_data(
                symbol,
                period,
                interval
            )

            if data.empty:

                st.error(
                    "No market data found. "
                    "Please check the symbol."
                )

                st.stop()

            data = calculate_indicators(
                data
            )

            # Need enough data
            if len(data) < 30:

                st.warning(
                    "Not enough data for reliable "
                    "indicator calculation."
                )

                st.stop()

            signal, score, breakdown = (
                generate_signal(data)
            )

            latest = data.iloc[-1]

            current_price = float(
                latest["Close"]
            )

            atr = float(
                latest["ATR"]
            )

            trade = calculate_trade_setup(
                signal,
                current_price,
                atr,
                capital,
                risk_percent
            )

            st.session_state.data = data
            st.session_state.signal = signal
            st.session_state.score = score
            st.session_state.breakdown = breakdown
            st.session_state.trade = trade

        except Exception as e:

            st.error(
                f"Error while analyzing stock: {e}"
            )

            st.stop()


# =========================================================
# GET SESSION DATA
# =========================================================

data = st.session_state.data
signal = st.session_state.signal
score = st.session_state.score
breakdown = st.session_state.breakdown
trade = st.session_state.trade

latest = data.iloc[-1]

current_price = float(
    latest["Close"]
)

previous_price = float(
    data["Close"].iloc[-2]
)

price_change = (
    current_price -
    previous_price
)

price_change_pct = (
    price_change /
    previous_price
) * 100


# =========================================================
# TOP SUMMARY
# =========================================================

c1, c2, c3, c4 = st.columns(
    [2.2, 1.3, 1.3, 1.8]
)

with c1:

    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-title'>
                STOCK
            </div>
            <div class='metric-value'>
                {symbol}
            </div>
            <div class='metric-sub'>
                NSE • {interval}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:

    change_class = (
        "green"
        if price_change >= 0
        else "red"
    )

    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-title'>
                LAST PRICE
            </div>
            <div class='metric-value'>
                ₹{current_price:,.2f}
            </div>
            <div class='metric-sub {change_class}'>
                {price_change:+.2f}
                ({price_change_pct:+.2f}%)
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:

    if signal == "BUY":

        trend = "UPTREND"
        trend_class = "green"

    elif signal == "SELL":

        trend = "DOWNTREND"
        trend_class = "red"

    else:

        trend = "SIDEWAYS / MIXED"
        trend_class = "yellow"

    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-title'>
                TREND
            </div>
            <div class='metric-value {trend_class}'>
                {trend}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c4:

    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-title'>
                SIGNAL SCORE
            </div>
            <div class='metric-value'>
                {score} / 100
            </div>
            <div class='score-bar'>
                <div style='
                    width:{score}%;
                    height:100%;
                    background:#00e676;
                    border-radius:10px;
                '></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown("")


# =========================================================
# INDICATOR CARDS
# =========================================================

i1, i2, i3, i4, i5, i6 = st.columns(6)

indicator_cards = [

    (
        "EMA 9",
        f"₹{latest['EMA_9']:,.2f}",
        "Bullish"
        if latest["EMA_9"] > latest["EMA_21"]
        else "Bearish"
    ),

    (
        "EMA 21",
        f"₹{latest['EMA_21']:,.2f}",
        "Uptrend"
        if latest["EMA_9"] > latest["EMA_21"]
        else "Downtrend"
    ),

    (
        "RSI (14)",
        f"{latest['RSI']:.2f}",
        "Healthy"
        if 50 <= latest["RSI"] <= 70
        else "Watch"
    ),

    (
        "VWAP",
        f"₹{latest['VWAP']:,.2f}",
        "Above"
        if current_price > latest["VWAP"]
        else "Below"
    ),

    (
        "MACD",
        f"{latest['MACD']:.2f}",
        "Bullish"
        if latest["MACD"] > latest["MACD_SIGNAL"]
        else "Bearish"
    ),

    (
        "Volume",
        f"{latest['Volume'] / 1_000_000:.2f}M",
        "Above Avg"
        if latest["Volume"] > latest["Volume_SMA"]
        else "Below Avg"
    )

]

for col, card in zip(
    [i1, i2, i3, i4, i5, i6],
    indicator_cards
):

    title, value, status = card

    with col:

        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-title'>
                    {title}
                </div>
                <div class='metric-value'
                     style='font-size:20px'>
                    {value}
                </div>
                <div class='metric-sub green'>
                    ↑ {status}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


st.markdown("")


# =========================================================
# MAIN LAYOUT
# =========================================================

chart_col, trade_col = st.columns(
    [2.5, 1]
)


# =========================================================
# CHART
# =========================================================

with chart_col:

    st.markdown(
        "### 📈 Price Action"
    )

    chart_df = data.tail(100)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25]
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=chart_df.index,
            open=chart_df["Open"],
            high=chart_df["High"],
            low=chart_df["Low"],
            close=chart_df["Close"],
            name="Price",
            increasing_line_color="#00d084",
            decreasing_line_color="#ff4961"
        ),
        row=1,
        col=1
    )

    # EMA 9
    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["EMA_9"],
            name="EMA 9",
            line=dict(
                color="#42a5f5",
                width=2
            )
        ),
        row=1,
        col=1
    )

    # EMA 21
    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["EMA_21"],
            name="EMA 21",
            line=dict(
                color="#ff9800",
                width=2
            )
        ),
        row=1,
        col=1
    )

    # VWAP
    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["VWAP"],
            name="VWAP",
            line=dict(
                color="#c44cff",
                width=2
            )
        ),
        row=1,
        col=1
    )

    # Volume
    volume_colors = np.where(
        chart_df["Close"] >= chart_df["Open"],
        "#00a878",
        "#d93654"
    )

    fig.add_trace(
        go.Bar(
            x=chart_df.index,
            y=chart_df["Volume"],
            name="Volume",
            marker_color=volume_colors
        ),
        row=2,
        col=1
    )

    fig.update_layout(

        height=600,

        template="plotly_dark",

        paper_bgcolor="#101722",

        plot_bgcolor="#0a1424",

        margin=dict(
            l=10,
            r=10,
            t=30,
            b=10
        ),

        xaxis_rangeslider_visible=False,

        legend=dict(
            orientation="h",
            y=1.02,
            x=0
        ),

        hovermode="x unified"
    )

    fig.update_xaxes(
        gridcolor="#17263a"
    )

    fig.update_yaxes(
        gridcolor="#17263a"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# TRADE SETUP
# =========================================================

with trade_col:

    st.markdown(
        "### 🎯 Trade Setup"
    )

    if signal == "BUY":

        st.markdown(
            "<div class='trade-buy'>🟢 BUY</div>",
            unsafe_allow_html=True
        )

    elif signal == "SELL":

        st.markdown(
            "<div class='trade-sell'>🔴 SELL</div>",
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            "<div class='trade-hold'>🟡 HOLD</div>",
            unsafe_allow_html=True
        )

    if trade["entry"] is not None:

        trade_rows = [

            (
                "Entry Price",
                f"₹{trade['entry']:,.2f}",
                "white"
            ),

            (
                "Target 1 (TP1)",
                f"₹{trade['target1']:,.2f}",
                "#00e676"
            ),

            (
                "Target 2 (TP2)",
                f"₹{trade['target2']:,.2f}",
                "#00e676"
            ),

            (
                "Stop Loss (SL)",
                f"₹{trade['sl']:,.2f}",
                "#ff4961"
            ),

            (
                "Exit If Reversal",
                f"₹{trade['exit']:,.2f}",
                "#42a5f5"
            ),

            (
                "Risk / Reward",
                f"1 : {trade['rr']:.1f}",
                "#ffffff"
            ),

            (
                "Position Size",
                f"{trade['quantity']} shares",
                "#ffc107"
            )
        ]

        for label, value, color in trade_rows:

            st.markdown(
                f"""
                <div style='
                    display:flex;
                    justify-content:space-between;
                    padding:11px 3px;
                    border-bottom:1px solid #243247;
                '>
                    <span style='color:#8fa3bd'>
                        {label}
                    </span>
                    <b style='color:{color}'>
                        {value}
                    </b>
                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.info(
            "No entry condition met.\n\n"
            "Wait for stronger indicator confirmation."
        )


# =========================================================
# LOWER SECTION
# =========================================================

st.markdown("")

left, middle, right = st.columns(3)


# =========================================================
# INDICATOR SCORE
# =========================================================

with left:

    st.markdown(
        "### 🏆 Indicator Score"
    )

    max_scores = {
        "EMA Trend": 25,
        "VWAP": 20,
        "RSI": 15,
        "MACD": 15,
        "Volume": 15,
        "Price Action": 10
    }

    for name, (points, description) in breakdown.items():

        maximum = max_scores[name]

        percentage = (
            points /
            maximum
        ) * 100

        st.markdown(
            f"""
            <div style='margin-bottom:12px'>

                <div style='
                    display:flex;
                    justify-content:space-between;
                '>

                    <span>{name}</span>

                    <span>
                        {points}/{maximum}
                    </span>

                </div>

                <div class='score-bar'>

                    <div style='
                        width:{percentage}%;
                        height:100%;
                        background:#00d084;
                        border-radius:10px;
                    '></div>

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        f"""
        <div class='info-box'>
            <b>Total Score</b>
            <h2 class='green'>
                {score}/100
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# WHY SIGNAL?
# =========================================================

with middle:

    st.markdown(
        "### 💡 Why This Signal?"
    )

    reasons = []

    if latest["EMA_9"] > latest["EMA_21"]:

        reasons.append(
            "EMA 9 is above EMA 21"
        )

    else:

        reasons.append(
            "EMA 9 is below EMA 21"
        )

    if current_price > latest["VWAP"]:

        reasons.append(
            "Price is above VWAP"
        )

    else:

        reasons.append(
            "Price is below VWAP"
        )

    if 50 <= latest["RSI"] <= 70:

        reasons.append(
            f"RSI shows healthy momentum ({latest['RSI']:.1f})"
        )

    if latest["MACD"] > latest["MACD_SIGNAL"]:

        reasons.append(
            "MACD is positive / bullish"
        )

    else:

        reasons.append(
            "MACD is bearish"
        )

    if latest["Volume"] > latest["Volume_SMA"]:

        reasons.append(
            "Volume is above average"
        )

    else:

        reasons.append(
            "Volume is below average"
        )

    for reason in reasons:

        st.markdown(
            f"""
            <div style='
                padding:7px 0;
                color:#d8e2f0;
            '>
                <span class='green'>✓</span>
                {reason}
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# WATCH OUT
# =========================================================

with right:

    st.markdown(
        "### ⚠️ Watch Out"
    )

    warnings = []

    if latest["RSI"] > 70:

        warnings.append(
            "RSI is in overbought zone."
        )

    if latest["RSI"] < 30:

        warnings.append(
            "RSI is in oversold zone."
        )

    if latest["Volume"] < latest["Volume_SMA"]:

        warnings.append(
            "Volume is below average."
        )

    if abs(
        current_price - latest["VWAP"]
    ) / current_price < 0.002:

        warnings.append(
            "Price is very close to VWAP."
        )

    if latest["ATR"] / current_price > 0.01:

        warnings.append(
            "Volatility is relatively high."
        )

    if not warnings:

        warnings.append(
            "No major warning detected from current indicators."
        )

    for warning in warnings:

        st.markdown(
            f"""
            <div style='
                padding:9px;
                margin-bottom:7px;
                background:#151b25;
                border-radius:7px;
                color:#ffc107;
            '>
                ⚠️ {warning}
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# RSI + MACD CHART
# =========================================================

st.markdown("---")

rsi_col, macd_col = st.columns(2)


with rsi_col:

    st.markdown("### 📊 RSI (14)")

    rsi_fig = go.Figure()

    rsi_fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["RSI"],
            name="RSI",
            line=dict(
                color="#c44cff",
                width=2
            )
        )
    )

    rsi_fig.add_hline(
        y=70,
        line_dash="dash",
        line_color="#ff4961"
    )

    rsi_fig.add_hline(
        y=30,
        line_dash="dash",
        line_color="#00d084"
    )

    rsi_fig.add_hline(
        y=50,
        line_dash="dot",
        line_color="#8fa3bd"
    )

    rsi_fig.update_layout(
        height=280,
        template="plotly_dark",
        paper_bgcolor="#101722",
        plot_bgcolor="#0a1424",
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10
        ),
        yaxis=dict(
            range=[0, 100]
        )
    )

    st.plotly_chart(
        rsi_fig,
        use_container_width=True
    )


with macd_col:

    st.markdown("### 📈 MACD")

    macd_fig = go.Figure()

    macd_fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MACD"],
            name="MACD",
            line=dict(
                color="#42a5f5",
                width=2
            )
        )
    )

    macd_fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MACD_SIGNAL"],
            name="Signal",
            line=dict(
                color="#ff9800",
                width=2
            )
        )
    )

    macd_fig.add_trace(
        go.Bar(
            x=data.index,
            y=data["MACD_HIST"],
            name="Histogram",
            marker_color=np.where(
                data["MACD_HIST"] >= 0,
                "#00a878",
                "#d93654"
            )
        )
    )

    macd_fig.update_layout(
        height=280,
        template="plotly_dark",
        paper_bgcolor="#101722",
        plot_bgcolor="#0a1424",
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10
        )
    )

    st.plotly_chart(
        macd_fig,
        use_container_width=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    f"""
    <div style='
        display:flex;
        justify-content:space-between;
        color:#71849d;
        font-size:13px;
    '>

        <span>
            🟢 Data from Yahoo Finance
        </span>

        <span>
            {symbol} • {interval}
        </span>

        <span>
            Technical Analysis Only
        </span>

    </div>
    """,
    unsafe_allow_html=True
)
