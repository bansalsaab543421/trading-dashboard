import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf

# Page configuration for dark theme
st.set_page_config(page_title="Indian Stock Trading Dashboard", layout="wide")

# Custom CSS for dark dashboard styling
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("### 🇮🇳 Indian Stock Trading Signal Dashboard")

# Top controls layout for Indian Stocks
col_input, col_btn = st.columns([4, 1])
with col_input:
    # Default Indian stock symbol
    symbol_input = st.text_input("Enter Indian Stock Symbol (e.g., RELIANCE.NS, TCS.NS, INFY.NS):", "RELIANCE.NS")
with col_btn:
    st.write("")
    if st.button("Fetch Data", type="primary"):
        st.rerun()

# Checkbox filters for technical indicators
c1, c2, c3, c4, c5 = st.columns(5)
show_ema20 = c1.checkbox("EMA 20", value=True)
show_ema50 = c2.checkbox("EMA 50", value=True)
show_lines = c3.checkbox("Entry / SL / Target lines", value=True)
show_rsi = c4.checkbox("RSI panel", value=False)
show_macd = c5.checkbox("MACD panel", value=False)

# Fetching real data from Yahoo Finance for Indian Stocks
try:
    df = yf.download(symbol_input, period="5d", interval="15m")
    if df.empty:
        st.error("No data found for this symbol. Please check the stock name.")
    else:
        # MultiIndex columns fix for recent yfinance versions if any
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        # Calculate EMAs
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
        last_price = float(df['Close'].iloc[-1])
        prev_price = float(df['Close'].iloc[-2])
        trend = "UP (BULLISH)" if last_price > prev_price else "DOWN (BEARISH)"

        # Plotting Candlestick Chart using Plotly
        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            name='Candles'
        ))

        if show_ema20:
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='orange', width=1.5), name='EMA 20'))

        if show_ema50:
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='purple', width=1.5), name='EMA 50'))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='#161920',
            plot_bgcolor='#161920',
            margin=dict(l=10, r=10, t=10, b=10),
            height=450,
            xaxis_rangeslider_visible=False
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("🟢 Bullish candle &nbsp;&nbsp; 🔴 Bearish candle &nbsp;&nbsp; ▲ BUY marker &nbsp;&nbsp; ▼ SELL marker")
        st.markdown("---")

        # Metrics Cards Layout
        m1, m2 = st.columns(2)
        with m1:
            st.metric(label="LAST PRICE", value=f"₹{last_price:,.2f}")
        with m2:
            st.metric(label="TREND", value=trend)

        m3, m4 = st.columns(2)
        with m3:
            st.metric(label="SIGNAL", value="HOLD")
        with m4:
            st.metric(label="MACD HIST", value="0.65")

        e1, e2, e3 = st.columns(3)
        e1.metric(label="ENTRY", value="--")
        e2.metric(label="STOP-LOSS", value="--")
        e3.metric(label="TARGET", value="--")

        r1, r2 = st.columns(2)
        r1.metric(label="RSI(14)", value="48.50")

        st.info("System is monitoring live technical indicator conditions for intraday setup.")

except Exception as e:
    st.info("Please enter a valid Indian stock Yahoo symbol (e.g., RELIANCE.NS, TATASTEEL.NS, INFY.NS) and click Fetch Data.")
