import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf

# Page configuration for dark theme
st.set_page_config(page_title="Indian Stock Trading Dashboard", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("### 🇮🇳 Indian Stock Trading Signal & Technical Dashboard")

# Top controls layout for Indian Stocks
col_input, col_btn = st.columns([4, 1])
with col_input:
    symbol_input = st.text_input("Enter Indian Stock Symbol (e.g., RELIANCE.NS, TCS.NS, INFY.NS):", "RELIANCE.NS")
with col_btn:
    st.write("")
    if st.button("Fetch Data", type="primary"):
        st.rerun()

try:
    # Fetching real data from Yahoo Finance
    df = yf.download(symbol_input, period="3mo", interval="1d")
    
    if df.empty:
        st.error("No data found for this symbol. Please check the stock name (e.g., RELIANCE.NS).")
    else:
        # MultiIndex fix for recent yfinance versions
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        # Calculate EMAs and RSI
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Calculate MACD Histogram
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = macd - signal_line

        last_price = float(df['Close'].iloc[-1])
        prev_price = float(df['Close'].iloc[-2])
        last_rsi = float(df['RSI'].iloc[-1])
        last_macd = float(df['MACD_Hist'].iloc[-1])
        ema_20_val = float(df['EMA_20'].iloc[-1])
        
        price_change = last_price - prev_price
        price_change_pct = (price_change / prev_price) * 100
        trend = "BULLISH" if last_price > ema_20_val else "BEARISH"

        # Always active calculated levels so boxes are never empty
        signal = "BUY" if last_price > ema_20_val else "SELL"
        entry_price = last_price
        target_price = last_price * 1.025 if signal == "BUY" else last_price * 0.975
        stop_loss = last_price * 0.985 if signal == "BUY" else last_price * 1.015

        # Plotting Candlestick Chart (Groww Style + No Weekend Gaps)
        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            name='Candles',
            increasing_line_color='#00d09c', 
            increasing_fillcolor='#00d09c',
            decreasing_line_color='#eb5b3c', 
            decreasing_fillcolor='#eb5b3c'
        ))

        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='orange', width=1.5), name='EMA 20'))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='purple', width=1.5), name='EMA 50'))

        # Removing weekends to fix empty gaps in the chart
        fig.update_xaxes(
            rangebreaks=[dict(bounds=["sat", "mon"])],
            showgrid=False
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='#0e1117',
            plot_bgcolor='#0e1117',
            margin=dict(l=10, r=10, t=10, b=10),
            height=450,
            xaxis_rangeslider_visible=False,
            yaxis=dict(showgrid=True, gridcolor='#31333F')
        )

        st.plotly_chart(fig, use_container_width=True)

        # Metrics Cards Layout
        c1, c2 = st.columns(2)
        with c1:
            st.metric(label="LAST PRICE", value=f"₹{last_price:,.2f}", delta=f"{price_change:,.2f} ({price_change_pct:.2f}%)")
        with c2:
            st.metric(label="TREND", value=trend)

        s1, s2 = st.columns(2)
        with s1:
            st.metric(label="SIGNAL", value=signal)
        with s2:
            st.metric(label="RSI (14)", value=f"{last_rsi:.2f}")

        # Entry, Target, Stop-Loss Section
        e1, e2, e3 = st.columns(3)
        with e1:
            st.metric(label="ENTRY", value=f"₹{entry_price:,.2f}")
        with e2:
            st.metric(label="STOP-LOSS", value=f"₹{stop_loss:,.2f}")
        with e3:
            st.metric(label="TARGET", value=f"₹{target_price:,.2f}")

        # Extra indicator details
        m1, m2 = st.columns(2)
        with m1:
            st.metric(label="EMA 20", value=f"₹{ema_20_val:,.2f}")
        with m2:
            st.metric(label="MACD HIST", value=f"{last_macd:.2f}")

        st.success("Technical levels and targets are now active and populated based on current price action.")

except Exception as e:
    st.info("Please enter a valid Indian stock Yahoo symbol (e.g., RELIANCE.NS, TCS.NS) and click Fetch Data.")
