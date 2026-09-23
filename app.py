import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf

# Page configuration
st.set_page_config(page_title="Indian Stock Trading Dashboard", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("### 🇮🇳 Pro Intraday Trading Dashboard")

# Search Bar
col1, col2 = st.columns([4, 1])
with col1:
    symbol = st.text_input("Enter NSE Stock Symbol:", "RELIANCE.NS")
with col2:
    st.write("")
    if st.button("Fetch Data", type="primary"):
        st.rerun()

try:
    # Fetch Data
    df = yf.download(symbol, period="1mo", interval="1d")
    
    if df.empty:
        st.error("Invalid symbol or no data available.")
    else:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        # Indicators Calculation
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))

        last_price = float(df['Close'].iloc[-1])
        prev_price = float(df['Close'].iloc[-2])
        rsi_val = float(df['RSI'].iloc[-1])
        ema20_val = float(df['EMA_20'].iloc[-1])
        
        chg = last_price - prev_price
        chg_pct = (chg / prev_price) * 100
        
        # Signals & Targets
        signal = "BUY" if last_price > ema20_val else "SELL"
        target = last_price * 1.02 if signal == "BUY" else last_price * 0.98
        stoploss = last_price * 0.99 if signal == "BUY" else last_price * 1.01

        # 1. CHART SECTION
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            increasing_line_color='#00d09c', increasing_fillcolor='#00d09c',
            decreasing_line_color='#eb5b3c', decreasing_fillcolor='#eb5b3c',
            name='Candles'
        ))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='orange', width=1.5), name='EMA 20'))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='purple', width=1.5), name='EMA 50'))
        
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], showgrid=False)
        fig.update_layout(
            template="plotly_dark", paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
            margin=dict(l=10, r=10, t=10, b=10), height=450, xaxis_rangeslider_visible=False,
            yaxis=dict(showgrid=True, gridcolor='#31333F')
        )
        st.plotly_chart(fig, use_container_width=True)

        # 2. METRICS & CARDS SECTION (Ye rha aapka dashboard cards!)
        st.markdown("---")
        st.markdown("### 📊 Market Overview & Signals")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("LAST PRICE", f"₹{last_price:,.2f}", f"{chg:+.2f} ({chg_pct:+.2f}%)")
        m2.metric("RSI (14)", f"{rsi_val:.2f}")
        m3.metric("EMA 20", f"₹{ema20_val:,.2f}")
        m4.metric("SIGNAL", signal)

        st.markdown("### 🎯 Actionable Levels")
        t1, t2, t3 = st.columns(3)
        t1.metric("ENTRY PRICE", f"₹{last_price:,.2f}")
        t2.metric("TARGET PRICE", f"₹{target:,.2f}")
        t3.metric("STOP LOSS", f"₹{stoploss:,.2f}")

except Exception as e:
    st.warning("Please enter a valid stock symbol ending with .NS (e.g., RELIANCE.NS, TCS.NS)")
