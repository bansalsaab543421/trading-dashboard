import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf

# Dark theme layout
st.set_page_config(page_title="Indian Stock Dashboard", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("### 📈 Indian Stock Trading Dashboard (Pro Charts)")

# Search bar
col_input, col_btn = st.columns([4, 1])
with col_input:
    symbol_input = st.text_input("Enter NSE Stock Symbol (e.g., RELIANCE.NS, ZOMATO.NS, HDFCBANK.NS):", "RELIANCE.NS")
with col_btn:
    st.write("")
    if st.button("Fetch Data", type="primary"):
        st.rerun()

try:
    # Fetching 3 months of daily data
    df = yf.download(symbol_input, period="3mo", interval="1d")
    
    if df.empty:
        st.error("No data found. Please check the stock symbol.")
    else:
        # MultiIndex fix for new yfinance versions
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        # Calculate EMA for trend line
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        last_price = float(df['Close'].iloc[-1])
        
        # ---------------- CHART DESIGN (Groww Style) ----------------
        fig = go.Figure()

        # Customizing Candles to look exactly like Groww (Bright Green & Red)
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            name='Candles',
            increasing_line_color='#00d09c', # Groww style Green
            increasing_fillcolor='#00d09c',
            decreasing_line_color='#eb5b3c', # Groww style Red
            decreasing_fillcolor='#eb5b3c'
        ))

        # Adding EMA line
        fig.add_trace(go.Scatter(
            x=df.index, y=df['EMA_20'], 
            line=dict(color='#4c8dff', width=2), 
            name='EMA 20'
        ))

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
            height=550,
            xaxis_rangeslider_visible=False,
            yaxis=dict(showgrid=True, gridcolor='#31333F')
        )

        # Display chart
        st.plotly_chart(fig, use_container_width=True)

        # Live Price Metric
        st.metric(label=f"{symbol_input} - LATEST CLOSE", value=f"₹{last_price:,.2f}")

except Exception as e:
    st.info("Please enter a valid stock symbol. Make sure to add '.NS' at the end.")
