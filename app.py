# ---------------- 2-Column Split Layout ----------------
    col_left, col_right = st.columns([7, 5])

    # LEFT COLUMN: Subplot Charts
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

        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            margin=dict(l=10, r=10, t=10, b=10),
            height=620,
            showlegend=False,
            xaxis_rangeslider_visible=False,
        )
        fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
        fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")

        st.plotly_chart(fig, use_container_width=True)

    # RIGHT COLUMN: Structured Horizontal Tables
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

        # Pre-format strings safely for None values
        entry_str = f"₹{sig['entry']:,.2f}" if sig["entry"] is not None else "--"
        target_str = f"₹{sig['target']:,.2f}" if sig["target"] is not None else "--"
        sl_str = f"₹{sig['sl']:,.2f}" if sig["sl"] is not None else "--"
        rr_str = f"1 : {sig['rr']:.2f}" if sig["rr"] else "--"

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
                    <td><b>{entry_str}</b></td>
                    <td style="color:#059669"><b>{target_str}</b></td>
                    <td style="color:#dc2626"><b>{sl_str}</b></td>
                    <td>{rr_str}</td>
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
        c_a, c_b = st.columns(2)
        with c_a:
            st.markdown("<br>".join(matrix_items[:4]), unsafe_allow_html=True)
        with c_b:
            st.markdown("<br>".join(matrix_items[4:]), unsafe_allow_html=True)
