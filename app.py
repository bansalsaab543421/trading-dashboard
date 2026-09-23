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

        # Helper formatting variables to prevent NoneType errors safely
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
