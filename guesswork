import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# -----------------------
# Streamlit Page Setup
# -----------------------
st.set_page_config(page_title="Intraday Buffer Reversal System", layout="wide")
st.title("📈 SPY Intraday Buffer Reversal System v2.1 – Delta Normalized Edition")

# -----------------------
# Sidebar Inputs
# -----------------------
symbol = st.sidebar.text_input("Enter Symbol (e.g. SPY, QQQ, AAPL)", value="SPY").upper()
manual_buffer = st.sidebar.number_input("Manual Buffer (optional, leave 0 to auto-calculate)", value=0.0, min_value=0.0)
run = st.sidebar.button("🔍 Run Prediction")

if run:
    st.subheader(f"Results for {symbol}")

    # -----------------------
    # Load Live Data
    # -----------------------
    df = yf.download(symbol, interval='5m', period='30d', auto_adjust=True, progress=False)
    df = df.reset_index()
    df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
    df = df[['Datetime', 'Open', 'High', 'Low', 'Close']]
    df = df.dropna()
    df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].astype(float)

    # -----------------------
    # Feature Engineering
    # -----------------------
    df['Delta_High'] = (df['High'] - df['Open']) / df['Open']
    df['Delta_Low'] = (df['Low'] - df['Open']) / df['Open']
    df['Delta_Close'] = (df['Close'] - df['Open']) / df['Open']

    features = ['Open']
    df['Target_High'] = df['Delta_High']
    df['Target_Low'] = df['Delta_Low']
    df['Target_Close'] = df['Delta_Close']

    df = df.dropna()

    X = df[features]
    y_high = df['Target_High']
    y_low = df['Target_Low']
    y_close = df['Target_Close']

    # -----------------------
    # Train Models
    # -----------------------
    model_high = GradientBoostingRegressor().fit(X, y_high)
    model_low = GradientBoostingRegressor().fit(X, y_low)
    model_close = GradientBoostingRegressor().fit(X, y_close)

    df['Pred_High'] = model_high.predict(X)
    df['Pred_Low'] = model_low.predict(X)
    df['Pred_Close'] = model_close.predict(X)

    # -----------------------
    # Reconstruct Price Predictions
    # -----------------------
    df['Pred_High'] = df['Open'] * (1 + df['Pred_High'])
    df['Pred_Low'] = df['Open'] * (1 + df['Pred_Low'])
    df['Pred_Close'] = df['Open'] * (1 + df['Pred_Close'])

    # -----------------------
    # Calculate Buffer (if not manual)
    # -----------------------
    buffer_pct = manual_buffer if manual_buffer > 0 else np.median(np.abs(df['High'] - df['Pred_High']))

    st.markdown(f"**Buffer used:** `{buffer_pct:.4f}`")

    # -----------------------
    # Today’s Signal – Use latest row
    # -----------------------
    latest = df.iloc[-1]
    open_price = latest['Open']
    pred_high = latest['Pred_High']
    pred_low = latest['Pred_Low']
    pred_close = latest['Pred_Close']

    long_entry = pred_high + buffer_pct
    short_entry = pred_low - buffer_pct
    stop_loss = buffer_pct
    target_R = 1.5
    trail_after_R = 0.5

    st.markdown("### 🟢 Long Setup")
    st.markdown(f"""
    - **Entry Above:** {long_entry:.2f}  
    - **Stop Loss:** {long_entry - stop_loss:.2f}  
    - **Target 1.5R:** {long_entry + stop_loss * target_R:.2f}  
    - **Trail Stop After:** 0.5R (i.e., trail below {long_entry + stop_loss * target_R:.2f} by {stop_loss * trail_after_R:.2f})
    """)

    st.markdown("### 🔴 Short Setup")
    st.markdown(f"""
    - **Entry Below:** {short_entry:.2f}  
    - **Stop Loss:** {short_entry + stop_loss:.2f}  
    - **Target 1.5R:** {short_entry - stop_loss * target_R:.2f}  
    - **Trail Stop After:** 0.5R (i.e., trail above {short_entry - stop_loss * target_R:.2f} by {stop_loss * trail_after_R:.2f})
    """)

    # -----------------------
    # Trade Log Summary
    # -----------------------
    trade_log = {
        'Setup': ['Long', 'Short'],
        'Entry': [long_entry, short_entry],
        'Stop Loss': [long_entry - stop_loss, short_entry + stop_loss],
        'Target (1.5R)': [long_entry + stop_loss * target_R, short_entry - stop_loss * target_R],
        'Trail Trigger Price': [
            long_entry + stop_loss * target_R,
            short_entry - stop_loss * target_R
        ],
        'Trail Buffer (0.5R)': [stop_loss * trail_after_R, stop_loss * trail_after_R]
    }

    st.markdown("### 📋 Trade Log")
    st.dataframe(pd.DataFrame(trade_log).set_index("Setup"))

