import streamlit as st
import json
import os
import pandas as pd

st.set_page_config(page_title="SMC Crypto Bot Dashboard", layout="wide")

st.title("🚀 SMC + XGBoost Crypto Trading Bot")

def load_trades():
    if os.path.exists('opening.json'):
        with open('opening.json', 'r') as f:
            try:
                return json.load(f)
            except:
                return {}
    return {}

trades = load_trades()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Starting Balance", "$1000")
with col2:
    st.metric("Open Trades", len(trades))
with col3:
    st.metric("Status", "Running" if len(trades) < 2 else "Max Trades Hit")

st.header("Open Positions")
if trades:
    df = pd.DataFrame.from_dict(trades, orient='index')
    st.dataframe(df)
else:
    st.info("No open trades at the moment.")

st.header("Recent Logs")
if os.path.exists('bot.log'):
    with open('bot.log', 'r') as f:
        st.text_area("Log Output", f.read(), height=200)
else:
    st.info("No logs found yet.")
