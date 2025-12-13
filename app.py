import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Valorant Winner Predictor", layout="centered")

st.markdown("<h1 style='text-align: center;'>🎮 Valorant Match Winner Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Select both teams and a map to predict who’s likely to win the match!</p>", unsafe_allow_html=True)

# -------- Load Model (silent) --------
if not os.path.exists("valorant_winner_final_xgb.pkl"):
    st.error("Model file not found.")
    st.stop()

model = joblib.load("valorant_winner_final_xgb.pkl")

# -------- Load Data (silent) --------
try:
    team_stats = pd.read_csv("dataset/team_aggregated_stats.csv")
    maps = pd.read_csv("dataset/maps_stats.csv")
except:
    st.error("Dataset not found.")
    st.stop()

# -------- Centered Selection UI --------
st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.subheader("Select Match Details")

    team_list = sorted(team_stats["team"].dropna().unique())
    map_list = sorted(maps["map_name"].dropna().unique())

    team1 = st.selectbox("Select Team 1", team_list)
    team2 = st.selectbox("Select Team 2", [t for t in team_list if t != team1])
    selected_map = st.selectbox("Select Map", map_list)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Predict Winner", use_container_width=True):
        t1 = team_stats[team_stats["team"] == team1].iloc[0]
        t2 = team_stats[team_stats["team"] == team2].iloc[0]

        input_data = np.array([[  
            t1["rating"], t1["acs"], t1["adr"], t1["kast"], t1["hs_percent"],
            t1["fk"], t1["fd"], t1["fk_fd_diff"],
            t2["rating"], t2["acs"], t2["adr"], t2["kast"], t2["hs_percent"],
            t2["fk"], t2["fd"], t2["fk_fd_diff"],
            0, 0, 0
        ]])

        pred = model.predict(input_data)[0]

        if pred == 1:
            st.success(f"🏆 {team1.upper()} is likely to WIN on {selected_map.upper()}!")
        else:
            st.error(f"🏆 {team2.upper()} is likely to WIN on {selected_map.upper()}!")
