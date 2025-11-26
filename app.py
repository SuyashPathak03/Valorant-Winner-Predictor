import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Valorant Winner Predictor", layout="centered")
st.title(" Valorant Match Winner Predictor")
st.markdown("Select both teams and a map to predict who’s likely to win the match!")

# ---- Load model ----
try:
    if os.path.exists("valorant_winner_final_xgb.pkl"):
        model = joblib.load("valorant_winner_final_xgb.pkl")
        st.success(" Model loaded successfully!")
    else:
        st.error(" Model file not found.")
except Exception as e:
    st.error(f" Error loading model: {e}")

# ---- Load dataset ----
try:
    path = "./valorant-champions-2024"  # Adjust if your path differs
    players = pd.read_csv(path + "/player_stats.csv")
    maps = pd.read_csv(path + "/maps_stats.csv")
    st.success(" Dataset loaded successfully!")
except Exception as e:
    st.error(f" Error loading dataset: {e}")

# ---- Compute average team stats ----
if 'players' in locals():
    team_stats = players.groupby("player_team").agg({
        "rating": "mean",
        "acs": "mean",
        "adr": "mean"
    }).reset_index().rename(columns={"player_team": "team"})

    # ---- Sidebar selections ----
    st.sidebar.header("Select Match Details")
    team_list = sorted(team_stats["team"].dropna().unique())
    map_list = sorted(maps["map_name"].dropna().unique())

    team1 = st.sidebar.selectbox("Select Team 1", team_list)
    team2 = st.sidebar.selectbox("Select Team 2", [t for t in team_list if t != team1])
    selected_map = st.sidebar.selectbox("Select Map", map_list)

    picked_by_code = 0  # Default: neutral map

    # ---- Prediction ----
    if st.button("Predict Winner"):
        t1 = team_stats.loc[team_stats["team"] == team1].mean()
        t2 = team_stats.loc[team_stats["team"] == team2].mean()

        input_data = np.array([[
            t1["rating"], t2["rating"],
            t1["acs"], t2["acs"],
            t1["adr"], t2["adr"],
            picked_by_code
        ]])

        try:
            pred = model.predict(input_data)[0]
            if pred == 1:
                st.success(f" {team1.upper()} is likely to WIN on {selected_map.upper()}!")
            else:
                st.error(f" {team2.upper()} might WIN on {selected_map.upper()}!")
        except Exception as e:
            st.error(f" Prediction error: {e}")
