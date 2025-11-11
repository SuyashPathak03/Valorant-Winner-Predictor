import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

#  Debugging info for dataset path
st.write(" Current working directory:", os.getcwd())
st.write(" Files here:", os.listdir("."))
if os.path.exists("dataset"):
    st.write(" dataset/ folder found. Contents:", os.listdir("dataset"))
else:
    st.error(" dataset folder not found in current directory.")


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
    path = "./dataset"  # Adjust if your path differs
    players = pd.read_csv(path + "/player_stats.csv")
    maps = pd.read_csv(path + "/maps_stats.csv")
    st.success(" Dataset loaded successfully!")
except Exception as e:
    st.error(f" Error loading dataset: {e}")


# ---- Build UI using precomputed team_aggregated_stats ----
if 'team_stats' in locals() and not team_stats.empty:
    st.info(" Using precomputed team_aggregated_stats.csv")

    # Sidebar selections
    st.sidebar.header("Select Match Details")
    team_list = sorted(team_stats["team"].dropna().unique())

    if len(team_list) == 0:
        st.error(" No team names found in team_aggregated_stats.csv.")
    else:
        # Detect map column
        maps_col = None
        for c in maps.columns:
            if c.lower() in ["map_name", "map", "mapname"]:
                maps_col = c
                break

        if maps_col:
            map_list = sorted(maps[maps_col].dropna().unique())
        else:
            map_list = ["ascent", "haven", "lotus", "split", "bind", "sunset", "breeze", "abyss"]

        team1 = st.sidebar.selectbox("Select Team 1", team_list)
        team2 = st.sidebar.selectbox("Select Team 2", [t for t in team_list if t != team1])
        selected_map = st.sidebar.selectbox("Select Map", map_list)

        # Prediction
        if st.button("Predict Winner"):
            t1 = team_stats.loc[team_stats["team"] == team1].select_dtypes(include='number').mean(numeric_only=True)
            t2 = team_stats.loc[team_stats["team"] == team2].select_dtypes(include='number').mean(numeric_only=True)

            # 🔹 Use all numeric features available in the model training
            features = [
                "rating", "acs", "adr", "kast", "hs_percent", "fk", "fd", "fk_fd_diff"
            ]
            input_data = np.array([[
                t1.get("rating", 1.0), t2.get("rating", 1.0),
                t1.get("acs", 200), t2.get("acs", 200),
                t1.get("adr", 120), t2.get("adr", 120),
                t1.get("kast", 70), t2.get("kast", 70),
                t1.get("hs_percent", 25), t2.get("hs_percent", 25),
                t1.get("fk", 10), t2.get("fk", 10),
                t1.get("fd", 10), t2.get("fd", 10),
                t1.get("fk_fd_diff", 0), t2.get("fk_fd_diff", 0),
                0  # picked_by_code
            ]])

            try:
                pred = model.predict(input_data)[0]
                st.subheader(f" Map: {selected_map.upper()}")
                if pred == 1:
                    st.success(f" {team1.upper()} is likely to WIN!")
                else:
                    st.error(f" {team2.upper()} might WIN!")
            except Exception as e:
                st.error(f" Prediction error: {e}")
else:
    st.error(" team_aggregated_stats.csv not loaded or empty.")
