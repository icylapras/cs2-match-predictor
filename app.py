"""Streamlit front-end for the CS2 Match Predictor.

Enter two teams of five FACEIT nicknames; the app fetches each player's recent
form, runs the trained model, and shows which team is favored. This is the web
version of src/predict.py.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.faceit_api import FaceitError
from src.features import FEATURE_KEYS
from src.predict import DEFAULT_MODEL, predict_match

st.set_page_config(page_title="CS2 Match Predictor", page_icon="🎯")
st.title("🎯 CS2 Match Predictor")
st.caption("Enter five FACEIT nicknames per team. Names must be exact.")

if not Path(DEFAULT_MODEL).exists():
    st.error(f"No trained model found at {DEFAULT_MODEL}. Run `python -m src.train` first.")
    st.stop()

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Team A")
    team_a = [st.text_input(f"A — player {i + 1}", key=f"a{i}") for i in range(5)]
with col_b:
    st.subheader("Team B")
    team_b = [st.text_input(f"B — player {i + 1}", key=f"b{i}") for i in range(5)]

if st.button("Predict winner", type="primary"):
    team_a = [n.strip() for n in team_a]
    team_b = [n.strip() for n in team_b]
    if not (all(team_a) and all(team_b)):
        st.warning("Please fill in all 10 nicknames.")
        st.stop()

    with st.spinner("Fetching player stats and predicting..."):
        try:
            result = predict_match(team_a, team_b)
        except FaceitError as exc:
            st.error(str(exc))
            st.stop()

    prob_a = result["prob_a"]
    favored, conf = ("Team A", prob_a) if prob_a >= 0.5 else ("Team B", 1 - prob_a)

    if abs(prob_a - 0.5) < 0.05:
        st.info(f"**Too close to call** — essentially a coin flip ({prob_a * 100:.0f}% Team A).")
    else:
        st.success(f"**{favored} favored — {conf * 100:.0f}% confidence**")
    st.progress(prob_a, text=f"Team A win probability: {prob_a * 100:.0f}%")

    table = pd.DataFrame(
        {"Team A": [result["avg_a"][k] for k in FEATURE_KEYS],
         "Team B": [result["avg_b"][k] for k in FEATURE_KEYS]},
        index=[k.replace("_", " ") for k in FEATURE_KEYS],
    )
    st.subheader("Team averages (recent form)")
    st.dataframe(table, use_container_width=True)

    st.caption(
        "Prediction reflects the skill gap between the two teams. The model is "
        "trained on a small dataset, so treat the confidence as a rough estimate."
    )
