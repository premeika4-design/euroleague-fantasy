import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Euroleague Fantasy Dashboard", page_icon="🏀", layout="wide")
st.title("🏀 Euroleague Fantasy Dashboard")

# ---- Load data ----
@st.cache_data
def load_data():
    csv_path = Path("data") / "players.csv"
    return pd.read_csv(csv_path)

df = load_data()

# ---- Sidebar filters ----
st.sidebar.header("Filter")
teams = ["All"] + sorted(df["Team"].unique().tolist())
selected_team = st.sidebar.selectbox("Select a team", teams)
selected_metric = st.sidebar.radio("Select metric", ["Points", "Rebounds", "Assists"])

# Apply team filter
filtered = df if selected_team == "All" else df[df["Team"] == selected_team]

# ---- Chart ----
fig = px.bar(
    filtered,
    x="Player",
    y=selected_metric,
    color="Team",
    title=f"{selected_metric} per Player",
)
st.plotly_chart(fig, use_container_width=True)

# ---- Table ----
st.dataframe(filtered, use_container_width=True)
