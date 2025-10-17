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

# ---- Tabs ----
players_tab, teams_tab = st.tabs(["Players", "Teams"])

# =========================
# Players tab
# =========================
with players_tab:
    st.sidebar.header("Player Filters")
    all_teams = sorted(df["Team"].unique().tolist())
    selected_teams = st.sidebar.multiselect("Select teams", options=all_teams, default=all_teams)
    selected_metric = st.sidebar.radio("Select metric", ["Points", "Rebounds", "Assists"], horizontal=True)
    top_n = st.sidebar.slider("Show top N players", min_value=3, max_value=20, value=10, step=1)

    filtered = df[df["Team"].isin(selected_teams)].copy()
    leaderboard = filtered.sort_values(by=selected_metric, ascending=False).head(top_n)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"Top {top_n} by {selected_metric}")
        fig = px.bar(
            leaderboard, x="Player", y=selected_metric, color="Team",
            text=selected_metric, title=None
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Leaderboard")
        st.dataframe(leaderboard.reset_index(drop=True), use_container_width=True, hide_index=True)
        st.download_button(
            "Download leaderboard CSV",
            data=leaderboard.to_csv(index=False).encode("utf-8"),
            file_name="leaderboard.csv",
            mime="text/csv",
        )

    with st.expander("See full filtered table"):
        st.dataframe(filtered.reset_index(drop=True), use_container_width=True)

# =========================
# Teams tab (fixed indentation)
# =========================
with teams_tab:
    st.subheader("Team summary")

    left, right = st.columns([2, 1])
    with left:
        team_metric = st.radio(
            "Team metric",
            ["Points", "Rebounds", "Assists"],
            horizontal=True,
            key="team_metric",
        )
    with right:
        agg_mode = st.toggle("Use totals (instead of averages)", value=True, key="agg_mode")

    agg_fn = "sum" if agg_mode else "mean"
    team_stats = (
        df.groupby("Team", as_index=False)[["Points", "Rebounds", "Assists"]]
        .agg(agg_fn)
        .sort_values(by=team_metric, ascending=False)
    )

    top3 = team_stats.head(3)[["Team", team_metric]].reset_index(drop=True)
    st.caption("🏆 Top 3 teams")
    st.dataframe(t
