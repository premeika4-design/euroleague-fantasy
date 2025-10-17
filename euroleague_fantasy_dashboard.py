import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Euroleague Fantasy – Stable Table", page_icon="🏀", layout="wide")
st.title("🏀 Euroleague Fantasy – Stable Table")

# ------------------------------------------------------------------
# 🔧 FILES
# ------------------------------------------------------------------
PLAYERS_PATH  = Path("data/players_latest.csv")     # your latest player stats
FALLBACK_PATH = Path("data/players_sample.csv")     # backup
FIXTURES_PATH = Path("data/fixtures.csv")           # your 10 GW fixtures
DEF_PATH      = Path("data/defense_by_pos.csv")     # defense table

# ------------------------------------------------------------------
# ⚙️  SETTINGS (you can tweak)
# ------------------------------------------------------------------
W_VALUE   = 1.00
W_USAGE   = 0.80
W_OPP     = 0.80
W_MINUTES = 0.90

# ------------------------------------------------------------------
# 🧩 LOAD DATA
# ------------------------------------------------------------------
def load_csv_if_exists(path: Path):
    return pd.read_csv(path) if path.exists() else None

players = load_csv_if_exists(PLAYERS_PATH) or load_csv_if_exists(FALLBACK_PATH)
if players is None:
    st.error("❌ No player file found. Please upload data/players_latest.csv or data/players_sample.csv.")
    st.stop()

fixtures = load_csv_if_exists(FIXTURES_PATH)
defense  = load_csv_if_exists(DEF_PATH)

# ------------------------------------------------------------------
# 🧮 CLEAN + METRICS
# ------------------------------------------------------------------
required = ["Player","Team","Minutes","Points","FGM","FGA","FTM","FTA","Rebounds","Assists","Steals","Blocks","Turnovers"]
for col in required:
    if col not in players.columns:
        st.error(f"Missing column: {col}")
        st.stop()

players = players.fillna(0)

# Euroleague EFF (PIR style)
players["EFF"] = (
    players["Points"] + players["Rebounds"] + players["Assists"]
    + players["Steals"] + players["Blocks"]
    - ((players["FGA"] - players["FGM"]) + (players["FTA"] - players["FTM"]) + players["Turnovers"])
)

# Usage %
team_tot = players.groupby("Team", as_index=False)[["Minutes","FGA","FTA","Turnovers"]].sum()
team_tot.rename(columns={"Minutes":"TeamMin","FGA":"TeamFGA","FTA":"TeamFTA","Turnovers":"TeamTOV"}, inplace=True)
players = players.merge(team_tot, on="Team", how="left")
players["USG%"] = np.where(
    (players["Minutes"]>0) & (players["TeamFGA"]+0.44*players["TeamFTA"]+players["TeamTOV"]>0),
    100*((players["FGA"]+0.44*players["FTA"]+players["Turnovers"])*(players["TeamMin"]/5))
    /(players["Minutes"]*(players["TeamFGA"]+0.44*players["TeamFTA"]+players["TeamTOV"])),
    0
)

players["Value"] = players["EFF"]

# Opponent (optional)
if fixtures is not None and "Gameweek" in fixtures.columns:
    gw = st.sidebar.number_input("Gameweek", min_value=1, value=1, step=1)
    fx = fixtures[fixtures["Gameweek"]==gw]
    players = players.merge(fx[["Team","Opponent"]], on="Team", how="left")
else:
    players["Opponent"] = ""

if defense is not None and "Team" in defense.columns:
    defense["OppBenefit"] = 1/defense[["Def_PG","Def_SG","Def_SF","Def_PF","Def_C"]].mean(axis=1)
    players = players.merge(defense[["Team","OppBenefit"]].rename(columns={"Team":"Opponent"}), on="Opponent", how="left")
    players["OppBenefit"] = players["OppBenefit"].fillna(1.0)
else:
    players["OppBenefit"] = 1.0

# ------------------------------------------------------------------
# 📊 SCORE + FILTERS
# ------------------------------------------------------------------
def zscore(x):
    x = pd.to_numeric(x, errors="coerce")
    return (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) != 0 else 0

players["zValue"]   = zscore(players["Value"])
players["zUsage"]   = zscore(players["USG%"])
players["zOpp"]     = zscore(players["OppBenefit"])
players["zMinutes"] = zscore(players["Minutes"])

players["PickScore"] = (
    W_VALUE*players["zValue"]
    + W_USAGE*players["zUsage"]
    + W_OPP*players["zOpp"]
    + W_MINUTES*players["zMinutes"]
)

# ------------------------------------------------------------------
# 🧱 TABLE
# ------------------------------------------------------------------
st.info(f"Loaded {len(players)} players across {players['Team'].nunique()} teams.")

min_min = st.sidebar.slider("Min minutes", 0, int(players["Minutes"].max()), 10, 1)
top_n   = st.sidebar.slider("Show top N", 5, 100, 25, 1)

picks = players[players["Minutes"]>=min_min].sort_values("PickScore", ascending=False)

st.subheader("Top Picks Table")
cols = ["Player","Team","Opponent","Minutes","EFF","USG%","Value","OppBenefit","PickScore"]
st.dataframe(picks[cols].reset_index(drop=True).head(top_n), use_container_width=True)

fig = px.bar(picks.head(top_n), x="Player", y="PickScore", color="Team",
             hover_data=["Team","EFF","USG%","Value","OppBenefit","Minutes"])
st.plotly_chart(fig, use_container_width=True)
