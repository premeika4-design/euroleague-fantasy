import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO
from pathlib import Path

st.set_page_config(page_title="Euroleague Fantasy – Weekly Picks", page_icon="🏀", layout="wide")
st.title("🏀 Euroleague Fantasy – Weekly Picks")

# ---------------------------------
# DEFAULTS (so you don't paste URLs)
# ---------------------------------
DEFAULT_PLAYERS_URL = "https://raw.githubusercontent.com/premeika4-design/euroleague-fantasy/main/data/players_sample.csv"
DEFAULT_FIXTURES_PATH = Path("data/fixtures.csv")
DEFAULT_DEF_POS_PATH  = Path("data/defense_by_pos.csv")

# --------------------------
# Helpers
# --------------------------
@st.cache_data
def read_csv_bytes(raw_bytes):
    return pd.read_csv(StringIO(raw_bytes.decode("utf-8")))

@st.cache_data
def read_csv_url(url):
    return pd.read_csv(url)

def coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

def normalize_players(df):
    """Map common givemestat-like columns to our schema; fall back if missing."""
    lookup = {c.lower().strip(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n.lower() in lookup:
                return lookup[n.lower()]
        return None

    out = pd.DataFrame()
    out["Player"]   = df.get(pick("Player","Name"), "")
    out["Team"]     = df.get(pick("Team"), "")
    out["Position"] = df.get(pick("Position","Pos"), "G")
    out["Price"]    = df.get(pick("Price","Cost","Value"), 0)
    out["Minutes"]  = df.get(pick("Minutes","MIN"), 0)
    out["FGM"]      = df.get(pick("FGM","FG Made"), 0)
    out["FGA"]      = df.get(pick("FGA","FG Att"), 0)
    out["FTM"]      = df.get(pick("FTM","FT Made"), 0)
    out["FTA"]      = df.get(pick("FTA","FT Att"), 0)
    out["Rebounds"] = df.get(pick("Rebounds","REB","Total Rebounds"), 0)
    out["Assists"]  = df.get(pick("Assists","AST"), 0)
    out["Steals"]   = df.get(pick("Steals","STL"), 0)
    out["Blocks"]   = df.get(pick("Blocks","BLK"), 0)
    out["Turnovers"]= df.get(pick("Turnovers","TOV"), 0)
    out["Points"]   = df.get(pick("Points","PTS"), 0)

    # Clean/text
    out["Player"]   = out["Player"].astype(str).str.strip()
    out["Team"]     = out["Team"].astype(str).str.strip()
    out["Position"] = out["Position"].astype(str).str.upper().str.replace(" ", "")
    out = coerce_numeric(out, ["Price","Minutes","FGM","FGA","FTM","FTA","Rebounds","Assists","Steals","Blocks","Turnovers","Points"])
    return out

def compute_metrics(players):
    # EFF (Euro/PIR-like)
    miss_fg = players["FGA"] - players["FGM"]
    miss_ft = players["FTA"] - players["FTM"]
    players["EFF"] = (
        players["Points"] + players["Rebounds"] + players["Assists"]
        + players["Steals"] + players["Blocks"]
        - miss_fg - miss_ft - players["Turnovers"]
    )

    # Usage% estimate
    team_tot = players.groupby("Team", as_index=False)[["Minutes","FGA","FTA","Turnovers"]].sum()
    team_tot = team_tot.rename(columns={"Minutes":"TeamMinutes","FGA":"TeamFGA","FTA":"TeamFTA","Turnovers":"TeamTOV"})
    team_tot["TeamMinutesPerPlayer"] = team_tot["TeamMinutes"] / 5.0
    players = players.merge(team_tot, on="Team", how="left")

    numer = (players["FGA"] + 0.44*players["FTA"] + players["Turnovers"]) * players["TeamMinutesPerPlayer"]
    denom = players["Minutes"] * (players["TeamFGA"] + 0.44*players["TeamFTA"] + players["TeamTOV"])
    players["USG%"] = np.where(denom > 0, 100 * numer / denom, 0.0)

    # Value = EFF / Price
    players["Value"] = np.where(players["Price"] > 0, players["EFF"] / players["Price"], 0.0)
    return players

def attach_fixtures(players, fixtures, gw):
    fx = fixtures[fixtures["Gameweek"] == gw].copy()
    return players.merge(fx, on="Team", how="left") if not fx.empty else players.assign(Opponent="")

def apply_opponent_by_position(df, defpos):
    needed = {"Team","Def_PG","Def_SG","Def_SF","Def_PF","Def_C"}
    if defpos is None or not needed.issubset(set(defpos.columns)):
        df["OppBenefit"] = 1.0
        return df

    pos_map = {"PG":"Def_PG","SG":"Def_SG","SF":"Def_SF","PF":"Def_PF","C":"Def_C"}

    def get_mult(row):
        opp = row.get("Opponent","")
        pos = str(row.get("Position","")).upper()
        if not opp or opp not in defpos["Team"].values:
            return 1.0
        row_d = defpos[defpos["Team"] == opp].iloc[0]
        buckets = []
        if "/" in pos:
            for p in pos.split("/"):
                p = p.strip()
                if p in pos_map:
                    buckets.append(float(row_d[pos_map[p]]))
        elif pos in pos_map:
            buckets.append(float(row_d[pos_map[pos]]))
        elif pos.startswith("G"):
            buckets.extend([row_d["Def_PG"], row_d["Def_SG"]])
        elif pos.startswith("F"):
            buckets.extend([row_d["Def_SF"], row_d["Def_PF"]])
        else:
            buckets.append(row_d["Def_C"])
        val = float(np.mean(buckets)) if buckets else 1.0
        return 1.0/val if val>0 else 1.0   # benefit: >1 easier, <1 tougher

    df["OppBenefit"] = df.apply(get_mult, axis=1)
    return df

def zscore(series):
    s = pd.to_numeric(series, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series([0]*len(s), index=s.index)
    return (s - mu) / sd

def load_repo_csv(path: Path):
    return pd.read_csv(path) if path.exists() else None

# --------------------------
# Load data (no user action needed)
# --------------------------
# Players: try URL → fallback to local sample
try:
    players_df = read_csv_url(DEFAULT_PLAYERS_URL)
except Exception:
    players_df = load_repo_csv(Path("data/players_sample.csv"))

if players_df is None:
    st.error("Could not load players data. Please add data/players_sample.csv in the repo.")
    st.stop()

fixtures_df = load_repo_csv(DEFAULT_FIXTURES_PATH)
defpos_df   = load_repo_csv(DEFAULT_DEF_POS_PATH)

# Normalize & metrics
players = normalize_players(players_df)
players = compute_metrics(players)

# Controls
left, right = st.columns([2,1])
with right:
    gameweek = st.number_input("Gameweek", min_value=1, value=1, step=1)

# Fixtures & opponent
if fixtures_df is not None:
    fixtures_df["Gameweek"] = pd.to_numeric(fixtures_df["Gameweek"], errors="coerce").fillna(0).astype(int)
    fixtures_df["Team"] = fixtures_df["Team"].astype(str).str.strip()
    fixtures_df["Opponent"] = fixtures_df["Opponent"].astype(str).str.strip()
    players = attach_fixtures(players, fixtures_df, gameweek)
else:
    players["Opponent"] = ""

# Opponent impact
if defpos_df is not None:
    defpos_df.columns = [c.strip() for c in defpos_df.columns]
    defpos_df["Team"] = defpos_df["Team"].astype(str).str.strip()
players = apply_opponent_by_position(players, defpos_df)

# Pick Score (weights fixed for simplicity; you can expose sliders later)
w_value, w_usg, w_opp = 1.0, 0.7, 0.8
players["zValue"] = zscore(players["Value"])
players["zUSG"]   = zscore(players["USG%"])
players["zOpp"]   = zscore(players["OppBenefit"])
players["PickScore"] = w_value*players["zValue"] + w_usg*players["zUSG"] + w_opp*players["zOpp"]

# Filters
teams_sorted = sorted(players["Team"].unique().tolist())
pos_options = ["PG","SG","SF","PF","C","G","F","G/F","F/C"]
sel_teams = st.multiselect("Filter teams", teams_sorted, default=teams_sorted)
sel_pos   = st.multiselect("Filter positions (optional)", pos_options, default=[])
min_min   = st.slider("Min minutes", 0, int(players["Minutes"].max()) if len(players) else 36, 18, 1)

mask = players["Team"].isin(sel_teams) & (players["Minutes"] >= min_min)
if sel_pos:
    mask &= players["Position"].isin(sel_pos)
picks = players[mask].copy().sort_values("PickScore", ascending=False)

# Outputs
with left:
    st.subheader("Top Picks this Gameweek")
    top_n = st.slider("Show top N", 5, 40, 15, 1)
    top = picks.head(top_n)
    fig = px.bar(top, x="Player", y="PickScore", color="Team",
                 hover_data=["Position","Opponent","Value","USG%","EFF","OppBenefit"])
    st.plotly_chart(fig, use_container_width=True)

st.markdown("### Picks Table")
show_cols = ["Player","Team","Position","Opponent","Minutes","Price","EFF","USG%","Value","OppBenefit","PickScore"]
st.dataframe(top[show_cols].reset_index(drop=True), use_container_width=True)

with st.expander("See full filtered list"):
    st.dataframe(picks[show_cols].reset_index(drop=True), use_container_width=True)

# Correlation
st.markdown("### Correlation (how factors relate)")
x_metric = st.selectbox("X", ["Minutes","EFF","USG%","Value","OppBenefit"])
y_metric = st.selectbox("Y", ["PickScore","Value","USG%","EFF"])
fig_sc = px.scatter(picks, x=x_metric, y=y_metric, color="Team", hover_data=["Player","Position","Opponent"])
st.plotly_chart(fig_sc, use_container_width=True)

corr = picks[[x_metric,y_metric]].corr().iloc[0,1] if len(picks) else np.nan
st.write(f"**Pearson correlation between `{x_metric}` and `{y_metric}`:** `{corr:.3f}`" if pd.notna(corr) else "Not enough data.")
