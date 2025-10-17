import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO
from pathlib import Path

st.set_page_config(page_title="Euroleague Fantasy – Weekly Picks", page_icon="🏀", layout="wide")
st.title("🏀 Euroleague Fantasy – Weekly Picks")

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
            if n.lower() in lookup: return lookup[n.lower()]
        return None

    out = pd.DataFrame()
    out["Player"]   = df[pick("Player","Name")] if pick("Player","Name") else ""
    out["Team"]     = df[pick("Team")] if pick("Team") else ""
    out["Position"] = df[pick("Position","Pos")] if pick("Position","Pos") else "G"
    out["Price"]    = df[pick("Price","Cost","Value")] if pick("Price","Cost","Value") else 0
    out["Minutes"]  = df[pick("Minutes","MIN")] if pick("Minutes","MIN") else 0
    out["FGM"]      = df[pick("FGM","FG Made")] if pick("FGM","FG Made") else 0
    out["FGA"]      = df[pick("FGA","FG Att")] if pick("FGA","FG Att") else 0
    out["FTM"]      = df[pick("FTM","FT Made")] if pick("FTM","FT Made") else 0
    out["FTA"]      = df[pick("FTA","FT Att")] if pick("FTA","FT Att") else 0
    out["Rebounds"] = df[pick("Rebounds","REB","Total Rebounds")] if pick("Rebounds","REB","Total Rebounds") else 0
    out["Assists"]  = df[pick("Assists","AST")] if pick("Assists","AST") else 0
    out["Steals"]   = df[pick("Steals","STL")] if pick("Steals","STL") else 0
    out["Blocks"]   = df[pick("Blocks","BLK")] if pick("Blocks","BLK") else 0
    out["Turnovers"]= df[pick("Turnovers","TOV")] if pick("Turnovers","TOV") else 0
    out["Points"]   = df[pick("Points","PTS")] if pick("Points","PTS") else 0

    # Clean
    out["Player"]   = out["Player"].astype(str).str.strip()
    out["Team"]     = out["Team"].astype(str).str.strip()
    out["Position"] = out["Position"].astype(str).upper().str.replace(" ", "")
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

def attach_fixtures(players, fixtures, gameweek):
    fx = fixtures[fixtures["Gameweek"] == gameweek].copy()
    return players.merge(fx, on="Team", how="left")  # adds Opponent

def apply_opponent_by_position(df, defpos):
    # Normalize columns
    defpos = defpos.rename(columns={c: c.strip() for c in defpos.columns})
    needed = {"Team","Def_PG","Def_SG","Def_SF","Def_PF","Def_C"}
    if not needed.issubset(set(defpos.columns)):
        df["OppPosImpact"] = 1.0
        df["OppBenefit"] = 1.0
        return df

    pos_map = {"PG":"Def_PG","SG":"Def_SG","SF":"Def_SF","PF":"Def_PF","C":"Def_C"}

    def get_def_mult(row):
        opp = row.get("Opponent", "")
        pos = str(row.get("Position","")).upper()
        if opp not in defpos["Team"].values or not pos:
            return 1.0
        row_d = defpos[defpos["Team"] == opp].iloc[0]
        slots = []
        if "/" in pos:
            for p in pos.split("/"):
                p = p.strip()
                if p in pos_map and pos_map[p] in row_d: slots.append(row_d[pos_map[p]])
        elif pos in pos_map and pos_map[pos] in row_d:
            slots.append(row_d[pos_map[pos]])
        else:
            if pos.startswith("G"):
                slots.extend([row_d["Def_PG"], row_d["Def_SG"]])
            elif pos.startswith("F"):
                slots.extend([row_d["Def_SF"], row_d["Def_PF"]])
            else:
                slots.append(row_d["Def_C"])
        return float(np.mean(slots)) if slots else 1.0

    df["OppPosImpact"] = df.apply(get_def_mult, axis=1)
    df["OppBenefit"] = np.where(df["OppPosImpact"] > 0, 1.0 / df["OppPosImpact"], 1.0)  # >1 softer, <1 tougher
    return df

def zscore(series):
    s = pd.to_numeric(series, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series([0]*len(s), index=s.index)
    return (s - mu) / sd

def try_load_repo_csv(path):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None

# --------------------------
# Sidebar: data inputs
# --------------------------
st.sidebar.header("Data Inputs")

players_url = st.sidebar.text_input("Players CSV URL (givemestat export)")
players_up  = st.sidebar.file_uploader("Or upload Players CSV", type=["csv"], key="pup")

fixtures_url = st.sidebar.text_input("Fixtures CSV URL (Gameweek,Team,Opponent)")
fixtures_up  = st.sidebar.file_uploader("Or upload Fixtures CSV", type=["csv"], key="fxup")

defpos_url = st.sidebar.text_input("Defense-by-Position CSV URL")
defpos_up  = st.sidebar.file_uploader("Or upload Defense-by-Position CSV", type=["csv"], key="dposup")

# --------------------------
# Load data
# --------------------------
players_df = None
try:
    if players_url:
        players_df = read_csv_url(players_url)
    elif players_up is not None:
        players_df = read_csv_bytes(players_up.getvalue())
except Exception as e:
    st.error(f"Could not read Players CSV: {e}")

fixtures_df = None
defpos_df = None

try:
    if fixtures_url:
        fixtures_df = read_csv_url(fixtures_url)
    elif fixtures_up is not None:
        fixtures_df = read_csv_bytes(fixtures_up.getvalue())
    else:
        fixtures_df = try_load_repo_csv("data/fixtures.csv")
except Exception as e:
    st.warning(f"Fixtures CSV unreadable: {e}")

try:
    if defpos_url:
        defpos_df = read_csv_url(defpos_url)
    elif defpos_up is not None:
        defpos_df = read_csv_bytes(defpos_up.getvalue())
    else:
        defpos_df = try_load_repo_csv("data/defense_by_pos.csv")
except Exception as e:
    st.warning(f"Defense-by-Position CSV unreadable: {e}")

if players_df is None:
    st.info("Paste a **Players CSV URL** from givemestat or upload a CSV in the sidebar.")
    st.stop()

# Normalize & compute player metrics
players = normalize_players(players_df)
players = compute_metrics(players)

# Controls
left, right = st.columns([2,1])
with right:
    gameweek = st.number_input("Gameweek", min_value=1, value=1, step=1)

# Attach fixtures & opponent
if fixtures_df is not None:
    if "Gameweek" in fixtures_df.columns:
        fixtures_df["Gameweek"] = pd.to_numeric(fixtures_df["Gameweek"], errors="coerce").fillna(0).astype(int)
    fixtures_df["Team"] = fixtures_df["Team"].astype(str).str.strip()
    fixtures_df["Opponent"] = fixtures_df["Opponent"].astype(str).str.strip()
    players = attach_fixtures(players, fixtures_df, gameweek)
else:
    players["Opponent"] = ""

# Opponent by position
if defpos_df is not None:
    defpos_df.columns = [c.strip() for c in defpos_df.columns]
    defpos_df["Team"] = defpos_df["Team"].astype(str).str.strip()
    players = apply_opponent_by_position(players, defpos_df)
else:
    players["OppPosImpact"] = 1.0
    players["OppBenefit"] = 1.0

# --------------------------
# Pick Score (tunable)
# --------------------------
st.sidebar.header("Pick Score Weights")
w_value = st.sidebar.slider("Weight: Value (EFF/Price)", 0.0, 2.0, 1.0, 0.05)
w_usg   = st.sidebar.slider("Weight: Usage %", 0.0, 2.0, 0.7, 0.05)
w_opp   = st.sidebar.slider("Weight: Opponent (by position)", 0.0, 2.0, 0.8, 0.05)

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

# --------------------------
# Outputs
# --------------------------
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

# Correlation view
st.markdown("### Correlation (how factors relate)")
x_metric = st.selectbox("X", ["Minutes","EFF","USG%","Value","OppBenefit"])
y_metric = st.selectbox("Y", ["PickScore","Value","USG%","EFF"])
fig_sc = px.scatter(picks, x=x_metric, y=y_metric, color="Team", hover_data=["Player","Position","Opponent"])
st.plotly_chart(fig_sc, use_container_width=True)

corr = picks[[x_metric,y_metric]].corr().iloc[0,1] if len(picks) else np.nan
st.write(f"**Pearson correlation between `{x_metric}` and `{y_metric}`:** `{corr:.3f}`" if pd.notna(corr) else "Not enough data for correlation.")
