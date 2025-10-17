import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Euroleague Fantasy – Weekly Picks", page_icon="🏀", layout="wide")
st.title("🏀 Euroleague Fantasy – Weekly Picks")

# ===============================
# FILE PATHS (no scraping; always works)
# ===============================
PLAYERS_PRIMARY   = Path("data/players_latest.csv")   # put the latest export here (optional)
PLAYERS_FALLBACK  = Path("data/players_sample.csv")   # already in your repo (used if latest not present)
DEF_POS_PATH      = Path("data/defense_by_pos.csv")
FIXTURES_PATH     = Path("data/fixtures.csv")

# ===============================
# FIXED WEIGHTS (tune here only)
# ===============================
# Value is EFF for now (until you add prices -> then Value = EFF / Price)
W_VALUE   = 1.00   # Efficiency weight
W_USAGE   = 0.80   # Usage% weight
W_OPP     = 0.80   # Opponent ease weight
W_MINUTES = 0.90   # Minutes weight (your request)

# ===============================
# HELPERS
# ===============================
def clean_cols(cols):
    """lower + remove symbols so we can map columns robustly."""
    out = []
    for c in cols:
        c0 = re.sub(r"\s+", " ", str(c)).strip()
        c1 = re.sub(r"[^A-Za-z0-9_/ ]+", "", c0)
        out.append(c1.lower())
    return out

def normalize_players_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Map common column names into our schema and coerce types."""
    df = df_raw.copy()
    df.columns = clean_cols(df.columns)

    def pick(*opts):
        for o in opts:
            if o in df.columns:
                return o
        return None

    col_player = pick("player", "name")
    col_team   = pick("team")
    col_min    = pick("min", "minutes")
    col_pts    = pick("pts", "points")
    col_fgm    = pick("fgm")
    col_fga    = pick("fga")
    col_ftm    = pick("ftm")
    col_fta    = pick("fta")
    col_reb    = pick("rebounds", "reb", "treb")
    col_ast    = pick("assists", "ast")
    col_stl    = pick("steals", "stl")
    col_blk    = pick("blocks", "blk")
    col_tov    = pick("turnovers", "tov")

    needed = [col_player, col_team, col_min, col_pts, col_fgm, col_fga, col_ftm, col_fta,
              col_reb, col_ast, col_stl, col_blk, col_tov]
    if any(c is None for c in needed):
        raise ValueError("Players CSV is missing required columns. Make sure it has Player/Team/MIN/PTS/FGM/FGA/FTM/FTA/REB/AST/STL/BLK/TOV (names can vary).")

    players = pd.DataFrame({
        "Player":   df[col_player].astype(str).str.strip(),
        "Team":     df[col_team].astype(str).str.strip(),
        "Minutes":  pd.to_numeric(df[col_min], errors="coerce").fillna(0),
        "Points":   pd.to_numeric(df[col_pts], errors="coerce").fillna(0),
        "FGM":      pd.to_numeric(df[col_fgm], errors="coerce").fillna(0),
        "FGA":      pd.to_numeric(df[col_fga], errors="coerce").fillna(0),
        "FTM":      pd.to_numeric(df[col_ftm], errors="coerce").fillna(0),
        "FTA":      pd.to_numeric(df[col_fta], errors="coerce").fillna(0),
        "Rebounds": pd.to_numeric(df[col_reb], errors="coerce").fillna(0),
        "Assists":  pd.to_numeric(df[col_ast], errors="coerce").fillna(0),
        "Steals":   pd.to_numeric(df[col_stl], errors="coerce").fillna(0),
        "Blocks":   pd.to_numeric(df[col_blk], errors="coerce").fillna(0),
        "Turnovers":pd.to_numeric(df[col_tov], errors="coerce").fillna(0),
    })
    return players

def compute_metrics(players: pd.DataFrame) -> pd.DataFrame:
    # PIR-style Efficiency (EFF)
    miss_fg = players["FGA"] - players["FGM"]
    miss_ft = players["FTA"] - players["FTM"]
    players["EFF"] = (
        players["Points"] + players["Rebounds"] + players["Assists"]
        + players["Steals"] + players["Blocks"]
        - miss_fg - miss_ft - players["Turnovers"]
    )

    # Usage % estimate
    team_tot = players.groupby("Team", as_index=False)[["Minutes","FGA","FTA","Turnovers"]].sum()
    team_tot = team_tot.rename(columns={"Minutes":"TeamMinutes","FGA":"TeamFGA","FTA":"TeamFTA","Turnovers":"TeamTOV"})
    team_tot["TeamMinutesPerPlayer"] = team_tot["TeamMinutes"] / 5.0
    players = players.merge(team_tot, on="Team", how="left")

    numer = (players["FGA"] + 0.44*players["FTA"] + players["Turnovers"]) * players["TeamMinutesPerPlayer"]
    denom = players["Minutes"] * (players["TeamFGA"] + 0.44*players["TeamFTA"] + players["TeamTOV"])
    players["USG%"] = np.where(denom > 0, 100 * numer / denom, 0.0)

    # Value (until you add prices) = EFF
    players["Value"] = players["EFF"]
    return players

def load_csv_if_exists(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None

def apply_fixtures(players: pd.DataFrame, fixtures: pd.DataFrame | None, gw: int) -> pd.DataFrame:
    if fixtures is None or "Gameweek" not in fixtures.columns:
        players["Opponent"] = ""
        return players
    fx = fixtures.copy()
    fx["Gameweek"] = pd.to_numeric(fx["Gameweek"], errors="coerce").fillna(0).astype(int)
    fx["Team"] = fx["Team"].astype(str).str.strip()
    fx["Opponent"] = fx["Opponent"].astype(str).str.strip()
    this = fx[fx["Gameweek"] == gw]
    if this.empty:
        players["Opponent"] = ""
        return players
    return players.merge(this, on="Team", how="left")

def apply_opponent_avg(players: pd.DataFrame, defpos: pd.DataFrame | None) -> pd.DataFrame:
    """Use team average defensive multiplier across positions.
       OppBenefit > 1 = easier opponent, < 1 = tougher."""
    if defpos is None or not {"Team","Def_PG","Def_SG","Def_SF","Def_PF","Def_C"}.issubset(defpos.columns):
        players["OppBenefit"] = 1.0
        return players
    d = defpos.copy()
    d["Team"] = d["Team"].astype(str).str.strip()
    d["AvgDef"] = d[["Def_PG","Def_SG","Def_SF","Def_PF","Def_C"]].mean(axis=1)
    d["OppBenefitTeam"] = np.where(d["AvgDef"] > 0, 1.0 / d["AvgDef"], 1.0)
    players = players.merge(d[["Team","OppBenefitTeam"]].rename(columns={"Team":"Opponent"}), on="Opponent", how="left")
    players["OppBenefit"] = players["OppBenefitTeam"].fillna(1.0)
    players.drop(columns=["OppBenefitTeam"], inplace=True)
    return players

def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(0, index=s.index)
    return (s - mu) / sd

# ===============================
# LOAD DATA (always works)
# ===============================
players_raw = load_csv_if_exists(PLAYERS_PRIMARY)
if players_raw is None:
    players_raw = load_csv_if_exists(PLAYERS_FALLBACK)

if players_raw is None:
    st.error("No players CSV found. Add either data/players_latest.csv or data/players_sample.csv.")
    st.stop()

try:
    players = normalize_players_columns(players_raw)
except Exception as e:
    st.error(f"Players CSV format error: {e}")
    st.stop()

players = compute_metrics(players)
fixtures = load_csv_if_exists(FIXTURES_PATH)
defpos   = load_csv_if_exists(DEF_POS_PATH)

# ===============================
# CONTROLS
# ===============================
right = st.sidebar
right.header("Controls")
gameweek = right.number_input("Gameweek", min_value=1, value=1, step=1)
min_min  = right.slider("Min minutes", 0, int(players["Minutes"].max()) if len(players) else 40, 10, 1)
top_n    = right.slider("Show top N", 5, 50, 20, 1)

# ===============================
# WEEKLY OPPONENT + RANKING
# ===============================
players = apply_fixtures(players, fixtures, gameweek)
players = apply_opponent_avg(players, defpos)

players["zValue"]   = zscore(players["Value"])
players["zUSG"]     = zscore(players["USG%"])
players["zOpp"]     = zscore(players["OppBenefit"])
players["zMinutes"] = zscore(players["Minutes"])

players["PickScore"] = (
    W_VALUE   * players["zValue"]   +
    W_USAGE   * players["zUSG"]     +
    W_OPP     * players["zOpp"]     +
    W_MINUTES * players["zMinutes"]
)

# ===============================
# OUTPUTS
# ===============================
teams_sorted = sorted(players["Team"].unique().tolist())
sel_teams = st.multiselect("Filter teams", teams_sorted, default=teams_sorted)
mask = players["Team"].isin(sel_teams) & (players["Minutes"] >= min_min)
picks = players[mask].sort_values("PickScore", ascending=False)

st.subheader("Top Picks this Gameweek")
fig = px.bar(
    picks.head(top_n),
    x="Player", y="PickScore", color="Team",
    hover_data=["Team","Opponent","EFF","USG%","Value","OppBenefit","Minutes"]
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("### Picks Table")
cols = ["Player","Team","Opponent","Minutes","EFF","USG%","Value","OppBenefit","PickScore"]
st.dataframe(picks[cols].reset_index(drop=True), use_container_width=True)

st.markdown("### Correlation")
x = st.selectbox("X", ["Minutes","EFF","USG%","Value","OppBenefit"])
y = st.selectbox("Y", ["PickScore","Value","USG%","EFF"])
fig2 = px.scatter(picks, x=x, y=y, color="Team", hover_data=["Player","Opponent"])
st.plotly_chart(fig2, use_container_width=True)
if not picks.empty:
    corr = picks[[x, y]].corr().iloc[0, 1]
    st.write(f"**Pearson r({x}, {y}) = {corr:.3f}**")
