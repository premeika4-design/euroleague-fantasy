import streamlit as st
import pandas as pd
import plotly.express as px

# App setup
st.set_page_config(page_title="Euroleague Fantasy Dashboard", page_icon="🏀", layout="wide")
st.title("🏀 Euroleague Fantasy Dashboard")

st.sidebar.header("Filter Players")
selected_team = st.sidebar.selectbox("Select a team", ["All", "Real Madrid", "FC Barcelona", "Olympiacos", "Panathinaikos", "Fenerbahçe"])
selected_metric = st.sidebar.radio("Select metric", ["Points", "Rebounds", "Assists"])

# Sample data (we’ll connect to real data later)
data = {
    "Player": ["Luka", "Nikola", "Vasilis", "Mario", "Toko", "Mike"],
    "Team": ["Real Madrid", "FC Barcelona", "Olympiacos", "Panathinaikos", "Fenerbahçe", "Monaco"],
    "Points": [18, 15, 12, 20, 17, 22],
    "Rebounds": [7, 5, 6, 4, 9, 3],
    "Assists": [6, 8, 4, 5, 7, 9],
}

df = pd.DataFrame(data)

# Filter by team
if selected_team != "All":
    df = df[df["Team"] == selected_team]

# Plot chart
fig = px.bar(df, x="Player", y=selected_metric, color="Team", title=f"{selected_metric} per Player")

# Display
st.plotly_chart(fig, use_container_width=True)
st.dataframe(df)
