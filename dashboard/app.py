import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="IPL Analytics Dashboard",
    page_icon="🏏",
    layout="wide"
)

st.title("🏏 IPL Data Engineering & Analytics Dashboard")
st.markdown(
    "Analytics built from an AWS S3 → Databricks → Apache Spark → "
    "Spark SQL data pipeline."
)

# -----------------------------
# Load Gold datasets
# -----------------------------
@st.cache_data
def load_data():
    batsmen = pd.read_csv("data/top_batsmen.csv")
    bowlers = pd.read_csv("data/top_bowlers.csv")
    teams = pd.read_csv("data/team_performance.csv")
    toss = pd.read_csv("data/toss_analysis.csv")

    return batsmen, bowlers, teams, toss


batsmen, bowlers, teams, toss = load_data()

# -----------------------------
# KPI section
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Matches", "1,243")
col2.metric("Deliveries", "295K+")
col3.metric("Players", "261")
col4.metric("Teams", "19")

st.divider()

# -----------------------------
# Top Batsmen
# -----------------------------
st.subheader("🏏 Top 10 Run Scorers")

top_batsmen = batsmen.sort_values(
    "total_runs",
    ascending=False
).head(10)

fig_batsmen = px.bar(
    top_batsmen,
    x="batter",
    y="total_runs",
    title="Top 10 IPL Run Scorers",
    labels={
        "batter": "Batsman",
        "total_runs": "Runs"
    }
)

st.plotly_chart(fig_batsmen, use_container_width=True)

# -----------------------------
# Top Bowlers
# -----------------------------
st.subheader("🎯 Top 10 Wicket Takers")

top_bowlers = bowlers.sort_values(
    "wickets",
    ascending=False
).head(10)

fig_bowlers = px.bar(
    top_bowlers,
    x="bowler",
    y="wickets",
    title="Top 10 IPL Wicket Takers",
    labels={
        "bowler": "Bowler",
        "wickets": "Wickets"
    }
)

st.plotly_chart(fig_bowlers, use_container_width=True)

# -----------------------------
# Team Performance
# -----------------------------
st.subheader("🏆 Team Performance")

fig_teams = px.bar(
    teams.sort_values("win_percentage", ascending=False),
    x="team",
    y="win_percentage",
    title="Team Win Percentage",
    labels={
        "team": "Team",
        "win_percentage": "Win Percentage"
    }
)

st.plotly_chart(fig_teams, use_container_width=True)

# -----------------------------
# Toss Analysis
# -----------------------------
st.subheader("🪙 Toss Decision Analysis")

fig_toss = px.bar(
    toss,
    x="toss_decision",
    y="win_percentage",
    title="Win Percentage Based on Toss Decision",
    labels={
        "toss_decision": "Toss Decision",
        "win_percentage": "Win Percentage"
    }
)

st.plotly_chart(fig_toss, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption(
    "Built with AWS S3 • Databricks • Apache Spark • PySpark • Spark SQL • Streamlit"
)