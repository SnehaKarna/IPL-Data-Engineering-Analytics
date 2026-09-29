from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from agentic.team_names import TEAM_NAME_VARIANTS, normalize_team_performance
from agentic.venue_names import VENUE_NAME_VARIANTS
from ml.chase_prediction import ChasePredictionTool


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
GOLD_DIR = DATA_DIR / "gold"
TABLES = (
    "player_matchups", "player_form", "team_performance", "toss_analysis",
    "venue_strategy", "phase_analysis",
)
PAGE_OPTIONS = {
    "Overview": "Overview",
    "Player matchup": "Player matchup",
    "Venue & teams": "Venue & teams",
    "Chase prediction": "Chase prediction",
    "Strategy assistant": "Strategy assistant",
    "Data & model": "Data & model",
}

st.set_page_config(
    page_title="IPL Insight | Strategy Intelligence",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root {
      color-scheme: dark; --bg:#050d18; --sidebar:#071321; --panel:#0a1929;
      --panel-hi:#0e2034; --line:#1b3550; --muted:#8ea5bd; --text:#edf5ff;
      --blue:#2584ff; --cyan:#41c8e8; --gold:#ffc857; --green:#40d6a0;
    }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp, [data-testid="stAppViewContainer"] { background:radial-gradient(ellipse at 67% -12%,#12345a 0%,var(--bg) 44%); color:var(--text); }
    [data-testid="stMain"] { background:transparent; }
    [data-testid="stMain"] h1,[data-testid="stMain"] h2,[data-testid="stMain"] h3,
    [data-testid="stMain"] h4,[data-testid="stMain"] p,[data-testid="stMain"] label,
    [data-testid="stMain"] li,[data-testid="stMain"] span { color:var(--text); }
    [data-testid="stSidebar"] { background:linear-gradient(180deg,#081728,#06111d 70%); border-right:1px solid #18304a; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:var(--muted); }
    [data-testid="stSidebar"] label,[data-testid="stSidebar"] label p,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] { color:#c4d6e9 !important; }
    [data-testid="stSidebar"] [role="radiogroup"] label { border-radius:9px; padding:7px 10px; }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover { background:#102740; }
    .block-container { max-width:1500px; padding-top:1.65rem; padding-bottom:3rem; }
    .eyebrow { color:var(--cyan)!important; text-transform:uppercase; letter-spacing:.16em; font:500 10px 'DM Mono',monospace; }
    .page-title { color:var(--text)!important; font:800 clamp(1.8rem,3.2vw,2.75rem) 'Manrope',sans-serif; letter-spacing:-.045em; line-height:1.08; margin:.25rem 0 .35rem; }
    .page-copy { color:var(--muted)!important; max-width:820px; line-height:1.6; margin-bottom:1.25rem; }
    .brand { font:800 1.08rem 'Manrope',sans-serif; letter-spacing:.02em; color:#f3f8ff; }
    .brand-mark { color:var(--gold); }
    .section-label { color:var(--cyan)!important; font:500 10px 'DM Mono',monospace; letter-spacing:.14em; text-transform:uppercase; margin:1rem 0 .45rem; }
    .hero { border:1px solid #24476b; border-radius:18px; padding:1.5rem 1.7rem; background:radial-gradient(ellipse at 82% 0%,rgba(37,132,255,.25),transparent 52%),linear-gradient(125deg,#0b2036,#0a1929 65%,#0a1c2c); box-shadow:0 14px 46px #0005; margin:.35rem 0 1rem; }
    .hero h2 { font:800 clamp(1.5rem,2.7vw,2.25rem) 'Manrope',sans-serif; letter-spacing:-.04em; margin:.2rem 0 .35rem; }
    .hero p { color:#a9bed4!important; margin:0; max-width:800px; }
    .panel { background:linear-gradient(145deg,#0b1b2c,#091625); border:1px solid var(--line); border-radius:14px; padding:1rem 1.1rem; height:100%; }
    .panel-kicker { color:var(--muted)!important; text-transform:uppercase; letter-spacing:.12em; font:500 10px 'DM Mono',monospace; }
    .source-tag { display:inline-block; border:1px solid #254562; border-radius:999px; padding:.2rem .6rem; margin:.15rem .25rem .15rem 0; color:#a9dafa!important; background:#10263b; font:11px 'DM Mono',monospace; }
    div[data-testid="stMetric"] { background:linear-gradient(145deg,#0d2135,#091827); border:1px solid var(--line); padding:.85rem 1rem; border-radius:12px; min-height:105px; }
    div[data-testid="stMetricLabel"],div[data-testid="stMetricLabel"] * { color:var(--muted)!important; font-size:.82rem; }
    div[data-testid="stMetricValue"],div[data-testid="stMetricValue"] * { color:#f2f7ff!important; font-family:'Manrope',sans-serif; font-weight:700; }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stNumberInput"] input,[data-testid="stTextInput"] input,
    [data-testid="stChatInput"] textarea { background:#0b1c2c!important; color:var(--text)!important; border-color:#27435e!important; }
    [data-testid="stChatInput"] textarea::placeholder,[data-testid="stTextInput"] input::placeholder { color:#7189a2!important; }
    [data-testid="stExpander"] { background:#0a1929; border:1px solid var(--line); border-radius:12px; }
    [data-testid="stChatMessage"] { border:1px solid var(--line); border-radius:14px; padding:.6rem 1rem; background:#0a1929; }
    [data-testid="stChatMessage"] p,[data-testid="stChatMessage"] li { color:var(--text)!important; }
    .stButton > button { border-radius:9px; background:linear-gradient(110deg,#1670e8,#2584ff); color:#fff!important; border:1px solid #338fff; font-weight:600; }
    .stButton > button:hover { background:#3d94ff; border-color:#5ba5ff; }
    [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:11px; overflow:hidden; }
    [data-testid="stAlert"] { background:#102338; border-color:#275276; }
    [data-testid="stCaptionContainer"] { color:var(--muted); }
    hr { border-color:#19324b; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / filename)


@st.cache_data(show_spinner=False)
def load_gold(table: str) -> pd.DataFrame:
    return pd.read_parquet(GOLD_DIR / f"{table}.parquet")


@st.cache_resource(show_spinner="Loading local Gold analytics…")
def load_agent():
    from agentic.strategy_agent import IPLStrategyAgent

    return IPLStrategyAgent()


def page_header(eyebrow: str, title: str, copy: str) -> None:
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-copy">{copy}</div>', unsafe_allow_html=True)


def navigate_to(page: str) -> None:
    st.session_state.page = page


def section_title(title: str, detail: str | None = None) -> None:
    st.markdown(f"### {title}")
    if detail:
        st.caption(detail)


def chart_theme(fig, height: int = 300):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#d9e8f7"),
        margin=dict(l=8, r=12, t=24, b=8), height=height,
        xaxis=dict(gridcolor="#173049", zerolinecolor="#173049"),
        yaxis=dict(gridcolor="#173049", zerolinecolor="#173049"),
    )
    return fig


def required_tables(names: tuple[str, ...] = TABLES) -> dict[str, pd.DataFrame] | None:
    missing = [name for name in names if not (GOLD_DIR / f"{name}.parquet").exists()]
    if missing:
        st.error("Some Gold analytics snapshots are missing: " + ", ".join(missing))
        st.caption(f"Expected in `{GOLD_DIR}`. Export those tables from the data pipeline and reload the app.")
        return None
    try:
        return {name: load_gold(name) for name in names}
    except Exception as exc:
        st.error(f"Could not read the local Gold Parquet snapshots: {exc}")
        return None


def weighted_mean(frame: pd.DataFrame, value: str, weight: str) -> float:
    clean = frame[[value, weight]].dropna()
    if clean.empty or clean[weight].sum() == 0:
        return 0.0
    return float((clean[value].astype(float) * clean[weight].astype(float)).sum() / clean[weight].sum())


def render_overview() -> None:
    data = required_tables()
    if data is None:
        return
    try:
        batsmen = load_csv("top_batsmen.csv")
        bowlers = load_csv("top_bowlers.csv")
        teams = normalize_team_performance(data["team_performance"])
        venue = data["venue_strategy"]
        phase = data["phase_analysis"].copy()
    except Exception as exc:
        st.error(f"Could not load overview data: {exc}")
        return

    st.markdown(
        '<div class="hero"><div class="eyebrow">IPL STRATEGY & PERFORMANCE INTELLIGENCE</div>'
        '<h2>Read the game behind the numbers.</h2>'
        '<p>Explore historical player matchups, franchise results, venue trends and innings phases. Ask the strategy assistant to connect the evidence.</p></div>',
        unsafe_allow_html=True,
    )
    delivery_count = int(pd.to_numeric(phase["deliveries"], errors="coerce").fillna(0).sum())
    match_count = int(pd.to_numeric(venue["matches"], errors="coerce").fillna(0).sum())
    metrics = st.columns(5)
    metrics[0].metric("Matches in venue data", f"{match_count:,}")
    metrics[1].metric("Franchises", f"{len(teams):,}")
    metrics[2].metric("Player records", f"{len(data['player_form']):,}")
    metrics[3].metric("Deliveries analyzed", f"{delivery_count:,}")
    metrics[4].metric("Avg chase wins by venue", f"{weighted_mean(venue, 'chasing_win_percentage', 'matches'):.1f}%")

    left, right = st.columns([1.08, .92], gap="medium")
    with left:
        section_title("Franchise performance", "Historical win percentage across the available match data")
        team_chart = px.bar(
            teams.nlargest(10, "win_percentage").sort_values("win_percentage"),
            x="win_percentage", y="team", orientation="h", text="win_percentage",
            color="win_percentage", color_continuous_scale=["#15558b", "#35b8e8", "#ffc857"],
            labels={"win_percentage": "Win %", "team": ""},
        )
        team_chart.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
        team_chart.update_layout(coloraxis_showscale=False, yaxis_title=None, xaxis_title="Win percentage", height=365)
        st.plotly_chart(chart_theme(team_chart, 365), width="stretch", config={"displayModeBar": False})
    with right:
        section_title("Venue chase trends", "Ground-wide historical aggregates; not a match forecast")
        venue_view = venue.sort_values("chasing_win_percentage", ascending=False).head(9).sort_values("chasing_win_percentage")
        venue_chart = px.bar(
            venue_view, x="chasing_win_percentage", y="venue", orientation="h", color="chasing_win_percentage",
            color_continuous_scale=["#1f5b8f", "#2584ff", "#41c8e8"],
            labels={"chasing_win_percentage": "Chase win %", "venue": ""},
        )
        venue_chart.update_layout(coloraxis_showscale=False, yaxis_title=None, xaxis_title="Chase win percentage", height=365)
        st.plotly_chart(chart_theme(venue_chart, 365), width="stretch", config={"displayModeBar": False})

    col1, col2, col3 = st.columns([1, 1, 1.05], gap="medium")
    with col1:
        section_title("Top run scorers")
        st.dataframe(batsmen.nlargest(7, "total_runs").rename(columns={"batter": "Player", "total_runs": "Runs", "matches_played": "Matches"}), hide_index=True, width="stretch", height=285)
    with col2:
        section_title("Top wicket takers")
        st.dataframe(bowlers.nlargest(7, "wickets").rename(columns={"bowler": "Player", "wickets": "Wickets"}), hide_index=True, width="stretch", height=285)
    with col3:
        section_title("Scoring by innings phase", "Runs per six deliveries")
        phase["runs_per_six"] = phase["runs"] / phase["deliveries"] * 6
        phase_chart = px.bar(phase, x="phase", y="runs_per_six", color="phase", text="runs_per_six", color_discrete_sequence=["#2584ff", "#7b6cff", "#ffc857"], labels={"phase": "", "runs_per_six": "Runs / 6 balls"})
        phase_chart.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        st.plotly_chart(chart_theme(phase_chart, 285), width="stretch", config={"displayModeBar": False})

    st.markdown('<div class="panel"><span class="panel-kicker">Strategy desk</span><h3>Ask a question that combines the evidence</h3><p style="color:#8ea5bd">Compare a player matchup, explore a ground, or ask whether the historical numbers support a tactical call.</p></div>', unsafe_allow_html=True)
    st.button("Open strategy assistant →", key="overview-to-assistant", on_click=navigate_to, args=("Strategy assistant",))


def render_player_matchup() -> None:
    data = required_tables(("player_matchups", "player_form"))
    if data is None:
        return
    matchups, form = data["player_matchups"], data["player_form"]
    page_header("ANALYTICS / PLAYER MATCHUPS", "Player matchup lab", "Inspect batter-versus-bowler history from the deterministic Gold matchup table.")
    batter_options = sorted(matchups["batter"].dropna().astype(str).unique())
    bowler_options = sorted(matchups["bowler"].dropna().astype(str).unique())
    default_batter = next((name for name in batter_options if name == "V Kohli"), batter_options[0] if batter_options else None)
    default_bowler = next((name for name in bowler_options if name == "JJ Bumrah"), bowler_options[0] if bowler_options else None)
    left, versus, right = st.columns([5, .7, 5], vertical_alignment="bottom")
    with left:
        batter = st.selectbox("BATTER", batter_options, index=batter_options.index(default_batter) if default_batter else None, placeholder="Select a batter")
    with versus:
        st.markdown("<div style='text-align:center;padding:10px 0;color:#8ea5bd;font-weight:700'>VS</div>", unsafe_allow_html=True)
    with right:
        bowler = st.selectbox("BOWLER", bowler_options, index=bowler_options.index(default_bowler) if default_bowler else None, placeholder="Select a bowler")
    if not batter or not bowler:
        st.info("Choose a batter and bowler to load the head-to-head record.")
        return
    rows = matchups[(matchups["batter"] == batter) & (matchups["bowler"] == bowler)]
    if rows.empty:
        st.warning(f"No recorded matchup was found for {batter} against {bowler}.")
        return
    row = rows.iloc[0]
    form_row = form[form["batter"] == batter]
    player_runs = form_row.iloc[0]["total_runs"] if not form_row.empty else "—"
    st.markdown(f'<div class="panel"><span class="panel-kicker">HEAD TO HEAD · {batter} vs {bowler}</span><h2 style="margin:.4rem 0">Matchup profile</h2><p style="color:#8ea5bd">The matchup is a historical aggregate from the available IPL deliveries.</p></div>', unsafe_allow_html=True)
    metrics = st.columns(6)
    metrics[0].metric("Meetings", f"{int(row['matches']):,}")
    metrics[1].metric("Balls faced", f"{int(row['balls_faced']):,}")
    metrics[2].metric("Runs", f"{int(row['runs']):,}")
    metrics[3].metric("Strike rate", f"{float(row['strike_rate']):.1f}")
    metrics[4].metric("Dismissals", f"{int(row['dismissals']):,}")
    metrics[5].metric("Batter career runs", f"{int(player_runs):,}")
    st.write("")
    left, right = st.columns([1, 1], gap="large")
    with left:
        section_title("Run outcome")
        outcomes = pd.DataFrame({"Outcome": ["Runs", "Boundary runs", "Dismissals"], "Count": [row["runs"], row["boundary_runs"], row["dismissals"]]})
        chart = px.bar(outcomes, x="Count", y="Outcome", orientation="h", color="Outcome", color_discrete_sequence=["#2584ff", "#ffc857", "#ed6682"], text="Count")
        chart.update_layout(showlegend=False, xaxis_title="Recorded count", yaxis_title="")
        st.plotly_chart(chart_theme(chart), width="stretch", config={"displayModeBar": False})
    with right:
        section_title("Record and scope")
        st.markdown(f'<div class="panel"><p><b>{batter}</b> scored <b>{int(row["runs"])} runs</b> from {int(row["balls_faced"])} balls against <b>{bowler}</b>, with {int(row["dismissals"])} recorded dismissals.</p><p style="color:#8ea5bd">Source: player_matchups. Career runs come from player_form and cover the player’s full available record; they are not matchup-specific.</p></div>', unsafe_allow_html=True)


def render_venue_teams() -> None:
    data = required_tables(("venue_strategy", "team_performance", "toss_analysis"))
    if data is None:
        return
    venue = data["venue_strategy"].copy()
    teams = normalize_team_performance(data["team_performance"])
    page_header("ANALYTICS / VENUE & TEAMS", "Ground and franchise intelligence", "Keep ground-wide trends distinct from franchise-wide results. These aggregates do not imply venue-specific team performance.")
    left, right = st.columns(2)
    with left:
        selected_venue = st.selectbox("GROUND", sorted(venue["venue"].dropna().unique()), index=(sorted(venue["venue"].dropna().unique()).index("Wankhede Stadium") if "Wankhede Stadium" in set(venue["venue"].dropna()) else 0))
        row = venue[venue["venue"] == selected_venue].iloc[0]
        st.markdown(f'<div class="panel"><span class="panel-kicker">VENUE STRATEGY · {selected_venue.upper()}</span></div>', unsafe_allow_html=True)
        cards = st.columns(3)
        cards[0].metric("Matches", f"{int(row['matches']):,}")
        cards[1].metric("Avg 1st innings", f"{float(row['avg_first_innings_score']):.1f}")
        cards[2].metric("Chase wins", f"{float(row['chasing_win_percentage']):.1f}%")
        st.metric("Average 2nd innings score", f"{float(row['avg_second_innings_score']):.1f}")
    with right:
        selected_team = st.selectbox("FRANCHISE", sorted(teams["team"].dropna().unique()))
        row_team = teams[teams["team"] == selected_team].iloc[0]
        st.markdown(f'<div class="panel"><span class="panel-kicker">OVERALL TEAM PERFORMANCE · {selected_team.upper()}</span></div>', unsafe_allow_html=True)
        cards = st.columns(3)
        cards[0].metric("Matches", f"{int(row_team['matches_played']):,}")
        cards[1].metric("Wins", f"{int(row_team['wins']):,}")
        cards[2].metric("Win rate", f"{float(row_team['win_percentage']):.1f}%")
        st.caption("Team record covers all available venues and match roles. It is not specific to the selected ground.")
    section_title("Venue comparison", "Historical chase-win percentage, with match counts available in hover details")
    chart = px.scatter(venue, x="avg_first_innings_score", y="chasing_win_percentage", size="matches", color="avg_second_innings_score", hover_name="venue", hover_data={"matches": True, "avg_first_innings_score": ":.1f", "avg_second_innings_score": ":.1f", "chasing_win_percentage": ":.1f"}, color_continuous_scale=["#205b8a", "#2584ff", "#ffc857"], labels={"avg_first_innings_score": "Average first innings", "chasing_win_percentage": "Chase win %", "avg_second_innings_score": "Avg second innings", "matches": "Matches"})
    st.plotly_chart(chart_theme(chart, 370), width="stretch", config={"displayModeBar": False})
    toss = data["toss_analysis"].copy()
    section_title("Toss decision context", "Toss winner outcome by decision; historical data only")
    toss_chart = px.bar(toss, x="toss_decision", y="win_percentage", color="toss_decision", text="win_percentage", color_discrete_map={"field": "#2584ff", "bat": "#ffc857"}, labels={"toss_decision": "Toss choice", "win_percentage": "Toss winner win %"})
    toss_chart.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(chart_theme(toss_chart, 270), width="stretch", config={"displayModeBar": False})

    section_title("Franchise comparison", "Overall win rates across the available history")
    franchise_names = sorted(teams["team"].dropna().unique())
    compare_left, compare_right = st.columns(2)
    with compare_left:
        team_a = st.selectbox("FIRST FRANCHISE", franchise_names, index=franchise_names.index("Chennai Super Kings") if "Chennai Super Kings" in franchise_names else 0)
    with compare_right:
        team_b_options = [name for name in franchise_names if name != team_a]
        default_b = next((name for name in team_b_options if name == "Mumbai Indians"), team_b_options[0] if team_b_options else team_a)
        team_b = st.selectbox("SECOND FRANCHISE", team_b_options or franchise_names, index=(team_b_options or franchise_names).index(default_b))
    compared = teams[teams["team"].isin([team_a, team_b])].copy()
    a, b = st.columns(2)
    for column, name in ((a, team_a), (b, team_b)):
        team_row = compared[compared["team"] == name].iloc[0]
        with column:
            st.markdown(f'<div class="panel"><span class="panel-kicker">{name.upper()}</span></div>', unsafe_allow_html=True)
            cells = st.columns(3)
            cells[0].metric("Matches", f"{int(team_row['matches_played']):,}")
            cells[1].metric("Wins", f"{int(team_row['wins']):,}")
            cells[2].metric("Win rate", f"{float(team_row['win_percentage']):.1f}%")
    comparison_chart = px.bar(compared, x="team", y="win_percentage", color="team", text="win_percentage", color_discrete_sequence=["#2584ff", "#ffc857"], labels={"team": "", "win_percentage": "Win %"})
    comparison_chart.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    comparison_chart.update_layout(showlegend=False, yaxis_title="Win percentage")
    st.plotly_chart(chart_theme(comparison_chart, 280), width="stretch", config={"displayModeBar": False})


def render_chase_prediction() -> None:
    page_header("ML / CHASE MODEL", "Chase probability lab", "Estimate the chasing side’s win probability from an explicit live innings state. This is a model estimate, not a toss-time chase-versus-bat recommendation.")
    model = ChasePredictionTool()
    if not model.available:
        st.error("No local model artifact is available at the configured chase model path.")
        st.code("python -m ml.train_chase_model")
        return
    st.markdown('<div class="panel"><span class="panel-kicker">LIVE INNINGS STATE</span><p style="color:#8ea5bd">Enter the current chase situation. Values should describe one moment in the second innings.</p></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        runs_required = st.number_input("Runs required", min_value=0, max_value=500, value=25, step=1)
        balls_remaining = st.number_input("Legal balls remaining", min_value=0, max_value=120, value=12, step=1)
    with col2:
        wickets_remaining = st.number_input("Wickets remaining", min_value=0, max_value=10, value=5, step=1)
        current_run_rate = st.number_input("Current run rate", min_value=0.0, max_value=40.0, value=8.0, step=0.1)
    with col3:
        required_run_rate = st.number_input("Required run rate", min_value=0.0, max_value=100.0, value=12.5, step=0.1)
        st.caption("The model features were built for a 20-over innings.")
    if st.button("Estimate chase probability", type="primary", key="predict-chase"):
        features = {
            "runs_required": float(runs_required), "balls_remaining": float(balls_remaining),
            "wickets_remaining": float(wickets_remaining), "current_run_rate": float(current_run_rate),
            "required_run_rate": float(required_run_rate),
        }
        try:
            prediction = model.predict(features)
            st.session_state.chase_prediction = prediction
        except Exception as exc:
            st.error(f"Could not score this innings state: {exc}")
    prediction = st.session_state.get("chase_prediction")
    if prediction:
        probability = prediction["win_probability"]
        st.write("")
        left, right = st.columns([1, 1.25], gap="large")
        with left:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=probability * 100, number={"suffix": "%", "font": {"color": "#edf5ff", "size": 42}}, title={"text": "Chasing side win probability", "font": {"color": "#a9bed4", "size": 15}}, gauge={"axis": {"range": [0, 100], "tickcolor": "#8ea5bd"}, "bar": {"color": "#2584ff"}, "bgcolor": "#0a1929", "bordercolor": "#1b3550", "steps": [{"range": [0, 35], "color": "#172a3b"}, {"range": [35, 65], "color": "#1a3550"}, {"range": [65, 100], "color": "#174238"}], "threshold": {"line": {"color": "#ffc857", "width": 3}, "thickness": .8, "value": 50}}))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=300, margin=dict(l=18, r=18, t=50, b=15))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        with right:
            section_title("Model inputs", "The estimate uses exactly these features")
            st.dataframe(pd.DataFrame([{"Feature": name.replace("_", " ").title(), "Value": value} for name, value in prediction["features"].items()]), hide_index=True, width="stretch")
            st.caption("Model: local scikit-learn chase classifier · validation split grouped by match · historical training data. It does not account for batter/bowler identities, pitch conditions, weather, or new information unless encoded in the provided features.")


def render_response_details(details: dict) -> None:
    metrics = details.get("supporting_metrics", [])
    if metrics:
        st.markdown("**Evidence metrics**")
        columns = st.columns(min(4, len(metrics)))
        for index, metric in enumerate(metrics):
            with columns[index % len(columns)]:
                st.metric(metric["label"], metric["value"])
    if details.get("figure") is not None:
        st.plotly_chart(details["figure"], width="stretch")
    sources = details.get("data_sources", [])
    if sources:
        st.markdown("**Sources**  " + " ".join(f'<span class="source-tag">{source}</span>' for source in sources), unsafe_allow_html=True)
    with st.expander("Agent trace · tools, sources & caveats"):
        st.markdown(f"**Intent:** {details.get('interpretation') or 'Strategy question'}")
        used = details.get("tools_used", [])
        st.markdown("**Tools called:** " + (" → ".join(used) if used else "None"))
        for limitation in details.get("limitations", []):
            st.caption(f"Scope note: {limitation}")


def render_assistant() -> None:
    missing = [name for name in TABLES if not (GOLD_DIR / f"{name}.parquet").exists()]
    page_header("AI / STRATEGY ASSISTANT", "Your IPL strategy desk", "Ask a cricket question. The agent selects deterministic Gold analytics, contextual retrieval, or the chase model and shows the evidence behind its answer.")
    if missing:
        st.error("Assistant Gold tables are missing: " + ", ".join(missing))
        return
    try:
        agent = load_agent()
    except Exception as exc:
        st.error(f"The strategy assistant could not load: {exc}")
        return
    examples = ["Should we chase at Wankhede?", "Show Kohli's record against Bumrah", "Which team has the best win percentage?", "Which phase scores fastest?"]
    st.markdown('<div class="panel"><span class="panel-kicker">SUGGESTED QUESTIONS</span></div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for col, prompt in zip(cols, examples):
        with col:
            if st.button(prompt, key=f"suggest-{prompt}", width="stretch"):
                st.session_state.pending_prompt = prompt
    if "assistant_messages" not in st.session_state:
        st.session_state.assistant_messages = []
    for item in st.session_state.assistant_messages:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item["role"] == "assistant" and item.get("details"):
                render_response_details(item["details"])
    prompt = st.chat_input("Ask about player form, matchups, venues or tactics…")
    if st.session_state.get("pending_prompt"):
        prompt = st.session_state.pop("pending_prompt")
    if prompt:
        st.session_state.assistant_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Selecting tools and gathering evidence…"):
                response = agent.answer(prompt)
            st.markdown(response.answer)
            details = response.as_dict()
            details["figure"] = response.figure
            render_response_details(details)
        st.session_state.assistant_messages.append({"role": "assistant", "content": response.answer, "details": details})
    if st.session_state.assistant_messages and st.button("Clear conversation", key="clear-chat"):
        st.session_state.assistant_messages = []
        st.rerun()


def render_data_model() -> None:
    page_header("SYSTEM / DATA & MODEL", "Data and model health", "See which local analytics snapshots and optional AI components are available on this machine.")
    rows = []
    for name in TABLES:
        path = GOLD_DIR / f"{name}.parquet"
        if path.exists():
            try:
                frame = load_gold(name)
                status, count = "Ready", f"{len(frame):,} rows"
            except Exception as exc:
                status, count = "Unreadable", str(exc)
        else:
            status, count = "Missing", "Export Gold snapshot to data/gold"
        rows.append({"Gold table": name, "Status": status, "Rows / note": count})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    model = ChasePredictionTool()
    ollama_model = "not configured"
    qdrant_status = "configured in local settings"
    try:
        from rag.settings import OLLAMA_CHAT_MODEL, OLLAMA_EMBED_MODEL, QDRANT_BACKEND

        ollama_model = f"{OLLAMA_CHAT_MODEL} · embeddings {OLLAMA_EMBED_MODEL}"
        qdrant_status = QDRANT_BACKEND
    except Exception:
        pass
    a, b, c = st.columns(3)
    a.metric("Chase model", "Ready" if model.available else "Not trained")
    b.metric("Ollama models", ollama_model)
    c.metric("Qdrant mode", qdrant_status)
    if model.available:
        st.caption(f"Model artifact is local at `{model.model_path.name}` and excluded from Git. Features: {', '.join(model.required_features())}.")
    with st.expander("Name cleanup and source scope"):
        st.markdown("**Franchises:** historical labels and spelling variants are consolidated for team comparisons.")
        st.dataframe(pd.DataFrame([{"Canonical franchise": canonical, "Source label": variant} for canonical, variants in TEAM_NAME_VARIANTS.items() for variant in variants]), hide_index=True, width="stretch")
        st.markdown("**Venues:** supported venue aliases are consolidated when querying. Source-level distinctions that cannot be verified remain separate.")
        st.dataframe(pd.DataFrame([{"Canonical venue": canonical, "Source label": variant} for canonical, variants in VENUE_NAME_VARIANTS.items() for variant in variants]), hide_index=True, width="stretch")


def main() -> None:
    with st.sidebar:
        st.markdown('<div class="brand"><span class="brand-mark">🏏</span> IPL INSIGHT</div>', unsafe_allow_html=True)
        st.caption("Strategy & performance intelligence")
        st.markdown('<div class="section-label">WORKSPACE</div>', unsafe_allow_html=True)
        choices = list(PAGE_OPTIONS.values())
        if st.session_state.get("page") not in choices:
            st.session_state.page = "Overview"
        page = st.radio("Navigate", choices, label_visibility="collapsed", key="page")
        st.markdown("---")
        st.markdown('<div class="section-label">LOCAL SERVICES</div>', unsafe_allow_html=True)
        st.caption("Gold snapshots · Ollama · Qdrant")
        st.button("Data & model status", width="stretch", key="sidebar-data-status", on_click=navigate_to, args=("Data & model",))
        st.markdown("---")
        st.caption("Historical IPL evidence is decision support, not a guarantee of match outcomes.")

    if page == "Overview":
        render_overview()
    elif page == "Player matchup":
        render_player_matchup()
    elif page == "Venue & teams":
        render_venue_teams()
    elif page == "Chase prediction":
        render_chase_prediction()
    elif page == "Strategy assistant":
        render_assistant()
    else:
        render_data_model()


main()
