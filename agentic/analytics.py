from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from agentic.team_names import (
    TEAM_NAME_ALIASES,
    TEAM_NAME_VARIANTS,
    canonical_team_name,
    normalize_team_performance,
)
from agentic.venue_names import canonical_venue_name, normalize_venue_strategy


GOLD_PATH = Path(__file__).resolve().parents[1] / "data" / "gold"
TABLE_NAMES = (
    "player_matchups",
    "player_form",
    "venue_strategy",
    "team_performance",
    "toss_analysis",
    "phase_analysis",
)

@dataclass
class QueryResult:
    """Structured result returned by the data query tool."""

    data: pd.DataFrame
    sources: list[str]
    description: str


class GoldDataQueryTool:
    """Read-only query tool over the checked-in Gold Parquet datasets."""

    def __init__(self, gold_path: Path = GOLD_PATH) -> None:
        self.gold_path = Path(gold_path)
        self.tables = {
            name: pd.read_parquet(self.gold_path / f"{name}.parquet")
            for name in TABLE_NAMES
        }
        self.tables["team_performance"] = normalize_team_performance(self.tables["team_performance"])
        self.tables["venue_strategy"] = normalize_venue_strategy(self.tables["venue_strategy"])

    def search_player(self, name: str) -> str | None:
        normalized = name.casefold().strip()
        matches = self.tables["player_form"]["batter"].astype(str)
        exact = matches[matches.str.casefold() == normalized]
        if not exact.empty:
            return str(exact.iloc[0])
        partial = matches[matches.str.casefold().str.contains(normalized, regex=False)]
        return str(partial.iloc[0]) if len(partial) == 1 else None

    def search_bowler(self, name: str) -> str | None:
        """Resolve a bowler label from the matchup table without guessing on ambiguity."""
        normalized = name.casefold().strip()
        bowlers = self.tables["player_matchups"]["bowler"].dropna().astype(str).drop_duplicates()
        exact = bowlers[bowlers.str.casefold() == normalized]
        if not exact.empty:
            return str(exact.iloc[0])
        partial = bowlers[bowlers.str.casefold().str.contains(normalized, regex=False)]
        if len(partial) == 1:
            return str(partial.iloc[0])
        # Common queries use a surname while source labels often use initials.
        surname_matches = bowlers[bowlers.str.casefold().str.split().str[-1] == normalized]
        return str(surname_matches.iloc[0]) if len(surname_matches) == 1 else None

    def search_team(self, name: str) -> str | None:
        normalized = canonical_team_name(name).casefold()
        teams = self.tables["team_performance"]["team"].astype(str)
        exact = teams[teams.str.casefold() == normalized]
        if not exact.empty:
            return str(exact.iloc[0])
        partial = teams[teams.str.casefold().str.contains(normalized, regex=False)]
        return str(partial.iloc[0]) if len(partial) == 1 else None

    def search_venue(self, name: str) -> str | None:
        normalized = canonical_venue_name(name).casefold()
        venues = self.tables["venue_strategy"]["venue"].astype(str)
        exact = venues[venues.str.casefold() == normalized]
        if not exact.empty:
            return str(exact.iloc[0])
        partial = venues[venues.str.casefold().str.contains(normalized, regex=False)]
        return str(partial.iloc[0]) if len(partial) == 1 else None

    def player_form(self, players: list[str]) -> QueryResult:
        frame = self.tables["player_form"]
        result = frame[frame["batter"].isin(players)].copy()
        return QueryResult(result, ["player_form"], "player-level batting form")

    def player_matchup(self, batter: str, bowler: str | None = None) -> QueryResult:
        frame = self.tables["player_matchups"]
        result = frame[frame["batter"] == batter].copy()
        if bowler:
            result = result[result["bowler"] == bowler]
        return QueryResult(result, ["player_matchups"], "batter matchup records")

    def phase_summary(self) -> QueryResult:
        return QueryResult(
            self.tables["phase_analysis"].copy(),
            ["phase_analysis"],
            "IPL batting phase aggregates",
        )

    def venue_strategy(self, venue: str) -> QueryResult:
        frame = self.tables["venue_strategy"]
        result = frame[frame["venue"] == canonical_venue_name(venue)].copy()
        return QueryResult(result, ["venue_strategy"], "venue strategy aggregates")

    def team_performance(self, team: str | None = None) -> QueryResult:
        frame = self.tables["team_performance"]
        canonical = canonical_team_name(team) if team is not None else None
        result = frame if canonical is None else frame[frame["team"] == canonical]
        return QueryResult(result.copy(), ["team_performance"], "team win aggregates")

    def toss_analysis(self) -> QueryResult:
        return QueryResult(
            self.tables["toss_analysis"].copy(),
            ["toss_analysis"],
            "toss decision aggregates",
        )


class StatisticsTool:
    """Perform transparent calculations on query results."""

    @staticmethod
    def compare_players(frame: pd.DataFrame) -> list[dict[str, Any]]:
        metrics = ["total_runs", "avg_runs_per_match", "fours", "sixes"]
        return [
            {"label": column.replace("_", " ").title(), "values": {
                str(row["batter"]): _number(row[column]) for _, row in frame.iterrows()
            }}
            for column in metrics
            if column in frame.columns
        ]

    @staticmethod
    def phase_rates(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        result["run_rate"] = (result["runs"] / result["deliveries"] * 6).round(2)
        result["boundary_rate"] = (
            (result["fours"] + result["sixes"]) / result["deliveries"] * 100
        ).round(2)
        return result

    @staticmethod
    def best_chasing_venue(frame: pd.DataFrame) -> dict[str, Any] | None:
        if frame.empty:
            return None
        row = frame.sort_values("chasing_win_percentage", ascending=False).iloc[0]
        return {key: _number(value) for key, value in row.to_dict().items()}

    @staticmethod
    def top_team(frame: pd.DataFrame) -> dict[str, Any] | None:
        if frame.empty:
            return None
        row = frame.sort_values("win_percentage", ascending=False).iloc[0]
        return {key: _number(value) for key, value in row.to_dict().items()}


class VisualizationTool:
    """Create Plotly figures from the same results used in the answer."""

    @staticmethod
    def player_comparison(frame: pd.DataFrame):
        chart = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Career runs", "Average runs per match"),
            horizontal_spacing=0.18,
        )
        colors = ["#137a52", "#d77b35", "#426f99", "#9b6bb3"]
        for index, (_, row) in enumerate(frame.iterrows()):
            color = colors[index % len(colors)]
            chart.add_trace(
                go.Bar(name=str(row["batter"]), x=[str(row["batter"])], y=[row["total_runs"]], marker_color=color, showlegend=False),
                row=1,
                col=1,
            )
            chart.add_trace(
                go.Bar(name=str(row["batter"]), x=[str(row["batter"])], y=[row["avg_runs_per_match"]], marker_color=color, showlegend=False),
                row=1,
                col=2,
            )
        chart.update_layout(title="Player batting comparison", margin=dict(l=10, r=10, t=60, b=10), height=360)
        chart.update_xaxes(title_text="Player")
        chart.update_yaxes(title_text="Runs", row=1, col=1)
        chart.update_yaxes(title_text="Runs per match", row=1, col=2)
        return chart

    @staticmethod
    def phase_performance(frame: pd.DataFrame):
        return px.bar(
            frame,
            x="phase",
            y="run_rate",
            color="phase",
            title="Scoring rate by innings phase",
            labels={"phase": "Phase", "run_rate": "Runs per six legal balls"},
        )


def _number(value: Any) -> int | float | str:
    if pd.isna(value):
        return "n/a"
    converted = float(value) if hasattr(value, "as_tuple") else value
    if isinstance(converted, float) and converted.is_integer():
        return int(converted)
    return converted
