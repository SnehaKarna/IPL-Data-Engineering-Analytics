from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from agentic.analytics import GoldDataQueryTool, StatisticsTool, VisualizationTool


@dataclass
class AgentResponse:
    question: str
    interpretation: str
    tools_used: list[str] = field(default_factory=list)
    answer: str = ""
    supporting_metrics: list[dict[str, Any]] = field(default_factory=list)
    data_sources: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    figure: Any = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "interpretation": self.interpretation,
            "tools_used": self.tools_used,
            "answer": self.answer,
            "supporting_metrics": self.supporting_metrics,
            "data_sources": self.data_sources,
            "limitations": self.limitations,
        }


class IPLAnalyticsAgent:
    """Plan-act-observe coordinator for the available IPL Gold datasets."""

    def __init__(self, query_tool: GoldDataQueryTool | None = None) -> None:
        self.query_tool = query_tool or GoldDataQueryTool()
        self.statistics_tool = StatisticsTool()
        self.visualization_tool = VisualizationTool()

    def answer(self, question: str) -> AgentResponse:
        normalized = question.strip()
        response = AgentResponse(normalized, "")
        if not normalized:
            response.answer = "Please enter an IPL question so I can query the Gold datasets."
            response.limitations.append("No question was provided.")
            return response

        lower = normalized.casefold()
        if self._is_matchup_question(lower):
            return self._matchup_question(normalized)
        players = self._players_in_question(normalized)
        if self._is_unsupported(lower):
            response.interpretation = "The requested dimension is not present in the available Gold tables."
            response.answer = (
                "I cannot answer that reliably from the available Gold datasets. "
                "The current data does not contain season-by-season player records."
            )
            response.limitations.append(
                "Available Gold tables contain aggregate player form, matchups, venue, team, toss, and phase data."
            )
            return response
        if self._is_player_comparison(lower, players):
            return self._compare_players(normalized, players)
        if self._is_venue_chase(lower):
            return self._venue_chase(normalized)
        if self._is_phase_question(lower):
            return self._phase_question(normalized)
        if self._is_toss_question(lower):
            return self._toss_question(normalized)
        if self._is_team_question(lower):
            return self._team_question(normalized)
        if players:
            return self._player_form(normalized, players[0])

        response.interpretation = "The question did not map to a supported Gold dataset."
        response.answer = (
            "I can answer questions about player form, player matchups, team win percentage, "
            "venue chasing performance, toss decisions, and scoring phases."
        )
        response.limitations.append("Try naming a player, team, venue, toss decision, or innings phase.")
        return response

    def _compare_players(self, question: str, players: list[str]) -> AgentResponse:
        response = AgentResponse(question, "Compare aggregate batting-form metrics for the named players.")
        result = self.query_tool.player_form(players)
        response.tools_used.append("Data Query")
        if result.data.empty:
            response.answer = "I could not find those players in the player_form Gold table."
            response.limitations.append("Use the player names recorded in the dataset, such as V Kohli or RG Sharma.")
            return response
        response.tools_used.append("Statistics")
        response.supporting_metrics = _flatten_metrics(self.statistics_tool.compare_players(result.data))
        response.data_sources = result.sources
        response.answer = self._comparison_text(result.data)
        response.figure = self.visualization_tool.player_comparison(result.data)
        if "powerplay" in question.casefold():
            response.limitations.append(
                "The Gold layer does not contain player-level powerplay splits; this uses overall batting form."
            )
        return response

    def _player_form(self, question: str, player: str) -> AgentResponse:
        response = AgentResponse(question, f"Retrieve aggregate batting form for {player}.")
        result = self.query_tool.player_form([player])
        response.tools_used.append("Data Query")
        if result.data.empty:
            response.answer = f"I could not find {player} in the player_form Gold table."
            response.limitations.append("The player is not present in the available Gold data.")
            return response
        row = result.data.iloc[0]
        response.tools_used.append("Statistics")
        response.supporting_metrics = _row_metrics(row, [
            ("Runs", "total_runs"),
            ("Matches", "matches_played"),
            ("Average runs/match", "avg_runs_per_match"),
            ("Fours", "fours"),
            ("Sixes", "sixes"),
        ])
        response.data_sources = result.sources
        response.answer = (
            f"{player} scored {_fmt(row['total_runs'])} runs in {_fmt(row['matches_played'])} matches, "
            f"averaging {_fmt(row['avg_runs_per_match'])} runs per match."
        )
        return response

    def _matchup_question(self, question: str) -> AgentResponse:
        response = AgentResponse(
            question,
            "Retrieve the batter-versus-bowler record from the player_matchups Gold table.",
        )
        batter_aliases = {"virat kohli": "V Kohli", "kohli": "V Kohli"}
        bowler_aliases = {"jasprit bumrah": "JJ Bumrah", "bumrah": "JJ Bumrah"}
        lower = question.casefold()
        sides = re.split(r"\b(?:against|versus|vs\.?)\b", lower, maxsplit=1)
        batter_text = sides[0]
        bowler_text = sides[1] if len(sides) == 2 else ""
        batter_names = self.query_tool.tables["player_form"]["batter"].dropna().astype(str).unique()
        bowler_names = self.query_tool.tables["player_matchups"]["bowler"].dropna().astype(str).unique()

        batter_mentions = [
            name for name in batter_names
            if _contains_name(batter_text, name)
        ]
        bowler_mentions = [
            name for name in bowler_names
            if _contains_name(bowler_text, name)
        ]
        for phrase, canonical in batter_aliases.items():
            if phrase in batter_text:
                if canonical in batter_names and canonical not in batter_mentions:
                    batter_mentions.append(canonical)
        for phrase, canonical in bowler_aliases.items():
            if phrase in bowler_text:
                if canonical in bowler_names and canonical not in bowler_mentions:
                    bowler_mentions.append(canonical)

        # Resolve an unambiguous surname such as "Kohli" or "Bumrah" when the
        # source stores the player as an initial plus surname.
        for text, names, found in (
            (batter_text, batter_names, batter_mentions),
            (bowler_text, bowler_names, bowler_mentions),
        ):
            if found:
                continue
            for token in re.findall(r"[a-z]+", text):
                surname_matches = [name for name in names if name.casefold().split()[-1] == token]
                if len(surname_matches) == 1 and surname_matches[0] not in found:
                    found.append(surname_matches[0])

        response.tools_used.append("Gold matchup query")
        if len(batter_mentions) != 1 or len(bowler_mentions) != 1:
            response.answer = (
                "Please name one batter and one bowler. I can only resolve a matchup when each name "
                "uniquely identifies a player in the Gold data."
            )
            response.limitations.append("Player surnames that match multiple records are treated as ambiguous.")
            return response

        batter = batter_mentions[0]
        bowler = bowler_mentions[0]
        result = self.query_tool.player_matchup(batter, bowler)
        response.data_sources = result.sources
        if result.data.empty:
            response.answer = f"No recorded matchup row was found for {batter} against {bowler}."
            response.limitations.append("The matchup table may not include a row for this batter-bowler pair.")
            return response

        row = result.data.iloc[0]
        response.supporting_metrics = _row_metrics(row, [
            ("Matches", "matches"),
            ("Balls faced", "balls_faced"),
            ("Runs", "runs"),
            ("Strike rate", "strike_rate"),
            ("Dismissals", "dismissals"),
            ("Boundary runs", "boundary_runs"),
        ])
        response.answer = (
            f"In the available IPL matchup data, {batter} scored {_fmt(row['runs'])} runs from "
            f"{_fmt(row['balls_faced'])} balls against {bowler} (strike rate {_fmt(row['strike_rate'])}), "
            f"and was dismissed {_fmt(row['dismissals'])} times across {_fmt(row['matches'])} matches."
        )
        return response

    def _venue_chase(self, question: str) -> AgentResponse:
        venue_text = _after_keyword(question, ("at ", "in "))
        venue = self.query_tool.search_venue(venue_text) if venue_text else None
        response = AgentResponse(question, "Find the venue with the strongest chasing result, or inspect the named venue.")
        result = self.query_tool.venue_strategy(venue) if venue else self.query_tool.venue_strategy("")
        response.tools_used.append("Data Query")
        if venue is None:
            frame = self.query_tool.tables["venue_strategy"].copy()
            response.limitations.append("No venue was resolved; showing the best venue across the available venue table.")
        else:
            frame = result.data
        if frame.empty:
            response.answer = "I could not find that venue in the venue_strategy Gold table."
            response.limitations.append("Use a venue name recorded in the dataset, such as Wankhede Stadium.")
            return response
        best = self.statistics_tool.best_chasing_venue(frame)
        response.tools_used.append("Statistics")
        response.supporting_metrics = _row_metrics(best, [
            ("Venue", "venue"), ("Matches", "matches"),
            ("Chasing win %", "chasing_win_percentage"),
            ("Avg first innings", "avg_first_innings_score"),
            ("Avg second innings", "avg_second_innings_score"),
        ])
        response.data_sources = result.sources if venue else ["venue_strategy"]
        response.answer = (
            f"{best['venue']} has a chasing win percentage of {_fmt(best['chasing_win_percentage'])}% "
            f"across {_fmt(best['matches'])} matches."
        )
        return response

    def _team_question(self, question: str) -> AgentResponse:
        team_text = _after_keyword(question, ("for ", "of "))
        team = self.query_tool.search_team(team_text) if team_text else None
        result = self.query_tool.team_performance(team)
        response = AgentResponse(question, "Query team win aggregates and rank by win percentage.")
        response.tools_used.append("Data Query")
        if result.data.empty:
            response.answer = "I could not find that team in the team_performance Gold table."
            response.limitations.append("Use a team name recorded in the dataset.")
            return response
        best = self.statistics_tool.top_team(result.data)
        response.tools_used.append("Statistics")
        response.supporting_metrics = _row_metrics(best, [
            ("Team", "team"), ("Matches", "matches_played"),
            ("Wins", "wins"), ("Win percentage", "win_percentage"),
        ])
        response.data_sources = result.sources
        response.answer = (
            f"{best['team']} has the highest win percentage in the selected result: "
            f"{_fmt(best['win_percentage'])}% ({_fmt(best['wins'])} wins in {_fmt(best['matches_played'])} matches)."
        )
        return response

    def _phase_question(self, question: str) -> AgentResponse:
        response = AgentResponse(question, "Calculate scoring rate for each innings phase and visualize the comparison.")
        result = self.query_tool.phase_summary()
        response.tools_used.extend(["Data Query", "Statistics", "Visualization"])
        frame = self.statistics_tool.phase_rates(result.data)
        response.supporting_metrics = [
            {"label": f"{row['phase']} run rate", "value": _fmt(row["run_rate"])}
            for _, row in frame.iterrows()
        ]
        response.data_sources = result.sources
        response.figure = self.visualization_tool.phase_performance(frame)
        best = frame.sort_values("run_rate", ascending=False).iloc[0]
        response.answer = f"The {best['phase'].casefold()} phase had the highest scoring rate at {_fmt(best['run_rate'])} runs per six legal balls."
        return response

    def _toss_question(self, question: str) -> AgentResponse:
        response = AgentResponse(question, "Compare toss-winner win percentage by toss decision.")
        result = self.query_tool.toss_analysis()
        response.tools_used.extend(["Data Query", "Statistics"])
        response.supporting_metrics = [
            {"label": f"{row['toss_decision'].title()} win %", "value": _fmt(row["win_percentage"])}
            for _, row in result.data.iterrows()
        ]
        response.data_sources = result.sources
        best = result.data.sort_values("win_percentage", ascending=False).iloc[0]
        response.answer = (
            f"Choosing to {best['toss_decision']} produced the higher toss-winner win percentage: "
            f"{_fmt(best['win_percentage'])}% across {_fmt(best['matches'])} matches."
        )
        return response

    def _comparison_text(self, frame: pd.DataFrame) -> str:
        parts = [
            f"{row['batter']} scored {_fmt(row['total_runs'])} runs, averaging {_fmt(row['avg_runs_per_match'])} per match"
            for _, row in frame.iterrows()
        ]
        return "; ".join(parts) + "."

    def _players_in_question(self, question: str) -> list[str]:
        aliases = {"virat kohli": "V Kohli", "rohit sharma": "RG Sharma", "shikhar dhawan": "S Dhawan"}
        found = [name for alias, name in aliases.items() if alias in question.casefold()]
        for name in self.query_tool.tables["player_form"]["batter"].astype(str):
            if _contains_name(question, name) and name not in found:
                found.append(name)
        for alias, name in aliases.items():
            if alias in question.casefold() and name not in found:
                found.append(name)
        return found[:2]

    @staticmethod
    def _is_matchup_question(question: str) -> bool:
        return bool(re.search(r"\b(?:against|versus|vs\.?)\b", question))

    @staticmethod
    def _is_player_comparison(question: str, players: list[str]) -> bool:
        return len(players) >= 2 or ("compare" in question and "player" in question)

    @staticmethod
    def _is_venue_chase(question: str) -> bool:
        return "venue" in question or ("chasing" in question and " at " in question)

    @staticmethod
    def _is_team_question(question: str) -> bool:
        return "team" in question or "win percentage" in question or "performed best" in question

    @staticmethod
    def _is_phase_question(question: str) -> bool:
        return any(word in question for word in ("phase", "powerplay", "middle overs", "death overs", "run rate"))

    @staticmethod
    def _is_toss_question(question: str) -> bool:
        return "toss" in question

    @staticmethod
    def _is_unsupported(question: str) -> bool:
        return "season" in question or "year by year" in question or "over the years" in question


def _after_keyword(question: str, keywords: tuple[str, ...]) -> str:
    lower = question.casefold()
    for keyword in keywords:
        index = lower.find(keyword)
        if index >= 0:
            value = question[index + len(keyword):].strip(" ?.,")
            value = re.split(r"\b(?:while|with|and|has|performed|best)\b", value, flags=re.I)[0]
            return value.strip(" ?.,")
    return ""


def _row_metrics(row: Any, fields: list[tuple[str, str]]) -> list[dict[str, Any]]:
    return [{"label": label, "value": _fmt(row[column])} for label, column in fields if column in row]


def _flatten_metrics(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened = []
    for metric in metrics:
        for subject, value in metric["values"].items():
            flattened.append({"label": f"{subject} {metric['label']}", "value": _fmt(value)})
    return flattened


def _fmt(value: Any) -> str:
    if pd.isna(value):
        return "n/a"
    number = float(value) if hasattr(value, "as_tuple") else value
    if isinstance(number, float):
        return f"{number:.2f}" if not number.is_integer() else str(int(number))
    return str(number)


def _contains_name(question: str, name: str) -> bool:
    normalized_question = f" {re.sub(r'[^a-z0-9]+', ' ', question.casefold()).strip()} "
    normalized_name = f" {re.sub(r'[^a-z0-9]+', ' ', name.casefold()).strip()} "
    return normalized_name in normalized_question
