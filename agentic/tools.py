"""Deterministic tool implementations used by the single strategy agent."""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any

import pandas as pd

from agentic.analytics import GoldDataQueryTool
from agentic.team_names import canonical_team_name
from agentic.venue_names import canonical_venue_name
from ml.chase_prediction import ChasePredictionTool


TABLES = (
    "player_matchups",
    "player_form",
    "team_performance",
    "toss_analysis",
    "venue_strategy",
    "phase_analysis",
)

TEAM_SHORT_NAMES = {
    "csk": "Chennai Super Kings",
    "mi": "Mumbai Indians",
    "rcb": "Royal Challengers Bengaluru",
    "kkr": "Kolkata Knight Riders",
    "srh": "Sunrisers Hyderabad",
    "dc": "Delhi Capitals",
    "rr": "Rajasthan Royals",
    "pbks": "Punjab Kings",
}


def _serializable(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, Decimal):
        return float(value)
    if hasattr(value, "item"):
        return value.item()
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{key: _serializable(value) for key, value in row.items()} for row in frame.to_dict("records")]


def _contains_name(text: str, name: str) -> bool:
    def clean(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
    return f" {clean(name)} " in f" {clean(text)} "


class GoldAnalyticsTool:
    """Query local Gold tables and return source-labelled evidence, never LLM math."""

    def __init__(self, query_tool: GoldDataQueryTool | None = None) -> None:
        self.query_tool = query_tool or GoldDataQueryTool()

    def run(self, question: str, requested_tables: list[str]) -> tuple[list[dict], list[str]]:
        evidence: list[dict] = []
        limitations: list[str] = []
        for table in dict.fromkeys(name for name in requested_tables if name in TABLES):
            method = getattr(self, f"_query_{table}")
            result, notes = method(question)
            limitations.extend(notes)
            if result:
                evidence.extend(result)
        return evidence, limitations

    def _query_player_matchups(self, question: str) -> tuple[list[dict], list[str]]:
        sides = re.split(r"\b(?:against|versus|vs\.?)\b", question, maxsplit=1, flags=re.I)
        if len(sides) != 2:
            return [], ["A batter-versus-bowler query needs one batter and one bowler, for example 'Kohli against Bumrah'."]
        batter = self._resolve_batter(sides[0])
        bowler = self._resolve_bowler(sides[1])
        if not batter or not bowler:
            return [], ["Could not uniquely resolve both players in the matchup question."]
        rows = self.query_tool.player_matchup(batter, bowler).data
        if rows.empty:
            return [], [f"No player_matchups row was found for {batter} against {bowler}."]
        return [{"table": "player_matchups", "rows": _records(rows)}], []

    def _query_player_form(self, question: str) -> tuple[list[dict], list[str]]:
        names = self._mentioned_names(question, self.query_tool.tables["player_form"]["batter"].dropna().astype(str).unique())
        for phrase, canonical in (("virat kohli", "V Kohli"), ("rohit sharma", "RG Sharma"), ("shikhar dhawan", "S Dhawan")):
            if phrase in question.casefold() and canonical not in names:
                names.append(canonical)
        if not names:
            frame = self.query_tool.tables["player_form"].nlargest(5, "total_runs")
        else:
            frame = self.query_tool.tables["player_form"][self.query_tool.tables["player_form"]["batter"].isin(names)]
        if frame.empty:
            return [], ["No matching player_form row was found."]
        return [{"table": "player_form", "rows": _records(frame)}], []

    def _query_team_performance(self, question: str) -> tuple[list[dict], list[str]]:
        frame = self.query_tool.tables["team_performance"]
        team = self._resolve_team(question)
        selected = frame[frame["team"] == team] if team else frame.nlargest(5, "win_percentage")
        if selected.empty:
            return [], ["No matching team_performance row was found."]
        notes = []
        if team and any(term in question.casefold() for term in ("strength", "strong", "weakness", "best at")):
            notes.append(
                "Gold team_performance contains win/loss outcomes only; it does not break team strengths down by batting, bowling, phase, or current roster."
            )
        return [{
            "table": "team_performance",
            "scope": "Franchise aggregate across all available matches, not restricted to a particular venue or batting/chasing role.",
            "rows": _records(selected),
        }], notes

    def _query_venue_strategy(self, question: str) -> tuple[list[dict], list[str]]:
        venues = self.query_tool.tables["venue_strategy"]["venue"].dropna().astype(str).unique()
        venue = self._resolve_named_location(question, venues)
        frame = self.query_tool.tables["venue_strategy"]
        selected = frame[frame["venue"] == venue] if venue else frame.nlargest(5, "chasing_win_percentage")
        if selected.empty:
            return [], ["No matching venue_strategy row was found."]
        return [{
            "table": "venue_strategy",
            "scope": "Aggregate across all teams at the venue; not specific to a named franchise or head-to-head matchup.",
            "rows": _records(selected),
        }], ([] if venue else ["No venue was uniquely resolved; showing the five highest historical chase-win percentages."])

    def _query_toss_analysis(self, question: str) -> tuple[list[dict], list[str]]:
        return [{"table": "toss_analysis", "rows": _records(self.query_tool.tables["toss_analysis"])}], []

    def _query_phase_analysis(self, question: str) -> tuple[list[dict], list[str]]:
        frame = self.query_tool.tables["phase_analysis"].copy()
        if "deliveries" in frame and "runs" in frame:
            frame["runs_per_six_deliveries"] = (frame["runs"] / frame["deliveries"] * 6).round(2)
        limitations = []
        if any(term in question.casefold() for term in ("bowlers", "bowler", "effective")):
            limitations.append(
                "phase_analysis is aggregated by innings phase and does not contain bowler-level death-over economy or wicket rates."
            )
        return [{"table": "phase_analysis", "rows": _records(frame)}], limitations

    def _resolve_batter(self, text: str) -> str | None:
        aliases = {"virat kohli": "V Kohli", "kohli": "V Kohli", "rohit sharma": "RG Sharma"}
        lower = text.casefold()
        for alias, name in aliases.items():
            if alias in lower:
                return name
        names = self.query_tool.tables["player_form"]["batter"].dropna().astype(str).unique()
        matches = self._mentioned_names(text, names)
        if len(matches) == 1:
            return matches[0]
        if not matches:
            tokens = re.findall(r"[a-z]+", lower)
            by_surname = [name for name in names if name.casefold().split()[-1] in tokens]
            if len(by_surname) == 1:
                return by_surname[0]
        return None

    def _resolve_bowler(self, text: str) -> str | None:
        lower = text.casefold()
        aliases = {"jasprit bumrah": "JJ Bumrah", "bumrah": "JJ Bumrah"}
        for alias, name in aliases.items():
            if alias in lower:
                return name
        names = self.query_tool.tables["player_matchups"]["bowler"].dropna().astype(str).unique()
        matches = self._mentioned_names(text, names)
        if len(matches) == 1:
            return matches[0]
        if not matches:
            tokens = re.findall(r"[a-z]+", lower)
            by_surname = [name for name in names if name.casefold().split()[-1] in tokens]
            if len(by_surname) == 1:
                return by_surname[0]
        return None

    def _resolve_team(self, question: str) -> str | None:
        frame = self.query_tool.tables["team_performance"]
        names = frame["team"].dropna().astype(str).unique()
        candidates = self._mentioned_names(question, names)
        lower = question.casefold()
        for short, full in TEAM_SHORT_NAMES.items():
            if re.search(rf"\b{short}\b", lower):
                canonical = canonical_team_name(full)
                if canonical in names and canonical not in candidates:
                    candidates.append(canonical)
        return candidates[0] if len(candidates) == 1 else None

    def _resolve_named_location(self, question: str, names) -> str | None:
        candidates = self._mentioned_names(question, names)
        if len(candidates) == 1:
            return candidates[0]
        lower = question.casefold()
        partial = [name for name in names if name.casefold() in lower]
        if len(partial) == 1:
            return canonical_venue_name(partial[0])
        for token in re.findall(r"[a-z]+", lower):
            partial = [name for name in names if token in name.casefold().split() and len(token) > 3]
            if len(partial) == 1:
                return canonical_venue_name(partial[0])
        return None

    @staticmethod
    def _mentioned_names(text: str, names) -> list[str]:
        return [name for name in names if _contains_name(text, str(name))]


class PredictionTool:
    """Call the existing model only when it and all named input features exist."""

    def __init__(self, model: ChasePredictionTool | None = None) -> None:
        self.model = model or ChasePredictionTool()

    def run(self, question: str) -> tuple[dict | None, str | None]:
        if not self.model.available:
            return None, (
                "Chase prediction was requested, but no fitted model artifact is present in this repo. "
                "Add the existing scikit-learn model at models/chase_model.joblib or set CHASE_MODEL_PATH."
            )
        try:
            required = self.model.required_features()
        except Exception as exc:
            return None, str(exc)
        supplied = self._key_value_features(question, required)
        missing = [name for name in required if name not in supplied]
        if missing:
            return None, (
                "Chase prediction needs explicit values for model features: " + ", ".join(missing)
                + ". Add them as feature=value in your question."
            )
        try:
            return self.model.predict(supplied), None
        except Exception as exc:
            return None, f"Chase model could not score the supplied inputs: {exc}"

    @staticmethod
    def _key_value_features(question: str, required_features: list[str] | None = None) -> dict[str, float]:
        values = {}
        number = r"(-?\d+(?:\.\d+)?)"
        if required_features:
            for feature in required_features:
                name_pattern = re.escape(feature).replace("_", r"[\s_]+")
                match = re.search(rf"(?<![a-zA-Z0-9]){name_pattern}\s*[:=]\s*{number}", question, re.I)
                if match:
                    values[feature] = float(match.group(1))
            return values
        for key, raw in re.findall(r"([a-zA-Z][a-zA-Z0-9_ ]*?)\s*[:=]\s*" + number, question):
            normalized = re.sub(r"\s+", "_", key.strip().casefold())
            values[normalized] = float(raw)
        return values
