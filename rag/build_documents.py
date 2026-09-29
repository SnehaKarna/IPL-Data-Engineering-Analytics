"""Build small, entity-focused text records from the local Gold snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from agentic.team_names import normalize_team_performance
from agentic.venue_names import normalize_venue_strategy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLD_PATH = PROJECT_ROOT / "data" / "gold"
REQUIRED_TABLES = (
    "player_matchups",
    "player_form",
    "team_performance",
    "toss_analysis",
    "venue_strategy",
    "phase_analysis",
)


def load_gold_tables(gold_path: Path = GOLD_PATH) -> dict[str, pd.DataFrame]:
    """Load the six local Gold snapshots used by analytics and retrieval."""
    path = Path(gold_path)
    missing = [name for name in REQUIRED_TABLES if not (path / f"{name}.parquet").is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing Gold Parquet file(s) under {path}: {', '.join(missing)}. "
            "Copy/export the latest Gold snapshots into data/gold/."
        )
    return {name: pd.read_parquet(path / f"{name}.parquet") for name in REQUIRED_TABLES}


def _value(row: pd.Series, column: str, digits: int = 2) -> str:
    value: Any = row[column]
    if pd.isna(value):
        return "unknown"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _document(source: str, key: str, text: str, **metadata: Any) -> dict[str, Any]:
    return {
        "source": source,
        "key": key,
        "text": text,
        "metadata": {name: str(value) for name, value in metadata.items() if pd.notna(value)},
    }


def build_documents(tables: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """Turn Gold rows into searchable text with stable source/entity metadata."""
    missing = sorted(set(REQUIRED_TABLES) - set(tables))
    if missing:
        raise ValueError(f"Cannot build RAG documents; missing tables: {', '.join(missing)}")

    # Pair-level matchups have over 31K rows in this snapshot. Exact matchup
    # questions use the deterministic player_matchups tool, so embedding every
    # pair would add a large, duplicated RAG index without helping retrieval.
    docs: list[dict[str, Any]] = []

    for _, row in tables["player_form"].iterrows():
        player = str(row["batter"])
        docs.append(_document(
            "player_form", player,
            f"IPL batting form: {player} played {_value(row, 'matches_played', 0)} matches and scored "
            f"{_value(row, 'total_runs', 0)} runs, averaging {_value(row, 'avg_runs_per_match')} per match. "
            f"Fours: {_value(row, 'fours', 0)}; sixes: {_value(row, 'sixes', 0)}.",
            batter=player,
        ))

    teams = normalize_team_performance(tables["team_performance"])
    for _, row in teams.iterrows():
        team = str(row["team"])
        docs.append(_document(
            "team_performance", team,
            f"IPL team record: {team} played {_value(row, 'matches_played', 0)} matches and won "
            f"{_value(row, 'wins', 0)}, for a win percentage of {_value(row, 'win_percentage')}%.",
            team=team,
        ))

    venues = normalize_venue_strategy(tables["venue_strategy"])
    for _, row in venues.iterrows():
        venue = str(row["venue"])
        docs.append(_document(
            "venue_strategy", venue,
            f"IPL venue record for {venue}: {_value(row, 'matches', 0)} matches; average first-innings "
            f"score {_value(row, 'avg_first_innings_score')}; average second-innings score "
            f"{_value(row, 'avg_second_innings_score')}; chasing teams won "
            f"{_value(row, 'chasing_win_percentage')}% of matches.",
            venue=venue,
        ))

    for _, row in tables["toss_analysis"].iterrows():
        decision = str(row["toss_decision"])
        docs.append(_document(
            "toss_analysis", decision,
            f"IPL toss record: after winning the toss, captains chose to {decision} in "
            f"{_value(row, 'matches', 0)} matches. The toss winner won "
            f"{_value(row, 'toss_winner_wins', 0)} of those ({_value(row, 'win_percentage')}%).",
            toss_decision=decision,
        ))

    for _, row in tables["phase_analysis"].iterrows():
        phase = str(row["phase"])
        docs.append(_document(
            "phase_analysis", phase,
            f"IPL innings phase {phase}: {_value(row, 'runs', 0)} runs from "
            f"{_value(row, 'deliveries', 0)} deliveries, {_value(row, 'fours', 0)} fours, "
            f"{_value(row, 'sixes', 0)} sixes, and {_value(row, 'wickets', 0)} wickets.",
            phase=phase,
        ))
    return docs


if __name__ == "__main__":
    datasets = load_gold_tables()
    documents = build_documents(datasets)
    print(f"Loaded {len(datasets)} Gold tables and built {len(documents)} documents.")
    print(documents[0]["text"])
