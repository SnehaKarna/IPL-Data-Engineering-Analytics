"""Canonical IPL franchise names shared by the UI and analytics layer."""

import pandas as pd


TEAM_NAME_ALIASES = {
    "royal challengers bangalore": "Royal Challengers Bengaluru",
    "royal challengers bengaluru": "Royal Challengers Bengaluru",
    "delhi daredevils": "Delhi Capitals",
    "delhi capitals": "Delhi Capitals",
    "kings xi punjab": "Punjab Kings",
    "punjab kings": "Punjab Kings",
    "rising pune supergiants": "Rising Pune Supergiant",
    "rising pune supergiant": "Rising Pune Supergiant",
}

TEAM_NAME_VARIANTS = {
    "Royal Challengers Bengaluru": ("Royal Challengers Bangalore",),
    "Delhi Capitals": ("Delhi Daredevils",),
    "Punjab Kings": ("Kings XI Punjab",),
    "Rising Pune Supergiant": ("Rising Pune Supergiants",),
}


def canonical_team_name(name: str) -> str:
    cleaned = str(name).strip()
    return TEAM_NAME_ALIASES.get(cleaned.casefold(), cleaned)


def normalize_team_performance(frame: pd.DataFrame) -> pd.DataFrame:
    """Merge spelling and franchise-renaming variants into one team row."""
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    result["team"] = result["team"].map(canonical_team_name)
    result["matches_played"] = pd.to_numeric(result["matches_played"], errors="coerce").fillna(0)
    result["wins"] = pd.to_numeric(result["wins"], errors="coerce").fillna(0)
    result = result.groupby("team", as_index=False)[["matches_played", "wins"]].sum()
    result["win_percentage"] = (result["wins"] / result["matches_played"] * 100).round(2)
    return result.sort_values("win_percentage", ascending=False).reset_index(drop=True)
