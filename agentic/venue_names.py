"""Canonical venue labels and weighted aggregation for IPL venue summaries."""

import pandas as pd


VENUE_NAME_ALIASES = {
    "feroz shah kotla": "Arun Jaitley Stadium",
    "arun jaitley stadium, delhi": "Arun Jaitley Stadium",
    "brabourne stadium, mumbai": "Brabourne Stadium",
    "dr dy patil sports academy, mumbai": "Dr DY Patil Sports Academy",
    "dr. y.s. rajasekhara reddy aca-vdca cricket stadium, visakhapatnam": "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium",
    "eden gardens, kolkata": "Eden Gardens",
    "himachal pradesh cricket association stadium, dharamsala": "Himachal Pradesh Cricket Association Stadium",
    "m.chinnaswamy stadium": "M Chinnaswamy Stadium",
    "m chinnaswamy stadium, bengaluru": "M Chinnaswamy Stadium",
    "ma chidambaram stadium, chepauk": "MA Chidambaram Stadium",
    "ma chidambaram stadium, chepauk, chennai": "MA Chidambaram Stadium",
    "maharashtra cricket association stadium, pune": "Maharashtra Cricket Association Stadium",
    "maharaja yadavindra singh international cricket stadium, new chandigarh": "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur",
    "narendra modi stadium, ahmedabad": "Narendra Modi Stadium",
    "sardar patel stadium, motera": "Narendra Modi Stadium",
    "punjab cricket association is bindra stadium, mohali": "Punjab Cricket Association IS Bindra Stadium",
    "punjab cricket association is bindra stadium, mohali, chandigarh": "Punjab Cricket Association IS Bindra Stadium",
    "punjab cricket association stadium, mohali": "Punjab Cricket Association IS Bindra Stadium",
    "rajiv gandhi international stadium, uppal": "Rajiv Gandhi International Stadium",
    "rajiv gandhi international stadium, uppal, hyderabad": "Rajiv Gandhi International Stadium",
    "sawai mansingh stadium, jaipur": "Sawai Mansingh Stadium",
    "shaheed veer narayan singh international stadium, raipur": "Shaheed Veer Narayan Singh International Stadium",
    "zayed cricket stadium, abu dhabi": "Sheikh Zayed Stadium",
    "wankhede stadium, mumbai": "Wankhede Stadium",
}

VENUE_NAME_VARIANTS = {
    "Arun Jaitley Stadium": ("Arun Jaitley Stadium, Delhi", "Feroz Shah Kotla"),
    "Brabourne Stadium": ("Brabourne Stadium, Mumbai",),
    "Dr DY Patil Sports Academy": ("Dr DY Patil Sports Academy, Mumbai",),
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium": (
        "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam",
    ),
    "Eden Gardens": ("Eden Gardens, Kolkata",),
    "Himachal Pradesh Cricket Association Stadium": (
        "Himachal Pradesh Cricket Association Stadium, Dharamsala",
    ),
    "M Chinnaswamy Stadium": (
        "M.Chinnaswamy Stadium",
        "M Chinnaswamy Stadium, Bengaluru",
    ),
    "MA Chidambaram Stadium": (
        "MA Chidambaram Stadium, Chepauk",
        "MA Chidambaram Stadium, Chepauk, Chennai",
    ),
    "Maharashtra Cricket Association Stadium": (
        "Maharashtra Cricket Association Stadium, Pune",
    ),
    "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur": (
        "Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh",
    ),
    "Narendra Modi Stadium": (
        "Narendra Modi Stadium, Ahmedabad",
        "Sardar Patel Stadium, Motera",
    ),
    "Punjab Cricket Association IS Bindra Stadium": (
        "Punjab Cricket Association IS Bindra Stadium, Mohali",
        "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh",
        "Punjab Cricket Association Stadium, Mohali",
    ),
    "Rajiv Gandhi International Stadium": (
        "Rajiv Gandhi International Stadium, Uppal",
        "Rajiv Gandhi International Stadium, Uppal, Hyderabad",
    ),
    "Sawai Mansingh Stadium": ("Sawai Mansingh Stadium, Jaipur",),
    "Shaheed Veer Narayan Singh International Stadium": (
        "Shaheed Veer Narayan Singh International Stadium, Raipur",
    ),
    "Sheikh Zayed Stadium": ("Zayed Cricket Stadium, Abu Dhabi",),
    "Wankhede Stadium": ("Wankhede Stadium, Mumbai",),
}


def canonical_venue_name(name: str) -> str:
    cleaned = str(name).strip()
    return VENUE_NAME_ALIASES.get(cleaned.casefold(), cleaned)


def normalize_venue_strategy(frame: pd.DataFrame) -> pd.DataFrame:
    """Merge venue aliases, weighting averages and win rates by match count."""
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    result["venue"] = result["venue"].map(canonical_venue_name)
    numeric = ["matches", "avg_first_innings_score", "avg_second_innings_score", "chasing_win_percentage"]
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["matches"] = result["matches"].fillna(0)
    result["first_weighted"] = result["avg_first_innings_score"] * result["matches"]
    result["second_weighted"] = result["avg_second_innings_score"] * result["matches"]
    result["chasing_wins_weighted"] = result["chasing_win_percentage"] * result["matches"]
    combined = result.groupby("venue", as_index=False).agg(
        matches=("matches", "sum"),
        first_weighted=("first_weighted", "sum"),
        second_weighted=("second_weighted", "sum"),
        chasing_wins_weighted=("chasing_wins_weighted", "sum"),
    )
    denominator = combined["matches"].replace(0, pd.NA)
    combined["avg_first_innings_score"] = (combined["first_weighted"] / denominator).round(2)
    combined["avg_second_innings_score"] = (combined["second_weighted"] / denominator).round(2)
    combined["chasing_win_percentage"] = (combined["chasing_wins_weighted"] / denominator).round(2)
    return combined.drop(columns=["first_weighted", "second_weighted", "chasing_wins_weighted"]).sort_values(
        "chasing_win_percentage", ascending=False
    ).reset_index(drop=True)
