"""Train a local innings-2 chase model from the project's raw IPL CSV files.

This is a reproducible fallback when the original fitted artifact is unavailable.
It does not reproduce or claim the reported baseline ROC-AUC without its original
features, split, and data. Split is by match to prevent deliveries from one match
appearing in both train and validation sets.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv


load_dotenv()


FEATURES = ["runs_required", "balls_remaining", "wickets_remaining", "current_run_rate", "required_run_rate"]


def _download_if_s3(source: str, directory: Path) -> Path:
    if not source.startswith("s3://"):
        return Path(source)
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("S3 input needs boto3. Install project dependencies with `pip install -r requirements.txt`.") from exc
    bucket_key = source[5:].split("/", 1)
    if len(bucket_key) != 2 or not bucket_key[0] or not bucket_key[1]:
        raise ValueError(f"Expected an S3 object URI like s3://bucket/path/file.csv, got {source!r}")
    bucket, key = bucket_key
    destination = directory / Path(key).name
    try:
        boto3.client("s3").download_file(bucket, key, str(destination))
    except Exception as exc:
        raise RuntimeError(
            f"Could not download {source}. Check the active AWS identity and s3:GetObject permission. "
            "The bucket must also allow access from this AWS account."
        ) from exc
    return destination


def build_training_rows(matches_path: Path, deliveries_path: Path) -> pd.DataFrame:
    matches = pd.read_csv(matches_path)
    balls = pd.read_csv(deliveries_path)
    required_matches = {"match_number", "winner"}
    required_balls = {"ID", "Innings", "Overs", "BallNumber", "BattingTeam", "TotalRun", "IsWicketDelivery"}
    if missing := required_matches - set(matches.columns):
        raise ValueError(f"Match_Info.csv is missing columns: {', '.join(sorted(missing))}")
    if missing := required_balls - set(balls.columns):
        raise ValueError(f"Ball_By_Ball_Match_Data.csv is missing columns: {', '.join(sorted(missing))}")

    matches["match_number"] = matches["match_number"].astype(str)
    matches["winner"] = matches["winner"].fillna("").astype(str).str.strip()
    balls["ID"] = balls["ID"].astype(str)
    for col in ("Innings", "Overs", "BallNumber", "TotalRun", "IsWicketDelivery"):
        balls[col] = pd.to_numeric(balls[col], errors="coerce").fillna(0)
    balls["BattingTeam"] = balls["BattingTeam"].fillna("").astype(str).str.strip()
    match_targets = balls.loc[balls["Innings"] == 1].groupby("ID")["TotalRun"].sum().rename("target")
    chasing = balls.loc[balls["Innings"] == 2].merge(match_targets, left_on="ID", right_index=True, how="inner")
    chasing = chasing.merge(matches[["match_number", "winner"]], left_on="ID", right_on="match_number", how="inner")
    if "ExtraType" in chasing:
        extra = chasing["ExtraType"].fillna("").astype(str).str.casefold()
        legal = ~extra.str.contains(r"wide|no.?ball", regex=True)
    else:
        legal = pd.Series(True, index=chasing.index)
    chasing["legal"] = legal

    rows = []
    for match_id, frame in chasing.sort_values(["ID", "Overs", "BallNumber"]).groupby("ID", sort=False):
        frame = frame.reset_index(drop=True)
        team = frame["BattingTeam"].iloc[0]
        winners = frame["winner"].dropna().unique()
        if not team or len(winners) != 1 or not winners[0]:
            continue
        target = float(frame["target"].iloc[0])
        score = 0.0
        wickets = 0
        legal_balls = 0
        for row in frame.itertuples(index=False):
            score += float(row.TotalRun)
            wickets += int(row.IsWicketDelivery)
            legal_balls += int(row.legal)
            balls_remaining = max(0, 120 - legal_balls)
            runs_required = max(0.0, target + 1 - score)
            rows.append({
                "match_id": str(match_id),
                "runs_required": runs_required,
                "balls_remaining": balls_remaining,
                "wickets_remaining": max(0, 10 - wickets),
                "current_run_rate": score * 6 / legal_balls if legal_balls else 0.0,
                "required_run_rate": runs_required * 6 / balls_remaining if balls_remaining else 0.0,
                "won": int(winners[0] == team),
            })
    result = pd.DataFrame(rows)
    if result.empty or result["won"].nunique() < 2:
        raise ValueError("Not enough valid second-innings match outcomes to train a binary model.")
    return result


def train(matches_path: Path, deliveries_path: Path, output_path: Path) -> float:
    data = build_training_rows(matches_path, deliveries_path)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(data[FEATURES], data["won"], groups=data["match_id"]))
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    model.fit(data.iloc[train_idx][FEATURES], data.iloc[train_idx]["won"])
    probabilities = model.predict_proba(data.iloc[test_idx][FEATURES])[:, list(model.classes_).index(1)]
    auc = float(roc_auc_score(data.iloc[test_idx]["won"], probabilities))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return auc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", default="data/raw/Match_Info.csv", help="Local CSV path or s3://bucket/key")
    parser.add_argument("--deliveries", default="data/raw/Ball_By_Ball_Match_Data.csv", help="Local CSV path or s3://bucket/key")
    parser.add_argument("--output", type=Path, default=Path("models/chase_model.joblib"))
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="ipl-chase-") as temporary_dir:
        temp_path = Path(temporary_dir)
        matches = _download_if_s3(args.matches, temp_path)
        deliveries = _download_if_s3(args.deliveries, temp_path)
        auc = train(matches, deliveries, args.output)
    print(f"Saved match-split chase model to {args.output}; validation ROC-AUC={auc:.3f}")


if __name__ == "__main__":
    main()
