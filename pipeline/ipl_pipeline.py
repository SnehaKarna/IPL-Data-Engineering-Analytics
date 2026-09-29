# Databricks notebook source
"""Incremental, fail-fast IPL ETL using Delta Lake tables in a UC Volume.

Run this notebook from a Databricks Repo that contains this project. The legacy
Parquet paths are intentionally left untouched; Delta outputs use *_delta paths.
"""

from datetime import datetime, timezone
import re

from delta.tables import DeltaTable
from pyspark.sql import functions as F

from pipeline.quality import (
    report_check,
    require_columns,
    require_no_orphans,
    require_non_null,
    require_player_references,
    require_row_count_change,
    require_schema,
    require_season_coverage,
    require_unique,
)


VOLUME_ROOT = "/Volumes/workspace/default/ipl_data"
RAW_ROOT = f"{VOLUME_ROOT}/Raw"
DELTA_ROOT = VOLUME_ROOT
SILVER_PATH = f"{DELTA_ROOT}/Silver/ball_by_ball_delta"
MANIFEST_PATH = f"{DELTA_ROOT}/Silver/match_manifest_delta"
RUN_METRICS_PATH = f"{DELTA_ROOT}/Silver/pipeline_run_metrics_delta"
GOLD_PATHS = {
    "top_batsmen": f"{DELTA_ROOT}/Gold/top_batsmen_delta",
    "top_bowlers": f"{DELTA_ROOT}/Gold/top_bowlers_delta",
    "player_matchups": f"{DELTA_ROOT}/Gold/player_matchups_delta",
    "player_form": f"{DELTA_ROOT}/Gold/player_form_delta",
    "team_performance": f"{DELTA_ROOT}/Gold/team_performance_delta",
    "toss_analysis": f"{DELTA_ROOT}/Gold/toss_analysis_delta",
    "venue_strategy": f"{DELTA_ROOT}/Gold/venue_strategy_delta",
    "phase_analysis": f"{DELTA_ROOT}/Gold/phase_analysis_delta",
}
MAX_DELIVERY_COUNT_CHANGE_FRACTION = 0.75
PLAYER_NAME_COLUMN_OVERRIDE = None  # Set to the exact roster name field if auto-detection is ambiguous.


def _delta_exists(spark, path):
    return DeltaTable.isDeltaTable(spark, path)


def _merge_snapshot(spark, frame, path, key):
    if not _delta_exists(spark, path):
        frame.write.format("delta").mode("overwrite").save(path)
        return
    condition = " AND ".join(f"target.`{column}` = source.`{column}`" for column in key)
    (DeltaTable.forPath(spark, path).alias("target")
     .merge(frame.alias("source"), condition)
     .whenMatchedUpdateAll()
     .whenNotMatchedInsertAll()
     .whenNotMatchedBySourceDelete()
     .execute())


def _player_name_column(players):
    if PLAYER_NAME_COLUMN_OVERRIDE:
        if PLAYER_NAME_COLUMN_OVERRIDE not in players.columns:
            raise AssertionError(
                f"Configured player name field {PLAYER_NAME_COLUMN_OVERRIDE!r} is missing; "
                f"available fields={players.columns}"
            )
        return PLAYER_NAME_COLUMN_OVERRIDE
    candidates = {re.sub(r"[^a-z0-9]", "", name.casefold()): name for name in players.columns}
    for name in ("player_name", "playername", "full_name", "fullname", "name", "player", "player_full_name"):
        key = re.sub(r"[^a-z0-9]", "", name.casefold())
        if key in candidates:
            return candidates[key]
    raise AssertionError(
        "Could not identify the player-name column in 2024_players_details.csv. "
        f"Set PLAYER_NAME_COLUMN_OVERRIDE. Available fields={players.columns}"
    )


def _prepare_sources(spark):
    teams = spark.read.option("header", True).option("inferSchema", True).csv(f"{RAW_ROOT}/teams_info.csv")
    players = spark.read.option("header", True).option("inferSchema", True).csv(f"{RAW_ROOT}/2024_players_details.csv")
    matches = spark.read.option("header", True).option("inferSchema", True).csv(f"{RAW_ROOT}/Match_Info.csv")
    balls = spark.read.option("header", True).option("inferSchema", True).csv(f"{RAW_ROOT}/Ball_By_Ball_Match_Data.csv")

    require_columns(matches, ["match_number", "match_date", "team1", "team2"], "raw matches")
    require_columns(balls, [
        "ID", "Innings", "Overs", "BallNumber", "Batter", "Bowler", "NonStriker",
        "BatsmanRun", "ExtrasRun", "TotalRun", "IsWicketDelivery", "BattingTeam",
    ], "raw deliveries")
    require_schema(players, {_player_name_column(players): "string"}, "player roster")

    matches = (matches
        .withColumn("match_number", F.col("match_number").cast("string"))
        .withColumn("match_date", F.to_date("match_date")))
    balls = (balls
        .withColumn("ID", F.col("ID").cast("string"))
        .withColumn("Innings", F.col("Innings").cast("int"))
        .withColumn("Overs", F.col("Overs").cast("int"))
        .withColumn("BallNumber", F.col("BallNumber").cast("int"))
        .withColumn("BatsmanRun", F.col("BatsmanRun").cast("int"))
        .withColumn("ExtrasRun", F.col("ExtrasRun").cast("int"))
        .withColumn("TotalRun", F.col("TotalRun").cast("int"))
        .withColumn("IsWicketDelivery", F.col("IsWicketDelivery").cast("int")))

    require_schema(matches, {"match_number": "string", "match_date": "date", "team1": "string", "team2": "string"}, "standardized matches")
    require_schema(balls, {"ID": "string", "Innings": "int", "Overs": "int", "BallNumber": "int", "BatsmanRun": "int", "ExtrasRun": "int", "TotalRun": "int", "IsWicketDelivery": "int"}, "standardized deliveries")
    require_non_null(matches, ["match_number", "match_date", "team1", "team2"], "matches")
    require_non_null(balls, ["ID", "Innings", "Overs", "BallNumber", "Batter", "Bowler", "NonStriker", "BatsmanRun", "ExtrasRun", "TotalRun", "IsWicketDelivery", "BattingTeam"], "deliveries")
    require_unique(matches, ["match_number"], "matches")
    require_unique(_with_delivery_key(balls), ["delivery_key"], "exact source delivery records")
    require_no_orphans(balls, matches, "ID", "match_number", "delivery match IDs")
    require_player_references(balls, players, _player_name_column(players))
    require_season_coverage(matches)
    print(f"[DQ INFO] source counts: players={players.count()}, matches={matches.count()}, deliveries={balls.count()}, teams={teams.count()}")
    return teams, players, matches, balls


def _with_delivery_key(balls):
    ordered_columns = sorted(balls.columns)
    payload = F.to_json(F.struct(*[F.col(f"`{name}`").alias(name) for name in ordered_columns]))
    return balls.withColumn("delivery_key", F.sha2(payload, 256))


def _silver_source(balls, matches):
    keyed = _with_delivery_key(balls)
    require_unique(keyed, ["delivery_key"], "exact source delivery records")
    b = keyed.alias("b")
    m = matches.alias("m")
    return (b.join(m, F.col("b.ID") == F.col("m.match_number"), "inner")
        .select(
            F.col("b.delivery_key"), F.col("b.ID").alias("match_id"), F.col("m.match_date"),
            F.col("m.team1"), F.col("m.team2"), F.col("m.toss_winner"), F.col("m.toss_decision"),
            F.col("m.winner"), F.col("m.player_of_match"), F.col("m.venue"), F.col("m.city"),
            F.col("b.Innings").alias("innings"), F.col("b.Overs").alias("over"),
            F.col("b.BallNumber").alias("ball_number"), F.col("b.Batter").alias("batter"),
            F.col("b.Bowler").alias("bowler"), F.col("b.NonStriker").alias("non_striker"),
            F.col("b.ExtraType").alias("extra_type"), F.col("b.BatsmanRun").alias("batsman_runs"),
            F.col("b.ExtrasRun").alias("extras_runs"), F.col("b.TotalRun").alias("total_runs"),
            F.col("b.IsWicketDelivery").alias("is_wicket"), F.col("b.PlayerOut").alias("player_out"),
            F.col("b.Kind").alias("dismissal_kind"), F.col("b.BattingTeam").alias("batting_team"),
        ))


def _source_manifest(balls, matches):
    keyed = _with_delivery_key(balls)
    deliveries = keyed.groupBy(F.col("ID").alias("match_id")).agg(
        F.count("*").alias("source_delivery_count"),
        F.sha2(F.concat_ws("|", F.sort_array(F.collect_list("delivery_key"))), 256).alias("delivery_fingerprint"),
    )
    metadata = matches.select(
        F.col("match_number").alias("match_id"),
        F.sha2(F.to_json(F.struct(*[F.col(name) for name in matches.columns])), 256).alias("metadata_fingerprint"),
    )
    return (metadata.join(deliveries, "match_id", "left")
        .fillna({"source_delivery_count": 0, "delivery_fingerprint": "NO_DELIVERIES"})
        .withColumn("source_fingerprint", F.sha2(F.concat_ws("|", "metadata_fingerprint", "delivery_fingerprint"), 256))
        .select("match_id", "source_fingerprint", "source_delivery_count"))


def _changed_matches(spark, current_manifest):
    if not _delta_exists(spark, MANIFEST_PATH):
        ids = current_manifest.select("match_id").distinct()
        return ids, current_manifest
    prior = spark.read.format("delta").load(MANIFEST_PATH)
    require_no_orphans(prior, current_manifest, "match_id", "match_id", "previously ingested matches still present in source")
    changed = (current_manifest.alias("new").join(prior.alias("old"), "match_id", "left")
        .filter(F.col("old.source_fingerprint").isNull() | (F.col("new.source_fingerprint") != F.col("old.source_fingerprint")))
        .select("new.match_id", "new.source_fingerprint", "new.source_delivery_count"))
    return changed.select("match_id"), changed


def _merge_silver(spark, changed_silver, changed_ids):
    if changed_ids.limit(1).count() == 0:
        print("[DQ INFO] Silver Delta is current; no matches changed.")
        return
    if not _delta_exists(spark, SILVER_PATH):
        changed_silver.write.format("delta").mode("overwrite").save(SILVER_PATH)
        return
    ids = [row[0] for row in changed_ids.distinct().collect()]
    literals = ",".join("'" + str(value).replace("'", "''") + "'" for value in ids)
    delete_scope = f"target.match_id IN ({literals})"
    (DeltaTable.forPath(spark, SILVER_PATH).alias("target")
     .merge(changed_silver.alias("source"), "target.delivery_key = source.delivery_key")
     .whenMatchedUpdateAll()
     .whenNotMatchedInsertAll()
     .whenNotMatchedBySourceDelete(condition=delete_scope)
     .execute())


def _build_gold(spark, silver):
    silver.createOrReplaceTempView("ipl_deliveries")
    batsmen = spark.sql("""
        SELECT batter, SUM(batsman_runs) AS total_runs,
               COUNT(DISTINCT match_id) AS matches_played,
               CASE WHEN COUNT(DISTINCT match_id) = 0 THEN 0
                    ELSE ROUND(SUM(batsman_runs) / COUNT(DISTINCT match_id), 2) END AS avg_runs_per_match,
               SUM(CASE WHEN batsman_runs = 4 THEN 1 ELSE 0 END) AS fours,
               SUM(CASE WHEN batsman_runs = 6 THEN 1 ELSE 0 END) AS sixes
        FROM ipl_deliveries GROUP BY batter
    """)
    bowlers = spark.sql("""
        SELECT bowler, COUNT(*) AS wickets
        FROM ipl_deliveries
        WHERE is_wicket = 1 AND dismissal_kind NOT IN ('run out', 'retired hurt', 'obstructing the field')
        GROUP BY bowler
    """)
    teams = spark.sql("""
        WITH participants AS (
            SELECT match_id, team1 AS team, winner FROM ipl_deliveries GROUP BY match_id, team1, winner
            UNION ALL
            SELECT match_id, team2 AS team, winner FROM ipl_deliveries GROUP BY match_id, team2, winner
        ), canonical AS (
            SELECT match_id,
                CASE LOWER(TRIM(team))
                    WHEN 'royal challengers bangalore' THEN 'Royal Challengers Bengaluru'
                    WHEN 'royal challengers bengaluru' THEN 'Royal Challengers Bengaluru'
                    WHEN 'delhi daredevils' THEN 'Delhi Capitals'
                    WHEN 'delhi capitals' THEN 'Delhi Capitals'
                    WHEN 'kings xi punjab' THEN 'Punjab Kings'
                    WHEN 'punjab kings' THEN 'Punjab Kings'
                    WHEN 'rising pune supergiants' THEN 'Rising Pune Supergiant'
                    WHEN 'rising pune supergiant' THEN 'Rising Pune Supergiant'
                    ELSE team END AS team,
                CASE LOWER(TRIM(winner))
                    WHEN 'royal challengers bangalore' THEN 'Royal Challengers Bengaluru'
                    WHEN 'royal challengers bengaluru' THEN 'Royal Challengers Bengaluru'
                    WHEN 'delhi daredevils' THEN 'Delhi Capitals'
                    WHEN 'delhi capitals' THEN 'Delhi Capitals'
                    WHEN 'kings xi punjab' THEN 'Punjab Kings'
                    WHEN 'punjab kings' THEN 'Punjab Kings'
                    WHEN 'rising pune supergiants' THEN 'Rising Pune Supergiant'
                    WHEN 'rising pune supergiant' THEN 'Rising Pune Supergiant'
                    ELSE winner END AS winner
            FROM participants
        )
        SELECT team, COUNT(*) AS matches_played,
               SUM(CASE WHEN winner = team THEN 1 ELSE 0 END) AS wins,
               ROUND(SUM(CASE WHEN winner = team THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS win_percentage
        FROM canonical GROUP BY team
    """)
    toss = spark.sql("""
        SELECT toss_decision, COUNT(*) AS matches,
               SUM(CASE WHEN toss_winner = winner THEN 1 ELSE 0 END) AS toss_winner_wins,
               ROUND(SUM(CASE WHEN toss_winner = winner THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS win_percentage
        FROM (SELECT DISTINCT match_id, toss_winner, toss_decision, winner FROM ipl_deliveries)
        WHERE toss_decision IS NOT NULL GROUP BY toss_decision
    """)

    player_matchups = spark.sql("""
        WITH matchup AS (
            SELECT batter, bowler, match_id, batsman_runs, extra_type, is_wicket,
                   player_out, dismissal_kind
            FROM ipl_deliveries
        )
        SELECT batter, bowler,
               COUNT(DISTINCT match_id) AS matches,
               SUM(CASE WHEN LOWER(COALESCE(extra_type, '')) NOT LIKE 'wide%'
                              AND LOWER(COALESCE(extra_type, '')) NOT LIKE 'no%ball%'
                        THEN 1 ELSE 0 END) AS balls_faced,
               SUM(batsman_runs) AS runs,
               CASE WHEN SUM(CASE WHEN LOWER(COALESCE(extra_type, '')) NOT LIKE 'wide%'
                                       AND LOWER(COALESCE(extra_type, '')) NOT LIKE 'no%ball%'
                                  THEN 1 ELSE 0 END) = 0 THEN 0
                    ELSE ROUND(SUM(batsman_runs) * 100.0 /
                               SUM(CASE WHEN LOWER(COALESCE(extra_type, '')) NOT LIKE 'wide%'
                                             AND LOWER(COALESCE(extra_type, '')) NOT LIKE 'no%ball%'
                                        THEN 1 ELSE 0 END), 2) END AS strike_rate,
               SUM(CASE WHEN is_wicket = 1 AND player_out = batter
                             AND LOWER(COALESCE(dismissal_kind, '')) NOT IN
                                 ('run out', 'retired hurt', 'obstructing the field')
                        THEN 1 ELSE 0 END) AS dismissals,
               SUM(CASE WHEN batsman_runs IN (4, 6) THEN batsman_runs ELSE 0 END) AS boundary_runs
        FROM matchup GROUP BY batter, bowler
    """)

    player_form = batsmen.select(
        "batter", "matches_played", "total_runs", "avg_runs_per_match", "fours", "sixes"
    )

    venue_strategy = spark.sql("""
        WITH innings_scores AS (
            SELECT match_id, venue, MAX(winner) AS winner,
                   SUM(CASE WHEN innings = 1 THEN total_runs END) AS first_score,
                   SUM(CASE WHEN innings = 2 THEN total_runs END) AS second_score,
                   MAX(CASE WHEN innings = 2 THEN batting_team END) AS chasing_team
            FROM ipl_deliveries
            WHERE venue IS NOT NULL
            GROUP BY match_id, venue
        ), match_outcomes AS (
            SELECT match_id, venue, first_score, second_score,
                   CASE WHEN winner = chasing_team THEN 1 ELSE 0 END AS chasing_win
            FROM innings_scores
            WHERE first_score IS NOT NULL AND second_score IS NOT NULL
        )
        SELECT venue, COUNT(*) AS matches,
               ROUND(AVG(first_score), 2) AS avg_first_innings_score,
               ROUND(AVG(second_score), 2) AS avg_second_innings_score,
               ROUND(SUM(chasing_win) * 100.0 / COUNT(*), 2) AS chasing_win_percentage
        FROM match_outcomes GROUP BY venue
    """)

    first_over = silver.agg(F.min("over").alias("first_over")).first()["first_over"]
    zero_based_overs = first_over == 0
    powerplay_last = 5 if zero_based_overs else 6
    middle_last = 14 if zero_based_overs else 15
    phased = silver.withColumn(
        "phase",
        F.when(F.col("over") <= powerplay_last, "Powerplay")
         .when(F.col("over") <= middle_last, "Middle")
         .otherwise("Death"),
    )
    phase_analysis = phased.groupBy("phase").agg(
        F.count("*").alias("deliveries"),
        F.sum("total_runs").alias("runs"),
        F.sum(F.when(F.col("batsman_runs") == 4, 1).otherwise(0)).alias("fours"),
        F.sum(F.when(F.col("batsman_runs") == 6, 1).otherwise(0)).alias("sixes"),
        F.sum(F.when(F.col("is_wicket") == 1, 1).otherwise(0)).alias("wickets"),
    )

    return {
        "top_batsmen": batsmen,
        "top_bowlers": bowlers,
        "player_matchups": player_matchups,
        "player_form": player_form,
        "team_performance": teams,
        "toss_analysis": toss,
        "venue_strategy": venue_strategy,
        "phase_analysis": phase_analysis,
    }


def run_pipeline(spark):
    print(f"Starting IPL Delta pipeline on Spark {spark.version}")
    teams_df, players_df, matches_df, ball_df = _prepare_sources(spark)
    source_delivery_count = ball_df.count()
    source_manifest = _source_manifest(ball_df, matches_df)
    changed_ids, changed_manifest = _changed_matches(spark, source_manifest)
    prior_silver_exists = _delta_exists(spark, SILVER_PATH)
    if not prior_silver_exists:
        changed_ids = source_manifest.select("match_id").distinct()
        changed_manifest = source_manifest

    previous_metrics = None
    if _delta_exists(spark, RUN_METRICS_PATH):
        previous_metrics = (spark.read.format("delta").load(RUN_METRICS_PATH)
            .orderBy(F.col("run_timestamp").desc())
            .select("delivery_count", "match_count", "player_count", "team_count").first())
    require_row_count_change(source_delivery_count, previous_metrics["delivery_count"] if previous_metrics else None, MAX_DELIVERY_COUNT_CHANGE_FRACTION)
    require_row_count_change(matches_df.count(), previous_metrics["match_count"] if previous_metrics else None, MAX_DELIVERY_COUNT_CHANGE_FRACTION)
    require_row_count_change(players_df.count(), previous_metrics["player_count"] if previous_metrics else None, MAX_DELIVERY_COUNT_CHANGE_FRACTION)
    require_row_count_change(teams_df.count(), previous_metrics["team_count"] if previous_metrics else None, MAX_DELIVERY_COUNT_CHANGE_FRACTION)

    if prior_silver_exists:
        changed_balls = ball_df.join(
            changed_ids, ball_df["ID"] == changed_ids["match_id"], "inner"
        ).drop("match_id")
        changed_matches = matches_df.join(
            changed_ids, matches_df["match_number"] == changed_ids["match_id"], "inner"
        ).drop("match_id")
        changed_silver = _silver_source(changed_balls, changed_matches)
        prior_silver = spark.read.format("delta").load(SILVER_PATH)
        unchanged_silver = prior_silver.join(
            changed_ids, prior_silver["match_id"] == changed_ids["match_id"], "left_anti"
        )
        all_silver = unchanged_silver.unionByName(changed_silver)
    else:
        changed_silver = _silver_source(ball_df, matches_df)
        all_silver = changed_silver

    require_schema(all_silver, {
        "delivery_key": "string", "match_id": "string", "match_date": "date",
        "innings": "int", "over": "int", "ball_number": "int", "batter": "string",
        "bowler": "string", "batsman_runs": "int", "extras_runs": "int",
        "total_runs": "int", "is_wicket": "int",
    }, "Silver delivery")
    require_non_null(all_silver, ["delivery_key", "match_id", "match_date", "innings", "over", "ball_number", "batter", "bowler", "batsman_runs", "extras_runs", "total_runs", "is_wicket"], "Silver")
    require_unique(all_silver, ["delivery_key"], "Silver candidate delivery key")
    silver_count = all_silver.count()
    report_check("Silver/source delivery reconciliation", silver_count == source_delivery_count,
                 f"source={source_delivery_count}; Silver candidate={silver_count}")
    source_runs = ball_df.agg(F.sum("BatsmanRun").alias("runs")).first()["runs"] or 0
    source_total_runs = ball_df.agg(F.sum("TotalRun").alias("runs")).first()["runs"] or 0
    gold = _build_gold(spark, all_silver)
    gold_runs = gold["top_batsmen"].agg(F.sum("total_runs").alias("runs")).first()["runs"] or 0
    report_check("Gold/source batter-run reconciliation", int(source_runs) == int(gold_runs),
                 f"source={source_runs}; Gold={gold_runs}")
    form_runs = gold["player_form"].agg(F.sum("total_runs").alias("runs")).first()["runs"] or 0
    matchup_runs = gold["player_matchups"].agg(F.sum("runs").alias("runs")).first()["runs"] or 0
    phase_runs = gold["phase_analysis"].agg(F.sum("runs").alias("runs")).first()["runs"] or 0
    phase_deliveries = gold["phase_analysis"].agg(F.sum("deliveries").alias("deliveries")).first()["deliveries"] or 0
    report_check("Player form/source batter-run reconciliation", int(source_runs) == int(form_runs),
                 f"source={source_runs}; player_form={form_runs}")
    report_check("Player matchup/source batter-run reconciliation", int(source_runs) == int(matchup_runs),
                 f"source={source_runs}; player_matchups={matchup_runs}")
    report_check("Phase/source run and delivery reconciliation",
                 int(source_total_runs) == int(phase_runs) and int(silver_count) == int(phase_deliveries),
                 f"source total runs={source_total_runs}; phase runs={phase_runs}; source deliveries={silver_count}; phase deliveries={phase_deliveries}")
    match_count = matches_df.select("match_number").distinct().count()
    season_count = matches_df.select(F.year("match_date").alias("season")).distinct().count()
    team_match_count = gold["team_performance"].agg(F.sum("matches_played")).first()[0] or 0
    report_check("Gold team-match reconciliation", int(team_match_count) == match_count * 2,
                 f"team appearances={team_match_count}; matches={match_count}; expected appearances={match_count * 2}")
    toss_expected = silver.filter(F.col("toss_decision").isNotNull()).select("match_id").distinct().count()
    toss_actual = gold["toss_analysis"].agg(F.sum("matches")).first()[0] or 0
    report_check("Gold toss-match reconciliation", int(toss_actual) == toss_expected,
                 f"Gold toss matches={toss_actual}; source matches with a decision={toss_expected}")

    gold_keys = {
        "top_batsmen": ["batter"],
        "top_bowlers": ["bowler"],
        "player_matchups": ["batter", "bowler"],
        "player_form": ["batter"],
        "team_performance": ["team"],
        "toss_analysis": ["toss_decision"],
        "venue_strategy": ["venue"],
        "phase_analysis": ["phase"],
    }
    for name, frame in gold.items():
        require_non_null(frame, gold_keys[name], f"Gold {name}")
        require_unique(frame, gold_keys[name], f"Gold {name}")

    # All quality assertions run before the first Delta table is changed.
    _merge_silver(spark, changed_silver, changed_ids)

    for name, frame in gold.items():
        _merge_snapshot(spark, frame, GOLD_PATHS[name], gold_keys[name])
        print(f"Published Delta Gold table: {GOLD_PATHS[name]} ({frame.count()} rows)")

    run_record = spark.createDataFrame([(
        datetime.now(timezone.utc).isoformat(), source_delivery_count, match_count,
        players_df.count(), teams_df.count(), season_count,
    )], "run_timestamp string, delivery_count long, match_count long, player_count long, team_count long, season_count long")
    if _delta_exists(spark, RUN_METRICS_PATH):
        run_record.write.format("delta").mode("append").save(RUN_METRICS_PATH)
    else:
        run_record.write.format("delta").mode("overwrite").save(RUN_METRICS_PATH)
    changed_manifest = changed_manifest.withColumn("processed_at", F.current_timestamp())
    if _delta_exists(spark, MANIFEST_PATH):
        (DeltaTable.forPath(spark, MANIFEST_PATH).alias("target")
         .merge(changed_manifest.alias("source"), "target.match_id = source.match_id")
         .whenMatchedUpdateAll().whenNotMatchedInsertAll().execute())
    else:
        changed_manifest.write.format("delta").mode("overwrite").save(MANIFEST_PATH)

    # Keep the existing local dashboard export format; Delta remains the source of truth.
    gold["top_batsmen"].orderBy(F.desc("total_runs")).limit(10).toPandas().to_csv("/tmp/top_batsmen.csv", index=False)
    gold["top_bowlers"].orderBy(F.desc("wickets")).limit(10).toPandas().to_csv("/tmp/top_bowlers.csv", index=False)
    gold["team_performance"].orderBy(F.desc("win_percentage")).toPandas().to_csv("/tmp/team_performance.csv", index=False)
    gold["toss_analysis"].toPandas().to_csv("/tmp/toss_analysis.csv", index=False)
    print("IPL Delta pipeline completed successfully.")


run_pipeline(spark)
