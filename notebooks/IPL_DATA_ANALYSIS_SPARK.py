# Databricks notebook source
print("Spark version:", spark.version)

# COMMAND ----------

teams_df = spark.read.csv(
    "s3://ipl-2010-data/Raw/teams_info.csv",
    header=True,
    inferSchema=True
)

display(teams_df)

# COMMAND ----------

players_df = spark.read.csv(
    "s3://ipl-2010-data/Raw/2024_players_details.csv",
    header=True,
    inferSchema=True
)

matches_df = spark.read.csv(
    "s3://ipl-2010-data/Raw/Match_Info.csv",
    header=True,
    inferSchema=True
)

ball_df = spark.read.csv(
    "s3://ipl-2010-data/Raw/Ball_By_Ball_Match_Data.csv",
    header=True,
    inferSchema=True
)

print("Players:", players_df.count())
print("Matches:", matches_df.count())
print("Balls:", ball_df.count())
print("Teams:", teams_df.count())

# COMMAND ----------

print("===== PLAYERS =====")
players_df.printSchema()

print("===== MATCHES =====")
matches_df.printSchema()

print("===== BALL BY BALL =====")
ball_df.printSchema()

print("===== TEAMS =====")
teams_df.printSchema()

# COMMAND ----------

print("Match IDs:")
matches_df.select("match_number").show(10, truncate=False)

print("Ball IDs:")
ball_df.select("ID").distinct().show(10, truncate=False)

# COMMAND ----------

from pyspark.sql.functions import col

silver_ball_df = (
    ball_df.alias("b")
    .join(
        matches_df.alias("m"),
        col("b.ID") == col("m.match_number"),
        "left"
    )
    .select(
        col("b.ID").alias("match_id"),
        col("m.match_date"),
        col("m.team1"),
        col("m.team2"),
        col("m.toss_winner"),
        col("m.toss_decision"),
        col("m.winner"),
        col("m.player_of_match"),
        col("m.venue"),
        col("m.city"),
        col("b.Innings").alias("innings"),
        col("b.Overs").alias("over"),
        col("b.BallNumber").alias("ball_number"),
        col("b.Batter").alias("batter"),
        col("b.Bowler").alias("bowler"),
        col("b.NonStriker").alias("non_striker"),
        col("b.ExtraType").alias("extra_type"),
        col("b.BatsmanRun").alias("batsman_runs"),
        col("b.ExtrasRun").alias("extras_runs"),
        col("b.TotalRun").alias("total_runs"),
        col("b.IsWicketDelivery").alias("is_wicket"),
        col("b.PlayerOut").alias("player_out"),
        col("b.Kind").alias("dismissal_kind"),
        col("b.BattingTeam").alias("batting_team")
    )
)

display(silver_ball_df.limit(10))

# COMMAND ----------

from pyspark.sql.functions import col

silver_ball_df = (
    ball_df.alias("b")
    .join(
        matches_df.alias("m"),
        col("b.ID") == col("m.match_number"),
        "left"
    )
    .select(
        col("b.ID").alias("match_id"),
        col("m.match_date"),
        col("m.team1"),
        col("m.team2"),
        col("m.toss_winner"),
        col("m.toss_decision"),
        col("m.winner"),
        col("m.player_of_match"),
        col("m.venue"),
        col("m.city"),
        col("b.Innings").alias("innings"),
        col("b.Overs").alias("over"),
        col("b.BallNumber").alias("ball_number"),
        col("b.Batter").alias("batter"),
        col("b.Bowler").alias("bowler"),
        col("b.NonStriker").alias("non_striker"),
        col("b.ExtraType").alias("extra_type"),
        col("b.BatsmanRun").alias("batsman_runs"),
        col("b.ExtrasRun").alias("extras_runs"),
        col("b.TotalRun").alias("total_runs"),
        col("b.IsWicketDelivery").alias("is_wicket"),
        col("b.PlayerOut").alias("player_out"),
        col("b.Kind").alias("dismissal_kind"),
        col("b.BattingTeam").alias("batting_team")
    )
)

print("Silver DataFrame created:", silver_ball_df.count())

# COMMAND ----------

silver_path = "s3://ipl-2010-data/Silver/ball_by_ball/"

silver_ball_df.write \
    .mode("overwrite") \
    .parquet(silver_path)

print("Silver data written successfully!")

# COMMAND ----------

silver_check_df = spark.read.parquet(
    "s3://ipl-2010-data/Silver/ball_by_ball/"
)

print("Silver rows:", silver_check_df.count())
display(silver_check_df.limit(10))

# COMMAND ----------

silver_check_df.createOrReplaceTempView("ipl_deliveries")

print("SQL view created!")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     batter,
# MAGIC     SUM(batsman_runs) AS total_runs,
# MAGIC     COUNT(DISTINCT match_id) AS matches_played
# MAGIC FROM ipl_deliveries
# MAGIC GROUP BY batter
# MAGIC ORDER BY total_runs DESC
# MAGIC LIMIT 10;

# COMMAND ----------

gold_top_batsmen = spark.sql("""
    SELECT
        batter,
        SUM(batsman_runs) AS total_runs,
        COUNT(DISTINCT match_id) AS matches_played
    FROM ipl_deliveries
    GROUP BY batter
    ORDER BY total_runs DESC
    LIMIT 10
""")

gold_top_batsmen.write \
    .mode("overwrite") \
    .parquet("s3://ipl-2010-data/Gold/top_batsmen/")

print("Gold table saved successfully!")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     bowler,
# MAGIC     COUNT(*) AS wickets
# MAGIC FROM ipl_deliveries
# MAGIC WHERE is_wicket = 1
# MAGIC   AND dismissal_kind NOT IN ('run out', 'retired hurt', 'obstructing the field')
# MAGIC GROUP BY bowler
# MAGIC ORDER BY wickets DESC
# MAGIC LIMIT 10;

# COMMAND ----------

gold_top_bowlers = spark.sql("""
    SELECT
        bowler,
        COUNT(*) AS wickets
    FROM ipl_deliveries
    WHERE is_wicket = 1
      AND dismissal_kind NOT IN (
          'run out',
          'retired hurt',
          'obstructing the field'
      )
    GROUP BY bowler
    ORDER BY wickets DESC
    LIMIT 10
""")

gold_top_bowlers.write \
    .mode("overwrite") \
    .parquet("s3://ipl-2010-data/Gold/top_bowlers/")

print("Gold top bowlers table saved successfully!")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     team,
# MAGIC     COUNT(*) AS matches_played,
# MAGIC     SUM(CASE WHEN winner = team THEN 1 ELSE 0 END) AS wins,
# MAGIC     ROUND(
# MAGIC         SUM(CASE WHEN winner = team THEN 1 ELSE 0 END) * 100.0
# MAGIC         / COUNT(*),
# MAGIC         2
# MAGIC     ) AS win_percentage
# MAGIC FROM (
# MAGIC     SELECT match_id, team1 AS team, winner
# MAGIC     FROM ipl_deliveries
# MAGIC     GROUP BY match_id, team1, winner
# MAGIC
# MAGIC     UNION ALL
# MAGIC
# MAGIC     SELECT match_id, team2 AS team, winner
# MAGIC     FROM ipl_deliveries
# MAGIC     GROUP BY match_id, team2, winner
# MAGIC )
# MAGIC GROUP BY team
# MAGIC ORDER BY win_percentage DESC;

# COMMAND ----------

gold_team_performance = spark.sql("""
    SELECT
        team,
        COUNT(*) AS matches_played,
        SUM(CASE WHEN winner = team THEN 1 ELSE 0 END) AS wins,
        ROUND(
            SUM(CASE WHEN winner = team THEN 1 ELSE 0 END) * 100.0
            / COUNT(*),
            2
        ) AS win_percentage
    FROM (
        SELECT match_id, team1 AS team, winner
        FROM ipl_deliveries
        GROUP BY match_id, team1, winner

        UNION ALL

        SELECT match_id, team2 AS team, winner
        FROM ipl_deliveries
        GROUP BY match_id, team2, winner
    )
    GROUP BY team
    ORDER BY win_percentage DESC
""")

gold_team_performance.write \
    .mode("overwrite") \
    .parquet("s3://ipl-2010-data/Gold/team_performance/")

print("Gold team performance saved successfully!")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     toss_decision,
# MAGIC     COUNT(*) AS matches,
# MAGIC     SUM(CASE WHEN toss_winner = winner THEN 1 ELSE 0 END) AS toss_winner_wins,
# MAGIC     ROUND(
# MAGIC         SUM(CASE WHEN toss_winner = winner THEN 1 ELSE 0 END) * 100.0
# MAGIC         / COUNT(*),
# MAGIC         2
# MAGIC     ) AS win_percentage
# MAGIC FROM (
# MAGIC     SELECT DISTINCT
# MAGIC         match_id,
# MAGIC         toss_winner,
# MAGIC         toss_decision,
# MAGIC         winner
# MAGIC     FROM ipl_deliveries
# MAGIC )
# MAGIC GROUP BY toss_decision
# MAGIC ORDER BY win_percentage DESC;

# COMMAND ----------

gold_toss_analysis = spark.sql("""
    SELECT
        toss_decision,
        COUNT(*) AS matches,
        SUM(CASE WHEN toss_winner = winner THEN 1 ELSE 0 END) AS toss_winner_wins,
        ROUND(
            SUM(CASE WHEN toss_winner = winner THEN 1 ELSE 0 END) * 100.0
            / COUNT(*),
            2
        ) AS win_percentage
    FROM (
        SELECT DISTINCT
            match_id,
            toss_winner,
            toss_decision,
            winner
        FROM ipl_deliveries
    )
    GROUP BY toss_decision
    ORDER BY win_percentage DESC
""")

display(gold_toss_analysis)

# COMMAND ----------

gold_toss_analysis.write \
    .mode("overwrite") \
    .parquet("s3://ipl-2010-data/Gold/toss_analysis/")

print("Gold toss analysis saved successfully!")

# COMMAND ----------

gold_batsmen = spark.read.parquet(
    "s3://ipl-2010-data/Gold/top_batsmen/"
)

gold_bowlers = spark.read.parquet(
    "s3://ipl-2010-data/Gold/top_bowlers/"
)

gold_teams = spark.read.parquet(
    "s3://ipl-2010-data/Gold/team_performance/"
)

gold_toss = spark.read.parquet(
    "s3://ipl-2010-data/Gold/toss_analysis/"
)

gold_batsmen.createOrReplaceTempView("gold_batsmen")
gold_bowlers.createOrReplaceTempView("gold_bowlers")
gold_teams.createOrReplaceTempView("gold_teams")
gold_toss.createOrReplaceTempView("gold_toss")

print("Gold SQL views created!")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     batter,
# MAGIC     total_runs,
# MAGIC     matches_played
# MAGIC FROM gold_batsmen
# MAGIC ORDER BY total_runs DESC
# MAGIC LIMIT 10;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     bowler,
# MAGIC     wickets
# MAGIC FROM gold_bowlers
# MAGIC ORDER BY wickets DESC
# MAGIC LIMIT 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM gold_batsmen ORDER BY total_runs DESC;
# MAGIC SELECT * FROM gold_bowlers ORDER BY wickets DESC;
# MAGIC SELECT * FROM gold_teams ORDER BY win_percentage DESC;
# MAGIC SELECT * FROM gold_toss;

# COMMAND ----------

gold_batsmen.toPandas().to_csv("/tmp/top_batsmen.csv", index=False)
gold_bowlers.toPandas().to_csv("/tmp/top_bowlers.csv", index=False)
gold_teams.toPandas().to_csv("/tmp/team_performance.csv", index=False)
gold_toss.toPandas().to_csv("/tmp/toss_analysis.csv", index=False)

print("All Gold datasets exported!")

# COMMAND ----------

print("Batsmen:", gold_batsmen.count())
print("Bowlers:", gold_bowlers.count())
print("Teams:", gold_teams.count())
print("Toss:", gold_toss.count())