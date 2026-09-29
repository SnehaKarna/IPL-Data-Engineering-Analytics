"""Fail-fast PySpark data-quality checks used by the Databricks pipeline."""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def report_check(name: str, passed: bool, details: str) -> None:
    status = "PASS" if passed else "FAIL"
    print(f"[DQ {status}] {name}: {details}")
    if not passed:
        raise AssertionError(f"Data-quality check failed: {name}. {details}")


def require_schema(frame: DataFrame, required: dict[str, str], label: str) -> None:
    actual = {field.name: field.dataType.simpleString() for field in frame.schema.fields}
    missing = sorted(set(required) - set(actual))
    wrong = {name: (actual[name], expected) for name, expected in required.items()
             if name in actual and actual[name] != expected}
    report_check(
        f"{label} schema",
        not missing and not wrong,
        f"missing={missing}; type mismatches={wrong}",
    )


def require_columns(frame: DataFrame, required: list[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    report_check(f"{label} required columns", not missing, f"missing={missing}")


def require_non_null(frame: DataFrame, columns: list[str], label: str) -> None:
    counts = frame.agg(*[
        F.sum(F.when(F.col(name).isNull(), 1).otherwise(0)).alias(name)
        for name in columns
    ]).first().asDict()
    failures = {name: int(count or 0) for name, count in counts.items() if count}
    report_check(f"{label} critical nulls", not failures, f"null counts={failures}")


def require_unique(frame: DataFrame, keys: list[str], label: str) -> None:
    duplicate = frame.groupBy(*keys).count().filter(F.col("count") > 1).limit(1).count() > 0
    report_check(f"{label} unique key", not duplicate, f"key={keys}; duplicate groups={int(duplicate)}")


def require_no_orphans(child: DataFrame, parent: DataFrame, child_key: str,
                       parent_key: str, label: str) -> None:
    child_values = child.select(F.col(child_key).alias("__child_key")).distinct()
    parent_values = parent.select(F.col(parent_key).alias("__parent_key")).distinct()
    orphan = child_values.join(
        parent_values,
        F.col("__child_key") == F.col("__parent_key"),
        "left_anti",
    ).limit(1).count() > 0
    report_check(label, not orphan, f"unmatched {child_key} values exist={orphan}")


def require_player_references(deliveries: DataFrame, players: DataFrame,
                              player_name_column: str) -> None:
    """Match full names and common initial-plus-surname delivery labels."""
    roster = players.select(F.trim(F.col(player_name_column).cast("string")).alias("name"))
    roster = roster.filter(F.col("name").isNotNull() & (F.length("name") > 0))
    roster = roster.withColumn("normalized", F.lower(F.regexp_replace("name", r"[^A-Za-z0-9 ]", "")))
    roster = roster.withColumn("first", F.element_at(F.split(F.trim("normalized"), r"\s+"), 1))
    roster = roster.withColumn("last", F.element_at(F.split(F.trim("normalized"), r"\s+"), -1))
    roster_exact = roster.select(F.regexp_replace("normalized", r"\s+", "").alias("name_key")).distinct()
    roster_initials = roster.select(F.substring("first", 1, 1).alias("first_initial"), "last").distinct()

    refs = deliveries.select(F.explode(F.array(
        F.col("batter"), F.col("bowler"), F.col("non_striker"), F.col("player_out")
    )).alias("name")).filter(F.col("name").isNotNull())
    refs = refs.withColumn("normalized", F.lower(F.regexp_replace("name", r"[^A-Za-z0-9 ]", "")))
    refs = refs.withColumn("name_key", F.regexp_replace("normalized", r"\s+", ""))
    refs = refs.withColumn("first", F.element_at(F.split(F.trim("normalized"), r"\s+"), 1))
    refs = refs.withColumn("last", F.element_at(F.split(F.trim("normalized"), r"\s+"), -1))
    unmatched = refs.select("name", "name_key", "first", "last").distinct().alias("r").join(
        roster_exact.alias("e"), F.col("r.name_key") == F.col("e.name_key"), "left_anti"
    )
    unresolved = unmatched.alias("u").join(
        roster_initials.alias("p"),
        (F.substring(F.col("u.first"), 1, 1) == F.col("p.first_initial"))
        & (F.col("u.last") == F.col("p.last")),
        "left_anti",
    ).select("u.name").distinct()
    sample = [row["name"] for row in unresolved.limit(10).collect()]
    report_check("delivery player references", not sample,
                 f"unresolved player labels (sample, max 10)={sample}; roster column={player_name_column}")


def require_season_coverage(matches: DataFrame) -> list[int]:
    years = [row[0] for row in matches.select(F.year("match_date").alias("season"))
             .distinct().orderBy("season").collect() if row[0] is not None]
    missing = sorted(set(range(min(years), max(years) + 1)) - set(years)) if years else []
    report_check("season coverage", bool(years) and not missing,
                 f"observed={years}; missing calendar seasons={missing}")
    return years


def require_row_count_change(current: int, previous: int | None,
                             max_change_fraction: float) -> None:
    if previous is None:
        report_check("delivery row-count baseline", current > 0,
                     f"initial accepted baseline={current}")
        return
    change = abs(current - previous) / max(previous, 1)
    report_check(
        "delivery row-count change",
        change <= max_change_fraction,
        f"previous={previous}; current={current}; change={change:.1%}; limit={max_change_fraction:.1%}",
    )
