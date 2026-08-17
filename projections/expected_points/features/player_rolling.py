
import polars as pl


def add_rolling_features(stats_df: pl.DataFrame, window: int = 6) -> pl.DataFrame:
    """
    :param stats_df: player-week stats df, must include gsis_id, season,
        week, fantasy_points, carries, targets, attempts
    :param window: number of trailing games to average over (default 6)
    :return: stats_df with rolling feature columns added

    rolling_avg_prior blends the current-season trailing average with last
    season's full-season average early in the season (see CLAUDE.md
    "Early-season handling"). Weight on the current-season value is
    games_this_season / 9, capped at 0.9 — so last season's average never
    drops below 0.1 weight once it's available, and ramps out as the
    current-season sample grows. Players with no prior-season data (true
    rookies) get the current-season value alone, which is None until it's
    populated — never a fabricated fallback.
    """

    required_cols = {"gsis_id", "season", "week", "fantasy_points", "carries", "targets", "attempts"}
    missing = required_cols - set(stats_df.columns)
    if missing:
        raise ValueError(f"stats_df missing required columns: {missing}")

    stats_df = stats_df.sort(["gsis_id", "season", "week"])

    stats_df = stats_df.with_columns(
        (pl.col("carries") + pl.col("targets")).alias("opportunities")
    )

    partition = ["gsis_id", "season"]

    stats_df = stats_df.with_columns([
        pl.col("fantasy_points")
            .shift(1)
            .rolling_mean(window_size=window)
            .over(partition)
            .alias("_current_season_rolling_avg"),

        pl.col("week")
            .shift(1)
            .cum_count()
            .over(partition)
            .alias("_games_this_season"),

        pl.col("opportunities")
            .shift(1)
            .rolling_mean(window_size=window)
            .over(partition)
            .alias("trailing_opportunities_avg"),

        pl.col("targets")
            .shift(1)
            .rolling_mean(window_size=window)
            .over(partition)
            .alias("trailing_targets_avg"),

        pl.col("attempts")
            .shift(1)
            .rolling_mean(window_size=window)
            .over(partition)
            .alias("trailing_attempts_avg"),
    ])

    last_season_avg = (
        stats_df.group_by(["gsis_id", "season"])
        .agg(pl.col("fantasy_points").mean().alias("_last_season_avg"))
        .with_columns((pl.col("season") + 1).alias("season"))
    )

    stats_df = stats_df.join(last_season_avg, on=["gsis_id", "season"], how="left")

    current_weight = (pl.col("_games_this_season") / 9.0).clip(upper_bound=0.9)

    stats_df = stats_df.with_columns(
        pl.when(
            pl.col("_current_season_rolling_avg").is_not_null()
            & pl.col("_last_season_avg").is_not_null()
        )
        .then(
            current_weight * pl.col("_current_season_rolling_avg")
            + (1 - current_weight) * pl.col("_last_season_avg")
        )
        .when(pl.col("_current_season_rolling_avg").is_not_null())
        .then(pl.col("_current_season_rolling_avg"))
        .otherwise(pl.col("_last_season_avg"))
        .alias("rolling_avg_prior")
    ).drop(["_current_season_rolling_avg", "_games_this_season", "_last_season_avg"])

    return stats_df