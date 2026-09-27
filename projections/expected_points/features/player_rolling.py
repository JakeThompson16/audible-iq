
import polars as pl

from projections.rolling_window import trailing_count, trailing_mean


def add_rolling_features(stats_df: pl.DataFrame, window: int = 8) -> pl.DataFrame:
    """
    :param stats_df: player-week stats df, must include gsis_id, season,
        week, fantasy_points, carries, targets, attempts
    :param window: number of trailing games to average over (default 8)
    :return: stats_df with rolling feature columns added:
        rolling_avg_prior, n_games_in_window, trailing_opportunities_avg,
        trailing_targets_avg, trailing_attempts_avg

    Continuous trailing window (projections/rolling_window.py): the last
    `window` games the player appeared in, spanning the season boundary, so a
    returning veteran's week-1 value is the tail of last season. There is no
    separate prior-season blend; n_games_in_window says how many games back the
    value. True rookies (no prior games) resolve to None — never a fabricated
    fallback.
    """

    required_cols = {"gsis_id", "season", "week", "fantasy_points", "carries", "targets", "attempts"}
    missing = required_cols - set(stats_df.columns)
    if missing:
        raise ValueError(f"stats_df missing required columns: {missing}")

    stats_df = stats_df.sort(["gsis_id", "season", "week"])

    stats_df = stats_df.with_columns(
        (pl.col("carries") + pl.col("targets")).alias("opportunities")
    )

    return stats_df.with_columns([
        trailing_mean("fantasy_points", window, "gsis_id").alias("rolling_avg_prior"),
        trailing_count("fantasy_points", window, "gsis_id").alias("n_games_in_window"),
        trailing_mean("opportunities", window, "gsis_id").alias("trailing_opportunities_avg"),
        trailing_mean("targets", window, "gsis_id").alias("trailing_targets_avg"),
        trailing_mean("attempts", window, "gsis_id").alias("trailing_attempts_avg"),
    ])
