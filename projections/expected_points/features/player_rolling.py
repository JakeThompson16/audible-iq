
import polars as pl

from projections.rolling_window import trailing_count, trailing_mean


def _add_window_features(stats_df: pl.DataFrame, window: int) -> pl.DataFrame:
    return stats_df.with_columns([
        trailing_mean("fantasy_points", window, "gsis_id").alias("rolling_avg_prior"),
        trailing_count("fantasy_points", window, "gsis_id").alias("n_games_in_window"),
        trailing_mean("opportunities", window, "gsis_id").alias("trailing_opportunities_avg"),
        trailing_mean("targets", window, "gsis_id").alias("trailing_targets_avg"),
        trailing_mean("attempts", window, "gsis_id").alias("trailing_attempts_avg"),
        pl.lit(window, dtype=pl.Int32).alias("rolling_window"),
    ])


def add_rolling_features(
        stats_df: pl.DataFrame,
        window: int | dict[str, int] = 8,
        default_window: int = 8) -> pl.DataFrame:
    """
    :param stats_df: player-week stats df, must include gsis_id, season,
        week, position, fantasy_points, carries, targets, attempts
    :param window: trailing games to average over, either one value for every
        position or a position -> window mapping (engine/expected_points.py
        POSITION_ROLLING_WINDOWS); positions missing from the mapping use
        default_window
    :return: stats_df with rolling feature columns added:
        rolling_avg_prior, n_games_in_window, trailing_opportunities_avg,
        trailing_targets_avg, trailing_attempts_avg, rolling_window

    Continuous trailing window (projections/rolling_window.py): the last
    `window` games the player appeared in, spanning the season boundary, so a
    returning veteran's week-1 value is the tail of last season. There is no
    separate prior-season blend; n_games_in_window says how many games back the
    value. True rookies (no prior games) resolve to None — never a fabricated
    fallback. A player's rows all carry one position, so computing per
    position never splits a player's history.
    """

    required_cols = {"gsis_id", "season", "week", "position", "fantasy_points", "carries", "targets", "attempts"}
    missing = required_cols - set(stats_df.columns)
    if missing:
        raise ValueError(f"stats_df missing required columns: {missing}")

    stats_df = stats_df.sort(["gsis_id", "season", "week"]).with_columns(
        (pl.col("carries") + pl.col("targets")).alias("opportunities")
    )

    if isinstance(window, int):
        return _add_window_features(stats_df, window)

    window_of = pl.col("position").replace_strict(window, default=default_window, return_dtype=pl.Int32)
    stats_df = stats_df.with_columns(window_of.alias("_window"))
    parts = [
        _add_window_features(stats_df.filter(pl.col("_window") == w), w)
        for w in stats_df["_window"].unique().to_list()
    ]
    return pl.concat(parts).drop("_window").sort(["gsis_id", "season", "week"])
