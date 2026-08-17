
import polars as pl


# Provisional — not yet calibrated against actual MAE per bucket. Once
# engine/metrics.py has real output, tune these so MAE is monotonically
# decreasing high < medium < low (see evaluate_projections' calibration check).
CONFIDENCE_TIER_THRESHOLDS = {
    "low": 4,
    "medium": 8,
}


def calculate_expected_points(stats_df: pl.DataFrame, skew_df: pl.DataFrame) -> pl.DataFrame:
    """
    :param stats_df: player-week stats, output of add_rolling_features() — must
        include gsis_id, season, week, opponent_team, position, rolling_avg_prior
    :param skew_df: output of calculate_all_position_skews() — defense, week,
        season, position, opponent_skew, n_games
    :return: stats_df with games_this_season, projection, confidence columns added

    projection = rolling_avg_prior + opponent_skew. Join keys (week, season,
    opponent_team/defense, position) are the same validated join as test.py —
    position must be in the join key or rows silently fan out (see CLAUDE.md).

    Confidence is derived solely from player-side games_this_season — not
    opponent_skew's n_games, not outcome volatility (that's boom/bust's job).
    See CLAUDE.md for the documented mid-season-return limitation and the
    year-over-year defensive-scheme-continuity assumption this implies.
    """
    required = {"gsis_id", "season", "week", "opponent_team", "position", "rolling_avg_prior"}
    missing = required - set(stats_df.columns)
    if missing:
        raise ValueError(f"stats_df missing required columns: {missing}")

    stats_df = stats_df.sort(["gsis_id", "season", "week"])

    stats_df = stats_df.with_columns(
        pl.col("week")
        .shift(1)
        .cum_count()
        .over(["gsis_id", "season"])
        .alias("games_this_season")
    )

    df = stats_df.join(
        skew_df.select(["defense", "week", "season", "position", "opponent_skew", "n_games"])
               .rename({"n_games": "opponent_skew_n_games"}),
        left_on=["week", "season", "opponent_team", "position"],
        right_on=["week", "season", "defense", "position"],
        how="left",
    )

    df = df.with_columns(
        pl.col("opponent_skew").fill_null(0.0)
    )

    df = df.with_columns(
        (pl.col("rolling_avg_prior") + pl.col("opponent_skew")).alias("projection")
    )

    df = df.with_columns(
        pl.when(pl.col("games_this_season") == 0)
        .then(pl.lit("insufficient_data"))
        .when(pl.col("games_this_season") < CONFIDENCE_TIER_THRESHOLDS["low"])
        .then(pl.lit("low"))
        .when(pl.col("games_this_season") < CONFIDENCE_TIER_THRESHOLDS["medium"])
        .then(pl.lit("medium"))
        .otherwise(pl.lit("high"))
        .alias("confidence")
    )

    return df
