
import polars as pl

from common.frames import assert_unique_key
from projections.rolling_window import trailing_count, trailing_mean


# Trailing window of defense games (one regular season). Continuous across the
# season boundary, so week 1 uses last season's games instead of a separate
# prior-season blend. Not tuned.
SKEW_WINDOW = 17


def _calculate_position_skew(
    stats_df: pl.DataFrame,
    games_df: pl.DataFrame,
    position: str,
    volume_col: str,
    volume_threshold: float,
    min_games: int,
) -> pl.DataFrame:

    pos_stats = stats_df.filter(
        (pl.col("position") == position)
        & (pl.col(volume_col) >= volume_threshold)
    )

    joined = pos_stats.join(
        assert_unique_key(games_df, ["team", "season", "week"], "games_df"),
        on=["team", "season", "week"], how="inner",
    )

    # Aggregate to one residual per (defense, season, week) BEFORE shifting, so
    # the trailing window operates over games (weeks), not player-rows: no
    # same-week leakage, and n_games counts games.
    joined = (
        joined.with_columns(
            (pl.col("fantasy_points") - pl.col("rolling_avg_prior")).alias("residual")
        )
        .group_by(["opponent", "season", "week"])
        .agg(pl.col("residual").mean())
        .sort(["opponent", "season", "week"])
    )

    # n_games = games actually in the window (<= SKEW_WINDOW); drives both the
    # min_games floor and the n / (n + k) shrinkage.
    skew = joined.with_columns(
        trailing_mean("residual", SKEW_WINDOW, "opponent").alias("opponent_skew"),
        trailing_count("residual", SKEW_WINDOW, "opponent").alias("n_games"),
    )

    skew = (
        skew.filter(pl.col("n_games") >= min_games)
        .select(["opponent", "week", "season", "opponent_skew", "n_games"])
        .rename({"opponent": "defense"})
        .with_columns(pl.lit(position).alias("position"))
    )

    return skew


def calculate_rb_skew(stats_df: pl.DataFrame, games_df: pl.DataFrame, min_games: int = 3) -> pl.DataFrame:
    return _calculate_position_skew(stats_df, games_df, "RB", volume_col="trailing_opportunities_avg", volume_threshold=6.0, min_games=min_games)


def calculate_wr_skew(stats_df: pl.DataFrame, games_df: pl.DataFrame, min_games: int = 3) -> pl.DataFrame:
    return _calculate_position_skew(stats_df, games_df, "WR", volume_col="trailing_opportunities_avg", volume_threshold=4.0, min_games=min_games)


def calculate_te_skew(stats_df: pl.DataFrame, games_df: pl.DataFrame, min_games: int = 3) -> pl.DataFrame:
    return _calculate_position_skew(stats_df, games_df, "TE", volume_col="trailing_opportunities_avg", volume_threshold=2.0, min_games=min_games)


def calculate_qb_skew(stats_df: pl.DataFrame, games_df: pl.DataFrame, min_games: int = 3) -> pl.DataFrame:
    return _calculate_position_skew(stats_df, games_df, "QB", volume_col="trailing_attempts_avg", volume_threshold=6.0, min_games=min_games)


def calculate_all_position_skews(
        stats_df: pl.DataFrame,
        games_df: pl.DataFrame,
        k: float = 16.0,
        sanity_clip: float | None = 12.0) -> pl.DataFrame:
    """
    :param k: shrinkage constant — skew_shrunk = skew_raw * (n / (n + k)),
        where n is n_games (defense games in the trailing SKEW_WINDOW
        backing that row's estimate). A single tunable constant, chosen by
        grid search against engine.metrics.evaluate_projections() output
        (see CLAUDE.md), not fit via optimization.
    :param sanity_clip: wide guardrail applied after shrinkage. This is NOT
        the flattening mechanism (shrinkage is) — it only catches
        pathological rows (e.g. n=0 edge cases) that shouldn't reach the
        projection un-bounded. Pass None to disable.
    """

    df = pl.concat([
        calculate_rb_skew(stats_df, games_df),
        calculate_wr_skew(stats_df, games_df),
        calculate_te_skew(stats_df, games_df),
        calculate_qb_skew(stats_df, games_df),
    ])

    assert_unique_key(df, ["defense", "week", "season", "position"], "skew_df")

    # Sample-size-weighted shrinkage toward 0, replacing the old hard ±8
    # clip. Low-n estimates (noisy, per the same low-n_games clustering
    # that motivated the old clip) get pulled hard toward 0; well-supported
    # estimates (large n_games) are trusted close to their raw value.
    df = df.with_columns(
        (
            pl.col("opponent_skew")
            * (pl.col("n_games") / (pl.col("n_games").cast(pl.Float64) + k))
        ).alias("opponent_skew")
    )

    if sanity_clip is not None:
        df = df.with_columns(
            pl.col("opponent_skew").clip(lower_bound=-sanity_clip, upper_bound=sanity_clip)
            .alias("opponent_skew")
        )

    return df