
import polars as pl


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

    joined = pos_stats.join(games_df, on=["team", "season", "week"], how="inner")

    joined = joined.with_columns(
        (pl.col("fantasy_points") - pl.col("rolling_avg_prior")).alias("residual")
    ).sort(["opponent", "season", "week"])

    joined = joined.with_columns(
        pl.col("residual").shift(1).over(["opponent", "season"]).alias("residual_prior")
    )

    joined = joined.with_columns([
        pl.col("residual_prior").cum_sum().over(["opponent", "season"]).alias("_cum_sum"),
        pl.col("residual_prior").cum_count().over(["opponent", "season"]).alias("n_games"),
    ])

    skew = joined.with_columns(
        (pl.col("_cum_sum") / pl.col("n_games")).alias("opponent_skew")
    )

    # Blend with last season's full-season average residual for this
    # defense/position, same early-season ramp as player_rolling.py: weight
    # on the current season is n_games / 9, capped at 0.9 (see CLAUDE.md
    # "Early-season handling"). Uses raw (unshifted) residual — the prior
    # season is already complete, so there's no leakage risk averaging
    # across all of it.
    last_season_skew = (
        joined.group_by(["opponent", "season"])
        .agg(pl.col("residual").mean().alias("_last_season_avg_skew"))
        .with_columns((pl.col("season") + 1).alias("season"))
    )

    skew = skew.join(last_season_skew, on=["opponent", "season"], how="left")

    current_weight = (pl.col("n_games") / 9.0).clip(upper_bound=0.9)

    skew = skew.with_columns(
        pl.when(pl.col("_last_season_avg_skew").is_not_null())
        .then(
            current_weight * pl.col("opponent_skew")
            + (1 - current_weight) * pl.col("_last_season_avg_skew")
        )
        .otherwise(pl.col("opponent_skew"))
        .alias("opponent_skew")
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
        where n is n_games (current-season games of defense/position data
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