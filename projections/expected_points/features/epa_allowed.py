
import polars as pl

from projections.boom_bust.features.aggregate_pbp import (
    _construct_passing_columns,
    _construct_rushing_columns,
)


KEY = ["defteam", "season", "week"]


def _weekly_residuals(plays: pl.DataFrame) -> pl.DataFrame:
    """
    :param plays: pbp rows for one play-type group (pass or run)
    :return: one row per (defteam, season, week): EPA/play allowed that week
        minus the league-wide EPA/play that same week, plus play count

    Centering on the same week's league mean is leak-free: the feature for
    week W only ever sees residuals from weeks strictly before W (shift
    below), and each of those was centered with its own week's data.
    """
    plays = plays.filter(pl.col("epa").is_not_null() & pl.col("defteam").is_not_null())

    league = plays.group_by(["season", "week"]).agg(pl.col("epa").mean().alias("_league_epa"))

    return (
        plays.group_by(KEY)
        .agg(pl.col("epa").mean().alias("_def_epa"), pl.len().alias("n_plays"))
        .join(league, on=["season", "week"], how="left")
        .with_columns((pl.col("_def_epa") - pl.col("_league_epa")).alias("residual"))
        .select(KEY + ["residual", "n_plays"])
    )


def _shift_blend_shrink(weekly: pl.DataFrame, k: float) -> pl.DataFrame:
    """
    Same shape as opponent_skew._calculate_position_skew, applied to EPA
    residuals: week-level rows, .shift(1) by week within (defteam, season),
    cumulative mean, blend with last season's full-season average at
    min(n_games / 9, 0.9), then shrink toward 0 (league average).

    Shrinkage uses an effective n: n_eff = n_games + (1 - w) * prior_season_games.
    Plain n_games (as opponent_skew uses) would zero out week-1 rows entirely
    even though they carry a full prior season of blended evidence.
    """
    weekly = weekly.sort(KEY).with_columns(
        pl.col("residual").shift(1).over(["defteam", "season"]).alias("_residual_prior")
    )

    weekly = weekly.with_columns([
        pl.col("_residual_prior").cum_sum().over(["defteam", "season"]).alias("_cum_sum"),
        pl.col("_residual_prior").cum_count().over(["defteam", "season"]).alias("n_games"),
    ]).with_columns(
        pl.when(pl.col("n_games") > 0)
        .then(pl.col("_cum_sum") / pl.col("n_games"))
        .otherwise(None)
        .alias("_current")
    )

    # Prior season uses raw (unshifted) residuals — that season is complete.
    last_season = (
        weekly.group_by(["defteam", "season"])
        .agg(
            pl.col("residual").mean().alias("_last_season"),
            pl.len().alias("_prior_games"),
        )
        .with_columns((pl.col("season") + 1).alias("season"))
    )

    weekly = weekly.join(last_season, on=["defteam", "season"], how="left")

    w = (pl.col("n_games") / 9.0).clip(upper_bound=0.9)

    weekly = weekly.with_columns(
        pl.when(pl.col("_current").is_not_null() & pl.col("_last_season").is_not_null())
        .then(w * pl.col("_current") + (1 - w) * pl.col("_last_season"))
        .when(pl.col("_current").is_not_null())
        .then(pl.col("_current"))
        .otherwise(pl.col("_last_season"))
        .alias("_blended"),

        (
            pl.col("n_games")
            + (1 - w) * pl.col("_prior_games").fill_null(0)
        ).cast(pl.Float64).alias("n_eff"),
    )

    # No evidence at all (n_eff == 0, e.g. first loaded season week 1) resolves
    # to 0 = league average, which is also where shrinkage sends it.
    return weekly.with_columns(
        (pl.col("_blended") * pl.col("n_eff") / (pl.col("n_eff") + k))
        .fill_null(0.0)
        .alias("epa_allowed")
    ).select(KEY + ["epa_allowed", "n_games", "n_eff", "n_plays"])


def calculate_epa_allowed(pbp_df: pl.DataFrame, k: float = 16.0) -> pl.DataFrame:
    """
    :param pbp_df: raw play-by-play (nflreadpy load_pbp), all loaded seasons
    :param k: shrinkage constant, same default as opponent skew. Not tuned for
        EPA; the downstream OLS coefficient absorbs overall scale.
    :return: one row per (defteam, season, week) the defense played:
        epa_allowed_pass, epa_allowed_rush (positive = defense allows more
        EPA/play than league average, i.e. a softer matchup), n_games,
        n_eff_pass/rush, n_plays_pass/rush. Values for week W use only
        games strictly before W.

    Pass plays use the corrected (sack-excluded) filter and run plays use
    play_type == 'run', both shared with aggregate_pbp.py.
    """
    pass_df = _shift_blend_shrink(_weekly_residuals(_construct_passing_columns(pbp_df)), k)
    rush_df = _shift_blend_shrink(_weekly_residuals(_construct_rushing_columns(pbp_df)), k)

    df = pass_df.rename({
        "epa_allowed": "epa_allowed_pass", "n_eff": "n_eff_pass", "n_plays": "n_plays_pass",
    }).join(
        rush_df.rename({
            "epa_allowed": "epa_allowed_rush", "n_eff": "n_eff_rush",
            "n_plays": "n_plays_rush", "n_games": "_n_games_rush",
        }),
        on=KEY,
        how="full",
        coalesce=True,
    ).with_columns(
        pl.coalesce("n_games", "_n_games_rush").alias("n_games"),
        pl.col("epa_allowed_pass").fill_null(0.0),
        pl.col("epa_allowed_rush").fill_null(0.0),
    ).drop("_n_games_rush").sort(KEY)

    assert not df.select(KEY).is_duplicated().any(), \
        "epa_allowed has duplicate (defteam, season, week) keys"

    return df


def join_epa_allowed(stats_df: pl.DataFrame, epa_df: pl.DataFrame) -> pl.DataFrame:
    """
    Left-join epa_allowed onto player-week rows via opponent_team. Rows with
    no matching defense-week get 0.0 (league average).
    """
    df = stats_df.join(
        epa_df.select(KEY + ["epa_allowed_pass", "epa_allowed_rush"]),
        left_on=["opponent_team", "season", "week"],
        right_on=KEY,
        how="left",
    ).with_columns(
        pl.col("epa_allowed_pass").fill_null(0.0),
        pl.col("epa_allowed_rush").fill_null(0.0),
    )

    assert df.height == stats_df.height, \
        f"join fan-out: {df.height} rows vs {stats_df.height} stats rows"

    return df
