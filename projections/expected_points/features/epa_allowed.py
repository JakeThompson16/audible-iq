
import polars as pl

from common.frames import assert_no_fanout, assert_unique_key
from projections.boom_bust.features.aggregate_pbp import (
    _construct_passing_columns,
    _construct_rushing_columns,
)
from projections.expected_points.features.opponent_skew import SKEW_WINDOW
from projections.rolling_window import trailing_count, trailing_mean


KEY = ["defteam", "season", "week"]


def _weekly_residuals(plays: pl.DataFrame) -> pl.DataFrame:
    """
    :param plays: pbp rows for one play-type group (pass or run)
    :return: one row per (defteam, season, week): EPA/play allowed that week
        minus the league-wide EPA/play that same week, plus play count

    Centering on the same week's league mean is leak-free: the feature for
    week W only ever sees residuals from weeks strictly before W (shift in
    the trailing window), and each of those was centered with its own week's data.
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


def _trailing_shrunk(weekly: pl.DataFrame, k: float) -> pl.DataFrame:
    """
    Same shape as opponent skew: week-level rows, continuous trailing window of
    the defense's last SKEW_WINDOW games (spanning the season boundary), then
    shrinkage toward 0 (league average) by n / (n + k), where n = games
    actually in the window.
    """
    weekly = weekly.sort(KEY).with_columns(
        trailing_mean("residual", SKEW_WINDOW, "defteam").alias("_trailing"),
        trailing_count("residual", SKEW_WINDOW, "defteam").alias("n_games"),
    )

    # No prior games (first loaded week) -> 0 = league average, which is also
    # where shrinkage sends it.
    return weekly.with_columns(
        (pl.col("_trailing") * pl.col("n_games") / (pl.col("n_games") + k))
        .fill_null(0.0)
        .alias("epa_allowed")
    ).select(KEY + ["epa_allowed", "n_games", "n_plays"])


def calculate_epa_allowed(pbp_df: pl.DataFrame, k: float = 16.0) -> pl.DataFrame:
    """
    :param pbp_df: raw play-by-play (nflreadpy load_pbp), all loaded seasons
    :param k: shrinkage constant, same default as opponent skew. Not tuned for
        EPA; downstream OLS coefficients absorb overall scale.
    :return: one row per (defteam, season, week) the defense played:
        epa_allowed_pass, epa_allowed_rush (positive = defense allows more
        EPA/play than league average, i.e. a softer matchup), n_games,
        n_plays_pass/rush. Values for week W use only games strictly before W.

    Pass plays use the corrected (sack-excluded) filter and run plays use
    play_type == 'run', both shared with aggregate_pbp.py.
    """
    pass_df = _trailing_shrunk(_weekly_residuals(_construct_passing_columns(pbp_df)), k)
    rush_df = _trailing_shrunk(_weekly_residuals(_construct_rushing_columns(pbp_df)), k)

    df = pass_df.rename({
        "epa_allowed": "epa_allowed_pass", "n_plays": "n_plays_pass",
    }).join(
        rush_df.rename({
            "epa_allowed": "epa_allowed_rush", "n_plays": "n_plays_rush", "n_games": "_n_games_rush",
        }),
        on=KEY,
        how="full",
        coalesce=True,
    ).with_columns(
        pl.coalesce("n_games", "_n_games_rush").alias("n_games"),
        pl.col("epa_allowed_pass").fill_null(0.0),
        pl.col("epa_allowed_rush").fill_null(0.0),
    ).drop("_n_games_rush").sort(KEY)

    return assert_unique_key(df, KEY, "epa_allowed")


def join_epa_allowed(stats_df: pl.DataFrame, epa_df: pl.DataFrame) -> pl.DataFrame:
    """
    Left-join epa_allowed onto player-week rows via opponent_team. Rows with
    no matching defense-week get 0.0 (league average).
    """
    df = stats_df.join(
        assert_unique_key(epa_df, KEY, "epa_allowed").select(
            KEY + ["epa_allowed_pass", "epa_allowed_rush"]
        ),
        left_on=["opponent_team", "season", "week"],
        right_on=KEY,
        how="left",
    ).with_columns(
        pl.col("epa_allowed_pass").fill_null(0.0),
        pl.col("epa_allowed_rush").fill_null(0.0),
    )

    return assert_no_fanout(stats_df, df, "join_epa_allowed")
