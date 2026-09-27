
from dataclasses import dataclass, field
from typing import Callable

import polars as pl

from common.frames import assert_no_fanout, assert_unique_key
from domain.scoring import ScoringSettings
from projections.expected_points.stat_vector.core import StatVectorModel


# Provisional — not yet calibrated against actual MAE per bucket. Once
# engine/metrics.py has real output, tune these so MAE is monotonically
# decreasing high < medium < low (see evaluate_projections' calibration check).
CONFIDENCE_TIER_THRESHOLDS = {
    "low": 4,
    "medium": 8,
}

KEY = ["gsis_id", "season", "week"]


@dataclass
class ExpectedPointsContext:
    """
    Everything an implementation may need beyond the stats rows. Fitted
    models are produced offline (stat_vector.core.fit_stat_vector) and passed
    in; this module never fits anything.
    """
    skew_df: pl.DataFrame
    epa_df: pl.DataFrame | None = None
    scoring: ScoringSettings | None = None
    stat_vector_models: dict[str, StatVectorModel] = field(default_factory=dict)
    # position -> rolling window the stats' rolling_avg_prior must have been
    # built with (add_rolling_features(window=...)); checked per row.
    rolling_windows: dict[str, int] = field(default_factory=lambda: dict(POSITION_ROLLING_WINDOWS))


# An implementation takes one position's player-week rows (full history,
# already joined with opponent_skew) and returns KEY + "projection".
Implementation = Callable[[pl.DataFrame, ExpectedPointsContext], pl.DataFrame]


def rolling_plus_skew(rows: pl.DataFrame, ctx: ExpectedPointsContext) -> pl.DataFrame:
    """
    projection = rolling_avg_prior + opponent_skew (the original formula),
    with rolling_avg_prior over the position's own window
    (ctx.rolling_windows; verified in calculate_expected_points).
    """
    return rows.select(KEY + [(pl.col("rolling_avg_prior") + pl.col("opponent_skew")).alias("projection")])


def stat_vector(rows: pl.DataFrame, ctx: ExpectedPointsContext) -> pl.DataFrame:
    """Predicted raw stat line scored by the league's settings (stat_vector/core.py)."""
    position = rows["position"][0]
    model = ctx.stat_vector_models.get(position)
    if model is None or ctx.epa_df is None or ctx.scoring is None:
        raise ValueError(
            f"{position} is mapped to 'stat_vector' but the context is missing a fitted "
            f"model for it, epa_df, or scoring. Fit one with fit_stat_vector(), or map "
            f"{position} to 'rolling_plus_skew' via position_implementations."
        )
    return model.predict_points(rows, ctx.epa_df, ctx.scoring).select(KEY + ["projection"])


# Swappable implementation behind a stable interface (VISION.md, Q-2): the
# implementation is chosen per position here, never by an inline branch.
# A position switches only after it is separately validated (README).
IMPLEMENTATIONS: dict[str, Implementation] = {
    "rolling_plus_skew": rolling_plus_skew,
    "stat_vector": stat_vector,
}

POSITION_IMPLEMENTATIONS: dict[str, str] = {
    "QB": "rolling_plus_skew",
    "RB": "stat_vector",
    "WR": "stat_vector",
    "TE": "stat_vector",
}
DEFAULT_IMPLEMENTATION = "rolling_plus_skew"

# Per-position continuous-window length (games) for rolling_avg_prior and the
# trailing volume features — config, not an implementation branch. Chosen by
# LOSO grid search over {8, 10, 12, 14, 16, 20} on mean MAE, 2019-2025
# (README "Rolling window length"). RB stays 8. RB and WR are on stat_vector,
# so their rolling_avg_prior only feeds skew and comparison baselines.
POSITION_ROLLING_WINDOWS: dict[str, int] = {
    "QB": 12,
    "RB": 8,
    "WR": 10,
    "TE": 12,
}
DEFAULT_ROLLING_WINDOW = 8


def calculate_expected_points(
        stats_df: pl.DataFrame,
        context: ExpectedPointsContext,
        position_implementations: dict[str, str] | None = None) -> pl.DataFrame:
    """
    :param stats_df: player-week stats, output of add_rolling_features() — must
        include gsis_id, season, week, opponent_team, position, rolling_avg_prior
        (and the raw stat columns, for stat-vector positions)
    :param context: skew_df (calculate_all_position_skews() output) plus, for
        stat-vector positions, epa_df, scoring, and fitted models
    :param position_implementations: per-call overrides of
        POSITION_IMPLEMENTATIONS, e.g. {"RB": "rolling_plus_skew"} to score
        the original formula for comparison
    :return: stats_df with games_this_season, opponent_skew,
        opponent_skew_n_games, projection, projection_method, confidence added

    opponent_skew is joined for every row (week, season, opponent_team/defense,
    position — position must be in the key or rows fan out, see CLAUDE.md),
    even where the position's implementation doesn't use it.

    Confidence is derived solely from player-side games_this_season — not
    opponent_skew's n_games, not the rolling window, not outcome volatility
    (that's boom/bust's job). See CLAUDE.md for the documented mid-season-return
    limitation and the defensive-scheme-continuity assumption this implies.
    """
    required = {"gsis_id", "season", "week", "opponent_team", "position", "rolling_avg_prior", "rolling_window"}
    missing = required - set(stats_df.columns)
    if missing:
        raise ValueError(f"stats_df missing required columns: {missing}")

    # The rolling window is per-position config held in the context; make sure
    # the stats were built with it rather than trusting the caller.
    expected_window = pl.col("position").replace_strict(
        context.rolling_windows, default=DEFAULT_ROLLING_WINDOW, return_dtype=pl.Int32)
    mismatched = stats_df.filter(pl.col("rolling_window") != expected_window)
    if mismatched.height:
        bad = mismatched.group_by("position").agg(pl.col("rolling_window").first()).rows()
        raise ValueError(
            f"rolling_avg_prior was built with windows {bad} but the context expects "
            f"{context.rolling_windows}; pass the same mapping to add_rolling_features()."
        )

    impls = {**POSITION_IMPLEMENTATIONS, **(position_implementations or {})}

    stats_df = stats_df.sort(["gsis_id", "season", "week"])

    stats_df = stats_df.with_columns(
        pl.col("week")
        .shift(1)
        .cum_count()
        .over(["gsis_id", "season"])
        .alias("games_this_season")
    )

    skew = assert_unique_key(
        context.skew_df.select(["defense", "week", "season", "position", "opponent_skew", "n_games"])
        .rename({"n_games": "opponent_skew_n_games"}),
        ["defense", "week", "season", "position"], "skew_df",
    )
    df = assert_no_fanout(stats_df, stats_df.join(
        skew,
        left_on=["week", "season", "opponent_team", "position"],
        right_on=["week", "season", "defense", "position"],
        how="left",
    ), "stats x skew")

    df = df.with_columns(
        pl.col("opponent_skew").fill_null(0.0)
    )

    parts = []
    for position in df["position"].unique().to_list():
        name = impls.get(position, DEFAULT_IMPLEMENTATION)
        rows = df.filter(pl.col("position") == position)
        parts.append(
            IMPLEMENTATIONS[name](rows, context).with_columns(pl.lit(name).alias("projection_method"))
        )
    projections = assert_unique_key(pl.concat(parts), KEY, "projections")

    df = assert_no_fanout(df, df.join(projections, on=KEY, how="left"), "stats x projections")

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
