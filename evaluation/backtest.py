
from dataclasses import dataclass

import polars as pl

from clients.nflreadpy.pbp_data import load_pbp_data
from clients.nflreadpy.player_data import load_player_stats
from clients.nflreadpy.team_data import pull_team_games
from domain.scoring import ScoringSettings
from engine.expected_points import (
    POSITION_IMPLEMENTATIONS, POSITION_ROLLING_WINDOWS, ExpectedPointsContext, calculate_expected_points,
)
from engine.scoring import calculate_points_vectorized
from projections.expected_points.features.epa_allowed import calculate_epa_allowed
from projections.expected_points.features.opponent_skew import calculate_all_position_skews
from projections.expected_points.features.player_rolling import add_rolling_features
from projections.expected_points.stat_vector.core import StatVectorSpec, fit_stat_vector
from projections.expected_points.stat_vector.specs import STAT_VECTOR_SPECS


# Backtest harness (offline evaluation only, never used at inference time).
# Leave-one-season-out: each full season is predicted by stat-vector models fit
# on the other full seasons; the in-progress season is a test-only fold fit on
# all full seasons. The first loaded season is history only (rolling windows).

HISTORY_SEASON = 2018
FULL_SEASONS = list(range(2019, 2026))
PARTIAL_SEASON = 2026
LOAD_SEASONS = [HISTORY_SEASON] + FULL_SEASONS + [PARTIAL_SEASON]
# Stat-vector feature windows live on each StatVectorSpec; rolling_avg_prior
# uses POSITION_ROLLING_WINDOWS.
STAT_VECTOR_WINDOW_GRID = [8, 10, 12, 14, 16, 20]


def select_by_policy(summaries: dict) -> object:
    """
    Model-selection policy (CLAUDE.md "Selection policy"): rank by mean
    Spearman, then mean R² (ties = within 0.001), then lower mean MAE.
    summaries: key -> {"spearman", "r2", "mae"} (LOSO means). Returns the key.
    """
    def rank(item):
        _, m = item
        return (round(m["spearman"], 3), round(m["r2"], 3), -m["mae"])
    return max(summaries.items(), key=rank)[0]

PBP_COLUMNS = ["season", "week", "defteam", "play_type", "sack", "epa",
               "yardline_100", "rushing_yards", "air_yards"]


@dataclass
class BacktestInputs:
    stats: pl.DataFrame      # player-weeks with fantasy_points + rolling features
    skew: pl.DataFrame
    epa: pl.DataFrame | None
    scoring: ScoringSettings
    rolling_windows: dict[str, int]


def load_inputs(
        scoring: ScoringSettings,
        seasons: list[int] = LOAD_SEASONS,
        rolling_windows: int | dict[str, int] = POSITION_ROLLING_WINDOWS,
        with_epa: bool = True,
        base_stats: pl.DataFrame | None = None) -> BacktestInputs:
    """
    :param with_epa: skip the play-by-play load when only rolling_plus_skew is evaluated
    :param base_stats: already-scored player stats to reuse (e.g. across a window grid)
    """
    stats = base_stats if base_stats is not None else \
        calculate_points_vectorized(load_player_stats(seasons), scoring)
    stats = add_rolling_features(stats, window=rolling_windows)

    skew = calculate_all_position_skews(stats, pull_team_games(seasons))

    epa = None
    if with_epa:
        pbp = pl.concat([load_pbp_data(s).select(PBP_COLUMNS) for s in seasons], how="vertical_relaxed")
        epa = calculate_epa_allowed(pbp)

    windows = rolling_windows if isinstance(rolling_windows, dict) else \
        {p: rolling_windows for p in stats["position"].unique().drop_nulls().to_list()}
    return BacktestInputs(stats=stats, skew=skew, epa=epa, scoring=scoring, rolling_windows=windows)


def loso_folds(full_seasons: list[int] = FULL_SEASONS, partial: int | None = PARTIAL_SEASON) -> list[tuple]:
    """[(label, train_seasons, test_season)]"""
    folds = [(f"{s}", [o for o in full_seasons if o != s], s) for s in full_seasons]
    if partial is not None:
        folds.append((f"{partial} partial", list(full_seasons), partial))
    return folds


def predict_fold(
        inputs: BacktestInputs,
        train_seasons: list[int],
        test_season: int,
        position_implementations: dict[str, str] | None = None,
        specs: dict[str, StatVectorSpec] = STAT_VECTOR_SPECS,
        window: int | None = None) -> tuple[pl.DataFrame, dict]:
    """
    Fit a stat-vector model on train_seasons for every position mapped to
    'stat_vector', run the engine, return (test-season predictions, models).
    window: None -> each spec's own window.
    """
    impls = {**POSITION_IMPLEMENTATIONS, **(position_implementations or {})}
    models = {
        pos: fit_stat_vector(inputs.stats, inputs.epa, specs[pos], train_seasons, window=window)
        for pos, name in impls.items() if name == "stat_vector"
    }
    ctx = ExpectedPointsContext(skew_df=inputs.skew, epa_df=inputs.epa, scoring=inputs.scoring,
                                stat_vector_models=models, rolling_windows=inputs.rolling_windows)
    preds = calculate_expected_points(inputs.stats, ctx, position_implementations)
    return preds.filter(pl.col("season") == test_season), models
