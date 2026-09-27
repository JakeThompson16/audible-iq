
from dataclasses import dataclass

import polars as pl

from clients.nflreadpy.pbp_data import load_pbp_data
from clients.nflreadpy.player_data import load_player_stats
from clients.nflreadpy.team_data import pull_team_games
from domain.scoring import ScoringSettings
from engine.expected_points import (
    POSITION_IMPLEMENTATIONS, ExpectedPointsContext, calculate_expected_points,
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
WINDOW = 8

PBP_COLUMNS = ["season", "week", "defteam", "play_type", "sack", "epa",
               "yardline_100", "rushing_yards", "air_yards"]


@dataclass
class BacktestInputs:
    stats: pl.DataFrame      # player-weeks with fantasy_points + rolling features
    skew: pl.DataFrame
    epa: pl.DataFrame
    scoring: ScoringSettings


def load_inputs(scoring: ScoringSettings, seasons: list[int] = LOAD_SEASONS, window: int = WINDOW) -> BacktestInputs:
    stats = load_player_stats(seasons)
    stats = calculate_points_vectorized(stats, scoring)
    stats = add_rolling_features(stats, window=window)

    skew = calculate_all_position_skews(stats, pull_team_games(seasons))

    pbp = pl.concat([load_pbp_data(s).select(PBP_COLUMNS) for s in seasons], how="vertical_relaxed")
    epa = calculate_epa_allowed(pbp)

    return BacktestInputs(stats=stats, skew=skew, epa=epa, scoring=scoring)


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
        window: int = WINDOW) -> tuple[pl.DataFrame, dict]:
    """
    Fit a stat-vector model on train_seasons for every position mapped to
    'stat_vector', run the engine, return (test-season predictions, models).
    """
    impls = {**POSITION_IMPLEMENTATIONS, **(position_implementations or {})}
    models = {
        pos: fit_stat_vector(inputs.stats, inputs.epa, specs[pos], train_seasons, window=window)
        for pos, name in impls.items() if name == "stat_vector"
    }
    ctx = ExpectedPointsContext(skew_df=inputs.skew, epa_df=inputs.epa, scoring=inputs.scoring,
                                stat_vector_models=models)
    preds = calculate_expected_points(inputs.stats, ctx, position_implementations)
    return preds.filter(pl.col("season") == test_season), models
