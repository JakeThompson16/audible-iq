"""
Shared, session-scoped fixtures. Everything heavy (nflverse loads, fits, one
retrain) happens once per test session.
"""
import polars as pl
import pytest

from evaluation.backtest import FULL_SEASONS, load_inputs
from pipeline.context import get_context
from pipeline.train import default_metrics_scoring, train_all
from projections.expected_points.stat_vector.core import build_features, fit_stat_vector
from projections.expected_points.stat_vector.specs import STAT_VECTOR_SPECS

POSITIONS = ["QB", "RB", "WR", "TE"]
PARITY_SEASONS = [2024, 2025]


@pytest.fixture(scope="session")
def scoring():
    return default_metrics_scoring(2026)


@pytest.fixture(scope="session")
def ctx():
    """The prediction pipeline's own context (2020 onward)."""
    return get_context()


@pytest.fixture(scope="session")
def harness(scoring):
    """The backtest harness inputs (2018 onward), exactly as test.py builds them."""
    return load_inputs(scoring)


@pytest.fixture(scope="session")
def fold_models(harness):
    """season -> {position -> model fit the way the backtest fits that held-out season}."""
    out = {}
    for season in PARITY_SEASONS:
        train = [s for s in FULL_SEASONS if s != season]
        out[season] = {pos: fit_stat_vector(harness.stats, harness.epa, STAT_VECTOR_SPECS[pos], train)
                       for pos in POSITIONS}
    return out


@pytest.fixture(scope="session")
def harness_lines(harness, fold_models):
    """(season, position) -> the harness's own predicted stat lines for every row of that position."""
    out = {}
    for season, models in fold_models.items():
        for pos, model in models.items():
            feats = build_features(harness.stats.filter(pl.col("position") == pos), harness.epa, model.window)
            out[(season, pos)] = model.predict_stat_vector(feats).filter(pl.col("season") == season)
    return out


@pytest.fixture(scope="session")
def trained(tmp_path_factory, ctx, scoring):
    """One full retrain written to a temp artifacts dir (never the repo's)."""
    directory = tmp_path_factory.mktemp("artifacts") / "stat_vector"
    result = train_all(save=True, artifacts_dir=directory, context=ctx, metrics_scoring=scoring)
    return directory, result
