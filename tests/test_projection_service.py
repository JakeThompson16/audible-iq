"""project_player: the card-level service over predict_player_stats."""
import json

import polars as pl
import pytest

from pipeline.predict import predict_player_stats
from search.player_search import build_player_index
from search.projection_service import project_player


@pytest.fixture(scope="session")
def mahomes(ctx):
    idx = build_player_index(context=ctx)
    return idx.suggest("patrick mahomes")[0]


def _played_weeks(ctx, team, season):
    return ctx.schedule.filter((pl.col("team") == team) & (pl.col("season") == season)
                               & (pl.col("game_type") == "REG"))["week"].to_list()


def test_default_target_is_pipeline_next_game(ctx, trained, mahomes):
    directory, _ = trained
    card = project_player(mahomes["gsis_id"], context=ctx, artifacts_dir=directory)
    pred = predict_player_stats(mahomes["gsis_id"], gsis_id=True, context=ctx, artifacts_dir=directory)
    for k in ("season", "week", "opponent"):
        assert card["target"][k] == pred["target"][k]
    assert card["target"]["home_away"] in ("home", "away")
    assert card["target"]["game_final"] is False and card["actual"] is None
    assert card["in_training_window"] is False            # next game is after data_through
    json.dumps(card)


def test_final_game_includes_actuals(ctx, trained, mahomes):
    directory, _ = trained
    week = next(w for w in range(9, 18) if w in _played_weeks(ctx, mahomes["team"], 2025))
    card = project_player(mahomes["gsis_id"], season=2025, week=week, context=ctx, artifacts_dir=directory)
    assert card["status"] == "ok" and card["target"]["game_final"] is True
    assert card["actual"] is not None and set(card["actual"]) == set(card["stats"])
    assert card["in_training_window"] is True             # 2025 is inside the production training window
    row = ctx.stats.filter((pl.col("gsis_id") == mahomes["gsis_id"]) & (pl.col("season") == 2025)
                           & (pl.col("week") == week))
    assert card["actual"]["passing_yards"] == float(row["passing_yards"][0])
    json.dumps(card)


def test_bye_week(ctx, trained, mahomes):
    directory, _ = trained
    bye = next(w for w in range(5, 15) if w not in _played_weeks(ctx, mahomes["team"], 2025))
    card = project_player(mahomes["gsis_id"], season=2025, week=bye, context=ctx, artifacts_dir=directory)
    assert card["status"] == "bye" and card["reason"]
    assert card["stats"] is None and card["target"]["week"] == bye and card["target"]["home_away"] is None
    json.dumps(card)


def test_rookie_first_game(ctx, trained):
    directory, _ = trained
    idx = build_player_index(context=ctx)
    first = (ctx.stats.group_by("gsis_id").agg(pl.col("season").min().alias("s"), pl.col("week").min().alias("w"))
             .filter(pl.col("s") == 2026))
    for gsis, week in first.select(["gsis_id", "w"]).iter_rows():
        if idx.get(gsis):
            card = project_player(gsis, season=2026, week=week, context=ctx, artifacts_dir=directory)
            assert card["status"] == "no_history" and card["stats"] is None and card["reason"]
            json.dumps(card)
            return
    pytest.skip("no indexed 2026 rookie found")


def test_unknown_player(ctx, trained):
    directory, _ = trained
    card = project_player("00-9999999", context=ctx, artifacts_dir=directory)
    assert card["status"] == "unknown_player" and card["stats"] is None
    json.dumps(card)
