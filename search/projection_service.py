"""
Lookup-and-project service: one call from a gsis_id (e.g. a PlayerIndex
selection) to a predicted stat line for a game. No UI code.

    project_player("00-0033873")                   # next unplayed game
    project_player("00-0033873", season=2025, week=10)

A thin layer over pipeline.predict.predict_player_stats: same statuses, same
stats keys, same model artifacts. It only adds the schedule context the
prediction dict doesn't carry (home/away, game date).
"""
import polars as pl

from pipeline.artifacts import STAT_VECTOR_DIR
from pipeline.context import FeatureContext, get_context
from pipeline.predict import predict_player_stats


def project_player(gsis_id: str, season: int | None = None, week: int | None = None, *,
                   context: FeatureContext | None = None, artifacts_dir=STAT_VECTOR_DIR) -> dict:
    """
    :return: predict_player_stats' JSON-serializable dict, with target extended
        by home_away ("home" / "away", or None without a game) and gameday.
    """
    ctx = context or get_context()
    result = predict_player_stats(gsis_id, gsis_id=True, season=season, week=week,
                                  context=ctx, artifacts_dir=artifacts_dir)
    target = result.get("target")
    if target:
        team = result["player"]["team"]
        game = ctx.schedule.filter((pl.col("team") == team) & (pl.col("season") == target["season"])
                                   & (pl.col("week") == target["week"]))
        if game.height:
            row = game.row(0, named=True)
            target = {**target, "home_away": "home" if row["home"] else "away", "gameday": row["gameday"]}
        else:
            target = {**target, "home_away": None, "gameday": None}
        result = {**result, "target": target}
    return result
