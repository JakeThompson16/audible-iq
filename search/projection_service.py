"""
Lookup-and-project service: one call from a gsis_id (e.g. a PlayerIndex
selection) to everything a projection card needs. No UI code.

    project_player("00-0033873")                   # next unplayed game
    project_player("00-0033873", season=2025, week=10)

A thin layer over pipeline.predict.predict_player_stats: same statuses and
reasons (ok | bye | no_game_scheduled | no_history | unknown_player |
unsupported_position), same stats keys, same model artifacts, never zeros for
missing data. It adds schedule context the prediction dict doesn't carry
(home/away, game date, whether the game is final) and, for a final game, the
player's actual stats for that week under the same keys. The feature context
and artifacts come from the pipeline's process-wide caches; nothing here
reloads nflreadpy per call.

Default target: the player's team's next unplayed scheduled game. A bye week
has no schedule row, so the next unplayed game is the following week; the
week number is always returned so the UI can label it.
"""
import polars as pl

from pipeline.artifacts import STAT_VECTOR_DIR
from pipeline.context import FeatureContext, get_context
from pipeline.predict import predict_player_stats


def _actuals(ctx: FeatureContext, gsis_id: str, season: int, week: int, keys: list[str]) -> dict | None:
    row = ctx.stats.filter((pl.col("gsis_id") == gsis_id) & (pl.col("season") == season) & (pl.col("week") == week))
    if row.height == 0:
        return None
    r = row.row(0, named=True)
    return {k: (float(r[k]) if r.get(k) is not None else None) for k in keys if k in r}


def project_player(gsis_id: str, season: int | None = None, week: int | None = None, *,
                   context: FeatureContext | None = None, artifacts_dir=STAT_VECTOR_DIR) -> dict:
    """
    :return: predict_player_stats' JSON-serializable dict, with
        target += home_away ("home" / "away" / None), gameday, game_final;
        actual = the player's stats for a final game (same keys as `stats`),
        or None with actual_note explaining why.
    """
    ctx = context or get_context()
    result = predict_player_stats(gsis_id, gsis_id=True, season=season, week=week,
                                  context=ctx, artifacts_dir=artifacts_dir)
    result = {**result, "actual": None, "actual_note": None}
    target = result.get("target")
    if not target:
        return result

    team = result["player"]["team"]
    game = ctx.schedule.filter((pl.col("team") == team) & (pl.col("season") == target["season"])
                               & (pl.col("week") == target["week"]))
    if game.height == 0:
        result["target"] = {**target, "home_away": None, "gameday": None, "game_final": False}
        return result

    g = game.row(0, named=True)
    final = bool(g["completed"])
    result["target"] = {**target, "home_away": "home" if g["home"] else "away",
                        "gameday": g["gameday"], "game_final": final}
    if final:
        keys = list(result["stats"]) if result.get("stats") else []
        if keys:
            actual = _actuals(ctx, gsis_id, target["season"], target["week"], keys)
            result["actual"] = actual
            if actual is None:
                result["actual_note"] = "game is final but the player has no stats row (did not record a stat)"
    return result
