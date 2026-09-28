"""
Future-prediction pipeline: a predicted stat line per player for a target game.

    predict_player_stats("4984")                          # Sleeper id, next game
    predict_player_stats("00-0034796", gsis_id=True)      # GSIS id, next game
    predict_player_stats("4984", season=2025, week=10)    # as of a past game
    predict_many(["4984", "6794"], season=2026, week=4)

As-of semantics: only data strictly before the target game is used. A
placeholder row for the target game is appended to the player's history and
run through the same feature builders the backtest uses (shift(1) makes the
placeholder's own stats irrelevant); epa_allowed for a defense-week that hasn't
been played is the value after that defense's last game. So a past week can
be predicted exactly as it would have been at the time, which is what the
parity tests rely on.

Returns JSON-serializable dicts; never zeros for missing data. `status` is
one of ok | bye | no_game_scheduled | no_history | unknown_player |
unsupported_position.
"""
import math
from functools import lru_cache
from pathlib import Path

import polars as pl

from clients.nflreadpy.player_data import load_player_metadata
from engine.expected_points import confidence_tier
from pipeline.artifacts import STAT_VECTOR_DIR, artifact_path, load_model, read_artifact
from pipeline.context import FeatureContext, get_context
from projections.expected_points.features.epa_allowed import epa_allowed_from_weekly
from projections.expected_points.stat_vector.core import StatVectorModel, build_features, unpredicted_categories
from projections.expected_points.stat_vector.specs import STAT_VECTOR_SPECS

SUPPORTED_POSITIONS = ("QB", "RB", "WR", "TE")
PLAYER_KEY = ["gsis_id", "season", "week"]


# ------------------------------------------------------------------ ids

def normalize_id(value) -> str | None:
    """
    IDs are compared as strings. The crosswalk may hold ints or floats
    (4984 / 4984.0) while Sleeper sends "4984"; normalize at the boundary.
    """
    if value is None:
        return None
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        if value.is_integer():
            return str(int(value))
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text or None


@lru_cache(maxsize=None)
def _player_maps(cutoff: int) -> tuple[dict, dict]:
    """(sleeper -> gsis, gsis -> {sleeper_id, name, position}) from load_player_metadata (deduped, asserted)."""
    meta = load_player_metadata(cutoff)
    sleeper_to_gsis, gsis_info = {}, {}
    for gsis, sleeper, name, position in meta.select(["gsis_id", "sleeper_id", "display_name", "position"]).iter_rows():
        g, s = normalize_id(gsis), normalize_id(sleeper)
        gsis_info[g] = {"sleeper_id": s, "name": name, "position": position}
        if s is not None:
            sleeper_to_gsis.setdefault(s, g)
    return sleeper_to_gsis, gsis_info


def refresh_player_maps() -> None:
    _player_maps.cache_clear()


def _maps(ctx: FeatureContext) -> tuple[dict, dict]:
    return _player_maps(min(ctx.seasons) - 1)


# ------------------------------------------------------------------ models

_MODELS: dict[tuple, StatVectorModel] = {}


def _production_model(position: str, directory) -> tuple[StatVectorModel, dict]:
    """Artifact-backed model, cached per file version. Spec mismatch raises (never applied silently)."""
    path = artifact_path(position, directory)
    key = (position, str(Path(directory)), path.stat().st_mtime_ns if path.exists() else None)
    if key not in _MODELS:
        _MODELS[key] = load_model(position, directory)
    art = read_artifact(position, directory)
    return _MODELS[key], {"fit_timestamp": art.get("fit_timestamp"), "data_through": art.get("data_through")}


def refresh_models() -> None:
    _MODELS.clear()


# ------------------------------------------------------------------ team / target

def _before(season: int, week: int) -> pl.Expr:
    return (pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") < week))


def _team_as_of(ctx: FeatureContext, gsis: str, season: int | None, week: int | None) -> tuple[str | None, str]:
    """
    Team for a game: nflverse weekly roster at the latest week <= target in the
    target season (pre-game information), else nflverse latest_team (current
    season only), else the team on the player's most recent stats row before
    the target. The ff_playerids crosswalk team is never used (different
    abbreviations and stale after trades). Returns (team, source).
    """
    current = max(ctx.seasons)
    season = current if season is None else season
    week = 99 if week is None else week
    r = ctx.rosters.filter((pl.col("gsis_id") == gsis) & (pl.col("season") == season) & (pl.col("week") <= week))
    if r.height:
        return r.sort("week")["team"][-1], "weekly_roster"
    if season == current:
        lt = ctx.latest_teams.filter(pl.col("gsis_id") == gsis)
        if lt.height and lt["latest_team"][0]:
            return lt["latest_team"][0], "players.latest_team"
    s = ctx.stats.filter((pl.col("gsis_id") == gsis) & _before(season, week + 1)).sort(["season", "week"])
    if s.height:
        return s["team"][-1], "last_stats_row"
    return None, "none"


def _target_game(ctx, team, season, week) -> tuple[str, str | None, dict | None]:
    """(status, reason, target) for the team's game; target = {season, week, opponent}."""
    sched = ctx.schedule
    if season is None:
        upcoming = sched.filter((pl.col("team") == team) & ~pl.col("completed")).sort(["season", "week"])
        if upcoming.height == 0:
            return "no_game_scheduled", f"{team} has no unplayed game on the loaded schedule", None
        g = upcoming.row(0, named=True)
        return "ok", None, {"season": g["season"], "week": g["week"], "opponent": g["opponent"]}

    season_games = sched.filter(pl.col("season") == season)
    if season_games.height == 0 or week > season_games["week"].max() or week < 1:
        return "no_game_scheduled", f"no {season} week {week} on the loaded schedule", None
    g = season_games.filter((pl.col("team") == team) & (pl.col("week") == week))
    if g.height == 0:
        reg_weeks = season_games.filter(pl.col("game_type") == "REG")["week"]
        if reg_weeks.len() and week <= reg_weeks.max():
            return "bye", f"{team} has no game in {season} week {week}", {"season": season, "week": week, "opponent": None}
        return "no_game_scheduled", f"{team} has no game in {season} week {week}", {"season": season, "week": week, "opponent": None}
    return "ok", None, {"season": season, "week": week, "opponent": g["opponent"][0]}


# ------------------------------------------------------------------ core

def _result(player, target, status, reason, **extra) -> dict:
    out = {
        "player": player,
        "target": target,
        "status": status,
        "reason": reason,
        "stats": None,
        "volume_rate_detail": None,
        "games_this_season": None,
        "confidence": None,
        "projection_method": "stat_vector",
        "model_version": None,
        "unpredicted": None,
    }
    out.update(extra)
    return out


def _confidence(games: int) -> str:
    return pl.DataFrame({"g": [games]}).select(confidence_tier("g")).item()


def predict_many(ids, gsis_id: bool = False, season: int | None = None, week: int | None = None, *,
                 context: FeatureContext | None = None, models: dict[str, StatVectorModel] | None = None,
                 artifacts_dir=STAT_VECTOR_DIR) -> list[dict]:
    """
    Predicted stat lines for many players, one dict per id in input order.

    :param gsis_id: False -> ids are Sleeper ids; True -> GSIS ids
    :param season, week: target game; both None -> each player's team's next unplayed game
    :param models: position -> fitted StatVectorModel, overriding the production
        artifacts (used by the parity tests with backtest-fold models)
    """
    if (season is None) != (week is None):
        raise ValueError("pass both season and week, or neither")
    ctx = context or get_context()
    sleeper_to_gsis, gsis_info = _maps(ctx)

    results: list[dict | None] = [None] * len(ids)
    pending = []  # (index, player, target, gsis, position)
    for i, raw in enumerate(ids):
        pid = normalize_id(raw)
        gsis = pid if gsis_id else sleeper_to_gsis.get(pid)
        info = gsis_info.get(gsis) if gsis else None
        if info is None:
            results[i] = _result({"gsis_id": gsis if gsis_id else None, "sleeper_id": None if gsis_id else pid,
                                  "name": None, "position": None, "team": None},
                                 None, "unknown_player", f"id {pid!r} not in the player crosswalk")
            continue

        rows = ctx.stats.filter(pl.col("gsis_id") == gsis)
        position = rows.sort(["season", "week"])["position"][-1] if rows.height else info["position"]
        team, team_source = _team_as_of(ctx, gsis, season, week)
        player = {"gsis_id": gsis, "sleeper_id": info["sleeper_id"], "name": info["name"],
                  "position": position, "team": team, "team_source": team_source}

        if position not in SUPPORTED_POSITIONS:
            results[i] = _result(player, None, "unsupported_position",
                                 f"position {position!r}; supported: {', '.join(SUPPORTED_POSITIONS)}")
            continue
        if team is None:
            results[i] = _result(player, None, "no_game_scheduled", "no current team found for this player")
            continue
        status, reason, target = _target_game(ctx, team, season, week)
        if status != "ok":
            results[i] = _result(player, target, status, reason)
            continue
        history = rows.filter(_before(target["season"], target["week"]))
        if history.height == 0:
            results[i] = _result(player, target, "no_history", "no games before the target game")
            continue
        pending.append((i, player, target, gsis, position, history))

    # The same player listed twice shares one computation (duplicated history
    # rows would double-count in the trailing windows).
    first_index = {}
    duplicates = []
    unique_pending = []
    for p in pending:
        if p[3] in first_index:
            duplicates.append((p[0], first_index[p[3]]))
        else:
            first_index[p[3]] = p[0]
            unique_pending.append(p)
    pending = unique_pending

    for position in SUPPORTED_POSITIONS:
        batch = [p for p in pending if p[4] == position]
        if not batch:
            continue
        if models is not None:
            model, version = models[position], {"fit_timestamp": None, "data_through": None,
                                                 "source": "in-memory model (not a production artifact)"}
        else:
            model, version = _production_model(position, artifacts_dir)

        placeholders = pl.DataFrame({
            "gsis_id": [b[3] for b in batch],
            "season": [b[2]["season"] for b in batch],
            "week": [b[2]["week"] for b in batch],
            "position": [position] * len(batch),
            "team": [b[1]["team"] for b in batch],
            "opponent_team": [b[2]["opponent"] for b in batch],
            "_target": [True] * len(batch),
        }).with_columns(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))
        frame = pl.concat([b[5] for b in batch] + [placeholders], how="diagonal_relaxed")

        epa = epa_allowed_from_weekly(
            ctx.pass_weekly, ctx.rush_weekly,
            as_of_keys=placeholders.select(pl.col("opponent_team").alias("defteam"), "season", "week"),
        )
        feats = build_features(frame, epa, model.window).filter(pl.col("_target").fill_null(False))
        vec = model.predict_stat_vector(feats)
        spec = STAT_VECTOR_SPECS[position]

        for i, player, target, gsis, _, history in batch:
            row = vec.filter(pl.col("gsis_id") == gsis).row(0, named=True)
            games = history.filter(pl.col("season") == target["season"]).height
            common = dict(games_this_season=games, confidence=_confidence(games), model_version=version,
                          unpredicted=unpredicted_categories(spec))
            stats = {c: row[c] for c in spec.recomposed_columns}
            if any(v is None for v in stats.values()):
                results[i] = _result(player, target, "no_history",
                                     "not enough prior games for the trailing features", **common)
                continue
            results[i] = _result(player, target, "ok", None, stats=stats,
                                 volume_rate_detail={t: row[f"pred_{t}"] for t in spec.models}, **common)
    for dup, original in duplicates:
        results[dup] = results[original]
    return results


def predict_player_stats(player_id, gsis_id: bool = False, season: int | None = None, week: int | None = None,
                         **kwargs) -> dict:
    """Single-player wrapper around predict_many (same code path)."""
    return predict_many([player_id], gsis_id=gsis_id, season=season, week=week, **kwargs)[0]
