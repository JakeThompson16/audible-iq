"""ID resolution and status paths of the prediction API (production-style artifacts from a temp retrain)."""
import json

import polars as pl

from pipeline.predict import _maps, normalize_id, predict_many, predict_player_stats


def _sample_players(ctx, n=5):
    """Active 2025 skill players with a Sleeper id, one or two per position."""
    sleeper_to_gsis, gsis_info = _maps(ctx)
    rows = (ctx.stats.filter((pl.col("season") == 2025) & (pl.col("week") == 10)
                             & pl.col("position").is_in(["QB", "RB", "WR", "TE"]))
            .sort(pl.col("carries") + pl.col("targets") + pl.col("attempts"), descending=True))
    out = []
    for gsis in rows["gsis_id"]:
        info = gsis_info.get(gsis)
        if info and info["sleeper_id"]:
            out.append((gsis, info["sleeper_id"]))
        if len(out) == n:
            break
    return out


def test_normalize_id():
    assert normalize_id(4984) == normalize_id("4984") == normalize_id(4984.0) == normalize_id(" 4984.0 ") == "4984"
    assert normalize_id(None) is None
    assert normalize_id(float("nan")) is None


def test_sleeper_and_gsis_paths_agree(ctx, trained):
    directory, _ = trained
    players = _sample_players(ctx)
    assert len(players) == 5
    for gsis, sleeper in players:
        by_str = predict_player_stats(str(sleeper), season=2025, week=11, context=ctx, artifacts_dir=directory)
        by_int = predict_player_stats(int(sleeper), season=2025, week=11, context=ctx, artifacts_dir=directory)
        by_gsis = predict_player_stats(gsis, gsis_id=True, season=2025, week=11, context=ctx, artifacts_dir=directory)
        assert by_str["player"]["gsis_id"] == gsis
        assert by_str == by_int == by_gsis
        assert by_str["status"] in ("ok", "bye"), by_str["reason"]
        json.dumps(by_str)  # JSON-serializable


def test_unknown_id(ctx, trained):
    directory, _ = trained
    r = predict_player_stats("not-a-real-id-123", context=ctx, artifacts_dir=directory)
    assert r["status"] == "unknown_player" and r["stats"] is None


def test_ok_shape(ctx, trained):
    directory, _ = trained
    gsis, _ = _sample_players(ctx, 1)[0]
    r = predict_player_stats(gsis, gsis_id=True, season=2025, week=12, context=ctx, artifacts_dir=directory)
    if r["status"] == "ok":
        assert r["projection_method"] == "stat_vector"
        assert set(r["model_version"]) >= {"fit_timestamp", "data_through"}
        assert r["confidence"] in ("high", "medium", "low", "insufficient_data")
        assert all(isinstance(v, float) for v in r["stats"].values())
        assert "fumbles_lost" not in r["stats"] and "rushing_2pt_conversions" not in r["stats"]
        assert "fum_lost" in r["unpredicted"]


def test_bye(ctx, trained):
    directory, _ = trained
    reg = ctx.schedule.filter((pl.col("season") == 2025) & (pl.col("game_type") == "REG"))
    played = set(reg.select(["team", "week"]).iter_rows())
    team, week = next((t, w) for t in sorted(reg["team"].unique()) for w in range(5, 15) if (t, w) not in played)
    _, gsis_info = _maps(ctx)
    on_team = (ctx.stats.filter((pl.col("season") == 2025) & (pl.col("team") == team) & (pl.col("week") < week)
                                & pl.col("position").is_in(["QB", "RB", "WR", "TE"]))
               .sort(pl.col("carries") + pl.col("targets") + pl.col("attempts"), descending=True))
    gsis = next(g for g in on_team["gsis_id"] if g in gsis_info)
    r = predict_player_stats(gsis, gsis_id=True, season=2025, week=week, context=ctx, artifacts_dir=directory)
    assert r["status"] == "bye", r
    assert r["stats"] is None


def test_no_history_rookie(ctx, trained):
    directory, _ = trained
    first = (ctx.stats.filter(pl.col("position").is_in(["QB", "RB", "WR", "TE"]))
             .group_by("gsis_id").agg(pl.col("season").min().alias("s"), pl.col("week").min().alias("w")))
    rookies = ctx.stats.join(first, on="gsis_id").filter(
        (pl.col("s") == 2025) & (pl.col("season") == 2025) & (pl.col("week") == pl.col("w")))
    for gsis, week in rookies.select(["gsis_id", "week"]).iter_rows():
        r = predict_player_stats(gsis, gsis_id=True, season=2025, week=week, context=ctx, artifacts_dir=directory)
        if r["status"] != "unknown_player":
            assert r["status"] == "no_history", r
            assert r["stats"] is None
            return
    raise AssertionError("no 2025 rookie found in the crosswalk")


def test_unsupported_position(ctx, trained):
    directory, _ = trained
    _, gsis_info = _maps(ctx)
    kicker = next(g for g, info in gsis_info.items() if info["position"] in ("K", "PK"))
    r = predict_player_stats(kicker, gsis_id=True, season=2025, week=5, context=ctx, artifacts_dir=directory)
    assert r["status"] == "unsupported_position", r


def test_week_beyond_schedule(ctx, trained):
    directory, _ = trained
    gsis, _ = _sample_players(ctx, 1)[0]
    r = predict_player_stats(gsis, gsis_id=True, season=2025, week=30, context=ctx, artifacts_dir=directory)
    assert r["status"] == "no_game_scheduled", r


def test_batch_matches_single(ctx, trained):
    directory, _ = trained
    players = [g for g, _ in _sample_players(ctx, 5)]
    batch = predict_many(players + [players[0]], gsis_id=True, season=2025, week=11,
                         context=ctx, artifacts_dir=directory)
    singles = [predict_player_stats(g, gsis_id=True, season=2025, week=11, context=ctx, artifacts_dir=directory)
               for g in players]
    assert batch[:5] == singles
    assert batch[5] == batch[0]  # duplicate id shares the first result
