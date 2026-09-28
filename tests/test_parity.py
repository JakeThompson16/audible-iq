"""
Parity: the prediction pipeline, given the model the backtest fit for a
held-out season, must reproduce the backtest harness's own stat line for that
player-week to floating-point tolerance. (Not compared to the production
artifact, which trained on these weeks.)
"""
import math

import polars as pl
import pytest

from engine.expected_points import ExpectedPointsContext, calculate_expected_points
from engine.scoring import calculate_points_vectorized
from pipeline.predict import predict_many
from projections.expected_points.stat_vector.specs import STAT_VECTOR_SPECS

from conftest import PARITY_SEASONS, POSITIONS

TOL = 1e-9


def _sample_targets(harness, ctx, season, pos, lines):
    """week 1, mid-season (week 9) and post-bye player-weeks that the harness can project."""
    rows = (harness.stats.filter((pl.col("season") == season) & (pl.col("position") == pos))
            .with_columns((pl.col("carries") + pl.col("targets") + pl.col("attempts")).alias("_vol"))
            .join(lines.filter(pl.col("carries").is_not_null()).select(["gsis_id", "season", "week"]),
                  on=["gsis_id", "season", "week"]))
    picks = []
    for week in (1, 9):
        wk = rows.filter(pl.col("week") == week).sort("_vol", descending=True).head(2)
        picks += [(g, week, "week 1" if week == 1 else "mid-season") for g in wk["gsis_id"]]

    reg = ctx.schedule.filter((pl.col("season") == season) & (pl.col("game_type") == "REG"))
    played = set(reg.select(["team", "week"]).iter_rows())
    post_bye = []
    for team, week in sorted(played):
        if 5 <= week <= 14 and (team, week - 1) not in played:
            cand = rows.filter((pl.col("team") == team) & (pl.col("week") == week)).sort("_vol", descending=True)
            if cand.height:
                post_bye.append((cand["gsis_id"][0], week, "post-bye"))
        if len(post_bye) == 2:
            break
    return picks + post_bye


def _parity_cases(harness, ctx, harness_lines):
    cases = []
    for season in PARITY_SEASONS:
        for pos in POSITIONS:
            for gsis, week, kind in _sample_targets(harness, ctx, season, pos, harness_lines[(season, pos)]):
                cases.append((season, pos, gsis, week, kind))
    return cases


def test_parity_with_backtest_harness(harness, ctx, fold_models, harness_lines):
    cases = _parity_cases(harness, ctx, harness_lines)
    assert len(cases) >= 30, f"too few parity cases sampled: {len(cases)}"
    kinds = {c[4] for c in cases}
    assert kinds == {"week 1", "mid-season", "post-bye"}

    mismatches, compared = [], 0
    for season, pos, gsis, week, kind in cases:
        result = predict_many([gsis], gsis_id=True, season=season, week=week,
                              context=ctx, models=fold_models[season])[0]
        expected = harness_lines[(season, pos)].filter((pl.col("gsis_id") == gsis) & (pl.col("week") == week)).row(0, named=True)
        actual_row = harness.stats.filter((pl.col("gsis_id") == gsis) & (pl.col("season") == season) & (pl.col("week") == week))
        if result["status"] != "ok":
            mismatches.append((season, pos, gsis, week, kind, f"status {result['status']}: {result['reason']}"))
            continue
        if result["target"]["opponent"] != actual_row["opponent_team"][0]:
            mismatches.append((season, pos, gsis, week, kind,
                               f"opponent {result['target']['opponent']} (team {result['player']['team']} via "
                               f"{result['player']['team_source']}) vs actual {actual_row['opponent_team'][0]}"))
            continue
        for col in STAT_VECTOR_SPECS[pos].recomposed_columns:
            compared += 1
            a, b = result["stats"][col], expected[col]
            if not math.isclose(a, b, rel_tol=TOL, abs_tol=TOL):
                mismatches.append((season, pos, gsis, week, kind, f"{col}: pipeline {a!r} vs harness {b!r}"))
    print(f"\nparity: {len(cases)} player-weeks, {compared} stat values compared, {len(mismatches)} mismatches")
    assert not mismatches, "\n".join(map(str, mismatches))


def test_round_trip_points(harness, ctx, fold_models, harness_lines, scoring):
    """pipeline dict -> one-row frame -> calculate_points_vectorized == the engine's projection."""
    cases = [c for c in _parity_cases(harness, ctx, harness_lines) if c[4] == "mid-season"]
    checked = 0
    for season in PARITY_SEASONS:
        engine = calculate_expected_points(
            harness.stats,
            ExpectedPointsContext(skew_df=harness.skew, epa_df=harness.epa, scoring=scoring,
                                  stat_vector_models=fold_models[season], rolling_windows=harness.rolling_windows),
        )
        for s, pos, gsis, week, _ in cases:
            if s != season:
                continue
            result = predict_many([gsis], gsis_id=True, season=season, week=week,
                                  context=ctx, models=fold_models[season])[0]
            assert result["status"] == "ok", result["reason"]
            points = calculate_points_vectorized(pl.DataFrame([result["stats"]]), scoring)["fantasy_points"][0]
            expected = engine.filter((pl.col("gsis_id") == gsis) & (pl.col("season") == season)
                                     & (pl.col("week") == week))["projection"][0]
            assert math.isclose(points, expected, rel_tol=TOL, abs_tol=TOL), (pos, gsis, week, points, expected)
            checked += 1
    assert checked >= 8
