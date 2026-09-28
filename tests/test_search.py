"""Player search index, the projection service, and prediction sanity bounds."""
import statistics
import time

import polars as pl
import pytest

from engine.expected_points import POSITION_IMPLEMENTATIONS
from pipeline.predict import predict_many
from projections.expected_points.stat_vector.core import StatVectorBoundsError, check_stat_line, stat_line_violations
from search.player_search import (
    PlayerIndex, build_player_index, name_keys, normalize_name, production_positions, refresh_player_index,
)


def _first(index, query):
    hits = index.suggest(query, size=8)
    return hits[0] if hits else None


def test_normalize_name():
    assert normalize_name("Amon-Ra St. Brown") == "amon ra st brown"
    assert normalize_name("D'Andre Swift Jr.") == "dandre swift"
    assert normalize_name("Kenneth Walker III") == "kenneth walker"
    assert normalize_name("José Ramírez") == "jose ramirez"
    assert normalize_name("A.J. Brown") == "aj brown"
    assert name_keys("Amon-Ra St. Brown") == ["amon ra st brown", "ra st brown", "st brown", "brown"]


def test_membership(index, ctx):
    positions = set(production_positions())
    assert positions == set(POSITION_IMPLEMENTATIONS)          # derived from the registry
    assert {e["position"] for e in index.entries.values()} <= positions
    current = ctx.rosters.filter(pl.col("season") == max(ctx.seasons))["gsis_id"].unique().to_list()
    assert set(index.entries) <= set(current)                  # only current-season rostered players
    assert index.excluded_not_in_crosswalk >= 0 and len(index) > 500


def test_every_suggestion_resolves(index, ctx, trained):
    directory, _ = trained
    results = predict_many(list(index.entries), gsis_id=True, context=ctx, artifacts_dir=directory)
    assert not [r for r in results if r["status"] in ("unknown_player", "unsupported_position")]


def test_team_matches_prediction_pipeline(index, ctx, trained):
    directory, _ = trained
    sample = list(index.entries)[:25]
    for r in predict_many(sample, gsis_id=True, context=ctx, artifacts_dir=directory):
        assert r["player"]["team"] == index.get(r["player"]["gsis_id"])["team"]


@pytest.mark.parametrize("query,expected_name", [
    ("mahom", "Patrick Mahomes"),            # prefix
    ("patrick mahomes", "Patrick Mahomes"),  # full name
    ("PaTrIcK MaHoMeS", "Patrick Mahomes"),  # mixed case
    ("mahomez", "Patrick Mahomes"),          # one-character typo
    ("mahomes", "Patrick Mahomes"),          # last name only
    ("patrik mahomes", "Patrick Mahomes"),   # typo
    ("pat mah", "Patrick Mahomes"),          # first + last prefixes
    ("st brown", "Amon-Ra St. Brown"),       # multi-part last name
    ("amon-ra", "Amon-Ra St. Brown"),        # hyphen
    ("jamarr", "Ja'Marr Chase"),             # apostrophe
    ("smith-njigba", "Jaxon Smith-Njigba"),
    ("kenneth walker", "Kenneth Walker III"),  # suffix
    ("a.j. brown", "A.J. Brown"),
])
def test_suggest_matches(index, query, expected_name):
    names = [h["name"] for h in index.suggest(query, size=8)]
    assert expected_name in names, (query, names)


def test_suggestion_shape_and_get(index):
    hit = _first(index, "mahomes")
    assert set(hit) == {"gsis_id", "sleeper_id", "name", "display", "position", "team"}
    assert hit["display"] == f"{hit['name']} · {hit['position']} · {hit['team']}"
    assert index.get(hit["gsis_id"]) == hit
    assert index.get("not-an-id") is None
    assert index.suggest("", 5) == [] and index.suggest("zzzzqqq", 5) == []


def test_shared_names_stay_separate(index):
    """Players who share a name key each come back with their own gsis_id."""
    shared = [(k, gs) for k, gs in index.key_to_gsis.items() if len(gs) > 1 and " " in k]
    if not shared:
        pytest.skip("no full-name collision among current rosters")
    key, gsis_ids = shared[0]
    got = {h["gsis_id"] for h in index.suggest(key, size=20)}
    assert set(gsis_ids) <= got


def test_duplicate_names_are_distinct_entries():
    """Two players with the same name are two entries with distinct gsis_ids (synthetic index)."""
    entries = {
        "00-1": {"gsis_id": "00-1", "sleeper_id": "1", "name": "Mike Williams", "display": "Mike Williams · WR · NYJ",
                 "position": "WR", "team": "NYJ"},
        "00-2": {"gsis_id": "00-2", "sleeper_id": "2", "name": "Mike Williams", "display": "Mike Williams · WR · PIT",
                 "position": "WR", "team": "PIT"},
    }
    key_to_gsis = {}
    for g, e in entries.items():
        for k in name_keys(e["name"]):
            key_to_gsis.setdefault(k, set()).add(g)
    idx = PlayerIndex(season=2026, entries=entries, relevance={"00-1": 30, "00-2": 5}, key_to_gsis=key_to_gsis,
                      excluded_not_in_crosswalk=0, excluded_other_position=0)
    hits = idx.suggest("mike williams")
    assert [h["gsis_id"] for h in hits] == ["00-1", "00-2"]      # both, ordered by relevance
    assert {h["display"] for h in hits} == {"Mike Williams · WR · NYJ", "Mike Williams · WR · PIT"}
    assert [h["gsis_id"] for h in idx.suggest("williams")] == ["00-1", "00-2"]


def test_refresh_rebuilds(ctx):
    first = build_player_index(context=ctx)
    assert build_player_index(context=ctx) is first               # cached
    refresh_player_index(reload_data=False)
    second = build_player_index(context=ctx)
    assert second is not first and set(second.entries) == set(first.entries)


def test_relevance_ranks_established_players_first(index):
    hits = index.suggest("josh", size=8)
    games = [index.relevance[h["gsis_id"]] for h in hits]
    assert games == sorted(games, reverse=True)


def test_suggest_latency(index):
    queries = [normalize_name(e["name"])[:4] for e in list(index.entries.values())[:300]]
    lat = []
    for q in queries:
        t = time.perf_counter()
        index.suggest(q)
        lat.append((time.perf_counter() - t) * 1000)
    assert statistics.median(lat) < 5.0, statistics.median(lat)


def test_bounds_check():
    good = pl.DataFrame({"attempts": [30.0], "completions": [20.0], "passing_inc": [10.0],
                         "targets": [5.0], "receptions": [3.0], "rushing_yards": [12.0]})
    assert stat_line_violations(good) == []
    check_stat_line(good)
    for bad in (good.with_columns(pl.lit(-1.0).alias("rushing_yards")),
                good.with_columns(pl.lit(35.0).alias("completions")),
                good.with_columns(pl.lit(6.0).alias("receptions"))):
        with pytest.raises(StatVectorBoundsError):
            check_stat_line(bad)
