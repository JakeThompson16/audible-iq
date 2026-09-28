"""
Player search / autocomplete over current-season rostered players. No UI code:
the UI, the agent and the Sleeper roster flow all call this.

    index = build_player_index()
    index.suggest("mahom")      # -> [{gsis_id, sleeper_id, name, display, position, team}, ...]
    index.get("00-0033873")

Identity comes from the same source as the prediction pipeline
(load_player_metadata via pipeline.predict's cached id maps), so every
suggestion's gsis_id resolves in predict_player_stats. Membership: players on an
nflverse weekly roster for the current season (the same roster source the
pipeline uses for current team), at a position that has a production
implementation in engine.expected_points.POSITION_IMPLEMENTATIONS.

Matching uses fast-autocomplete 0.9.0 (checked against its installed source):
- AutoComplete(words={key: {"count": n}}) builds a directed word graph; keys are
  normalized by the package (lowercase, a-z / 0-9 / space kept, '-' -> ' ',
  anything else dropped, 40-char cap), so we pre-normalize ourselves:
  accents folded to ASCII, apostrophes and periods removed, Jr./Sr./II/III/IV/V
  dropped.
- search(word, max_cost, size) returns lists of matched keys: prefix matches
  from the graph (descendants sorted by `count`) plus Levenshtein fuzzy matches
  (distance < max_cost) once 3+ characters are unmatched. Results are LFU-cached.
- Keys are name strings, so players sharing a name share a key; we keep a
  key -> {gsis_id} map and never resolve a selection from its display string.
"""
import re
import time
import unicodedata
from dataclasses import dataclass, field

import polars as pl
from fast_autocomplete import AutoComplete

from engine.expected_points import IMPLEMENTATIONS, POSITION_IMPLEMENTATIONS
from pipeline.context import FeatureContext, get_context, refresh_context
from pipeline.predict import _maps, refresh_player_maps

NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
# fast-autocomplete counts a key as a fuzzy match when edit distance < max_cost,
# so 3 tolerates up to 2 typos in the unmatched part of the query.
MAX_COST = 3
# Relevance weight = games played (stats rows) over this many seasons ending
# with the index season. Used as the autocomplete `count`, so established
# players outrank depth-chart names sharing a prefix.
RELEVANCE_SEASONS = 2


def production_positions() -> list[str]:
    """Positions whose registered implementation exists (derived, not hardcoded)."""
    return [p for p, impl in POSITION_IMPLEMENTATIONS.items() if impl in IMPLEMENTATIONS]


def normalize_name(name: str | None) -> str:
    """'Amon-Ra St. Brown' -> 'amon ra st brown'; "D'Andre Swift Jr." -> 'dandre swift'; 'José' -> 'jose'."""
    if not name:
        return ""
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"['’.`]", "", text)          # A.J. -> aj, O'Shaughnessy -> oshaughnessy
    text = re.sub(r"[^a-z0-9]+", " ", text)     # hyphens and anything else -> space
    tokens = text.split()
    while len(tokens) > 1 and tokens[-1] in NAME_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def name_keys(name: str) -> list[str]:
    """Full name plus every trailing part ('amon ra st brown', 'ra st brown', 'st brown', 'brown')."""
    tokens = normalize_name(name).split()
    return [" ".join(tokens[i:]) for i in range(len(tokens))]


@dataclass
class PlayerIndex:
    season: int
    entries: dict[str, dict]                  # gsis_id -> suggestion dict
    relevance: dict[str, int]                 # gsis_id -> games over RELEVANCE_SEASONS
    key_to_gsis: dict[str, set]               # normalized name key -> gsis_ids
    excluded_not_in_crosswalk: int
    excluded_other_position: int
    built_at: float = field(default_factory=time.time)

    def __post_init__(self):
        words = {key: {"count": max(self.relevance[g] for g in gs)} for key, gs in self.key_to_gsis.items()}
        self._ac = AutoComplete(words=words)

    def __len__(self) -> int:
        return len(self.entries)

    def get(self, gsis_id: str) -> dict | None:
        entry = self.entries.get(gsis_id)
        return dict(entry) if entry else None

    def _rank(self, gsis_ids) -> list[str]:
        return sorted(gsis_ids, key=lambda g: (-self.relevance[g], self.entries[g]["name"]))

    def suggest(self, query: str, size: int = 8) -> list[dict]:
        """
        Up to `size` players matching `query` (prefix of first / last / full
        name, or a small typo). Order: the autocomplete's order (exact and
        prefix before fuzzy; within that by relevance), players sharing a
        name key ordered by relevance.
        """
        q = normalize_name(query)
        if not q or size <= 0:
            return []
        out, seen = [], set()

        def add(gsis_ids):
            for g in self._rank(gsis_ids):
                if g not in seen:
                    seen.add(g)
                    out.append(self.get(g))
                    if len(out) == size:
                        return True
            return False

        for tokens in self._ac.search(word=q, max_cost=MAX_COST, size=max(size * 3, 20)):
            sets = [self.key_to_gsis[t] for t in tokens if t in self.key_to_gsis]
            if sets and add(set.intersection(*sets) or sets[-1]):
                return out

        # "pat mah": first-name prefix + last-name prefix. The word graph only
        # follows one path, so look up by the last word and keep players whose
        # first name starts with the first word.
        words = q.split()
        if len(words) >= 2:
            first, last = words[0], words[-1]
            matches = {g for key, gs in self.key_to_gsis.items() if key.startswith(last) for g in gs
                       if normalize_name(self.entries[g]["name"]).startswith(first)}
            add(matches)
        return out


def _build(ctx: FeatureContext, season: int) -> PlayerIndex:
    _, gsis_info = _maps(ctx)
    positions = set(production_positions())

    roster = ctx.rosters.filter(pl.col("season") == season)
    # Current team = latest roster week this season (same rule as pipeline.predict._team_as_of).
    latest = roster.sort("week").group_by("gsis_id").agg(pl.col("team").last(), pl.col("position").last())
    # Position = the player's stats-row position (what the backtest and predict use), else roster position.
    stats_pos = ctx.stats.sort(["season", "week"]).group_by("gsis_id").agg(pl.col("position").last().alias("stats_position"))
    latest = latest.join(stats_pos, on="gsis_id", how="left").with_columns(
        pl.coalesce("stats_position", "position").alias("position"))

    games = (ctx.stats.filter(pl.col("season").is_between(season - RELEVANCE_SEASONS + 1, season))
             .group_by("gsis_id").len().rename({"len": "games"}))
    latest = latest.join(games, on="gsis_id", how="left").with_columns(pl.col("games").fill_null(0))

    entries, relevance, key_to_gsis = {}, {}, {}
    not_in_crosswalk = other_position = 0
    for gsis, team, position, games_n in latest.select(["gsis_id", "team", "position", "games"]).iter_rows():
        if position not in positions:
            other_position += 1
            continue
        info = gsis_info.get(gsis)
        if info is None or not info["name"]:
            not_in_crosswalk += 1
            continue
        entries[gsis] = {
            "gsis_id": gsis,
            "sleeper_id": info["sleeper_id"],
            "name": info["name"],
            "display": f"{info['name']} · {position} · {team}",
            "position": position,
            "team": team,
        }
        relevance[gsis] = int(games_n)
        for key in name_keys(info["name"]):
            key_to_gsis.setdefault(key, set()).add(gsis)

    return PlayerIndex(season=season, entries=entries, relevance=relevance, key_to_gsis=key_to_gsis,
                       excluded_not_in_crosswalk=not_in_crosswalk, excluded_other_position=other_position)


_INDEXES: dict[int, PlayerIndex] = {}


def build_player_index(season: int | None = None, context: FeatureContext | None = None) -> PlayerIndex:
    """Cached per season for the life of the process."""
    ctx = context or get_context()
    season = season or max(ctx.seasons)
    if season not in _INDEXES:
        _INDEXES[season] = _build(ctx, season)
    return _INDEXES[season]


def refresh_player_index(reload_data: bool = True) -> None:
    """
    Drop cached indexes. With reload_data (default), also reload rosters and
    the id maps, so roster moves since the last load are picked up.
    """
    _INDEXES.clear()
    if reload_data:
        refresh_context()
        refresh_player_maps()
