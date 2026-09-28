"""
Feature context shared by training and prediction, loaded once per process.

Everything nflreadpy provides that the stat-vector features need: raw
player-week stats, weekly EPA residual tables (the input to epa_allowed),
the team schedule with completion flags, weekly rosters (current team) and
nflverse's latest_team. Per-player prediction calls never reload nflreadpy;
call refresh_context() when a new week of data lands.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone

import nflreadpy as nfl
import polars as pl

from clients.nflreadpy.pbp_data import load_pbp_data
from clients.nflreadpy.player_data import load_latest_teams, load_player_stats, load_weekly_rosters
from clients.nflreadpy.team_data import load_team_schedule
from projections.expected_points.features.epa_allowed import weekly_epa_residuals

DEFAULT_START_SEASON = 2022
# Seasons loaded before start_season as history only, so trailing windows
# (player up to 14 games, defense 17) and epa_allowed are populated from
# week 1 of start_season.
HISTORY_SEASONS = 2

PBP_COLUMNS = ["season", "week", "defteam", "play_type", "sack", "epa",
               "yardline_100", "rushing_yards", "air_yards"]


@dataclass
class FeatureContext:
    seasons: list[int]
    stats: pl.DataFrame            # raw player-week stats (every row nflverse has)
    pass_weekly: pl.DataFrame      # weekly EPA residuals, pass plays
    rush_weekly: pl.DataFrame      # weekly EPA residuals, run plays
    schedule: pl.DataFrame         # load_team_schedule(): team-level games + completed flag
    rosters: pl.DataFrame          # load_weekly_rosters(): gsis_id, season, week, team, position
    latest_teams: pl.DataFrame     # gsis_id, latest_team
    data_through: tuple[int, int]  # last (season, week) whose scheduled games are ALL complete
    loaded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


def completed_through(schedule: pl.DataFrame, season: int) -> tuple[int, int]:
    """
    Last (season, week) such that every scheduled game in that week and all
    earlier weeks of `season` has a final score. If week 1 of `season` isn't
    complete yet, it's the last week of the previous season.
    """
    weeks = (
        schedule.filter(pl.col("season") == season)
        .group_by("week").agg(pl.col("completed").all().alias("all_done"))
        .sort("week")
    )
    last = None
    for week, done in weeks.iter_rows():
        if not done:
            break
        last = week
    if last is not None:
        return season, last
    prev = schedule.filter(pl.col("season") == season - 1)
    return season - 1, int(prev["week"].max())


def through_mask(through: tuple[int, int]) -> pl.Expr:
    """Rows at or before (season, week)."""
    s, w = through
    return (pl.col("season") < s) | ((pl.col("season") == s) & (pl.col("week") <= w))


def load_context(start_season: int = DEFAULT_START_SEASON, end_season: int | None = None) -> FeatureContext:
    end_season = end_season or nfl.get_current_season()
    seasons = list(range(start_season - HISTORY_SEASONS, end_season + 1))

    pbp = pl.concat([load_pbp_data(s).select(PBP_COLUMNS) for s in seasons], how="vertical_relaxed")
    pass_weekly, rush_weekly = weekly_epa_residuals(pbp)
    schedule = load_team_schedule(seasons)

    return FeatureContext(
        seasons=seasons,
        stats=load_player_stats(seasons),
        pass_weekly=pass_weekly,
        rush_weekly=rush_weekly,
        schedule=schedule,
        rosters=load_weekly_rosters(seasons).filter(pl.col("position").is_in(["QB", "RB", "WR", "TE"])),
        latest_teams=load_latest_teams(),
        data_through=completed_through(schedule, end_season),
    )


_CONTEXTS: dict[tuple, FeatureContext] = {}


def get_context(start_season: int = DEFAULT_START_SEASON, end_season: int | None = None) -> FeatureContext:
    """Cached per (start_season, end_season) for the life of the process."""
    key = (start_season, end_season)
    if key not in _CONTEXTS:
        _CONTEXTS[key] = load_context(start_season, end_season)
    return _CONTEXTS[key]


def refresh_context() -> None:
    """Drop cached contexts; the next call reloads from nflreadpy."""
    _CONTEXTS.clear()
