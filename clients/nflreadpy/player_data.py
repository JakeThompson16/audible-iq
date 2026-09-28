
import nflreadpy as nfl
import polars as pl
from polars import Series

from common.frames import assert_unique_key
from config import PLAYER_METADATA

# nflreadpy 0.1.5 (latest on PyPI) builds the dynastyprocess player-ID
# crosswalk URL via github.com/.../raw/master/..., which now 404s — GitHub
# dropped that redirect. raw.githubusercontent.com still serves the file.
# Patched here (the module that calls load_ff_playerids) rather than in
# site-packages so it survives reinstalls, and so every caller gets it
# without importing test.py.
import nflreadpy.downloader as _nflreadpy_downloader
_nflreadpy_downloader.NflverseDownloader.BASE_URLS["dynastyprocess"] = (
    "https://raw.githubusercontent.com/dynastyprocess/data/master/files/"
)


RAW_STAT_COLUMNS = [
    'passing_yards', 'passing_tds', 'passing_interceptions', 'passing_2pt_conversions', 'attempts', 'completions',
    'passing_first_downs', 'sacks_suffered',
    
    'rushing_yards', 'rushing_tds', 'rushing_2pt_conversions', 'carries', 'rushing_first_downs',

    'rushing_fumbles_lost', 'receiving_fumbles_lost', 'sack_fumbles_lost',
    
    'receptions', 'targets', 'receiving_yards', 'receiving_tds', 'receiving_2pt_conversions', 'receiving_first_downs'
]

PROCESSED_STAT_COLUMNS = [
    'passing_inc', 'over_300_passing_yards', 'over_400_passing_yards',
    'over_25_completions', 'over_100_rushing_yards', 'over_200_rushing_yards',
    'over_20_carries', 'te_receptions', 'wr_receptions', 'rb_receptions', 'fumbles_lost',
    'opportunities'
]

GAME_ID_COLUMNS = [
    'season', 'week', 'opponent_team'
]


def load_player_metadata(cutoff: int = 2023) -> pl.DataFrame:
    """Returns nfl player metadata, players who have played at least one game since cutoff"""

    ff_ids = nfl.load_ff_playerids()

    players = nfl.load_players()
    players = players.filter(pl.col('last_season') > cutoff)
    assert_unique_key(players, ['gsis_id'], 'load_players')

    df = ff_ids.join(
        players,
        left_on='gsis_id',
        right_on='gsis_id',
        how='inner'
    )

    # The dynastyprocess crosswalk maps a few gsis_ids to two rows (e.g. Justin
    # Hamilton, Corey Moore; ~10 ids total), which fanned out every stats join
    # for those players. Keep the crosswalk row whose name matches nflverse's
    # player record, then the one with a sleeper_id, then the first.
    df = (
        df.with_columns(
            (pl.col('name').str.to_lowercase() == pl.col('display_name').str.to_lowercase())
            .fill_null(False).alias('_name_match'),
            pl.col('sleeper_id').is_not_null().alias('_has_sleeper'),
        )
        .sort(['gsis_id', '_name_match', '_has_sleeper'], descending=[False, True, True])
        .unique(subset=['gsis_id'], keep='first', maintain_order=True)
    )

    return assert_unique_key(df.select(PLAYER_METADATA), ['gsis_id'], 'load_player_metadata')


def get_positional_ids(
        seasons: int | list[int]) -> tuple[Series, Series, Series, Series]:
    """
    :param seasons: Season's to get positional ids for
    :return: qbs, rbs, wrs, tes
    """

    if isinstance(seasons, int):
        seasons = [seasons]

    metadata = load_player_metadata(min(seasons) - 1)

    qbs = metadata.filter(
        pl.col('position') == 'QB'
    ).get_column('gsis_id')

    rbs = metadata.filter(
        pl.col('position') == 'RB'
    ).get_column('gsis_id')

    wrs = metadata.filter(
        pl.col('position') == 'WR'
    ).get_column('gsis_id')

    tes = metadata.filter(
        pl.col('position') == 'TE'
    ).get_column('gsis_id')

    return qbs, rbs, wrs, tes


def _build_stats(df: pl.DataFrame) -> pl.DataFrame:
    """
    :param df: Stats dataframe
    :return: Adds non-native stats columns to stats df
    """

    df = df.with_columns(
        # Passing
        (pl.col('attempts') - pl.col('completions'))
        .alias('passing_inc'),

        pl.when(
            pl.col('passing_yards') >= 300.0
        )
        .then(1)
        .otherwise(0)
        .alias('over_300_passing_yards'),

        pl.when(
            pl.col('passing_yards') >= 400.0
        )
        .then(1)
        .otherwise(0)
        .alias('over_400_passing_yards'),

        pl.when(
            pl.col('completions') >= 25
        )
        .then(1)
        .otherwise(0)
        .alias('over_25_completions'),

        pl.when(
            pl.col('rushing_yards') >= 100.0
        )
        .then(1)
        .otherwise(0)
        .alias('over_100_rushing_yards'),

        pl.when(
            pl.col('rushing_yards') >= 200.0
        )
        .then(1)
        .otherwise(0)
        .alias('over_200_rushing_yards'),

        pl.when(
            pl.col('carries') >= 20
        )
        .then(1)
        .otherwise(0)
        .alias('over_20_carries'),

        pl.when(
            pl.col('position') == 'TE'
        )
        .then(pl.col('receptions'))
        .otherwise(0)
        .alias('te_receptions'),

        pl.when(
            pl.col('position') == 'WR'
        )
        .then(pl.col('receptions'))
        .otherwise(0)
        .alias('wr_receptions'),

        pl.when(
            pl.col('position') == 'RB'
        )
        .then(pl.col('receptions'))
        .otherwise(0)
        .alias('rb_receptions'),

        (
            pl.col('rushing_fumbles_lost') +
            pl.col('receiving_fumbles_lost') +
            pl.col('sack_fumbles_lost')
        )
        .alias('fumbles_lost'),

        # Opportunities to show volume for WR/RB
        (
            pl.col('targets') +
            pl.col('carries')
        )
        .alias('opportunities')
    )

    return df


def load_player_stats(seasons: int | list[int]) -> pl.DataFrame:
    """Loads generic player stats df for seasons in seasons"""

    if isinstance(seasons, int):
        seasons = [seasons]

    metadata_df = load_player_metadata(min(seasons) - 1)
    stats_df = nfl.load_player_stats(seasons)

    df = stats_df.join(
        metadata_df,
        left_on='player_id',
        right_on='gsis_id',
        how='inner'
    ).with_columns(
        pl.col('player_id')
        .alias('gsis_id')
    )

    df = _build_stats(df)

    assert_unique_key(df, ['gsis_id', 'season', 'week'], 'load_player_stats')

    return df.select(
        PLAYER_METADATA +
        GAME_ID_COLUMNS +
        RAW_STAT_COLUMNS +
        PROCESSED_STAT_COLUMNS
    )


def load_weekly_rosters(seasons: int | list[int]) -> pl.DataFrame:
    """
    :return: nflverse weekly rosters, one row per (gsis_id, season, week):
        gsis_id, season, week, team, position. Team abbreviations match the
        schedule and the stats' opponent_team (unlike the ff_playerids
        crosswalk's team field, which uses e.g. KCC / LAR / JAC).
    """
    if isinstance(seasons, int):
        seasons = [seasons]

    df = nfl.load_rosters_weekly(seasons).filter(pl.col('gsis_id').is_not_null())
    return (
        df.select(['gsis_id', 'season', 'week', 'team', 'position'])
        .unique(subset=['gsis_id', 'season', 'week'], keep='last', maintain_order=True)
        .sort(['gsis_id', 'season', 'week'])
    )


def load_latest_teams() -> pl.DataFrame:
    """:return: gsis_id, latest_team from nflverse players (same abbreviations as the schedule)."""
    return nfl.load_players().select(['gsis_id', 'latest_team']).filter(pl.col('gsis_id').is_not_null())


def get_snap_counts(seasons: int | list[int]) -> pl.DataFrame:
    """
    :param seasons: Seasons to get snap counts from
    :return: snap counts DataFrame
    """
    if isinstance(seasons, int):
        seasons = [seasons]

    df = nfl.load_player_stats(seasons)

    return df
