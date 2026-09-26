
import nflreadpy as nfl
import polars as pl
from polars import Series

from config import PLAYER_METADATA


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

    df = ff_ids.join(
        players,
        left_on='gsis_id',
        right_on='gsis_id',
        how='inner'
    )
    return df.select(PLAYER_METADATA)


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

    return df.select(
        PLAYER_METADATA +
        GAME_ID_COLUMNS +
        RAW_STAT_COLUMNS +
        PROCESSED_STAT_COLUMNS
    )


def get_snap_counts(seasons: int | list[int]) -> pl.DataFrame:
    """
    :param seasons: Seasons to get snap counts from
    :return: snap counts DataFrame
    """
    if isinstance(seasons, int):
        seasons = [seasons]

    df = nfl.load_player_stats(seasons)

    return df
