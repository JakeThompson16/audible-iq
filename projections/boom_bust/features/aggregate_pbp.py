
import polars as pl

from clients.nflreadpy.pbp_data import load_pbp_data
from clients.nflreadpy.player_data import load_player_metadata, get_positional_ids


DEEP_PASS_THRESHOLD = 20
BIG_RUSH_THRESHOLD = 15

HIGH_VALUE_THRESHOLD = 10
REDZONE_YARDLINE = 20


def _build_cross_walk(stats_df: pl.DataFrame, seasons: int | list[int]) -> pl.DataFrame:
    """
    :param stats_df:
    :return:
    """
    if isinstance(seasons, int):
        seasons = [seasons]

    metadata = load_player_metadata(min(seasons) - 1)

    df = stats_df.join(
        metadata,
        left_on='gsis_id',
        right_on='gsis_id',
        how='inner'
    )

    return df


def _construct_general_columns(pbp_data: pl.DataFrame) -> pl.DataFrame:
    """
    :param pbp_data: Raw play-by-play DataFrame
    :return: DataFrame with play-by-play columns used for all positions
    """

    pbp_data = pbp_data.with_columns(

        # Red Zone opportunity
        pl.when(pl.col('yardline_100') <= REDZONE_YARDLINE)
        .then(1)
        .otherwise(0)
        .alias('redzone_opportunity'),

        # Red Zone touchdowns
        pl.when(pl.col('touchdown') == 1)
        .then(
            pl.when(pl.col('yardline_100') <= REDZONE_YARDLINE)
            .then(1)
            .otherwise(0)
        )
        .otherwise(0)
        .alias('redzone_td')
    )

    return pbp_data


def _construct_rushing_columns(pbp_data: pl.DataFrame) -> pl.DataFrame:
    """
    :param pbp_data: Raw play-by-play DataFrame
    :return: play-by-play DataFrame with columns needed for rushing boom/bust aggregation
    """

    rushes = pbp_data.filter(
        pl.col('play_type') == 'run'
    )

    rushes = rushes.with_columns(

        # Big rush
        pl.when(pl.col('rushing_yards') >= BIG_RUSH_THRESHOLD)
        .then(1)
        .otherwise(0)
        .alias('big_rush'),

        # Value of rushing opportunity
        pl.when(pl.col('yardline_100') <= HIGH_VALUE_THRESHOLD)
        .then(HIGH_VALUE_THRESHOLD / pl.col('yardline_100'))
        .otherwise(1)
        .alias('rush_value'),


    )

    return rushes


def _construct_passing_columns(pbp_data: pl.DataFrame) -> pl.DataFrame:
    """
    :param pbp_data: Raw play-by-play DataFrame
    :return: play-by-play DataFrame with columns needed for passing boom/bust aggregation
    """

    passes = pbp_data.filter(
        pl.col('play_type') == 'pass'
    )

    passes = passes.with_columns(

        # Deep pass attempt
        pl.when(pl.col('air_yards') >= DEEP_PASS_THRESHOLD)
        .then(1)
        .otherwise(0)
        .alias('deep_pass_attempt'),

        # Deep pass completion
        pl.when(pl.col('air_yards') >= DEEP_PASS_THRESHOLD)
        .then(1)
        .otherwise(0)
        .alias('deep_pass_completion'),

        # Value of passing opportunity
        pl.when(pl.col('yardline_100') <= HIGH_VALUE_THRESHOLD)
        .then(HIGH_VALUE_THRESHOLD / (pl.col('yardline_100')))
        .otherwise(1)
        .alias('pass_value')
    )

    return passes


def _build_qb_df(passes: pl.DataFrame, rushes: pl.DataFrame, qbs: pl.Series) -> pl.DataFrame:
    """
    :param passes: df of passing plays
    :param rushes: df of rushing plays
    :param qbs: Series of GSIS id's of QB's
    :return: Data frame of plays where QB was a direct actor (pass thrown or carry)
    """

    passes = passes.filter(
        pl.col('passer_player_id').is_in(qbs)
    ).with_columns(
        pl.col('passer_player_id')
        .alias('actor_id')
    )

    rushes = rushes.filter(
        pl.col('rusher_player_id').is_in(qbs)
    ).with_columns(
        pl.when(pl.col('qb_scramble') == 0)
        .then(1)
        .otherwise(0)
        .alias('designed_run'),

        pl.col('rusher_player_id')
        .alias('actor_id')
    )

    qb_df = pl.concat(
        [passes, rushes],
        how='diagonal'
    ).with_columns(
        pl.lit('QB')
        .alias('actor_position')
    )

    return qb_df

def _build_skill_position_df(
        passes: pl.DataFrame, rushes: pl.DataFrame, player_ids: pl.Series, position: str) -> pl.DataFrame:
    """
    Reusable code to compute DF for skill position players
    :param passes: df of passing plays
    :param rushes: df of rushing plays
    :param player_ids: Series of GSIS id's of players at that position
    :param position: String of player position
    :return: Data frame of plays where player of position was a direct actor (carry or reception)
    """

    receptions = passes.filter(
        pl.col('receiver_player_id').is_in(player_ids)
    ).with_columns(
        pl.col('receiver_player_id')
        .alias('actor_id')
    )

    rushes = rushes.filter(
        pl.col('rusher_player_id').is_in(player_ids)
    ).with_columns(
        pl.col('rusher_player_id')
        .alias('actor_id')
    )

    position_df = pl.concat(
        [receptions, rushes],
        how='diagonal'
    ).with_columns(
        pl.lit(position)
        .alias('actor_position')
    )

    return position_df

def _build_rb_df(passes: pl.DataFrame, rushes: pl.DataFrame, rbs: pl.Series) -> pl.DataFrame:
    """
    :param passes: df of passing plays
    :param rushes: df of rushing plays
    :param rbs: Series of GSIS id's of RB's
    :return: Data frame of plays where RB was a direct actor (carry or reception)
    """

    rb_df = _build_skill_position_df(
        passes, rushes, rbs, 'RB'
    )

    return rb_df


def _build_wr_df(passes: pl.DataFrame, rushes: pl.DataFrame, wrs: pl.Series) -> pl.DataFrame:
    """
    :param passes: df of passing plays
    :param rushes: df of rushing plays
    :param wrs: Series of GSIS id's of WR's
    :return: Data frame of plays where WR was a direct actor (carry or reception)
    """

    wr_df = _build_skill_position_df(
        passes, rushes, wrs, 'WR'
    )

    return wr_df



def _build_te_df(passes: pl.DataFrame, rushes: pl.DataFrame, tes: pl.Series) -> pl.DataFrame:
    """
    :param passes: df of passing plays
    :param rushes: df of rushing plays
    :param tes: Series of GSIS id's of TE's
    :return: Data frame of plays where TE was a direct actor (carry or reception)
    """

    te_df = _build_skill_position_df(
        passes, rushes, tes, 'TE'
    )

    return te_df


def _aggregate_pbp_data(pbp_data: pl.DataFrame, seasons: int | list[int]) -> dict:
    """
    :param seasons:
    :param pbp_data:
    :return:
    """

    qbs, rbs, wrs, tes = get_positional_ids(seasons)

    pbp_data = _construct_general_columns(pbp_data)

    rushes = _construct_rushing_columns(pbp_data)
    passes = _construct_passing_columns(pbp_data)


    return {
        'qb_df': _build_qb_df(passes, rushes, qbs),
        'rb_df': _build_rb_df(passes, rushes, rbs),
        'wr_df': _build_wr_df(passes, rushes, wrs),
        'te_df': _build_te_df(passes, rushes, tes)
    }


def pull_pbp_features(seasons: int | list[int]) -> pl.DataFrame:
    """
    :param seasons: Seasons to get data from
    :return: Dict of
    """
    if isinstance(seasons, int):
        seasons = [seasons]

    pbp_df = load_pbp_data(seasons)

    positional_data = _aggregate_pbp_data(pbp_df, seasons)


