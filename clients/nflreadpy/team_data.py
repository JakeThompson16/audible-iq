
import nflreadpy as nfl
import polars as pl


GAMES_COLS = [
    'team', 'opponent', 'week', 'season'
]


def _de_pivot_df(schedule: pl.DataFrame) -> pl.DataFrame:
    """
    :param schedule: Schedule df
    :return: DF with 2 rows for each game, one where team=home_team, opponent=away_team;
    one where team=away_team, opponent=home_team
    """
    left = schedule.with_columns(
        pl.col('home_team')
        .alias('team'),

        pl.col('away_team')
        .alias('opponent')
    )

    right = schedule.with_columns(
        pl.col('away_team')
        .alias('team'),

        pl.col('home_team')
        .alias('opponent')
    )

    df = pl.concat([left, right])

    return df.select(GAMES_COLS)


def pull_team_games(seasons: int | list[int]) -> pl.DataFrame:
    """
    :param seasons: Seasons to pull games from
    :return: Data frame of all games for every team
    """

    if isinstance(seasons, int):
        seasons = [seasons]

    schedule = nfl.load_schedules(seasons).select([
        'home_team', 'away_team', 'season', 'week', 'game_type'
    ])

    schedule = schedule.filter(
        pl.col('game_type') == 'REG'
    )

    games = _de_pivot_df(schedule)

    return games