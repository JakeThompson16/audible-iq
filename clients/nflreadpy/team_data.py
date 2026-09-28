
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


def load_team_schedule(seasons: int | list[int]) -> pl.DataFrame:
    """
    :return: one row per (team, season, week) game, all game types:
        team, opponent, season, week, game_type, game_id, completed
        (completed = the game has a final score), home (the team is the
        listed home team; neutral-site games still have a listed home team),
        gameday.
    """
    if isinstance(seasons, int):
        seasons = [seasons]

    schedule = nfl.load_schedules(seasons).select([
        'game_id', 'home_team', 'away_team', 'season', 'week', 'game_type', 'home_score', 'gameday'
    ]).with_columns(pl.col('home_score').is_not_null().alias('completed'))

    home = schedule.with_columns(pl.col('home_team').alias('team'), pl.col('away_team').alias('opponent'),
                                 pl.lit(True).alias('home'))
    away = schedule.with_columns(pl.col('away_team').alias('team'), pl.col('home_team').alias('opponent'),
                                 pl.lit(False).alias('home'))

    return pl.concat([home, away]).select(
        ['team', 'opponent', 'season', 'week', 'game_type', 'game_id', 'completed', 'home', 'gameday']
    ).sort(['season', 'week', 'team'])


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