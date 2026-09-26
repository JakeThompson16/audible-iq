

import nflreadpy as nfl
import polars as pl

from clients.nflreadpy.player_data import load_player_metadata


def load_pbp_data(seasons: int | list[int]) -> pl.DataFrame:
    """
    :param seasons: Season(s) to pull data from
    :return: Raw play-by-play DataFrame from all games in seasons
    """

    if isinstance(seasons, int):
        seasons = [seasons]

    pbp_df = nfl.load_pbp(seasons)

    return pbp_df