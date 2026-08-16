

import polars as pl

from config import PLAYER_METADATA

def metadata_keys() -> list[str]:
    """Keys that a valid metadata dict should include"""
    return PLAYER_METADATA


class Player:
    """Class for a player which stores performance history and metadata"""

    def __init__(self, data: dict):
        """
        Call metadata_keys() for list[str] of expected keys
        :param data: Dict with players metadata
        """

        ex_keys = metadata_keys()
        missing = set(ex_keys) - set(data.keys())

        if missing:
            raise ValueError(f'Missing keys: {missing}')

        self.sleeper_id = data['sleeper_id']
        self.gsis_id = data['gsis_id']
        self.display_name = data['display_name']
        self.position = data['position']
        self.team = data['team']

        self.rolling_avg: float | None = None
        self.boom_prob: float | None = None
        self.bust_prob: float | None = None
        self.recent_cache_week: int | None = None


    def reset_cache(self, stats_df: pl.DataFrame):
        """Updates rolling average and boom/bust prob"""
        history = self.get_player_history(stats_df).sort('week')

        if history.is_empty():
            raise LookupError('Could not find player history.')

        latest = history.tail(1)

        self.rolling_avg = latest['rolling_avg'].item()
        self.boom_prob = latest['boom_prob'].item()
        self.bust_prob = latest['bust_prob'].item()
        self.recent_cache_week = int(latest['week'].item())


    def pull_cache(self) -> dict:
        """Returns dict with boom_prob, bust_prob, rolling_avg keys, recent_cache_week"""

        return {
            'boom_prob': self.boom_prob,
            'bust_prob': self.bust_prob,
            'rolling_avg': self.rolling_avg,
            'recent_cache_week': self.recent_cache_week
        }


    def get_player_history(self, stats_df: pl.DataFrame) -> pl.DataFrame:
        """Returns players stats as they are in stats_df"""

        player_stats = stats_df.filter(
            pl.col('gsis_id') == self.gsis_id
        )

        return player_stats


    def __eq__(self, other) -> bool:
        """Returns true if two players have the same id's"""

        if isinstance(other, Player):
            return (
                self.sleeper_id == other.sleeper_id and
                self.gsis_id == other.gsis_id
            )

        return NotImplemented


    def __hash__(self) -> int:
        return hash((self.sleeper_id, self.gsis_id))


    def __repr__(self) -> str:
        return (
            f"GSIS: {self.gsis_id}, SLEEPER: {self.sleeper_id}, NAME: {self.display_name}"
        )