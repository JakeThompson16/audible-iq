
import polars as pl


VOLUME_STATS = ["carries", "targets"]

# rate name -> (numerator stat, denominator stat)
RATE_STATS = {
    "ypc": ("rushing_yards", "carries"),
    "rush_td_rate": ("rushing_tds", "carries"),
    "catch_rate": ("receptions", "targets"),
    "ypr": ("receiving_yards", "receptions"),
    "rec_td_rate": ("receiving_tds", "receptions"),
}

DELTA_SHORT = 3
DELTA_LONG = 8


def add_stat_rolling_features(stats_df: pl.DataFrame, window: int = 8) -> pl.DataFrame:
    """
    :param stats_df: player-week stats (load_player_stats output)
    :param window: trailing games for the per-stat rolling features
    :return: stats_df with, per stat:
        roll_<volume>        trailing mean of carries / targets
        roll_<rate>          trailing rate as a ratio of rolling sums
        delta_<volume>       trailing DELTA_SHORT-game mean minus DELTA_LONG-game mean
        <rate>               this game's realized rate (the rate models' target;
                             null when the denominator is 0)

    Every trailing feature uses .shift(1) first, so week W only sees games
    before W. Partitioned by gsis_id only, NOT (gsis_id, season): the window
    rolls across the season boundary, so a player's week-2 value still pulls
    from last season's tail instead of starting from one game (the B-2
    failure mode). A player's first career game has no history and stays null.

    Rates are sum(numerator) / sum(denominator) over the window, not the mean
    of per-game rates, so a 1-carry, 20-yard game doesn't count as a 20-ypc
    data point.
    """
    required = {"gsis_id", "season", "week"} | set(VOLUME_STATS) | {
        c for pair in RATE_STATS.values() for c in pair
    }
    missing = required - set(stats_df.columns)
    if missing:
        raise ValueError(f"stats_df missing required columns: {missing}")

    df = stats_df.sort(["gsis_id", "season", "week"])

    def trailing(col: str, n: int, agg: str) -> pl.Expr:
        shifted = pl.col(col).shift(1)
        rolled = (
            shifted.rolling_mean(window_size=n, min_samples=1) if agg == "mean"
            else shifted.rolling_sum(window_size=n, min_samples=1)
        )
        return rolled.over("gsis_id")

    exprs = []
    for stat in VOLUME_STATS:
        exprs.append(trailing(stat, window, "mean").alias(f"roll_{stat}"))
        exprs.append(
            (trailing(stat, DELTA_SHORT, "mean") - trailing(stat, DELTA_LONG, "mean"))
            .alias(f"delta_{stat}")
        )

    for rate, (num, den) in RATE_STATS.items():
        num_sum = trailing(num, window, "sum")
        den_sum = trailing(den, window, "sum")
        exprs.append(
            pl.when(den_sum > 0).then(num_sum / den_sum).otherwise(None).alias(f"roll_{rate}")
        )
        exprs.append(
            pl.when(pl.col(den) > 0).then(pl.col(num) / pl.col(den)).otherwise(None).alias(rate)
        )

    return df.with_columns(exprs)
