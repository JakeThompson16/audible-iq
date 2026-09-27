
import polars as pl

from projections.rolling_window import trailing_count, trailing_mean, trailing_sum


VOLUME_STATS = ["carries", "targets", "attempts"]

# rate name -> (numerator stat, denominator stat). Passing yards and TDs accrue
# on completions; interceptions are per attempt (an INT is never a completion).
RATE_STATS = {
    "ypc": ("rushing_yards", "carries"),
    "rush_td_rate": ("rushing_tds", "carries"),
    "catch_rate": ("receptions", "targets"),
    "ypr": ("receiving_yards", "receptions"),
    "rec_td_rate": ("receiving_tds", "receptions"),
    "completion_rate": ("completions", "attempts"),
    "yards_per_completion": ("passing_yards", "completions"),
    "pass_td_rate": ("passing_tds", "completions"),
    "int_rate": ("passing_interceptions", "attempts"),
}

DELTA_SHORT = 3
DELTA_LONG = 8


def add_stat_rolling_features(stats_df: pl.DataFrame, window: int = 8) -> pl.DataFrame:
    """
    :param stats_df: player-week stats (load_player_stats output)
    :param window: trailing games for the per-stat rolling features
    :return: stats_df with, per stat:
        roll_<volume>          trailing mean of carries / targets / attempts
        roll_<rate>            trailing rate as a ratio of rolling sums
        delta_<volume>         trailing DELTA_SHORT-game mean minus DELTA_LONG-game mean
        stat_games_in_window   games backing the roll_* values
        <rate>                 this game's realized rate (the rate models' target;
                               null when the denominator is 0)

    Uses the shared continuous trailing window (projections/rolling_window.py):
    shift(1) first, partitioned by gsis_id only, so windows span the season
    boundary. A player's first career game has no history and stays null.

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

    exprs = [trailing_count("carries", window, "gsis_id").alias("stat_games_in_window")]
    for stat in VOLUME_STATS:
        exprs.append(trailing_mean(stat, window, "gsis_id").alias(f"roll_{stat}"))
        exprs.append(
            (trailing_mean(stat, DELTA_SHORT, "gsis_id") - trailing_mean(stat, DELTA_LONG, "gsis_id"))
            .alias(f"delta_{stat}")
        )

    for rate, (num, den) in RATE_STATS.items():
        num_sum = trailing_sum(num, window, "gsis_id")
        den_sum = trailing_sum(den, window, "gsis_id")
        exprs.append(
            pl.when(den_sum > 0).then(num_sum / den_sum).otherwise(None).alias(f"roll_{rate}")
        )
        exprs.append(
            pl.when(pl.col(den) > 0).then(pl.col(num) / pl.col(den)).otherwise(None).alias(rate)
        )

    return df.with_columns(exprs)
