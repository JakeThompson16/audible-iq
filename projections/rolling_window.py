
import polars as pl


# Continuous trailing windows, shared by every rolling feature in the project
# (player points, opponent skew, epa_allowed, stat-vector features).
#
# - `.shift(1)` first, so week W only ever sees games strictly before W
#   (the leakage rule in CLAUDE.md).
# - Partitioned by entity only (player or defense), NOT by season: the window
#   spans the season boundary freely, so week 1 uses the tail of last season.
#   There is no separate current-season/prior-season blend weight.
# - `min_samples=1`: the window is populated from the first prior game. How
#   much evidence backs a value is carried explicitly by trailing_count
#   (n_games_in_window), never by a formula.
#
# Callers must sort by [entity, season, week] before applying these.


def trailing_mean(col: str, window: int, by: str | list[str]) -> pl.Expr:
    return pl.col(col).shift(1).rolling_mean(window_size=window, min_samples=1).over(by)


def trailing_sum(col: str, window: int, by: str | list[str]) -> pl.Expr:
    return pl.col(col).shift(1).rolling_sum(window_size=window, min_samples=1).over(by)


def trailing_count(col: str, window: int, by: str | list[str]) -> pl.Expr:
    """Number of non-null prior values in the window (0 for the first row)."""
    return (
        pl.col(col).is_not_null().cast(pl.Int32).shift(1)
        .rolling_sum(window_size=window, min_samples=1).over(by)
        .fill_null(0)
    )
