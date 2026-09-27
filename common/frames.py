
import polars as pl


def assert_unique_key(df: pl.DataFrame, keys: list[str], name: str) -> pl.DataFrame:
    """
    Raise if `keys` do not uniquely identify rows of `df`; return `df` unchanged
    so it can be used inline (e.g. on the right side of a join).

    Use on the right-hand side of every join whose key is supposed to be
    one-to-one. A non-unique right side silently fans out the left side
    (see CLAUDE.md "Known bugs already hit once") instead of erroring.
    """
    dup = df.filter(pl.struct(keys).is_duplicated())
    if dup.height:
        sample = dup.select(keys).unique().head(5).rows()
        raise ValueError(
            f"{name}: {dup.height} rows share a duplicate {keys} key "
            f"({dup.select(keys).n_unique()} distinct keys), e.g. {sample}"
        )
    return df


def assert_no_fanout(before: pl.DataFrame, after: pl.DataFrame, name: str) -> pl.DataFrame:
    """Raise if a left join changed the row count; return `after`."""
    if after.height != before.height:
        raise ValueError(f"{name}: join fan-out, {after.height} rows vs {before.height} before")
    return after
