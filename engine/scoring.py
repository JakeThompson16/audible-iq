
from dataclasses import fields
import polars as pl

from domain.scoring import ScoringSettings


SCORING_TO_STAT_COLUMN = {
    "pass_yd": "passing_yards",
    "pass_td": "passing_tds",
    "pass_int": "passing_interceptions",
    "pass_2pt": "passing_2pt_conversions",
    "pass_att": "attempts",
    "pass_cmp": "completions",
    "pass_fd": "passing_first_downs",
    "pass_inc": "passing_inc",
    "pass_sack": "sacks_suffered",
    "bonus_pass_yd_300": "over_300_passing_yards",
    "bonus_pass_yd_400": "over_400_passing_yards",
    "bonus_pass_cmp_25": "over_25_completions",

    "rush_yd": "rushing_yards",
    "rush_td": "rushing_tds",
    "rush_2pt": "rushing_2pt_conversions",
    "rush_att": "carries",
    "rush_fd": "rushing_first_downs",
    "bonus_rush_yd_100": "over_100_rushing_yards",
    "bonus_rush_yd_200": "over_200_rushing_yards",
    "bonus_rush_att_20": "over_20_carries",

    "rec": "receptions",
    "rec_yd": "receiving_yards",
    "rec_td": "receiving_tds",
    "rec_2pt": "receiving_2pt_conversions",
    "rec_fd": "receiving_first_downs",
    "bonus_rec_wr": "wr_receptions",
    "bonus_rec_rb": "rb_receptions",
    "bonus_rec_te": "te_receptions",

    "fum_lost": "fumbles_lost",
}


def calculate_points_vectorized(stats_df: pl.DataFrame, scoring: ScoringSettings) -> pl.DataFrame:
    """Computes points scored based on stats and scoring settings"""
    weights = {f.name: getattr(scoring, f.name) for f in fields(scoring)}

    terms = [
        pl.col(stat_col) * weight
        for field_name, stat_col in SCORING_TO_STAT_COLUMN.items()
        if (weight := weights.get(field_name, 0.0)) != 0.0
        and stat_col in stats_df.columns
    ]

    if not terms:
        return stats_df.with_columns(pl.lit(0.0).alias("fantasy_points"))

    expr = terms[0]
    for term in terms[1:]:
        expr = expr + term

    return stats_df.with_columns(expr.alias("fantasy_points"))