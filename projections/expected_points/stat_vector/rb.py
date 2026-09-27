
from projections.expected_points.stat_vector.core import StatVectorSpec


# RB stat vector (validated 2026-09-27, LOSO 2019-2025; see README).
# Volume models: trailing average + recent-usage delta, no matchup term —
# epa_allowed on carries flipped sign between seasons and removing it did not
# change accuracy. Rate models: trailing rate + delta + the matchup term for
# the play type that produces the stat (rush-side: epa_allowed_rush,
# receiving-side: epa_allowed_pass).
RB_SPEC = StatVectorSpec(
    position="RB",
    models={
        "carries":      (["roll_carries", "delta_carries"], None),
        "targets":      (["roll_targets", "delta_targets"], None),
        "ypc":          (["roll_ypc", "delta_carries", "epa_allowed_rush"], "carries"),
        "rush_td_rate": (["roll_rush_td_rate", "delta_carries", "epa_allowed_rush"], "carries"),
        "catch_rate":   (["roll_catch_rate", "delta_targets", "epa_allowed_pass"], "targets"),
        "ypr":          (["roll_ypr", "delta_targets", "epa_allowed_pass"], "receptions"),
        "rec_td_rate":  (["roll_rec_td_rate", "delta_targets", "epa_allowed_pass"], "receptions"),
    },
    reception_bonus_col="rb_receptions",
)
