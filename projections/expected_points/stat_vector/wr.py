
from projections.expected_points.stat_vector.core import StatVectorSpec


# WR stat vector (UNDER REVIEW — not wired into POSITION_IMPLEMENTATIONS).
# Same structure as RB: targets is the primary volume model; carries is kept
# (expected near-zero for most WRs, covers jet sweeps / gadget usage). Matchup
# terms live in the rate models, by the play type that produces the stat.
# Whether epa_allowed_pass belongs in the targets volume model is re-tested
# for WR (stat_vector_eval.py WR --targets-epa), not inherited from RB.
WR_SPEC = StatVectorSpec(
    position="WR",
    models={
        "targets":      (["roll_targets", "delta_targets"], None),
        "carries":      (["roll_carries", "delta_carries"], None),
        "catch_rate":   (["roll_catch_rate", "delta_targets", "epa_allowed_pass"], "targets"),
        "ypr":          (["roll_ypr", "delta_targets", "epa_allowed_pass"], "receptions"),
        "rec_td_rate":  (["roll_rec_td_rate", "delta_targets", "epa_allowed_pass"], "receptions"),
        "ypc":          (["roll_ypc", "delta_carries", "epa_allowed_rush"], "carries"),
        "rush_td_rate": (["roll_rush_td_rate", "delta_carries", "epa_allowed_rush"], "carries"),
    },
    reception_bonus_col="wr_receptions",
)
