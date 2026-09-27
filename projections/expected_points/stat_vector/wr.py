
from projections.expected_points.stat_vector.core import StatVectorSpec, rush_receive_derivations


# WR stat vector (LOSO 2019-2025, see README). Same structure as RB: targets is
# the primary volume model; carries is kept (near-zero for most WRs, covers
# jet sweeps / gadget usage).
# Matchup terms: epa_allowed_pass is ON only for catch_rate, the one receiving
# term that was positive and significant in every fold. It is OFF for the
# targets volume model (re-tested for WR: sign flips, p >= 0.48 every fold, no
# accuracy change) and for ypr / rec_td_rate (not significant); those stay as
# candidate terms for re-testing on more data. The rushing-side rates keep
# epa_allowed_rush as specified for completeness (WR carries are tiny).
WR_SPEC = StatVectorSpec(
    position="WR",
    models={
        "targets":      (["roll_targets", "delta_targets"], None),
        "carries":      (["roll_carries", "delta_carries"], None),
        "catch_rate":   (["roll_catch_rate", "delta_targets", "epa_allowed_pass"], "targets"),
        "ypr":          (["roll_ypr", "delta_targets"], "receptions"),
        "rec_td_rate":  (["roll_rec_td_rate", "delta_targets"], "receptions"),
        "ypc":          (["roll_ypc", "delta_carries", "epa_allowed_rush"], "carries"),
        "rush_td_rate": (["roll_rush_td_rate", "delta_carries", "epa_allowed_rush"], "carries"),
    },
    derivations=rush_receive_derivations("wr_receptions"),
    window=8,
    candidate_terms={
        "targets": ["epa_allowed_pass"],
        "ypr": ["epa_allowed_pass"],
        "rec_td_rate": ["epa_allowed_pass"],
    },
)
