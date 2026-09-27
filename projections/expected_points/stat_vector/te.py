
from projections.expected_points.stat_vector.core import StatVectorSpec, rush_receive_derivations


# TE stat vector (LOSO 2019-2025, see README): WR's final spec refit on TE
# data, no structural changes, every matchup term re-tested on TE's own folds.
# Matchup terms ON (validated on TE):
#   catch_rate  + epa_allowed_pass  positive in 7/7 folds, significant in 5/7
#   ypr         + epa_allowed_pass  positive and significant in 7/7 (p <= 0.008)
#   rec_td_rate + epa_allowed_pass  positive and significant in 7/7 (p <= 0.025)
# The ypr / rec_td_rate terms did NOT clear significance for WR; TEs differ.
# OFF (candidate): epa_allowed_pass in the targets volume model — positive in
# 7/7 but significant in only 3/7 (p 0.03-0.13). Rushing-side rates keep
# epa_allowed_rush for completeness (noise; TE carries are tiny).
# Window 10: chosen by select_by_policy over {8, 10, 12, 14, 16, 20}.
TE_SPEC = StatVectorSpec(
    position="TE",
    models={
        "targets":      (["roll_targets", "delta_targets"], None),
        "carries":      (["roll_carries", "delta_carries"], None),
        "catch_rate":   (["roll_catch_rate", "delta_targets", "epa_allowed_pass"], "targets"),
        "ypr":          (["roll_ypr", "delta_targets", "epa_allowed_pass"], "receptions"),
        "rec_td_rate":  (["roll_rec_td_rate", "delta_targets", "epa_allowed_pass"], "receptions"),
        "ypc":          (["roll_ypc", "delta_carries", "epa_allowed_rush"], "carries"),
        "rush_td_rate": (["roll_rush_td_rate", "delta_carries", "epa_allowed_rush"], "carries"),
    },
    derivations=rush_receive_derivations("te_receptions"),
    candidate_terms={
        "targets": ["epa_allowed_pass"],
    },
    window=10,
)
