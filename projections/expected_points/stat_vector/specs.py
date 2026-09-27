
from projections.expected_points.stat_vector.rb import RB_SPEC
from projections.expected_points.stat_vector.wr import WR_SPEC


# Positions with a stat-vector spec. Adding a position here does not switch it
# over; engine/expected_points.py POSITION_IMPLEMENTATIONS does that (WR is
# specced for evaluation only, still rolling_plus_skew in production).
STAT_VECTOR_SPECS = {
    "RB": RB_SPEC,
    "WR": WR_SPEC,
}
