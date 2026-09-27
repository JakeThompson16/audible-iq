
from projections.expected_points.stat_vector.qb import QB_SPEC
from projections.expected_points.stat_vector.rb import RB_SPEC
from projections.expected_points.stat_vector.te import TE_SPEC
from projections.expected_points.stat_vector.wr import WR_SPEC


# Positions with a stat-vector spec. Adding a position here does not switch it
# over; engine/expected_points.py POSITION_IMPLEMENTATIONS does that, only
# after LOSO validation (README).
STAT_VECTOR_SPECS = {
    "RB": RB_SPEC,
    "WR": WR_SPEC,
    "TE": TE_SPEC,
    "QB": QB_SPEC,
}
