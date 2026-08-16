
from dataclasses import dataclass, fields


@dataclass
class ScoringSettings:
    """Internal scoring rules for offensive skill positions (QB/RB/WR/TE).
    Kicker, team defense, IDP, and return/special-teams scoring are
    intentionally out of scope, see README."""

    # passing
    pass_yd: float = 0.0
    pass_td: float = 0.0
    pass_int: float = 0.0
    pass_2pt: float = 0.0
    pass_att: float = 0.0
    pass_cmp: float = 0.0
    pass_inc: float = 0.0
    pass_fd: float = 0.0
    pass_int_td: float = 0.0
    pass_sack: float = 0.0
    pass_td_40p: float = 0.0
    pass_td_50p: float = 0.0
    pass_cmp_40p: float = 0.0
    bonus_pass_yd_300: float = 0.0
    bonus_pass_yd_400: float = 0.0
    bonus_pass_cmp_25: float = 0.0
    bonus_fd_qb: float = 0.0

    # rushing
    rush_yd: float = 0.0
    rush_td: float = 0.0
    rush_2pt: float = 0.0
    rush_att: float = 0.0
    rush_fd: float = 0.0
    bonus_rush_yd_100: float = 0.0
    bonus_rush_yd_200: float = 0.0
    bonus_rush_att_20: float = 0.0
    bonus_fd_rb: float = 0.0

    # receiving
    rec: float = 0.0
    rec_yd: float = 0.0
    rec_td: float = 0.0
    rec_2pt: float = 0.0
    rec_fd: float = 0.0
    rec_td_40p: float = 0.0
    rec_td_50p: float = 0.0
    rec_40p: float = 0.0
    rec_0_4: float = 0.0
    rec_5_9: float = 0.0
    rec_10_19: float = 0.0
    rec_20_29: float = 0.0
    rec_30_39: float = 0.0
    bonus_rec_yd_100: float = 0.0
    bonus_rec_yd_200: float = 0.0
    bonus_rush_rec_yd_100: float = 0.0
    bonus_rush_rec_yd_200: float = 0.0
    bonus_rec_wr: float = 0.0
    bonus_rec_rb: float = 0.0
    bonus_rec_te: float = 0.0
    bonus_fd_wr: float = 0.0
    bonus_fd_te: float = 0.0

    # fumbles
    fum: float = 0.0
    fum_lost: float = 0.0
    fum_rec: float = 0.0
    fum_rec_td: float = 0.0
    fum_ret_yd: float = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> "ScoringSettings":
        """Builds ScoringSettings from a raw settings dict (e.g. Sleeper's
        scoring_settings payload). Unknown keys (kicker, defense, IDP, etc.)
        are silently dropped; missing keys fall back to the 0.0 default."""
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)