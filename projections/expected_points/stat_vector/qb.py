
from projections.expected_points.stat_vector.core import PASS_RUSH_DERIVATIONS, StatVectorSpec


# QB stat vector: its own spec, not a copy of RB/WR.
# Volume: pass attempts and rush attempts. Rush attempts include scrambles
# (play_type == 'run' upstream counts them as carries, matching official
# stats). epa_allowed on both volume models is re-tested via candidate_terms.
# Rates and the denominators they are per:
#   completion_rate       per attempt      + epa_allowed_pass  (expect +)
#   yards_per_completion  per completion   + epa_allowed_pass  (expect +)
#   pass_td_rate          per completion   + epa_allowed_pass  (expect +)
#   int_rate              per attempt      + epa_allowed_pass  (expect NEGATIVE:
#                         a softer pass defense forces fewer interceptions)
#   ypc                   per rush attempt + epa_allowed_rush  (expect +)
#   rush_td_rate          per rush attempt + epa_allowed_rush  (expect +)
# Derivation: completions = attempts x completion_rate; passing yards/TDs =
# completions x rate; interceptions = attempts x int_rate; incompletions =
# attempts - completions; rushing yards/TDs = rush attempts x rate. pass_int's
# negative league weight is applied by calculate_points_vectorized as usual.
# Known open item: scrambles and designed runs share one rush volume and one
# ypc model (not split in this pass).
# STATUS: evaluated, NOT switched on (LOSO 2019-2025: better R² 0.238 vs 0.220
# and bias, but Spearman 0.483 vs 0.487 with 2/7 fold wins, MAE +0.016; see
# README / OPEN_QUESTIONS Q-14). Window 12 chosen by select_by_policy.
# Validated signs: completion_rate +, yards_per_completion +, pass_td_rate +,
# int_rate NEGATIVE (7/7 folds significant) — the expected direction.
QB_SPEC = StatVectorSpec(
    position="QB",
    models={
        "attempts":             (["roll_attempts", "delta_attempts"], None),
        "carries":              (["roll_carries", "delta_carries"], None),
        "completion_rate":      (["roll_completion_rate", "delta_attempts", "epa_allowed_pass"], "attempts"),
        "yards_per_completion": (["roll_yards_per_completion", "delta_attempts", "epa_allowed_pass"], "completions"),
        "pass_td_rate":         (["roll_pass_td_rate", "delta_attempts", "epa_allowed_pass"], "completions"),
        "int_rate":             (["roll_int_rate", "delta_attempts", "epa_allowed_pass"], "attempts"),
        "ypc":                  (["roll_ypc", "delta_carries", "epa_allowed_rush"], "carries"),
        "rush_td_rate":         (["roll_rush_td_rate", "delta_carries", "epa_allowed_rush"], "carries"),
    },
    derivations=PASS_RUSH_DERIVATIONS,
    candidate_terms={
        "attempts": ["epa_allowed_pass"],
        "carries": ["epa_allowed_rush"],
    },
    window=12,
)
