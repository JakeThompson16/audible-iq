
from dataclasses import dataclass, field, replace

import numpy as np
import polars as pl
from scipy import stats as sp_stats

from domain.scoring import ScoringSettings
from engine.scoring import calculate_points_vectorized
from projections.expected_points.features.epa_allowed import join_epa_allowed
from projections.expected_points.features.stat_rolling import RATE_STATS, add_stat_rolling_features


# Portable expected points: predict raw stat production (volume + rate), then
# score it with the unmodified calculate_points_vectorized. Fantasy points are
# never predicted directly — that would lock the model to one league's scoring.
#
# Position-specific pieces (which features feed which model, how the stat
# line is derived from volumes and rates, the feature window) live in a
# StatVectorSpec per position (rb.py, wr.py, te.py, qb.py); everything else
# here is shared.

KEY_COLUMNS = ["gsis_id", "season", "week", "position"]


# Rates bounded to [0, 1], and rates that can't be negative. Volumes are
# floored at 0 too. Yardage rates are left unclipped (negative ypc is real).
PROBABILITY_RATES = {"catch_rate", "completion_rate"}
NONNEGATIVE_RATES = {"rush_td_rate", "rec_td_rate", "pass_td_rate", "int_rate"}


def rush_receive_derivations(reception_bonus_col: str) -> tuple:
    """Derivation chain for RB/WR/TE: rushing + receiving stat lines."""
    return (
        ("rushing_yards", "carries", "*", "ypc"),
        ("rushing_tds", "carries", "*", "rush_td_rate"),
        ("receptions", "targets", "*", "catch_rate"),
        (reception_bonus_col, "receptions", "=", None),
        ("receiving_yards", "receptions", "*", "ypr"),
        ("receiving_tds", "receptions", "*", "rec_td_rate"),
    )


# QB: yards and TDs accrue on completions, interceptions on attempts.
PASS_RUSH_DERIVATIONS = (
    ("completions", "attempts", "*", "completion_rate"),
    ("passing_inc", "attempts", "-", "completions"),
    ("passing_yards", "completions", "*", "yards_per_completion"),
    ("passing_tds", "completions", "*", "pass_td_rate"),
    ("passing_interceptions", "attempts", "*", "int_rate"),
    ("rushing_yards", "carries", "*", "ypc"),
    ("rushing_tds", "carries", "*", "rush_td_rate"),
)


@dataclass(frozen=True)
class StatVectorSpec:
    position: str
    # target -> (features, denominator the target is a rate over, or None for volume).
    # Volume target names are stat column names (carries, targets, attempts).
    models: dict
    # Ordered (output stat, left, op, right): "*" -> left * pred_<right rate>,
    # "-" -> left - right, "=" -> copy of left. `left`/`right` are volume
    # stats or earlier outputs. Multiplied, never fitted.
    derivations: tuple
    # target -> extra features that were evaluated but are OFF by default
    # (not validated: unstable sign or not significant across folds). Kept so
    # they can be re-tested on more data: spec.with_candidates().
    candidate_terms: dict = field(default_factory=dict)
    # Trailing window (games) for this position's stat-vector features,
    # chosen per position (README).
    window: int = 8

    def with_candidates(self) -> "StatVectorSpec":
        """This spec with every candidate term switched on (for re-testing only)."""
        models = {
            t: (feats + [c for c in self.candidate_terms.get(t, []) if c not in feats], den)
            for t, (feats, den) in self.models.items()
        }
        return replace(self, models=models, candidate_terms={})

    @property
    def volume_targets(self) -> list[str]:
        return [t for t, (_, den) in self.models.items() if den is None]

    @property
    def rate_targets(self) -> list[str]:
        return [t for t, (_, den) in self.models.items() if den is not None]

    @property
    def recomposed_columns(self) -> list[str]:
        # Anything calculate_points_vectorized maps that is NOT here (2-pt
        # conversions, first downs, fumbles_lost, sacks, yardage/carry/
        # completion threshold bonuses) is absent from the scored frame and
        # contributes 0 — see OPEN_QUESTIONS.md Q-8.
        return self.volume_targets + [out for out, *_ in self.derivations]


@dataclass
class OLSFit:
    target: str
    features: list[str]
    coef: np.ndarray        # [intercept, *features]
    se: np.ndarray
    n: int
    r2: float

    def summary(self) -> pl.DataFrame:
        dof = self.n - len(self.coef)
        t = self.coef / self.se
        p = 2 * sp_stats.t.sf(np.abs(t), dof)
        crit = sp_stats.t.ppf(0.975, dof)
        return pl.DataFrame({
            "term": ["intercept"] + self.features,
            "coef": self.coef,
            "se": self.se,
            "ci_low": self.coef - crit * self.se,
            "ci_high": self.coef + crit * self.se,
            "t": t,
            "p": p,
        })

    def predict(self, df: pl.DataFrame) -> pl.Expr:
        expr = pl.lit(float(self.coef[0]))
        for name, b in zip(self.features, self.coef[1:]):
            expr = expr + pl.col(name) * float(b)
        return expr


def _design(df: pl.DataFrame, features: list[str]) -> np.ndarray:
    x = df.select(features).to_numpy().astype(float)
    return np.column_stack([np.ones(len(x)), x])


def fit_ols(df: pl.DataFrame, target: str, features: list[str], weight: str | None = None) -> OLSFit:
    """
    OLS with intercept (WLS when `weight` is given). SE from the classical
    covariance sigma^2 (X'WX)^-1. Rows with any null in target/features are
    dropped.
    """
    df = df.drop_nulls(subset=[target] + features)
    X = _design(df, features)
    y = df[target].to_numpy().astype(float)
    w = np.ones(len(y)) if weight is None else df[weight].to_numpy().astype(float)

    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    resid = y - X @ coef
    n, p = X.shape
    sigma2 = (w * resid) @ resid / (n - p)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ (X * w[:, None]))))
    y_bar = np.average(y, weights=w)
    r2 = 1 - ((w * resid) @ resid) / ((w * (y - y_bar)) @ (y - y_bar))

    return OLSFit(target=target, features=features, coef=coef, se=se, n=n, r2=float(r2))


def vif(df: pl.DataFrame, features: list[str]) -> dict[str, float]:
    """Variance inflation factor per feature: 1 / (1 - R^2 of it on the others)."""
    df = df.drop_nulls(subset=features)
    out = {}
    for f in features:
        others = [o for o in features if o != f]
        X = _design(df, others)
        y = df[f].to_numpy().astype(float)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
        out[f] = float("inf") if r2 >= 1 else 1 / (1 - r2)
    return out


def rate_priors(train_df: pl.DataFrame) -> dict[str, float]:
    """Pooled rate per rate stat over the training rows: sum(num) / sum(den)."""
    priors = {}
    for rate, (num, den) in RATE_STATS.items():
        den_sum = train_df[den].sum()
        priors[rate] = train_df[num].sum() / den_sum if den_sum else None
    return priors


def apply_rate_priors(df: pl.DataFrame, priors: dict[str, float]) -> pl.DataFrame:
    """
    roll_<rate> is null when the player had no attempts of that kind in the
    window (e.g. no receptions in 8 games -> no ypr). Fill it with the pooled
    training rate for the position so a small predicted volume can still be
    scored, but only for players with history (roll_carries non-null). True
    rookies stay null — no projection, per the CLAUDE.md rookie rule.
    """
    return df.with_columns([
        pl.when(pl.col("roll_carries").is_not_null())
        .then(pl.col(f"roll_{rate}").fill_null(prior))
        .otherwise(pl.col(f"roll_{rate}"))
        .alias(f"roll_{rate}")
        for rate, prior in priors.items()
    ])


def build_features(rows: pl.DataFrame, epa_df: pl.DataFrame, window: int) -> pl.DataFrame:
    """Per-stat trailing features + epa_allowed for one position's player-week rows (full history)."""
    return join_epa_allowed(add_stat_rolling_features(rows, window=window), epa_df)


def fit_models(train_df: pl.DataFrame, spec: StatVectorSpec, weight_rates: bool = True) -> dict[str, OLSFit]:
    """
    :param train_df: feature rows (build_features + apply_rate_priors) for training seasons
    :param weight_rates: weight each rate model's rows by its denominator
        (carries / targets / receptions). Default on. Unweighted, a 1-carry
        game counts as much as a 25-carry game, so the fitted "rate" is a
        per-game average that runs below the per-carry rate actually multiplied
        by predicted volume (RB 2024 ypc: 4.18 per-game vs 4.39 pooled), which
        biases recomposed points low.
    Rate models fit only on rows where the rate is defined (denominator > 0).
    """
    fits = {}
    for target, (features, den) in spec.models.items():
        rows = train_df if den is None else train_df.filter(pl.col(den) > 0)
        weight = den if (weight_rates and den is not None) else None
        fits[target] = fit_ols(rows, target, features, weight=weight)
    return fits


def recompose_points(stat_vector_df: pl.DataFrame, spec: StatVectorSpec, scoring: ScoringSettings) -> pl.DataFrame:
    """
    Score a (predicted or actual) stat vector with the unmodified league
    scoring. Only KEY_COLUMNS + the spec's recomposed columns are passed in, so
    no other real stat can leak into the points.
    """
    return calculate_points_vectorized(
        stat_vector_df.select(KEY_COLUMNS + spec.recomposed_columns), scoring
    )


@dataclass
class StatVectorModel:
    """A fitted stat-vector model for one position: what engine/expected_points.py consumes."""
    spec: StatVectorSpec
    fits: dict
    priors: dict[str, float]
    window: int
    train_seasons: list[int]

    def predict_stat_vector(self, features: pl.DataFrame) -> pl.DataFrame:
        """
        :param features: build_features output for this position
        :return: KEY_COLUMNS + pred_<target> + the recomposed stat vector
            (derived by multiplication, not fitted). Null wherever an input
            feature is null (no history -> no projection).
        """
        spec = self.spec
        preds = [f"pred_{t}" for t in spec.models]
        df = apply_rate_priors(features, self.priors)
        df = df.with_columns([self.fits[t].predict(df).alias(f"pred_{t}") for t in spec.models])

        # Keep predictions physically possible. clip(lower_bound, upper_bound).
        clips = []
        for t in spec.models:
            if t in PROBABILITY_RATES:
                clips.append(pl.col(f"pred_{t}").clip(lower_bound=0.0, upper_bound=1.0))
            elif t in NONNEGATIVE_RATES or t in spec.volume_targets:
                clips.append(pl.col(f"pred_{t}").clip(lower_bound=0.0))
        df = df.with_columns(clips).select(KEY_COLUMNS + preds)

        # Volume predictions become the stat columns, then the derivation chain
        # runs in order (each step may use an earlier output).
        df = df.with_columns([pl.col(f"pred_{v}").alias(v) for v in spec.volume_targets])
        for out, left, op, right in spec.derivations:
            if op == "*":
                expr = pl.col(left) * pl.col(f"pred_{right}")
            elif op == "-":
                expr = pl.col(left) - pl.col(right)
            elif op == "=":
                expr = pl.col(left)
            else:
                raise ValueError(f"unknown derivation op {op!r} for {out}")
            df = df.with_columns(expr.alias(out))

        return df.select(KEY_COLUMNS + preds + spec.recomposed_columns)

    def predict_points(self, rows: pl.DataFrame, epa_df: pl.DataFrame, scoring: ScoringSettings) -> pl.DataFrame:
        """
        :param rows: this position's player-week rows (full history, so the
            trailing windows can reach back)
        :return: KEY_COLUMNS + projection (league-scored predicted stat vector)
        """
        vec = self.predict_stat_vector(build_features(rows, epa_df, self.window))
        return recompose_points(vec, self.spec, scoring) \
            .select(KEY_COLUMNS + ["fantasy_points"]).rename({"fantasy_points": "projection"})


def fit_stat_vector(
        rows: pl.DataFrame,
        epa_df: pl.DataFrame,
        spec: StatVectorSpec,
        train_seasons: list[int],
        window: int | None = None,
        weight_rates: bool = True) -> StatVectorModel:
    """
    :param rows: this position's player-week rows (all loaded seasons)
    :param train_seasons: seasons whose rows the models are fit on
    :param window: stat-vector feature window; None -> spec.window
    Offline step: the engine only consumes the returned StatVectorModel.
    """
    window = spec.window if window is None else window
    features = build_features(rows.filter(pl.col("position") == spec.position), epa_df, window)
    train = features.filter(pl.col("season").is_in(train_seasons))
    priors = rate_priors(train)
    fits = fit_models(apply_rate_priors(train, priors), spec, weight_rates=weight_rates)
    return StatVectorModel(spec=spec, fits=fits, priors=priors, window=window, train_seasons=train_seasons)
