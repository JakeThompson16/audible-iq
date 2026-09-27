
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import stats as sp_stats

from domain.scoring import ScoringSettings
from engine.scoring import calculate_points_vectorized
from projections.expected_points.features.stat_rolling import RATE_STATS


# Portable expected points, RB proof of concept: predict raw stat production
# (volume + rate), then score it with the unmodified calculate_points_vectorized.
# Fantasy points are never predicted directly — that would lock the model to
# one league's scoring settings.
#
# target -> (features, denominator the target is a rate over, or None for volume).
# Rush-side rate models use epa_allowed_rush, receiving-side epa_allowed_pass.
# Volume models carry no matchup term: epa_allowed on carries flipped sign
# between fits (+15.1 vs -1.0), and dropping it did not change recomposed
# accuracy (see README "RB stat-vector projection").
RB_MODELS = {
    "carries":      (["roll_carries", "delta_carries"], None),
    "targets":      (["roll_targets", "delta_targets"], None),
    "ypc":          (["roll_ypc", "delta_carries", "epa_allowed_rush"], "carries"),
    "rush_td_rate": (["roll_rush_td_rate", "delta_carries", "epa_allowed_rush"], "carries"),
    "catch_rate":   (["roll_catch_rate", "delta_targets", "epa_allowed_pass"], "targets"),
    "ypr":          (["roll_ypr", "delta_targets", "epa_allowed_pass"], "receptions"),
    "rec_td_rate":  (["roll_rec_td_rate", "delta_targets", "epa_allowed_pass"], "receptions"),
}

# First-iteration spec (epa_allowed in the volume models too), kept only so the
# eval driver can report what removing it changed.
LEGACY_RB_MODELS = {
    **RB_MODELS,
    "carries": (["roll_carries", "delta_carries", "epa_allowed_rush"], None),
    "targets": (["roll_targets", "delta_targets", "epa_allowed_pass"], None),
}

VOLUME_TARGETS = [t for t, (_, den) in RB_MODELS.items() if den is None]
RATE_TARGETS = [t for t, (_, den) in RB_MODELS.items() if den is not None]

# Stat columns produced by predict_rb_stat_vector and scored downstream. Anything
# calculate_points_vectorized maps that is NOT here (2-pt conversions, first
# downs, fumbles_lost, 100/200-yard and 20-carry bonuses) is absent from the
# frame and therefore contributes 0 — see OPEN_QUESTIONS.md Q-8.
KEY_COLUMNS = ["gsis_id", "season", "week", "position"]

RECOMPOSED_STAT_COLUMNS = [
    "carries", "rushing_yards", "rushing_tds",
    "targets", "receptions", "rb_receptions", "receiving_yards", "receiving_tds",
]


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


@dataclass
class ConstantRate:
    """Rate 'model' with no regression: the pooled training-season RB rate."""
    target: str
    value: float

    def predict(self, df: pl.DataFrame) -> pl.Expr:
        return pl.lit(self.value)


@dataclass
class SeasonToDateRate:
    """
    Rate 'model' with no regression: the player's own season-to-date rate
    (std_<rate>, prior games this season only), falling back to the pooled
    training-season RB rate until the player has attempts this season.
    """
    target: str
    fallback: float

    def predict(self, df: pl.DataFrame) -> pl.Expr:
        return pl.col(f"std_{self.target}").fill_null(self.fallback)


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
    """Pooled RB rate per rate stat over the training rows: sum(num) / sum(den)."""
    return {
        rate: train_df[num].sum() / train_df[den].sum()
        for rate, (num, den) in RATE_STATS.items()
    }


def apply_rate_priors(df: pl.DataFrame, priors: dict[str, float]) -> pl.DataFrame:
    """
    roll_<rate> is null when the player had no attempts of that kind in the
    window (e.g. no receptions in 8 games -> no ypr). Fill it with the pooled
    training-season RB rate so a small predicted volume can still be scored,
    but only for players with history (roll_carries non-null). True rookies
    stay null — no projection, per the CLAUDE.md rookie rule.
    """
    return df.with_columns([
        pl.when(pl.col("roll_carries").is_not_null())
        .then(pl.col(f"roll_{rate}").fill_null(prior))
        .otherwise(pl.col(f"roll_{rate}"))
        .alias(f"roll_{rate}")
        for rate, prior in priors.items()
    ])


def fit_rb_models(
        train_df: pl.DataFrame,
        weight_rates: bool = True,
        models: dict = RB_MODELS) -> dict[str, OLSFit]:
    """
    :param train_df: RB player-week rows with stat_rolling + epa_allowed features
    :param weight_rates: weight each rate model's rows by its denominator
        (carries / targets / receptions). Default on. Unweighted, a 1-carry
        game counts as much as a 25-carry game, so the fitted "rate" is a
        per-game average that runs below the per-carry rate actually multiplied
        by predicted volume (2024 ypc: 4.18 per-game vs 4.39 pooled; rush TD
        rate: 0.027 vs 0.033), which biases recomposed points low.
    :param models: model spec (RB_MODELS, or LEGACY_RB_MODELS for comparison)
    Rate models fit only on rows where the rate is defined (denominator > 0).
    """
    fits = {}
    for target, (features, den) in models.items():
        rows = train_df if den is None else train_df.filter(pl.col(den) > 0)
        weight = den if (weight_rates and den is not None) else None
        fits[target] = fit_ols(rows, target, features, weight=weight)
    return fits


def with_unfitted_rates(fits: dict, priors: dict[str, float], mode: str) -> dict:
    """
    Replace the five fitted rate models with a no-regression rate, keeping the
    fitted volume models. mode: "league" (ConstantRate at the pooled training
    rate) or "season_to_date" (SeasonToDateRate, falling back to that rate).
    """
    cls = {"league": lambda t: ConstantRate(t, priors[t]),
           "season_to_date": lambda t: SeasonToDateRate(t, priors[t])}[mode]
    return {**fits, **{t: cls(t) for t in RATE_TARGETS}}


def predict_rb_stat_vector(df: pl.DataFrame, fits: dict) -> pl.DataFrame:
    """
    :param fits: target -> anything with .predict(df) -> pl.Expr (OLSFit,
        ConstantRate, SeasonToDateRate)
    :return: df with pred_<target> for every model, and the recomposed stat
        vector in RECOMPOSED_STAT_COLUMNS (derived by multiplication, not
        fitted), keyed by KEY_COLUMNS. A row is null wherever any input
        feature is null (no history -> no projection).
    """
    df = df.with_columns([
        fits[t].predict(df).alias(f"pred_{t}") for t in RB_MODELS
    ])

    # Keep predictions physically possible. clip(lower_bound, upper_bound).
    df = df.with_columns(
        pl.col("pred_carries").clip(lower_bound=0.0),
        pl.col("pred_targets").clip(lower_bound=0.0),
        pl.col("pred_rush_td_rate").clip(lower_bound=0.0),
        pl.col("pred_rec_td_rate").clip(lower_bound=0.0),
        pl.col("pred_catch_rate").clip(lower_bound=0.0, upper_bound=1.0),
    )

    df = df.select(KEY_COLUMNS + [f"pred_{t}" for t in RB_MODELS])

    df = df.with_columns(
        pl.col("pred_carries").alias("carries"),
        (pl.col("pred_carries") * pl.col("pred_ypc")).alias("rushing_yards"),
        (pl.col("pred_carries") * pl.col("pred_rush_td_rate")).alias("rushing_tds"),
        pl.col("pred_targets").alias("targets"),
        (pl.col("pred_targets") * pl.col("pred_catch_rate")).alias("receptions"),
    ).with_columns(
        pl.col("receptions").alias("rb_receptions"),
        (pl.col("receptions") * pl.col("pred_ypr")).alias("receiving_yards"),
        (pl.col("receptions") * pl.col("pred_rec_td_rate")).alias("receiving_tds"),
    )

    return df.select(KEY_COLUMNS + [f"pred_{t}" for t in RB_MODELS] + RECOMPOSED_STAT_COLUMNS)


def recompose_points(stat_vector_df: pl.DataFrame, scoring: ScoringSettings) -> pl.DataFrame:
    """
    Score a (predicted or actual) stat vector with the unmodified league
    scoring. Only KEY_COLUMNS + RECOMPOSED_STAT_COLUMNS are passed in, so no
    other real stat can leak into the points.
    """
    return calculate_points_vectorized(
        stat_vector_df.select(KEY_COLUMNS + RECOMPOSED_STAT_COLUMNS), scoring
    )
