# Audible IQ

A fantasy football decision-support tool that pulls live league data from Sleeper, computes league-accurate fantasy scoring, and (in progress) builds matchup-adjusted projections and boom/bust probabilities to power start/sit recommendations.

## Status: Early development

Currently implemented:
- **Sleeper integration**: pulls leagues, rosters, users, and scoring settings via Sleeper's public API
- **Player data pipeline**: pulls weekly stats and player metadata via `nflreadpy` (nflverse), joined against a Sleeper/GSIS ID crosswalk
- **League-accurate scoring engine**: computes fantasy points via a vectorized dot product of each league's actual scoring settings against raw stat lines (supports PPR variants, TE premium, yardage/completion bonus thresholds, etc.)
- **Validated**: scoring output cross-checked against real Sleeper league data (Trey McBride, full season) with exact matches
- **Weekly point projection model**: trailing rolling average (blended with prior-season data early in the season) plus an opponent-adjusted matchup skew (Adjusted Points Allowed), with a confidence tier per projection — see [Model Evaluation](#model-evaluation) below for how well this actually performs

In progress:
- Boom/bust probability classifier
- Agentic start/sit recommendation layer

## Architecture

The project is split into layers to keep external API/data source dependencies isolated from core logic:

- `domain/` — core objects (`Player`, `Team`, `League`, `ScoringSettings`), platform-agnostic
- `clients/` — raw API/data-source wrappers (Sleeper API, nflreadpy)
- `engine/` — computation logic that operates across domain objects (scoring calculation, upcoming: projections, boom/bust)

This separation means adding a new platform (e.g. ESPN, Yahoo) or data source only requires a new adapter/client, not changes to core logic.

## Scope

Currently focused on offensive skill positions (QB/RB/WR/TE). Kicker, team defense/IDP, and play-level long-touchdown bonus categories are intentionally out of scope, see code comments for details. The experimental RB stat-vector projection also omits 2-pt conversions, first downs, fumbles, and yardage/carry threshold bonuses (they score 0).

## Model Evaluation

Projection = `rolling_avg_prior` (trailing form, blended with prior-season data early in the season) + `opponent_skew` (Adjusted Points Allowed, an opponent-adjusted matchup residual). Evaluated via `engine/metrics.py` against 2024/25 season data (43-44 weeks per position, ~13.6k scored player-weeks), comparing the full projection against a `rolling_avg_prior`-only baseline to test whether opponent_skew is actually earning its complexity.

### Does opponent_skew help?

`opponent_skew` was originally hard-clipped at ±8 to control extreme values, which turned out to concentrate at low sample sizes (`n_games` 3-5). That clip was replaced with sample-size-weighted shrinkage — `skew_shrunk = skew_raw * (n / (n + k))`, pulling low-`n` estimates toward 0 while trusting well-supported ones — with a wide ±12 clip kept only as a guardrail against pathological edge cases, not as the flattening mechanism.

`k` was chosen by grid search (k = 2, 4, 8, 16) against held-out 2024/25 metrics, not fit as a model parameter — projection is the reference point boom/bust will measure deviation against, so it stays free of anything resembling a fitted parameter. Results were monotonic across every k tested:

| k | Pos | Projection MAE | Baseline MAE | Projection R² | Baseline R² | Skew helps MAE? | Spearman (rank corr.) |
|---|-----|----------------:|--------------:|----------------:|--------------:|:----------------:|----------------------:|
| 2  | QB | 8.465 | 8.497 | 0.108 | 0.101 | **Yes** | 0.399 |
| 2  | RB | 4.767 | 4.713 | 0.387 | 0.392 | No  | 0.701 |
| 2  | WR | 4.815 | 4.740 | 0.301 | 0.314 | No  | 0.615 |
| 2  | TE | 4.966 | 4.929 | 0.304 | 0.311 | No  | 0.586 |
| 8  | QB | 8.470 | 8.497 | 0.107 | 0.101 | **Yes** | 0.397 |
| 8  | RB | 4.723 | 4.713 | 0.393 | 0.392 | No  | 0.705 |
| 8  | WR | 4.762 | 4.740 | 0.312 | 0.314 | No  | 0.619 |
| 8  | TE | 4.934 | 4.929 | 0.311 | 0.311 | No (tie) | 0.588 |
| **16** | QB | 8.479 | 8.497 | 0.105 | 0.101 | **Yes** | 0.395 |
| **16** | RB | 4.711 | 4.713 | **0.395** | 0.392 | **Yes** | 0.706 |
| **16** | WR | 4.743 | 4.740 | **0.315** | 0.314 | No (tie) | 0.621 |
| **16** | TE | 4.928 | 4.929 | **0.313** | 0.311 | **Yes** | 0.589 |

**`k = 16`** is the current default: RB and TE flip to net-positive, WR is a statistical tie (R² edges ahead), and QB retains a consistent advantage across every k tested. Confidence-tier calibration (`monotonic_decreasing_mae`) was `False` for every position at every k — a pre-existing, separately-tracked issue with the provisional confidence thresholds, unaffected by this change.

### Why does opponent_skew barely move the needle for RB/WR/TE?

Segmenting the R² delta (projection − baseline) by `n_games` (the sample size backing each opponent_skew estimate) at k=16, for RB/WR/TE combined:

| n_games bucket | rows | Projection R² | Baseline R² | Δ R² |
|----------------|-----:|---------------:|--------------:|------:|
| 3-5   | 1,782 | 0.393 | 0.392 | +0.002 |
| 6-8   | 1,721 | 0.374 | 0.369 | +0.004 |
| 9-11  | 1,390 | 0.384 | 0.383 | +0.001 |
| 12+   | 1,406 | 0.334 | 0.330 | +0.003 |

If the pre-shrinkage problem had been low-sample noise, the delta should be largest where `n_games` is smallest and shrink toward the raw (unshrunk) gap as sample size grows. It doesn't — the improvement is small and roughly flat across every bucket, including the best-supported one. That points to a different conclusion than "shrinkage fixed a noise problem": **opponent-matchup residual carries very little signal for RB/WR/TE at the current feature granularity, at any sample size** — shrinkage mostly neutralized the harm rather than unlocking real predictive value. QB is the exception, where skew is a consistent, real improvement over the baseline.

**Takeaway**: treat `opponent_skew` as reliable for QB. For RB/WR/TE it's now harmless rather than genuinely helpful — worth leaving enabled (it doesn't hurt) but revisiting with finer-grained matchup features rather than trusting it as a meaningful signal.

## QB rush-attempt weight

Learned with `qb_rush_weight.py`: OLS of QB weekly fantasy points on pass attempts and rush attempts (2023-25, QB player-weeks with 10+ attempts, n=1,779, league scoring settings), weight = rush coefficient / pass coefficient.

| | Points per attempt | Weight (pass attempt = 1.0) |
|---|---:|---:|
| Pass attempt | 0.402 | 1.00 |
| Rush attempt | 0.942 | **2.34** (95% bootstrap CI 1.95-2.78) |

By season: 2023 1.96, 2024 2.69, 2025 2.31. Without an intercept: 2.14. R² of the two-variable fit is 0.21.

Re-run on standard PPR scoring (nflreadpy's `fantasy_points_ppr`, same 1,779 QB-weeks):

| | Points per attempt | Weight (pass attempt = 1.0) |
|---|---:|---:|
| Pass attempt | 0.320 | 1.00 |
| Rush attempt | 0.952 | **2.98** (95% bootstrap CI 2.52-3.49) |

By season: 2023 2.48, 2024 3.47, 2025 2.90. Without an intercept: 2.71. R² 0.24. The rush yield is nearly identical across scoring systems (0.94 vs 0.95 points per rush); the difference comes from pass attempts being worth less under standard PPR (0.32 vs 0.40), so the weight is sensitive to how the league scores passing.

Limits: the weight is the average scoring yield per attempt under this league's scoring, so it changes with scoring settings (e.g. pass TD value). It is descriptive (same-week), not a causal or predictive weight, and it is not yet used in the pipeline; QB skew still filters on `trailing_attempts_avg` only.

## RB stat-vector projection (experimental, not in production)

Instead of projecting fantasy points (which ties the model to one league's scoring), predict the raw RB stat line and score it with the unmodified `calculate_points_vectorized`.

- **Volume models** (carries, targets): OLS on the trailing 8-game average and a recent-usage delta (3-game minus 8-game mean). No matchup term. `epa_allowed` was removed after its coefficient flipped sign between seasons, and removing it didn't change accuracy (LOSO mean MAE 4.671 vs 4.672).
- **Rate models** (ypc, rush TD/carry, catch rate, yards/reception, rec TD/reception): OLS weighted by attempts (the default), on the trailing rate, the usage delta, and `epa_allowed_rush`/`epa_allowed_pass`. That's opponent EPA/play allowed vs league, shifted and blended like opponent skew.
- Yards and TDs are derived by multiplication.

Run: `python rb_stat_vector_eval.py`.

Held-out RB results, leave-one-season-out: each season 2019-2025 is predicted by a model fit on the other six. Same rows for every method in a fold, `engine/metrics.py` harness, one Sleeper league's scoring. Each cell is MAE / R² / Spearman.

| Test season | Stat vector | Current `rolling_avg_prior + opponent_skew` | `rolling_avg_prior` only |
|---|---|---|---|
| 2019 | **4.851** / **0.388** / **0.702** | 4.949 / 0.365 / 0.683 | 4.922 / 0.364 / 0.685 |
| 2020 | 4.803 / **0.353** / **0.662** | 4.787 / 0.344 / 0.650 | **4.780** / 0.338 / 0.651 |
| 2021 | 5.100 / **0.307** / 0.619 | 5.062 / 0.299 / **0.621** | **5.032** / 0.299 / 0.620 |
| 2022 | **4.668** / **0.374** / **0.649** | 4.706 / 0.340 / 0.629 | 4.693 / 0.337 / 0.628 |
| 2023 | 4.577 / **0.380** / **0.706** | 4.548 / 0.353 / 0.683 | **4.520** / 0.352 / 0.685 |
| 2024 | 4.526 / 0.422 / 0.720 | 4.400 / 0.434 / 0.727 | **4.373** / **0.436** / **0.728** |
| 2025 | 4.499 / **0.425** / **0.745** | 4.513 / 0.407 / 0.732 | **4.441** / 0.408 / 0.735 |
| 2026 wk 1-3 (172 rows) | 4.765 / 0.437 / 0.722 | **4.376** / **0.505** / **0.753** | same as current (no skew rows yet) |

Over the 7 full-season folds, the stat vector:
- has the highest R² in 6 of 7 folds and the best Spearman (ranking) in 5 of 7, against both benchmarks;
- has worse MAE than rolling-only in 5 of 7 folds, because its predictions are unbiased on the mean and RB points are right-skewed;
- loses on every metric in 2024 only. That season was behind the earlier two-season tie.

Replacing the fitted rate models with constants was worse in every fold:
- league-average rates: MAE +0.032 on average;
- player season-to-date rates: MAE +0.09, Spearman −0.03.

With six training seasons the matchup terms in the rate models are consistently positive (ypc × `epa_allowed_rush` p < 0.001 in every fold), so the fitted rates carry real, if small, signal.

Known limitations:
- 2-pt conversions, first downs, fumbles, and 100/200-yard / 20-carry bonuses are not predicted and contribute 0. On actual 2024/25 RB stats that omission costs 0.11 points/game on average for the tested league.
- RB only; QB/WR/TE not built.

## Tech

Python, Polars, Sleeper API, nflreadpy (nflverse)