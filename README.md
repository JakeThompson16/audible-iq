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

Currently focused on offensive skill positions (QB/RB/WR/TE). Kicker, team defense/IDP, and play-level long-touchdown bonus categories are intentionally out of scope, see code comments for details.

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

## Tech

Python, Polars, Sleeper API, nflreadpy (nflverse)