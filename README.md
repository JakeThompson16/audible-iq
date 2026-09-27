# Audible IQ

A fantasy football decision-support tool that pulls live league data from Sleeper, computes league-accurate fantasy scoring, and (in progress) builds matchup-adjusted projections and boom/bust probabilities to power start/sit recommendations.

## Status: Early development

Currently implemented:
- **Sleeper integration**: pulls leagues, rosters, users, and scoring settings via Sleeper's public API
- **Player data pipeline**: pulls weekly stats and player metadata via `nflreadpy` (nflverse), joined against a Sleeper/GSIS ID crosswalk
- **League-accurate scoring engine**: computes fantasy points via a vectorized dot product of each league's actual scoring settings against raw stat lines (supports PPR variants, TE premium, yardage/completion bonus thresholds, etc.)
- **Validated**: scoring output cross-checked against real Sleeper league data (Trey McBride, full season) with exact matches
- **Weekly point projection engine** with a per-position implementation registry:
  - **RB**: a league-portable *stat vector* (predicted carries/targets and per-attempt rates, scored through the league's own settings).
  - **QB/WR/TE**: a trailing 8-game average plus an opponent-adjusted matchup skew.
  - Every projection gets a confidence tier.
  - See [Model Evaluation](#model-evaluation).

In progress:
- Boom/bust probability classifier
- Agentic start/sit recommendation layer

## Architecture

The project is split into layers to keep external API/data source dependencies isolated from core logic:

- `domain/` — core objects (`Player`, `Team`, `League`, `ScoringSettings`), platform-agnostic
- `clients/` — raw API/data-source wrappers (Sleeper API, nflreadpy)
- `engine/` — computation logic that operates across domain objects (scoring calculation, expected points with a position -> implementation registry; upcoming: boom/bust)
- `projections/` — feature engineering (continuous trailing windows, opponent skew, opponent EPA allowed) and the stat-vector models
- `evaluation/` — offline leave-one-season-out backtest harness

This separation means adding a new platform (e.g. ESPN, Yahoo) or data source only requires a new adapter/client, not changes to core logic.

## Scope

Currently focused on offensive skill positions (QB/RB/WR/TE). Kicker, team defense/IDP, and play-level long-touchdown bonus categories are intentionally out of scope, see code comments for details. The RB stat-vector projection also omits 2-pt conversions, first downs, fumbles, and yardage/carry threshold bonuses (they score 0).

## Model Evaluation

All numbers come from `engine/metrics.py` on a **leave-one-season-out** backtest (`evaluation/backtest.py`):
- each season 2019-2025 is predicted by stat-vector models fit on the other six;
- 2026 (weeks 1-3 so far) is a test-only fold;
- one Sleeper league's scoring;
- every rolling feature is a continuous trailing window that spans the season boundary (per-position player windows QB 12 / RB 8 / WR 10 / TE 12, 17 games for defenses).

Run `python test.py` (all positions) or `python stat_vector_eval.py RB`.

### Production engine, per position (mean of 7 full-season folds)

| Pos | Implementation | Window | MAE | R² | Spearman | Rolling-avg-alone MAE / R² |
|---|---|---:|---:|---:|---:|---|
| QB | rolling_avg_prior + opponent_skew | 12 | 7.895 | 0.220 | 0.487 | 7.869 / 0.221 |
| RB | **stat vector** | 8 | **4.717** | **0.378** | **0.686** | 4.733 / 0.349 |
| WR | rolling_avg_prior + opponent_skew | 10 | 4.697 | 0.321 | 0.635 | 4.692 / 0.326 |
| TE | rolling_avg_prior + opponent_skew | 12 | 4.318 | 0.295 | 0.570 | 4.286 / 0.301 |

On corrected data, opponent_skew doesn't beat rolling-average-alone at any position (OPEN_QUESTIONS Q-4). The confidence-tier calibration is still non-monotonic.

### Rolling window length (per position)

`rolling_window_eval.py` grid-searches the continuous window for `rolling_plus_skew` (lowest mean LOSO MAE). Each cell is MAE / R² / Spearman, same rows; the weeks 1-3 column is MAE.

| Pos | Chosen window | At chosen window | Window 8 | Old season blend | Weeks 1-3: chosen / 8 / blend |
|---|---:|---|---|---|---|
| QB | 12 | 7.894 / 0.216 / 0.483 | 7.925 / 0.208 / 0.475 | 7.893 / 0.202 / 0.473 | 8.060 / 8.187 / 8.078 |
| WR | 10 | 4.700 / 0.321 / 0.635 | 4.709 / 0.314 / 0.634 | 4.661 / 0.332 / 0.644 | 5.086 / 5.145 / 5.063 |
| TE | 12 | 4.321 / 0.295 / 0.569 | 4.344 / 0.283 / 0.564 | 4.282 / 0.301 / 0.573 | 4.555 / 4.644 / 4.481 |

- QB recovers fully: it ties the old blend on MAE and beats it on R², Spearman, and early weeks.
- WR and TE stay about 0.04 MAE behind the old blend (Q-13).
- WR's curve is flat: windows 10, 12 and 14 are within 0.007 MAE of each other.

### RB: stat vector vs current formula vs rolling alone

Instead of projecting fantasy points (which ties the model to one league's scoring), the RB model predicts the raw stat line and scores it with the unmodified `calculate_points_vectorized`:
- **Volume models** (carries, targets): OLS on the trailing average and a recent-usage delta (3-game minus 8-game mean). No matchup term: it flipped sign between seasons, and removing it didn't change accuracy.
- **Rate models** (ypc, rush TD/carry, catch rate, yards/reception, rec TD/reception): OLS weighted by attempts, on the trailing rate, the usage delta, and opponent EPA/play allowed for that play type.
- Yards and TDs are derived by multiplication.

Each cell is MAE / R² / Spearman; same rows per fold.

| Test season | Stat vector | rolling_avg_prior + opponent_skew | Rolling alone |
|---|---|---|---|
| 2019 | **4.855** / **0.387** / **0.701** | 5.000 / 0.353 / 0.672 | 4.921 / 0.360 / 0.680 |
| 2020 | **4.809** / **0.351** / **0.660** | 4.855 / 0.325 / 0.637 | 4.873 / 0.321 / 0.641 |
| 2021 | **5.085** / **0.309** / **0.620** | 5.150 / 0.274 / 0.611 | 5.127 / 0.277 / 0.613 |
| 2022 | **4.667** / **0.374** / **0.650** | 4.798 / 0.316 / 0.617 | 4.772 / 0.317 / 0.619 |
| 2023 | 4.579 / **0.379** / **0.705** | 4.541 / 0.349 / 0.677 | **4.522** / 0.349 / 0.684 |
| 2024 | 4.524 / 0.422 / **0.720** | 4.485 / 0.417 / 0.711 | **4.413** / **0.425** / 0.719 |
| 2025 | **4.502** / **0.424** / **0.743** | 4.552 / 0.390 / 0.716 | 4.504 / 0.394 / 0.723 |
| **Mean (7 folds)** | **4.717** / **0.378** / **0.685** | 4.769 / 0.346 / 0.663 | 4.733 / 0.349 / 0.668 |
| 2026 wk 1-3 (175 rows) | 4.721 / 0.440 / **0.722** | 4.649 / 0.442 / 0.719 | **4.617** / **0.450** / 0.623 |

How the stat vector compares over the 7 folds:
- vs the current formula: better MAE in 5, R² in 7, Spearman in 7;
- vs rolling alone: better MAE in 5, R² in 6, Spearman in 7;
- mean bias is −0.04 points/game;
- the coefficients are stable across folds (e.g. carries trailing average 0.874-0.887, ypc × `epa_allowed_rush` +4.3 to +5.2, p < 1e-9 in every fold).

Early season (2026 weeks 1-3) is the one place it trails on MAE, by about 0.07-0.10 on 175 rows, while ranking as well as the current formula. That's treated as an information ceiling for three games of data, not something to chase.

Earlier iterations tested and rejected: unweighted rate fits (under-project by 0.31/game), league-average or season-to-date rates instead of fitted rates (worse in every fold), and `epa_allowed` in the volume models (no effect).

Known limitations:
- 2-pt conversions, first downs, fumbles, and 100/200-yard / 20-carry bonuses are not predicted and score 0 (about 0.11 points/game on actual RB stats for the tested league).
- RB only so far.
- This is a backtest-validated path; there is no live upcoming-week projection path yet.

### Continuous windows vs the old season blend

The old early-season blend (current-season window mixed with last season's full average) was replaced by one continuous trailing window everywhere. At a shared 8-game window this cost the incumbent formula MAE:

| Position | MAE change |
|---|---|
| QB | +0.03 |
| RB | +0.06 |
| WR | +0.05 |
| TE | +0.06 |

The cost was larger in weeks 1-3 (+0.08 to +0.16). Per-position window tuning (above) removes it for QB and reduces it for WR/TE. See OPEN_QUESTIONS Q-13.

Earlier README results (the k grid search on 2024/25) were produced on a pipeline with the B-1..B-3 defects and the retired blend, and have been removed; they're in git history.

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

## Tech

Python, Polars, Sleeper API, nflreadpy (nflverse)