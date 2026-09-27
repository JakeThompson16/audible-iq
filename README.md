# Audible IQ

A fantasy football decision-support tool that pulls live league data from Sleeper, computes league-accurate fantasy scoring, and (in progress) builds matchup-adjusted projections and boom/bust probabilities to power start/sit recommendations.

## Status: Early development

Currently implemented:
- **Sleeper integration**: pulls leagues, rosters, users, and scoring settings via Sleeper's public API
- **Player data pipeline**: pulls weekly stats and player metadata via `nflreadpy` (nflverse), joined against a Sleeper/GSIS ID crosswalk
- **League-accurate scoring engine**: computes fantasy points via a vectorized dot product of each league's actual scoring settings against raw stat lines (supports PPR variants, TE premium, yardage/completion bonus thresholds, etc.)
- **Validated**: scoring output cross-checked against real Sleeper league data (Trey McBride, full season) with exact matches
- **Weekly point projection engine** with a per-position implementation registry:
  - **RB, WR and TE**: a league-portable *stat vector* (predicted carries/targets and per-attempt rates, scored through the league's own settings).
  - **QB**: a trailing 12-game average plus an opponent-adjusted matchup skew. A QB stat vector was evaluated and narrowly didn't clear the bar.
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

Currently focused on offensive skill positions (QB/RB/WR/TE). Kicker, team defense/IDP, and play-level long-touchdown bonus categories are intentionally out of scope, see code comments for details. The RB/WR/TE stat-vector projections also omit 2-pt conversions, first downs, fumbles, and yardage/carry threshold bonuses (they score 0).

## Model Evaluation

All numbers come from `engine/metrics.py` on a **leave-one-season-out** backtest (`evaluation/backtest.py`):
- each season 2019-2025 is predicted by stat-vector models fit on the other six;
- 2026 (weeks 1-3 so far) is a test-only fold;
- one Sleeper league's scoring;
- every rolling feature is a continuous trailing window that spans the season boundary (per-position player windows QB 14 / RB 8 / WR 10 / TE 12, 17 games for defenses).

Run `python test.py` (all positions + calibration), `python stat_vector_eval.py RB|WR|TE|QB [--grid] [--candidates]`, or `python rolling_window_eval.py`.

Selection policy: start/sit is a ranking decision, so choices prioritize mean Spearman, then R². A position switches implementation when both are comparable-or-better at comparable-or-better MAE; a strict MAE win isn't required.

### Production engine, per position (mean of 7 full-season folds)

| Pos | Implementation | Window | MAE | R² | Spearman | Rolling-avg-alone MAE / R² |
|---|---|---:|---:|---:|---:|---|
| QB | rolling_avg_prior + opponent_skew | 14 | 7.915 | 0.217 | 0.492 | 7.897 / 0.218 |
| RB | **stat vector** | 8 | **4.717** | **0.378** | **0.686** | 4.733 / 0.349 |
| WR | **stat vector** | 10 | **4.690** | **0.344** | **0.651** | 4.692 / 0.326 |
| TE | **stat vector** | 10 (stat-vector) | **4.233** | **0.326** | **0.590** | 4.286 / 0.301 |

The Window column is the rolling-average window, except for TE, where it's the stat vector's own feature window. RB and WR stat vectors use 8.

On corrected data, opponent_skew doesn't beat rolling-average-alone at any position; it now only feeds QB (OPEN_QUESTIONS Q-4).

**Confidence calibration** (pooled held-out 2019-2025, high / medium / low / insufficient):

| Pos | MAE | Mean actual points | Normalized MAE (MAE ÷ mean actual) |
|---|---|---|---|
| QB | 7.889 / 7.764 / 7.951 / 8.280 | 20.4 / 18.3 / 14.5 / 12.4 | **0.386 / 0.425 / 0.547 / 0.668** |
| RB | 5.093 / 4.507 / 4.430 / 4.543 | 9.31 / 7.71 / 6.87 / 6.37 | **0.547 / 0.585 / 0.645 / 0.713** |
| WR | 4.926 / 4.561 / 4.450 / 4.678 | 8.53 / 7.29 / 6.63 / 6.52 | **0.578 / 0.626 / 0.671 / 0.717** |
| TE | 4.690 / 4.106 / 3.881 / 3.899 | 8.48 / 6.70 / 5.43 / 4.99 | **0.553 / 0.613 / 0.715 / 0.781** |

Raw MAE looks inverted only because higher tiers hold higher-scoring players. Normalized by points scored, error falls monotonically from low to high confidence at every position, so the tiers are correctly ordered (Q-15, verified).

### Rolling window length (per position)

`rolling_window_eval.py` grid-searches the continuous window for `rolling_plus_skew`. WR and TE were picked on mean MAE; QB was reselected under the later Spearman-first policy. Each cell is MAE / R² / Spearman, same rows; the weeks 1-3 column is MAE.

| Pos | Chosen window | At chosen window | Window 8 | Old season blend | Weeks 1-3: chosen / 8 / blend |
|---|---:|---|---|---|---|
| QB | 14 | 7.913 / 0.213 / 0.488 | 7.925 / 0.208 / 0.475 | 7.893 / 0.202 / 0.473 | 8.035 / 8.187 / 8.078 |
| WR | 10 | 4.700 / 0.321 / 0.635 | 4.709 / 0.314 / 0.634 | 4.661 / 0.332 / 0.644 | 5.086 / 5.145 / 5.063 |
| TE | 12 | 4.321 / 0.295 / 0.569 | 4.344 / 0.283 / 0.564 | 4.282 / 0.301 / 0.573 | 4.555 / 4.644 / 4.481 |

- QB: 12 was the MAE optimum (7.894, tying the old blend). 14 is the Spearman optimum and is what's configured: it gives up 0.02 MAE and beats the blend on R², Spearman and early weeks.
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
- RB, WR and TE; QB stays on the rolling formula (Q-14).
- This is a backtest-validated path; there is no live upcoming-week projection path yet.

### WR: stat vector vs current formula vs rolling alone

Same structure as RB: targets and carries volume models, five attempt-weighted rate models. `epa_allowed_pass` is on for catch rate only, the one receiving matchup term significant in every fold. Baselines use the tuned WR window (10). Each cell is MAE / R² / Spearman; same rows per fold.

| Test season | Stat vector | rolling_avg_prior + opponent_skew | Rolling alone |
|---|---|---|---|
| 2019 | **4.836** / **0.325** / **0.657** | 4.906 / 0.287 / 0.628 | 4.941 / 0.287 / 0.628 |
| 2020 | 4.959 / 0.317 / 0.640 | 4.954 / 0.314 / 0.643 | **4.926** / **0.319** / **0.646** |
| 2021 | 4.772 / **0.349** / **0.669** | 4.785 / 0.330 / 0.659 | **4.762** / 0.339 / 0.665 |
| 2022 | 4.747 / **0.337** / **0.628** | **4.736** / 0.311 / 0.618 | 4.741 / 0.314 / 0.623 |
| 2023 | 4.557 / **0.367** / **0.644** | **4.543** / 0.353 / 0.626 | 4.574 / 0.354 / 0.625 |
| 2024 | 4.611 / **0.348** / **0.659** | 4.570 / 0.328 / 0.641 | **4.543** / 0.338 / 0.646 |
| 2025 | **4.345** / **0.362** / **0.660** | 4.388 / 0.324 / 0.630 | 4.358 / 0.333 / 0.637 |
| **Mean (7 folds)** | **4.689** / **0.344** / **0.651** | 4.697 / 0.321 / 0.635 | 4.692 / 0.326 / 0.638 |
| 2026 wk 1-3 (286 rows) | **4.745** / **0.328** / **0.595** | 4.772 / 0.287 / 0.561 | 4.782 / 0.291 / 0.560 |

How the stat vector compares over the 7 folds:
- vs the current formula: better R² in 7, Spearman in 6, MAE in 3;
- vs rolling alone: better R² in 6, Spearman in 6, MAE in 3;
- lowest mean MAE of the three;
- mean bias +0.03 (current formula +0.04, rolling alone −0.17).

Coefficients are stable across folds (targets trailing average 0.872-0.876; catch rate × `epa_allowed_pass` +0.16 to +0.24, significant in every fold). `epa_allowed` in the targets model and in yards/reception and TD rate was tested and left off (not significant or unstable sign).

### TE: stat vector vs current formula vs rolling alone

WR's structure refit on TE data. The stat-vector window is 10, picked by the selection policy from {8, 10, 12, 14, 16, 20}. `epa_allowed_pass` is on for catch rate, yards/reception and TD rate; the last two cleared significance for TE though not for WR. Baselines use the tuned TE window (12). Each cell is MAE / R² / Spearman.

| Test season | Stat vector | rolling_avg_prior + opponent_skew | Rolling alone |
|---|---|---|---|
| 2019 | **4.292** / **0.302** / **0.547** | 4.401 / 0.251 / 0.531 | 4.343 / 0.261 / 0.543 |
| 2020 | **4.539** / **0.283** / **0.544** | 4.633 / 0.256 / 0.536 | 4.580 / 0.267 / 0.542 |
| 2021 | **4.186** / **0.332** / **0.541** | 4.247 / 0.314 / 0.530 | 4.225 / 0.315 / 0.530 |
| 2022 | **4.156** / **0.318** / **0.606** | 4.248 / 0.284 / 0.564 | 4.226 / 0.290 / 0.578 |
| 2023 | **4.084** / **0.369** / **0.639** | 4.179 / 0.326 / 0.612 | 4.180 / 0.334 / 0.617 |
| 2024 | 4.256 / **0.336** / **0.632** | 4.274 / 0.320 / 0.610 | **4.254** / 0.326 / 0.620 |
| 2025 | **4.117** / **0.341** / **0.621** | 4.244 / 0.316 / 0.606 | 4.197 / 0.314 / 0.611 |
| **Mean (7 folds)** | **4.233** / **0.326** / **0.590** | 4.318 / 0.295 / 0.570 | 4.286 / 0.301 / 0.577 |
| 2026 wk 1-4 (189 rows) | **4.539** / 0.246 / 0.561 | 4.587 / **0.260** / **0.576** | 4.573 / 0.242 / 0.564 |

How the stat vector compares over the 7 folds:
- vs the current formula: better on all three metrics in all 7 folds;
- vs rolling alone: better R² and Spearman in 7, MAE in 6;
- mean bias +0.02.

Coefficients are stable across folds (targets trailing average 0.870-0.885; yards/reception × `epa_allowed_pass` +3.7 to +5.7 and TD rate × `epa_allowed_pass` +0.10 to +0.13, both significant in every fold).

### QB: stat vector evaluated, not switched

The QB model has its own spec:
- **Volume models:** pass attempts and rush attempts (scrambles included).
- **Rate models:** completion rate and INT rate per attempt; yards and TDs per completion; yards and TDs per rush attempt.
- **Stat window:** 12.

Each cell is MAE / R² / Spearman.

| Test season | Stat vector | rolling_avg_prior + opponent_skew (w=12; production now w=14) | Rolling alone (w=12) |
|---|---|---|---|
| 2019 | **8.063** / **0.191** / 0.424 | 8.166 / 0.156 / 0.424 | 8.197 / 0.146 / 0.396 |
| 2020 | 8.129 / 0.275 / 0.519 | 7.869 / 0.299 / 0.542 | **7.825** / **0.304** / **0.543** |
| 2021 | 8.089 / 0.247 / 0.498 | 7.938 / 0.237 / **0.528** | **7.850** / **0.251** / 0.524 |
| 2022 | **7.257** / **0.258** / 0.502 | 7.324 / 0.234 / **0.518** | 7.358 / 0.227 / 0.509 |
| 2023 | 7.641 / **0.226** / 0.487 | 7.623 / 0.218 / **0.492** | **7.571** / 0.220 / 0.485 |
| 2024 | 7.917 / **0.260** / **0.517** | 7.929 / 0.238 / 0.496 | **7.874** / 0.244 / 0.498 |
| 2025 | **8.279** / **0.207** / **0.437** | 8.416 / 0.155 / 0.412 | 8.410 / 0.152 / 0.405 |
| **Mean (7 folds)** | 7.911 / **0.238** / 0.483 | 7.895 / 0.220 / **0.487** | **7.869** / 0.221 / 0.480 |
| 2026 wk 1-4 (96 rows) | 8.575 / 0.152 / 0.305 | **8.284** / 0.162 / **0.317** | 8.291 / **0.165** / 0.316 |

- At the production rolling window (14), the current formula averages 7.915 / 0.217 / 0.492, which widens the Spearman gap.
- Better R² in 6 of 7 folds and smaller bias (−0.10 vs −0.31).
- Spearman, the primary metric, is slightly worse (wins 2 of 7 folds), MAE is +0.016, and the early-season fold is worse. So QB stays on rolling_avg_prior + opponent_skew (OPEN_QUESTIONS Q-14).
- The passing matchup terms are real and correctly signed: completion rate rises against weak pass defenses, and **interception rate falls** (a negative coefficient, significant in every fold, as expected).
- Candidate refinements:
  - improve the rush component (e.g. split scrambles from designed runs). Segmenting QBs by rushing volume shows the stat vector under-projects high-rushing QBs by 1.6 points/game (current formula 0.7) and orders them worse among themselves (Spearman 0.312 vs 0.363). A week-10 hint that it ranks rushing QBs lower league-wide didn't hold up;
  - predict fumbles and 2-pt conversions, which currently cost QBs 0.47 points/game in recomposition.

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