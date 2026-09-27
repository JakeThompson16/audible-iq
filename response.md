# RB stat vector, second iteration (prompt.txt)

Same worktree and branch (`worktree-rb-stat-vector`). Nothing merged to main,
`engine/expected_points.py` unchanged. Re-run with `python rb_stat_vector_eval.py`; the
final run matches the previous one exactly.

## Headline
Across 7 held-out full seasons (2019–2025), the new model **ranks players better** and
**explains more variance** than the current formula. It has **slightly worse MAE than
rolling-average-alone**. 2024 was an unusual season, and it caused the earlier two-season
tie. Constant rates lost to fitted rates in every fold, so I kept the fitted rates.

## Folds (step 4)
- **What's loadable:** nflreadpy loads every season I checked back to 2016, plus 2026
  weeks 1–3.
- **Design:** leave-one-season-out over 2019–2025. Each season is predicted by a model fit
  on the other six.
- **Partial 2026:** a test-only fold, fit on 2019–2025.
- **2018:** history only, feeding the prior-season blend and the rolling windows.
- **Window:** fixed at the earlier winner, n=8.

## Main table: new model / current formula / rolling-average-alone
New model = steps (1)+(3): no `epa_allowed` in volume models, rate models weighted by
attempts. Same rows for every method in a fold, unmodified `engine/metrics.py`. Each cell
is MAE / R² / Spearman.

| Test season | New model | Current (rolling + skew) | Rolling alone |
|---|---|---|---|
| 2019 | **4.851** / **0.388** / **0.702** | 4.949 / 0.365 / 0.683 | 4.922 / 0.364 / 0.685 |
| 2020 | 4.803 / **0.353** / **0.662** | 4.787 / 0.344 / 0.650 | **4.780** / 0.338 / 0.651 |
| 2021 | 5.100 / **0.307** / 0.619 | 5.062 / 0.299 / **0.621** | **5.032** / 0.299 / 0.620 |
| 2022 | **4.668** / **0.374** / **0.649** | 4.706 / 0.340 / 0.629 | 4.693 / 0.337 / 0.628 |
| 2023 | 4.577 / **0.380** / **0.706** | 4.548 / 0.353 / 0.683 | **4.520** / 0.352 / 0.685 |
| 2024 | 4.526 / 0.422 / 0.720 | 4.400 / 0.434 / 0.727 | **4.373** / **0.436** / **0.728** |
| 2025 | 4.499 / **0.425** / **0.745** | 4.513 / 0.407 / 0.732 | **4.441** / 0.408 / 0.735 |
| 2026 wk 1–3 (172 rows) | 4.765 / 0.437 / 0.722 | **4.376** / **0.505** / **0.753** | = current (no skew rows yet) |

**Folds won by the new model (out of 7):**

| Metric | vs current formula | vs rolling alone |
|---|---|---|
| R² | 6 | 6 |
| Spearman | 5 | 5 |
| MAE | 3 | 2 |

**LOSO means:**

| Method | MAE | R² | Spearman |
|---|---:|---:|---:|
| New model | 4.718 | **0.378** | **0.686** |
| Current formula | 4.709 | 0.363 | 0.675 |
| Rolling alone | **4.680** | 0.362 | 0.676 |

- **Does the third season break the tie?** Yes, toward the new model on ranking and R².
  2024 is the only full season where it loses every metric, and that was one of the two
  original folds.
- **Why MAE disagrees with R²:** the new model is unbiased on the mean. RB points are
  right-skewed, so MAE rewards predicting a bit low, while squared error and ranking reward
  the new model.
- **Early season:** the partial 2026 fold (weeks 1–3) is a clear loss. It's a small sample,
  but early-season projection needs a look before any promotion. There, rolling-alone
  leans on last season's full average, while the new model uses the 8-game window that
  carries over from the end of last season.

## Step 1: `epa_allowed` removed from the volume models
- **No change, as expected.**
  - LOSO mean MAE 4.671 → 4.672.
  - Spearman 0.687 → 0.686.
  - Per fold, MAE moves by 0.011 at most.
- **Volume coefficients are now very stable** across folds:

  | Model | Trailing average | Delta |
  |---|---|---|
  | carries | 0.874–0.887 | 0.429–0.454 |
  | targets | 0.781–0.805 | 0.232–0.249 |

- **The rate-model matchup terms:** with six training seasons, all five `epa_allowed`
  coefficients are **positive in every fold**:

  | Rate model | Coefficient range | p-values |
  |---|---|---|
  | ypc | +4.9 to +5.7 | < 0.001 in every fold |
  | rush_td_rate | +0.03 to +0.05 | 0.007–0.14 |
  | catch_rate | +0.13 to +0.17 | 0.014–0.053 |
  | ypr | +1.4 to +2.9 | 0.04–0.33 |
  | rec_td_rate | +0.05 to +0.07 | 0.04–0.15 |

## Step 3: attempt-weighted rates are now the default
- **What I changed:** the prompt says "attempt-weighted as the default for volume models".
  The weighting (and its 0.055 MAE cost) applies to the rate models; volume models have no
  attempt count to weight by. So I made weighted rate fits the default
  (`fit_rb_models(weight_rates=True)`). Tell me if you meant something else.
- **Effect over 7 folds, vs unweighted:**
  - mean bias (actual − predicted): +0.31 → −0.03
  - MAE +0.046
  - R² +0.005
  - Spearman unchanged
- **Recomposed points after (1)+(3):** the "New model" column in the main table.

## Step 2: constant rates instead of fitted rate models
MAE / Spearman per LOSO fold:

| Fold | Fitted (weighted) | League-average constant | Player season-to-date |
|---|---|---|---|
| 2019 | **4.851 / 0.702** | 4.876 / 0.700 | 5.029 / 0.657 |
| 2020 | **4.803 / 0.662** | 4.855 / 0.659 | 4.955 / 0.629 |
| 2021 | **5.100 / 0.619** | 5.116 / 0.614 | 5.159 / 0.607 |
| 2022 | **4.668 / 0.649** | 4.678 / 0.647 | 4.780 / 0.595 |
| 2023 | **4.577 / 0.706** | 4.621 / 0.704 | 4.610 / 0.683 |
| 2024 | **4.526 / 0.720** | 4.576 / 0.716 | 4.557 / 0.703 |
| 2025 | **4.499 / 0.745** | 4.531 / 0.742 | 4.580 / 0.724 |

- **League-average constant:** worse in **7/7** folds. MAE is +0.010 to +0.052 (mean
  +0.032) and Spearman −0.002 to −0.005.
- **Season-to-date:** clearly worse (MAE +0.09, Spearman −0.03). Players' own early-season
  rates are noise.
- **My recommendation: keep the fitted rates** for now. Your rule was to prefer the
  constant version at comparable accuracy. The gap is small, but it runs the same way in
  every fold, and the positive matchup terms above explain it. The fitted rates carry a
  little real matchup signal, which is about the size of the whole opponent-skew effect.
- **Your call if 0.03 MAE counts as "comparable".** Switching is one line
  (`with_unfitted_rates(fits, priors, "league")`). A middle option I haven't tested: rate =
  league average + b·`epa_allowed`, dropping the rolling-rate and delta terms that add
  little.

## Other findings
- **Original splits moved slightly.** The old 2024↔2025 splits shift by up to 0.02 MAE
  versus the first iteration: 2024→2025 is now 4.420 (was 4.430), 2025→2024 is 4.463 (was
  4.444). Loading 2018 onward gives 2023 its own prior-season blend and lets rolling windows
  for sparse players reach back into 2022. I confirmed the 2023–25 stat rows are identical
  either way, so this is more history, not different data.
- **Latent loader bug.** `load_player_metadata` fans out rows for two players listed twice
  in the ID crosswalk: Justin Hamilton and Corey Moore, both defensive, visible only when
  loading pre-2023 seasons. That's 48 duplicate player-week keys over 2018–2026. It has no
  effect on QB/RB/WR/TE. The driver now asserts RB keys are unique. I didn't change the
  loader (logged in OPEN_QUESTIONS Q-11).
- **Partial 2026 fold:** current and rolling-alone are identical there because opponent skew
  needs 3 games, so it's 0 through week 3.

## Changed
- `projections/expected_points/stat_vector/rb.py`:
  - no `epa_allowed` in the volume models; the old spec is kept as `LEGACY_RB_MODELS` for
    comparison;
  - weighted rates are the default;
  - new `ConstantRate` / `SeasonToDateRate` and `with_unfitted_rates`.
- `projections/expected_points/features/stat_rolling.py`: adds the season-to-date rates
  (`std_<rate>`).
- `rb_stat_vector_eval.py`: rewritten for the folds, five variants, and head-to-head
  summaries.
- README, CLAUDE.md, OPEN_QUESTIONS (Q-11 update), and STATUS updated.

## Decisions for you
1. Constant vs fitted rates: is 0.03 MAE "comparable"? (Recommended: keep fitted.)
2. Should I look into the early-season weakness (weeks 1–3) next, before QB/WR/TE?
3. Promotion still needs the "no fitted parameter" stance revisited (Q-11b).
