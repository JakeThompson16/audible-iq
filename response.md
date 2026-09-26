# RB stat-vector expected points (prompt.txt, Prompt 2 took priority)

Branch `worktree-rb-stat-vector` (worktree `.claude/worktrees/rb-stat-vector`). Re-run with
`python rb_stat_vector_eval.py`. `engine/expected_points.py`, `opponent_skew.py`,
`engine/scoring.py` and `engine/metrics.py` are unchanged. QB/WR/TE were not built.

## How the two prompts were reconciled
- **Kept from Prompt 1:** the sack fix, the kneel-gap note, and `epa_allowed`. Prompt 2 says
  `epa_allowed` was "already built", but it wasn't, so I built it.
- **Dropped:** Prompt 1's step 3, a fantasy-points OLS for all four positions. Prompt 2
  forbids predicting points directly and forbids QB/WR/TE.
- **`opponent_skew.py`:** Prompt 1 says reuse it, Prompt 2 says don't touch it. Its helper is
  hard-wired to fantasy residuals, so `epa_allowed.py` copies the same structure instead.
- **RB `epa_allowed_pass`:** Prompt 2 settles it. Rush models use `epa_allowed_rush` and
  receiving models use `epa_allowed_pass` (documented in CLAUDE.md).
- **Also kept:** the n∈{4,6,8} grid, VIF, and the naive-mean and last-week benchmarks.

## Step 1: pass filter
- Pass plays are now `play_type == 'pass' & sack == 0`. nflfastR's `pass_attempt` column also
  includes sacks (1,314), so it can't be used instead.
- REG 2024: **17,839** vs 17,811 official. The +28 is exactly 99 two-point tries (included)
  minus 71 spikes (excluded).
- Rush filter unchanged. The 370-carry gap is fully explained: 14,317 − 36 two-point runs +
  405 kneels (`play_type == 'qb_kneel'`) = 14,686 vs 14,687.
- Both gaps are accepted as-is (OPEN_QUESTIONS Q-12). Excluding 2-point tries would be a
  further filter change, which Prompt 1 ruled out.

## Assumptions and deviations to review
1. **Effective-n shrinkage.** The "effective-n fix" Prompt 1 refers to doesn't exist in
   `opponent_skew.py`. `epa_allowed` uses n_eff = n_games + (1 − w)·prior_season_games,
   shrunk by n_eff/(n_eff+16). k=16 is borrowed from skew and not tuned for EPA.
2. **Rolling windows.** Rates are sum(num)/sum(den) over the window, not the mean of
   per-game rates. Delta = 3-game minus 8-game mean.
3. **Rookies and missing rates.** A player with history but no attempts of a kind in the
   window gets the pooled training-season RB rate. Rookies with no history stay None.
4. **Clamping.** Volumes and TD rates are floored at 0, and catch rate is clamped to [0, 1].
5. **Extra variant (not in the prompt).** Because plain OLS is biased (below), I also ran
   the rate models weighted by attempts (WLS). It is reported separately; plain OLS is
   the primary result.
6. **Uncommitted main work.** The B-1..B-4 fixes on main were uncommitted, so I copied them
   into this branch as commit `ea60f13` first. Your checkout was not modified.

## Sanity checks
- **`epa_allowed` table:** 1,710 rows, unique keys.
  - 2024 week 1 is nonzero for all 32 defenses (the prior-season blend works).
  - 2023 week 1 is all 0 (no earlier season loaded).
  - Toughest pass defenses at the end of 2025: HOU, JAX, PHI. Softest: DAL, WAS, NYJ.
- **Season boundary:** Barkley's 2025 week-1 `roll_carries` (23.75) comes from his 2024 tail.
- **Cost of the omitted categories:** scoring actual RB stats through the same stat set:
  - mean(actual − recomposed) = −0.053 points/game
  - mean |gap| = 0.108 points/game
  - 94.9% of rows are exact

## Results: recomposed RB points
Same rows for every method, unmodified `evaluate_projections`, league "Top Tier Fantasy".
Bias = actual − predicted.

| n | Direction | Method | MAE | R² | Bias | Spearman |
|---|---|---|---:|---:|---:|---:|
| 8 | 2024→2025 (1,602 rows) | **stat vector (OLS)** | **4.430** | 0.425 | +0.21 | **0.743** |
| | | stat vector (WLS rates) | 4.487 | **0.427** | −0.12 | 0.742 |
| | | current rolling + skew | 4.513 | 0.407 | −0.17 | 0.732 |
| | | rolling_avg_prior only | 4.441 | 0.408 | −0.11 | 0.735 |
| | | last week | 5.435 | 0.009 | −0.04 | 0.649 |
| | | naive mean | 6.484 | 0.000 | −0.08 | n/a |
| 8 | 2025→2024 (1,570 rows) | stat vector (OLS) | 4.444 | 0.413 | +0.48 | 0.721 |
| | | stat vector (WLS rates) | 4.499 | 0.421 | +0.06 | 0.722 |
| | | current rolling + skew | 4.396 | 0.434 | +0.01 | 0.727 |
| | | **rolling_avg_prior only** | **4.373** | **0.436** | +0.07 | **0.728** |
| | | last week | 5.257 | 0.138 | −0.04 | 0.607 |
| | | naive mean | 6.464 | −0.002 | +0.36 | n/a |

- **Winning n = 8.** Stat-vector mean MAE / Spearman over both directions:
  - n=4: 4.493 / 0.727
  - n=6: 4.485 / 0.728
  - n=8: 4.437 / 0.732
- **Verdict: roughly a tie with the current formula.** It wins one direction and loses the
  other. Averaged over both directions at n=8:

  | Method | MAE | R² | Spearman |
  |---|---:|---:|---:|
  | Stat vector (OLS) | 4.437 | 0.419 | 0.732 |
  | Current formula | 4.455 | 0.421 | 0.730 |
  | Rolling only | 4.407 | 0.422 | 0.732 |

  So portability costs essentially nothing, but it doesn't gain accuracy yet either.
- **Plain OLS under-projects** by 0.2–0.5 points/game. Cause: an unweighted per-game rate fit
  lets 1-carry games count as much as 25-carry games. In training, per-game ypc is 4.18 vs
  pooled 4.39, and rush TD rate is 0.027 vs 0.033. Predicted rushing yards and TDs are where
  the shortfall shows up.
- **WLS removes the bias** and improves R², but MAE is about 0.055 worse because RB points
  are right-skewed.

## Coefficients (n=8)
Format: coef (p-value), fit on 2024 / fit on 2025. VIF is about 1.0 for every feature in
every model, so collinearity isn't an issue.

**Volume models (plain OLS):**

| Model (in-sample R²) | Own rolling avg | delta | epa_allowed |
|---|---|---|---|
| carries (0.58 / 0.57) | 0.902 / 0.887 (p<.001) | 0.402 / 0.412 (p<.001) | rush: **+15.10 (p<.001) / −0.98 (p=.79)**, sign flips |
| targets (0.34 / 0.39) | 0.784 / 0.833 (p<.001) | 0.227 / 0.274 (p<.001) | pass: 0.88 (.39) / 0.48 (.60) |

**Rate models, plain OLS (R² ≤ 0.008):**

| Model | Intercept | Own rolling rate | delta | epa_allowed |
|---|---|---|---|---|
| ypc | 3.40 / 3.74 | 0.188 (.003) / 0.098 (.10) | ~0 (n.s.) | rush: 1.59 (.51) / **5.22 (.03)** |
| rush_td_rate | 0.021 / 0.025 | 0.223 (.003) / 0.102 (.12) | n.s. | n.s. |
| catch_rate | 0.721 / 0.687 | 0.093 (.17) / 0.103 (.13) | n.s. | n.s. |
| ypr | 6.99 / 7.63 | 0.079 (.24) / −0.036 (.63) | n.s. | n.s. |
| rec_td_rate | 0.030 / 0.041 | 0.127 (.02) / −0.016 (.85) | n.s. | n.s. |

**Rate models, WLS:**
- ypc × `epa_allowed_rush` = **+3.74 (p=.03) / +5.74 (p=.002)**. This is the only matchup
  effect that is significant with the same sign in both directions.
- Rolling ypc = 0.194 (<.001) / 0.119 (.04).
- Rolling rush TD rate = 0.224 (<.001) / 0.091 (.09).

Full tables (SE, 95% CI, t) for every n and direction are printed by the driver.

**Reading the coefficients:**
- Volume carries the model: last-8 usage plus the recent trend.
- Per-game efficiency barely persists: rolling-rate coefficients are 0.0–0.22, so the rate
  predictions mostly fall back to the league-average rate.
- Matchup EPA doesn't reliably move RB volume. It does move yards per carry.

## Documented
- **OPEN_QUESTIONS:**
  - Q-8 extended: 2-point conversions, first downs, fumbles, and the 100/200-yard and
    20-carry bonuses are not predicted and score 0. The bonuses need a distribution
    (links to Q-3).
  - New Q-11: whether to promote, OLS vs WLS, and the fact that this adds fitted
    coefficients, which conflicts with CLAUDE.md's "no fitted model" stance.
  - New Q-12: PBP count reconciliation and the skew shrinkage gap.
- **CLAUDE.md:** new modules and the RB `epa_allowed` mapping decision.
- **README:** results section and known limitations.
- **STATUS.md:** new in-progress entry.

## Decisions for you
1. Promote, iterate, or park? It ties the current formula today. Promoting means revisiting
   the no-fitted-parameter stance (Q-11b).
2. OLS or attempt-weighted rates (Q-11a)?
3. Proceed to QB/WR/TE only after 1 and 2.
