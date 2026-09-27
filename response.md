# Pre-merge checks, QB window 14, merge status

**Summary:**
- **Both checks are done:**
  - The confidence-tier inversion is confirmed to be a scale effect: the tiers are fine
    (Q-15 resolved).
  - QB's rushing issue is real but different from the week-10 anecdote (Q-14 updated).
- **QB's rolling window is now 14,** committed and pushed on `worktree-qb-te-stat-vector`
  (`4795c89`).
- **Superseded branch deleted:** `worktree-wr-stat-vector`, locally and on origin.
- **Steps 3–5 (merge to main, push main, delete the merged branches): not executed.** This
  background session's rules forbid merging and pushing to main, and the sandbox blocks git in
  your main checkout. I checked that everything is ready; the three commands are below.

## 1. Mean actual points per tier (Q-15). Verified
Pooled held-out predictions 2019–2025, final registry (QB rolling window 14). Each cell is
MAE / mean actual points / **normalized MAE** (MAE ÷ mean actual).

| Pos | high | medium | low | insufficient | Raw MAE monotonic? | Normalized monotonic? |
|---|---|---|---|---|---|---|
| QB | 7.889 / 20.4 / **0.386** | 7.764 / 18.3 / **0.425** | 7.951 / 14.5 / **0.547** | 8.280 / 12.4 / **0.668** | No | **Yes** |
| RB | 5.093 / 9.31 / **0.547** | 4.507 / 7.71 / **0.585** | 4.430 / 6.87 / **0.645** | 4.543 / 6.37 / **0.713** | No | **Yes** |
| WR | 4.926 / 8.53 / **0.578** | 4.561 / 7.29 / **0.626** | 4.450 / 6.63 / **0.671** | 4.678 / 6.52 / **0.717** | No | **Yes** |
| TE | 4.690 / 8.48 / **0.553** | 4.106 / 6.70 / **0.613** | 3.881 / 5.43 / **0.715** | 3.899 / 4.99 / **0.781** | No | **Yes** |
| All | 5.235 / 9.97 / 0.525 | 4.774 / 8.39 / 0.569 | 4.751 / 7.40 / 0.642 | 4.977 / 6.97 / 0.714 | No | **Yes** |

- **High-tier players score 30–55% more than low-tier ones,** as expected.
- **Normalizing fully resolves the inversion.** Relative error falls monotonically from
  insufficient → low → medium → high at every position.
- **So the tiers are correctly ordered.** The raw-MAE check was the wrong metric. That's now
  a verified finding, not a hypothesis, and **it isn't a reason to call the tiers
  unreliable.**
- **For the agent:** the tiers are *relative*-uncertainty labels (errors of roughly 39–58% of
  a high-tier player's points vs 55–72% for low tier), not absolute point ranges.
- **Code:** `engine/metrics.py` now reports `mean_actual`, `normalized_mae` and
  `monotonic_decreasing_normalized_mae`, and its note points to the normalized check.
- **Docs:** CLAUDE.md "Confidence tiers" updated, Q-15 marked resolved.

## 2. QB Spearman segmented by rushing volume (Q-14)
- **Method:** within each held-out season, QBs are split into thirds by rush attempts per
  game.
- **Metrics:**
  - Spearman within each week and third;
  - bias (actual − predicted);
  - mean rank error = the projection's weekly percentile rank minus the actual's, among
    all QBs. Negative means underranked.

| Rushing third | Rush att/g | Spearman: stat vector / current | Stat vector wins | Bias: stat vector / current | Rank error: stat vector / current |
|---|---:|---|---:|---|---|
| Low | 1.4 | 0.539 / **0.561** | 1/7 | −2.33 / −2.03 | −0.018 / +0.013 |
| Mid | 2.5 | **0.403** / 0.375 | 5/7 | −0.76 / −0.39 | −0.035 / −0.031 |
| High | 5.3 | 0.312 / **0.363** | 1/7 | **+1.60 / +0.68** | +0.034 / +0.014 |

- **The week-10 anecdote doesn't generalize as stated.** Across all QBs, the stat vector
  doesn't rank high-rushing QBs *below* where they finish. Their mean rank error is
  positive: slightly over-ranked, as with the current formula.
- **But two rushing-specific weaknesses are consistent:**
  - **Bias:** it under-projects high-rushing QBs by 1.60 points/game, against 0.68 for the
    current formula, and does so more than the current formula in 7 of 7 folds.
  - **Ranking:** it orders high-rushing QBs worse among themselves (Spearman 0.312 vs
    0.363; the current formula is better in 6 of 7 folds).
- **Low-rushing third:** it also favours the current formula; the middle third favours the
  stat vector.
- **Conclusion:** improving the QB rush component (the scramble / designed-run split, or
  better rushing rates) targets a real, repeatable gap, not one-week noise. It's the
  recommended next QB step.
- **Production unchanged:** QB stays on `rolling_plus_skew`, as you said.

## 3. QB rolling window reselected to 14. Done
- **Change:** `POSITION_ROLLING_WINDOWS["QB"] = 14`, from the existing grid table, the
  Spearman-first pick.
- **Full-engine backtest at 14:** QB 7.915 / 0.217 / 0.492 (MAE / R² / Spearman), against
  7.895 / 0.220 / 0.487 at 12. That trades 0.02 MAE for +0.005 Spearman.
- **Side effect:** the Spearman gap to the QB stat vector (0.483) widens, which reinforces the
  no-switch decision.
- **RB, WR and TE are unchanged:** 4.717 / 0.378 / 0.686, 4.690 / 0.344 / 0.651 and
  4.233 / 0.326 / 0.590.
- **Docs:** README production and window tables, CLAUDE.md policy note and windows,
  OPEN_QUESTIONS and STATUS updated.

## 4–5. Merge and cleanup. Ready; you need to run it
- **Fast-forward:** `worktree-wr-stat-vector-tuned` is an ancestor of
  `worktree-qb-te-stat-vector`, and both descend from `main` (`fac0d4b`). One fast-forward to
  the QB/TE tip merges both.
- **Your main checkout is clean for every file these branches touch.** I compared all 17
  changed files with `fac0d4b`, and none of the three new files (`wr.py`, `te.py`, `qb.py`)
  exist there yet. No stash needed this time.
- **Superseded branch:** `worktree-wr-stat-vector` is already deleted, locally and on origin.
  Its one commit had been cherry-picked into the tuned branch.
- **In `C:\Users\jak3t\Audible IQ`:**
  ```
  git merge --ff-only worktree-qb-te-stat-vector
  git push origin main
  git branch -d worktree-wr-stat-vector-tuned worktree-qb-te-stat-vector
  git push origin --delete worktree-wr-stat-vector-tuned worktree-qb-te-stat-vector
  git worktree remove .claude/worktrees/rb-stat-vector
  ```
- **Last line:** it removes this session's worktree. That also frees the
  `worktree-qb-te-stat-vector` branch for `-d`, so if `-d` complains that the branch is
  checked out, run the worktree removal first.

## Open after merge
- **Q-14:** QB rush component (scramble split) is the next QB experiment.
- **Q-4:** opponent skew, now QB-only; k grid not re-run.
- **Q-5:** no live projection path.
- **Q-8:** unpredicted categories. They cost QB the most: 0.47 points/game in recomposition.
