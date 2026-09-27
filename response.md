# Window tuning, merge check, tool explanation fix, WR switch

**Summary:**
- **Steps 1, 2 and 4** are done on `worktree-rb-stat-vector` (commits `75e1fc6`, `fac0d4b`,
  pushed).
- **Step 3:** I confirmed the merge is still a clean fast-forward, but **I didn't run it.**
  This background session's rules forbid merging and pushing to main, and the sandbox
  blocks git in your main checkout. The commands are below; nothing about them changed.
- **Step 5:** WR still beats the newly tuned baseline, so it's switched to `stat_vector` on a
  new branch, `worktree-wr-stat-vector-tuned` (commit `0c0c8fb`, pushed, not merged).

## Step 1: window length per position
- **Method:** same continuous-window utility, no blend weights. Grid {8, 10, 12, 14} plus 16
  and 20, so an optimum at the edge of the grid would show up.
- **Selection rule, stated before looking:** lowest mean MAE over the 7 full-season folds.
  All variants are scored on the same rows.
- **Reproduce:** `python rolling_window_eval.py`.

Each cell is MAE / R² / Spearman (mean of 7 folds); the weeks 1–3 and 2026 columns are MAE.

| Pos | Best window | At best window | Window 8 | Old blend | Weeks 1–3: best / 8 / blend | 2026 partial: best / 8 / blend |
|---|---:|---|---|---|---|---|
| QB | **12** | 7.894 / 0.216 / 0.483 | 7.925 / 0.208 / 0.475 | 7.893 / 0.202 / 0.473 | 8.060 / 8.187 / 8.078 | 8.342 / 8.108 / 8.232 |
| WR | **10** | 4.700 / 0.321 / 0.635 | 4.709 / 0.314 / 0.634 | 4.661 / 0.332 / 0.644 | 5.086 / 5.145 / 5.063 | 4.783 / 4.789 / 4.700 |
| TE | **12** | 4.321 / 0.295 / 0.569 | 4.344 / 0.283 / 0.564 | 4.282 / 0.301 / 0.573 | 4.555 / 4.644 / 4.481 | 4.643 / 4.847 / 4.591 |

- **QB fully recovers at 12.** MAE ties the old blend, and R², Spearman and weeks 1–3 all
  beat it. The 76-row 2026 fold is worse, but it's too small to tune on.
- **WR and TE recover a third to a half of the gap.** They're still about 0.04 MAE behind
  the old blend.
- **WR's curve is flat:** 10, 12 and 14 are within 0.007 MAE, and 14 is best on R² and
  Spearman. I stuck to the MAE rule.
- **Where the optimum is on other metrics:** Spearman peaks at 14 for QB and WR, 12 for TE.
- **RB stays at 8.** It's on the stat vector, so its rolling average only feeds the baselines.

## Step 2: per-position windows as config
- **Registry:** `POSITION_ROLLING_WINDOWS = {"QB": 12, "RB": 8, "WR": 10, "TE": 12}` sits next
  to `POSITION_IMPLEMENTATIONS` in `engine/expected_points.py`.
- **Context:** `ExpectedPointsContext.rolling_windows` defaults to that mapping.
- **Features:** `add_rolling_features(window=<mapping>)` builds each position with its own
  window and stamps a `rolling_window` column.
- **Enforced, not trusted:** the engine rejects stats built with a different window. I tested
  this: matching windows pass, and a shared 8 gets a clear `ValueError`.
- **No position branches:** the window is a lookup (`replace_strict`), not an `if`.
- **Harness:** `evaluation/backtest.load_inputs` uses the registry mapping by default.
- **Full-engine backtest** (`test.py`), mean of 7 folds:

  | Position | MAE | R² | Spearman |
  |---|---:|---:|---:|
  | QB | 7.895 | 0.220 | 0.487 |
  | RB | 4.717 | 0.378 | 0.686 |
  | WR (still rolling + skew at this point) | 4.697 | 0.321 | 0.635 |
  | TE | 4.318 | 0.295 | 0.570 |

## Step 3: merge. Clean fast-forward confirmed, not executed
- **Fast-forward:** `main` (`afa22a6`) is an ancestor of `worktree-rb-stat-vector`, which is
  now 7 commits ahead (tip `fac0d4b`).
- **Your uncommitted work:** all 17 dirty files in your main checkout still match the
  snapshot commit, so the merge loses nothing.
- **Why I didn't run it:** this job's rules say background sessions never merge or push to
  main. The worktree sandbox also refuses git commands aimed at your checkout.
- **Run these in `C:\Users\jak3t\Audible IQ`** (same as before):
  ```
  git stash push -u -m "pre-rb-merge backup"
  git merge --ff-only worktree-rb-stat-vector
  git restore --source="stash@{0}^3" -- prompt.txt
  git push origin main
  git branch -d worktree-rb-stat-vector
  git push origin --delete worktree-rb-stat-vector
  ```
  Then run `git stash drop` once everything looks right.
- **What's included:** RB, the window tuning and the tool fix (step 4). To leave the tool fix
  out, fast-forward to `75e1fc6` instead.

## Step 4: tool explanation (Q-6). Fixed
- **What changed:** `tools/expected_points.py` now picks its explanation from the row's
  `projection_method` (`_METHOD_EXPLANATIONS`, same keys as the engine registry):
  - **`rolling_plus_skew`:** "projection = rolling_avg_prior (12.30, last 12 games) +
    opponent_skew (0.80) vs DAL."
  - **`stat_vector`:** "projection = this league's scoring applied to a predicted stat line
    (…per-attempt rates adjusted for DAL's EPA allowed). opponent_skew is not part of this
    projection."
- **Metadata:** now includes `projection_method` and `rolling_window`.
- **Compatibility:** older outputs without the column fall back to the rolling text.
- **Tested** with rows of both kinds, a rookie with no projection, and a legacy frame.
- **Still open:** the rest of Q-6 (typed payload, per-tool confidence).
- **Recorded in** OPEN_QUESTIONS and CLAUDE.md.

## Step 5: WR against the tuned baseline. Switched
- **Spec, as you asked:** `epa_allowed` is off for targets and on for catch_rate only among
  the receiving rates. The yards-per-reception and TD-rate EPA terms stay in the spec as
  `candidate_terms`: off by default, re-testable with `stat_vector_eval.py WR --candidates`.
- **Rushing-side WR rates** keep `epa_allowed_rush` as originally specified; it's noise but
  harmless.
- **Baselines** below use the tuned WR window of 10.

Each cell is MAE / R² / Spearman:

| Test season | WR stat vector | Current formula (w=10) | Rolling alone (w=10) |
|---|---|---|---|
| 2019 | **4.836** / **0.325** / **0.657** | 4.906 / 0.287 / 0.628 | 4.941 / 0.287 / 0.628 |
| 2020 | 4.959 / 0.317 / 0.640 | 4.954 / 0.314 / 0.643 | **4.926** / **0.319** / **0.646** |
| 2021 | 4.772 / **0.349** / **0.669** | 4.785 / 0.330 / 0.659 | **4.762** / 0.339 / 0.665 |
| 2022 | 4.747 / **0.337** / **0.628** | **4.736** / 0.311 / 0.618 | 4.741 / 0.314 / 0.623 |
| 2023 | 4.557 / **0.367** / **0.644** | **4.543** / 0.353 / 0.626 | 4.574 / 0.354 / 0.625 |
| 2024 | 4.611 / **0.348** / **0.659** | 4.570 / 0.328 / 0.641 | **4.543** / 0.338 / 0.646 |
| 2025 | **4.345** / **0.362** / **0.660** | 4.388 / 0.324 / 0.630 | 4.358 / 0.333 / 0.637 |
| **Mean** | **4.689** / **0.344** / **0.651** | 4.697 / 0.321 / 0.635 | 4.692 / 0.326 / 0.638 |
| 2026 wk 1–3 | **4.745** / **0.328** / **0.595** | 4.772 / 0.287 / 0.561 | 4.782 / 0.291 / 0.560 |

**Folds won by the stat vector (out of 7):**

| Metric | vs current formula | vs rolling alone |
|---|---:|---:|
| R² | 7 | 6 |
| Spearman | 6 | 6 |
| MAE | 3 | 3 |

- **MAE:** it wins only 3 of 7 folds but has the lowest mean MAE of the three.
- **Bias:** +0.03, against +0.04 for the current formula and −0.17 for rolling-alone.
- **Your criterion is met:** R² and Spearman win at comparable-or-better MAE. So
  `"WR": "stat_vector"` is flipped in `POSITION_IMPLEMENTATIONS`.
- **Through the full engine:** WR 4.690 / 0.344 / 0.651; the other positions are unchanged.
- **Candidate terms:** turning them on gives mean 4.690 / 0.343 / 0.651, no gain.
  - targets EPA: −0.31 to +0.27, p ≥ 0.48 in every fold
  - yards-per-reception EPA: significant in 1 of 7 folds
  - TD-rate EPA: significant in 0 of 7 folds
- **Stable coefficients:**
  - targets trailing average 0.872–0.876
  - catch rate × `epa_allowed_pass` +0.16 to +0.24, significant in all 7 folds
- **CLAUDE.md** features only catch rate as a validated matchup finding. The other EPA terms
  are described as off and not validated.

## Branches
- **`worktree-rb-stat-vector`** (`fac0d4b`): ready to fast-forward into main. The merge
  commands are in step 3.
- **`worktree-wr-stat-vector-tuned`** (`0c0c8fb` plus this report): WR switch plus docs, on
  top of the RB branch. After the RB merge it's a fast-forward too:
  `git merge --ff-only worktree-wr-stat-vector-tuned`, then `git push origin main`.
- **`worktree-wr-stat-vector`:** superseded. Delete it with
  `git push origin --delete worktree-wr-stat-vector` and
  `git branch -D worktree-wr-stat-vector`.
- **The worktree** `.claude/worktrees/rb-stat-vector` is on the tuned WR branch.

## Open items
- **Q-13:** only TE is still about 0.04 MAE behind the old blend. Options: accept it, or
  validate a TE stat vector next.
- **Q-4:** opponent skew doesn't beat rolling-alone for QB or TE, and the k grid hasn't been
  re-run.
- **Q-5:** there's still no live projection path.
