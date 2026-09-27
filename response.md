# Part 1 (RB finalize and merge) + Part 2 (WR stat vector)

**Short version:**
- **RB:** Part 1 steps 1–7 are done, committed and pushed on `worktree-rb-stat-vector`
  (commit `5481cce`).
- **Step 8, the merge, needs you.** This session can't run git in your main checkout, and
  background jobs don't merge on their own. It's a clean fast-forward; commands are at the
  bottom.
- **One finding you should see before merging:** the continuous-window change you asked for
  makes the formula QB/WR/TE still use about 1% worse (details in step 1).
- **WR:** the stat vector beats both baselines on R² and ranking. It's on
  `worktree-wr-stat-vector`, not merged.

## Part 1: RB

### Step 1: continuous rolling window everywhere. Done, with a measured cost
- **One shared utility.** `projections/rolling_window.py` (`trailing_mean`, `trailing_sum`,
  `trailing_count`). It applies `.shift(1)`, is partitioned by player or defense only (so it
  spans the season boundary), and has no blend weights. `n_games_in_window` is the real
  number of games in the window.
- **Used by everything:**
  - player points and volume rolling averages (8 games)
  - opponent skew (17 defense games)
  - `epa_allowed` (17)
  - every stat-vector feature
- **Retired:** the effective-n proposal and the old `min(games/9, 0.9)` prior-season blend.
  Both are recorded as history in `CLAUDE.md`.
- **Cost:** for the incumbent `rolling_avg_prior + opponent_skew` formula, same rows, mean of
  7 leave-one-season-out folds, the continuous window is worse than the old blend:

  | Position | MAE change | Weeks 1–3 MAE change | R² change |
  |---|---|---|---|
  | QB | +0.03 | +0.11 | +0.006 |
  | RB | +0.06 | +0.16 | −0.017 |
  | WR | +0.05 | +0.08 | −0.018 |
  | TE | +0.06 | +0.16 | −0.018 |

- **Likely cause:** the old blend leaned on a returning player's full prior season (about
  17 games), not his last 8 games.
- **A 12-game window** fully recovers QB (MAE 7.895 vs 7.893, with better R² and ranking),
  but WR/TE stay about 0.04 MAE worse. I left the window at 8 (untuned) and logged it as
  **Q-13**.
- **RB isn't affected in production,** since it no longer uses this formula. QB/WR/TE are.
  Your call before merging: accept it, tune the player window, or move WR/TE to the stat
  vector (Part 2 suggests WR would come out ahead).

### Step 2: early season (2026 weeks 1–3, 175 RB rows)
Each cell is MAE / R² / Spearman:

| | New model | Current formula | Rolling alone |
|---|---|---|---|
| Before (last iteration) | 4.765 / 0.437 / 0.722 | 4.376 / 0.505 / 0.753 | 4.376 / 0.505 / 0.753 |
| **Now** | 4.721 / 0.440 / **0.722** | 4.649 / 0.442 / 0.719 | **4.617** / **0.450** / 0.623 |

- **The MAE gap closed from 0.39 to 0.07–0.10.** Mostly the baselines lost their
  full-prior-season advantage; the new model only improved by 0.04.
- **Ranking:** the new model now ties the current formula and is far ahead of
  rolling-alone.
- **Verdict:** the remaining ~0.1 MAE on 175 rows over three weeks is an honest information
  ceiling, not something to chase. It's written up that way in the README.

**Full-season folds** (new model vs current formula vs rolling alone):

| Metric | New model | Current formula | Rolling alone |
|---|---:|---:|---:|
| Mean MAE | **4.717** | 4.769 | 4.733 |
| Mean R² | **0.378** | 0.346 | 0.349 |
| Mean Spearman | **0.685** | 0.663 | 0.668 |
| Mean bias | −0.04 | | |

Folds won by the new model (out of 7):

| Metric | vs current formula | vs rolling alone |
|---|---:|---:|
| MAE | 5 | 5 |
| R² | 7 | 6 |
| Spearman | 7 | 7 |

The final per-season table is in the README.

### Step 3: confidence tiers unaffected. Confirmed
Tiers still come only from `games_this_season`, computed in `engine/expected_points.py`
from the stats rows, independent of any window. I ran the pipeline with the old and new
rolling code: `games_this_season` and `confidence` are identical on all 141,434 rows. The
tier code is unchanged.

### Step 4: shared `assert_unique_key`. Built and applied
- **Where it lives:** `common/frames.py` has `assert_unique_key(df, keys, name)` and
  `assert_no_fanout`. The earlier local checks (skew, `epa_allowed`, the RB driver) were
  replaced with it.
- **Applied at:**
  - `load_player_metadata`: the Hamilton/Moore bug, now fixed
  - `load_players`
  - `load_player_stats` output
  - games → skew join
  - skew → engine join
  - `epa_allowed` → player join
  - actuals → `evaluate_projections`
  - the engine's projection assembly
- **Root cause:** the dynastyprocess crosswalk maps about 10 `gsis_id`s to two rows, sometimes
  two different people. `load_player_metadata` now keeps the row whose name matches
  nflverse's record, then prefers one with a Sleeper ID, then asserts uniqueness.
- **Also moved** the nflreadpy crosswalk URL patch into `clients/nflreadpy/player_data.py`,
  so callers no longer need `test.py`'s copy.

### Step 5: RB wired in through a registry. Done
- **`engine/expected_points.py`:**
  - `IMPLEMENTATIONS = {"rolling_plus_skew": fn, "stat_vector": fn}`
  - `POSITION_IMPLEMENTATIONS = {"QB": "rolling_plus_skew", "RB": "stat_vector", "WR":
    "rolling_plus_skew", "TE": "rolling_plus_skew"}`
  - No position `if` branches.
- **Signature:** `calculate_expected_points(stats_df, context, position_implementations=None)`.
  - `ExpectedPointsContext` carries `skew_df`, `epa_df`, `scoring`, and fitted models; the
    engine never fits anything.
  - Per-call overrides are how the harness scores the old formula for comparison.
- **Output:** adds `projection_method` and keeps every column the tools layer reads.
- **Stat vector generalized** into `stat_vector/core.py` (`StatVectorSpec`,
  `fit_stat_vector`, `StatVectorModel`); RB is just `RB_SPEC`.
- **Removed** the rejected variants (constant and season-to-date rates, the legacy
  EPA-in-volume spec).
- **Harness:** `evaluation/backtest.py` (`load_inputs`, `loso_folds`, `predict_fold`) is used
  by `test.py` (all positions through the production registry) and by
  `stat_vector_eval.py <POS>`. `rb_stat_vector_eval.py` was removed.
- **Checked:** `test.py` runs end to end; its RB numbers match `stat_vector_eval.py RB`
  exactly, and re-running gave identical output.

### Step 6: scope. Confirmed
`tools/expected_points.py` and `tools/registry.py` have no diff against main. The tool's
explanation text still says "rolling + skew" even for RB rows. It doesn't break anything and
is logged under Q-6.

### Step 7: docs. Done
- **CLAUDE.md:**
  - final RB feature set
  - per-position projection formula, including the explicit decision to accept fitted
    coefficients for RB
  - "Swappable implementations" pattern
  - "Continuous rolling windows" section, with the retired blend and effective-n history
    and the measured cost
  - new modules
  - crosswalk bug added to the known-bugs list
- **README:** "Model Evaluation" rewritten with the final LOSO tables. The stale k-grid
  tables were removed; they're in git history.
- **OPEN_QUESTIONS:**
  - **Q-11 resolved** (including Q-11b).
  - **Q-4 updated:** docs rewritten; skew doesn't help at any position; the k grid is still
    not re-run.
  - **Q-5 explicitly still OPEN:** this is backtest-only, with no live projection path.
  - **Q-6 note** about the tool's explanation text.
  - **Q-12:** the effective-n proposal is superseded.
  - **New Q-13:** the window cost.
- **STATUS.md:** updated.

### Step 8: merge. Needs you
- **It's a clean fast-forward:** `main` (`afa22a6`) is an ancestor of the branch.
- **Your uncommitted work is already inside it.** I compared all 17 dirty files in your main
  checkout against the snapshot commit `ea60f13`, and every one matches. The only thing the
  branch doesn't have is `prompt.txt` (plus `.pyc` churn and an empty `docs/`).
- **In `C:\Users\jak3t\Audible IQ`:**
  ```
  git stash push -u -m "pre-rb-merge backup"
  git merge --ff-only worktree-rb-stat-vector
  git restore --source="stash@{0}^3" -- prompt.txt
  git push origin main
  git branch -d worktree-rb-stat-vector
  git push origin --delete worktree-rb-stat-vector
  ```
- **After checking things look right,** run `git stash drop` to delete the backup.
- **Keep the worktree** `.claude/worktrees/rb-stat-vector`: it's now on the WR branch. Remove
  it with `git worktree remove .claude/worktrees/rb-stat-vector` after the WR review.
- **If you've decided against continuous windows for QB/WR/TE (Q-13), tell me before
  merging** and I'll change it on the branch first.

## Part 2: WR stat vector (branch `worktree-wr-stat-vector`, not merged)

- **Setup:** same structure as RB (`stat_vector/wr.py`), in `STAT_VECTOR_SPECS` for
  evaluation only; still `rolling_plus_skew` in production.
  - **Volume models:** targets and carries.
  - **Rate models:** catch_rate, ypr and rec_td_rate use `epa_allowed_pass`; ypc and
    rush_td_rate use `epa_allowed_rush`.
  - **Rates** are attempt-weighted.
  - **Recomposition:** through `calculate_points_vectorized` with `wr_receptions`.
- **Unpredicted categories** cost 0.054 points/game on actual 2024/25 WR stats.
- **Re-run:** `python stat_vector_eval.py WR --targets-epa`.

### Leave-one-season-out table
Each cell is MAE / R² / Spearman.

| Test season | New model | Current formula | Rolling alone |
|---|---|---|---|
| 2019 | **4.836** / **0.325** / **0.657** | 4.885 / 0.287 / 0.625 | 4.911 / 0.288 / 0.625 |
| 2020 | 4.956 / **0.317** / 0.641 | 4.956 / 0.307 / 0.642 | **4.932** / 0.312 / **0.646** |
| 2021 | 4.773 / **0.348** / **0.669** | 4.776 / 0.327 / 0.659 | **4.750** / 0.338 / 0.667 |
| 2022 | **4.747** / **0.337** / **0.628** | 4.791 / 0.298 / 0.608 | 4.786 / 0.301 / 0.613 |
| 2023 | 4.556 / **0.368** / **0.643** | **4.532** / 0.347 / 0.622 | 4.548 / 0.350 / 0.623 |
| 2024 | 4.614 / **0.347** / **0.659** | 4.594 / 0.321 / 0.647 | **4.560** / 0.331 / 0.652 |
| 2025 | **4.344** / **0.362** / **0.660** | 4.405 / 0.311 / 0.632 | 4.373 / 0.319 / 0.636 |
| **Mean** | **4.689** / **0.343** / **0.651** | 4.706 / 0.314 / 0.634 | 4.694 / 0.320 / 0.637 |
| 2026 wk 1–3 (286 rows) | **4.745** / **0.328** / **0.595** | 4.775 / 0.293 / 0.561 | 4.755 / 0.301 / 0.580 |

**Folds won by the new model (out of 7):**

| Metric | vs current formula | vs rolling alone |
|---|---:|---:|
| R² | 7 | 7 |
| Spearman | 6 | 6 |
| MAE | 5 | 3 |

- **Mean bias:** +0.03, against +0.08 for the current formula and −0.14 for rolling-alone.
- **Early season:** unlike RB, WR also wins the partial 2026 fold on every metric.
- **Caveat:** these baselines carry the continuous-window cost from step 1. Against the old
  blended formula (WR 4.661 / 0.332 / 0.644), the new model's MAE is 0.03 worse, while R²
  and Spearman are still better.

### Coefficient stability across the 7 folds

| Model | Term | Range | Positive in | p < .05 in |
|---|---|---|---:|---:|
| targets | trailing avg | 0.872–0.876 | 7 | 7 |
| targets | delta | 0.228–0.253 | 7 | 7 |
| carries | trailing avg | 0.70–0.78 | 7 | 7 |
| carries | delta | 0.10–0.20 | 7 | 7 |
| catch_rate | trailing rate | 0.245–0.257 | 7 | 7 |
| catch_rate | **epa_allowed_pass** | **+0.164 to +0.237** | 7 | **7** |
| ypr | trailing rate | 0.279–0.294 | 7 | 7 |
| ypr | epa_allowed_pass | −0.55 to +2.78 | 6 | 1 |
| rec_td_rate | trailing rate | 0.117–0.148 | 7 | 7 |
| rec_td_rate | epa_allowed_pass | −0.007 to +0.040 | 6 | 0 |
| ypc | all terms | noise (trailing rate negative, EPA −4.3 to +1.1) | – | 0 |
| rush_td_rate | all terms | noise | – | 0 |

- **Receiving efficiency persists much more for WRs than RBs.** The trailing-rate
  coefficients are 0.13–0.29, against 0.05–0.16 for RBs, and every one is p < 1e-8.
- **The matchup signal is in catch rate.** Weaker pass defenses reliably raise it; the EPA
  terms on yards per reception and TD rate aren't reliable.
- **The rushing side is noise, as expected.** It's harmless because WR carries are tiny.

### Does `epa_allowed` belong in the targets volume model? No
- **Coefficient:** −0.31 to +0.27, sign flips between folds, p = 0.48–0.89 in every fold.
- **Accuracy:** mean MAE 4.690 vs 4.689, identical R² and Spearman.
- **Conclusion:** target share among WRs doesn't follow the matchup either. Same conclusion
  as RB, but tested here independently.

**Stopping here as instructed:** WR is not merged and main is untouched. If you approve, the
change is one line: `"WR": "stat_vector"` in `POSITION_IMPLEMENTATIONS`.

## Decisions for you
1. Q-13: keep continuous windows for QB/WR/TE despite about +0.03–0.06 MAE, or change it
   before the RB merge?
2. Run the merge commands above (or tell me what to change first).
3. WR: approve the switch to the stat vector (with or without `epa_allowed` on yards per
   reception and TD rate), or send back for changes.
