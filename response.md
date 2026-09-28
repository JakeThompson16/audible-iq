# QB on the stat vector, training pipeline, prediction API, tests

**Summary:**
- **Status:** all five parts are done on branch **`worktree-stat-pipeline`** (commit `1049438`,
  pushed, **not merged**). It's a fast-forward from main (`2fbc712`).
- **No modeling changes:** the specs, windows, attempt-weighting and `fit_stat_vector` are
  untouched.
- **Engine unaffected:** after all the refactors, `test.py` output is byte-identical to the
  Part 1 run.
- **Tests:** the suite passes, 14 tests.
- **Parity with the backtest harness is exact:** 48 player-weeks, 396 stat values,
  0 mismatches.

## Part 1: QB to the stat vector
- **Change:** `POSITION_IMPLEMENTATIONS["QB"] = "stat_vector"`.
  - The QB spec keeps its own window, 12.
  - `rolling_plus_skew` stays registered as the alternative implementation and evaluation
    baseline.
  - opponent_skew isn't deleted.
- **Recorded in CLAUDE.md as an explicit owner override with the numbers,** not as a policy
  pass.

**`test.py`, mean of 7 folds (MAE / R² / Spearman):**

| Pos | Result | Expected | Match |
|---|---|---|---|
| QB | 7.911 / 0.238 / 0.483 | 7.911 / 0.238 / 0.483 | exact |
| RB | 4.717 / 0.378 / 0.686 | same | unchanged |
| WR | 4.690 / 0.344 / 0.651 | same | unchanged |
| TE | 4.233 / 0.326 / 0.590 | same | unchanged |

**Portability** (the same fitted models run through the full engine; 678 QB rows, 2025):
- **Pass TD 4 → 6 points:** every QB projection rises (mean +2.48, minimum +0.47). That's
  +2 × predicted passing TDs.
- **INT −1 → −2:** every QB projection falls (mean −0.62, max −0.15). That's −1 × predicted
  INTs.

**Tool text:** a QB row shows the stat-vector explanation: "projection = this league's scoring
applied to a predicted stat line (stat vector: predicted volume — pass attempts, targets,
carries — …) opponent_skew is not part of this projection." (Justin Herbert, 2025 wk 10.)

**QB 2026 partial fold, same 104 rows (weeks 1–3), before vs after:**

| QB 2026 partial | MAE | R² | Spearman | Bias |
|---|---:|---:|---:|---:|
| Before: rolling + skew (window 14) | 8.406 | 0.160 | 0.364 | +0.13 |
| After: stat vector | 8.559 | 0.154 | 0.323 | +0.99 |

Early-season QB gets worse with the override. That's recorded in CLAUDE.md with it.

## Part 2: training pipeline
- **Entry points:** `python -m pipeline.train retrain [--start 2022] [--end YYYY] [--dry-run]`,
  or `train_all(start_season=2022, end_season=None, save=True)`.
- **Data window:**
  - Fit on 2022 through **2026 week 2**. Week 3 has 14 of 16 games final, so it's excluded.
  - 2020–2021 are loaded as history only.
  - Postseason weeks are included for complete seasons, matching the backtest.

**Training rows per position (player-weeks, rows per sub-model):**

| Pos | Player-weeks | Sub-model rows |
|---|---:|---|
| QB | 2,817 | attempts / carries 2,768; completion & INT rate 2,651; yards/completion & pass-TD rate 2,606; rush rates 2,443 |
| RB | 6,635 | carries / targets 6,487; rush rates 5,430; catch rate 4,452; ypr & rec-TD rate 4,060 |
| WR | 10,604 | targets / carries 10,380; catch rate 8,969; ypr & rec-TD rate 8,031; rush rates 1,356 |
| TE | 5,269 | targets / carries 5,160; catch rate 4,594; ypr & rec-TD rate 4,104; **rush rates 189** (smallest) |

- **Artifacts:** `artifacts/stat_vector/<POS>.json` are committed.
  - Contents: schema version, spec name and sha256 spec hash, spec config, ordered features
    per sub-model, intercept, coefficients, SEs, p-values, n, in-sample R², rate priors,
    weighting flag, window, seasons and weeks trained on, `data_through`, row counts, fit
    timestamp.
  - **The loader raises `ArtifactSpecMismatch`** on any difference in features, order,
    denominators, derivations, window or weighting.
- **Safe overwrite:** write to a temp dir, validate, swap in, and keep `<POS>.json.prev`
  (git-ignored).
  - Tested: a NaN coefficient or a row count below the floor leaves every file byte-identical
    and no temp dir behind.
  - **Errors (block the write):** NaN/inf; a sub-model below its row floor (1,000 volume,
    50 rate); a coefficient beyond its 2019–2025 range by more than the range's own width; a
    previously significant coefficient (p < 0.01) jumping more than 5 SE and 50%.
  - **Warnings:** everything milder.
- **Idempotent:** a second real retrain produced artifacts identical to the previous
  generation except `fit_timestamp`. The determinism test checks the same thing.
  - Floats are rounded to 12 significant digits because Polars' multi-threaded aggregation
    flips the last bit of some floats between runs. I saw that happen once while checking
    the `epa_allowed` refactor.
- **Scheduling:** not set up. It's documented in CLAUDE.md: run the CLI Tuesday morning, after
  Monday Night Football is final (Task Scheduler or cron). The CLI exits non-zero on a
  validation error.

### Coefficient drift vs the 2019–2025 leave-one-season-out ranges
- **46 of 106 coefficients are inside their fold range; 60 are outside** (QB 15, RB 16,
  WR 15, TE 14). **None are gross.**
- **Why so many flags:** the fold ranges are narrow, because each fold shares five of six
  seasons, and production trains on a different window (2022–2026). So many small flags are
  expected. Notable ones:

| Pos | Term | Production (2022+) | 2019–2025 range | Note |
|---|---|---:|---|---|
| QB | int_rate × trailing INT rate | 0.069 | 0.117 … 0.176 | INT persistence weaker in recent seasons |
| QB | yards/completion × epa_allowed_pass | 0.68 | 1.15 … 3.56 | matchup effect on yardage smaller |
| QB | rush_td_rate × epa_allowed_rush | 0.010 | 0.038 … 0.110 | was never significant |
| TE | carries × delta | −0.217 | +0.088 … +0.423 | **sign flip**, tiny-volume model |
| TE | ypr × epa_allowed_pass | 3.09 | 3.68 … 5.74 | still clearly positive |
| WR | ypc × epa_allowed_rush | 3.28 | −4.29 … +1.08 | noise model (WR carries) |

The full table is in `artifacts/MODEL_METRICS.md`.

### Out-of-sample accuracy (from MODEL_METRICS.md)
- **Method:** each season 2022–2025 is scored by models fit on the window's other full seasons;
  2026 weeks 1–2 is test-only.
- **Numbers:** mean over the 4 full seasons. Each cell is MAE / R² / Spearman / bias.

| Pos | Stat vector | Rolling alone | 2026 partial: stat vector / rolling alone |
|---|---|---|---|
| QB | **7.746 / 0.240 / 0.487 / −0.12** | 7.793 / 0.215 / 0.480 / −0.20 | 8.757 / 0.176 / 0.323 vs 8.550 / 0.206 / 0.409 (74 rows) |
| RB | **4.533 / 0.401 / 0.705 / −0.04** | 4.553 / 0.371 / 0.686 / −0.13 | 4.655 / 0.428 / 0.684 vs 4.611 / 0.427 / 0.684 |
| WR | **4.533 / 0.353 / 0.647 / 0.02** | 4.554 / 0.334 / 0.633 / −0.24 | 4.670 / 0.323 / 0.617 vs 4.780 / 0.275 / 0.582 |
| TE | **4.140 / 0.341 / 0.625 / −0.00** | 4.212 / 0.316 / 0.607 / −0.07 | 4.600 / 0.250 / 0.542 vs 4.629 / 0.253 / 0.560 |

- **Full seasons:** the stat vector beats rolling-alone on mean MAE, R² and Spearman at every
  position. These are fewer and shorter folds than the README's 2019–2025 table, so the
  numbers aren't identical.
- **Calibration:** normalized MAE by tier is monotonic at every position.

## Part 3: prediction API (`pipeline/predict.py`)
- **Calls:** `predict_player_stats(player_id, gsis_id=False, season=None, week=None)` and
  `predict_many(ids, ...)`. The single call is a thin wrapper over the batch.
- **IDs:** everything is compared as a string. `normalize_id` makes 4984, 4984.0 and "4984.0"
  all "4984".
  - The maps come from `load_player_metadata` (its dedupe and `assert_unique_key` kept),
    cached with `lru_cache`.
  - `refresh_player_maps()` clears them.
- **Features:**
  - A placeholder row for the target game is appended to the player's history and run through
    the same `build_features` the backtest uses.
  - `epa_allowed` for a defense-week not yet played is looked up as of the defense's last game
    (`epa_allowed_from_weekly(..., as_of_keys=)`).
  - Checked: the as-of value for a future week equals the value the full table later gives
    that week (max difference 1e-17). The refactored `calculate_epa_allowed` is identical to
    the old one on 2023–25 data.
- **Context:** loaded once per process and cached (`refresh_context()` reloads it). Per-player
  calls never touch nflreadpy.
- **Output:** the JSON dict as specified.
  - Statuses: `ok | bye | no_game_scheduled | no_history | unknown_player |
    unsupported_position`. Missing data gets a status, never zeros.
  - `stats` uses the exact `calculate_points_vectorized` column names and contains only
    predicted categories. `unpredicted` lists the rest.
- **Example:** Lamar Jackson's next game (2026 week 4 vs TEN; team from the weekly roster):
  - 25.8 attempts, 17.0 completions, 205 passing yards, 1.28 passing TDs, 0.51 INTs
  - 5.1 carries, 23.8 rushing yards
  - confidence "low" (3 games this season)

### Current-team source: nflverse weekly rosters
- **Rule:** weekly roster at the latest week not after the target (pre-game information), then
  `players.latest_team`, then the last stats row.
- **What I compared:**
  - **ff_playerids `team`:** unusable. Different abbreviations (KCC, LAR, JAC): 128 of 446
    2026 players "mismatch" on abbreviation alone, and those codes don't join to the
    schedule.
  - **Last stats row:** stale for anyone who hasn't played yet this season. 85 of 174 players
    whose last game was in 2025 are on a different 2026 roster team.
  - **Weekly rosters:** equal to the team on the actual stats row in **every** played
    QB/RB/WR/TE week, 2020–2026 (0 mismatches). 37 player-weeks have no roster row and fall
    back.
- **Traded-player mismatch found:** 1 player. Jimmy Horn Jr.'s last stats row is with CAR, but
  his latest 2026 roster team is CLE (a move after his last game). The roster rule projects
  him with CLE, which is correct for his next game.

## Part 4: tests (`python -m pytest`, 14 pass)
- **Runtime:** about 50 s once nflreadpy's cache is warm; much longer on a first-ever run while
  it downloads.

| Test | Result |
|---|---|
| **Parity** | 2024 and 2025, all four positions, with week-1, mid-season and post-bye samples. The pipeline uses the model fit exactly as the backtest fits that held-out season, not the production artifact. **48 player-weeks, 396 values, 0 mismatches at 1e-9**, opponents all match. |
| **Round-trip** | Output dict → one-row frame → `calculate_points_vectorized` with the real league scoring equals the engine's projection for that player-week at 1e-9. 8+ cases. |
| **IDs** | Sleeper str == Sleeper int == GSIS for 5 players; an unknown id → `unknown_player`; a batch equals the single calls; a duplicate id in a batch shares its result. |
| **Status paths** | bye, rookie first game → `no_history`, kicker → `unsupported_position`, week 30 → `no_game_scheduled`. |
| **Artifact safety** | A changed window, swapped feature order or renamed feature raises `ArtifactSpecMismatch`; a failed validation leaves every file byte-identical. |
| **Determinism** | Two consecutive fits are identical, and equal to the saved artifact apart from the timestamp. |

- **One test fix during the run:** my first bye test picked a roster player who isn't in the ID
  crosswalk. The API correctly returned `unknown_player`; the test now picks a crosswalk
  player.

**Timings:**

| Call | Time |
|---|---:|
| Cold single call (loads the context once, nflreadpy cache warm) | 15.2 s |
| Warm single call | 0.14 s |
| 20-player batch (all `ok`) | 0.45 s |

## Part 5: docs
- **CLAUDE.md:**
  - the QB owner override with its numbers;
  - the registry;
  - a new "Production pipeline" section: retraining, artifact format, safe overwrite,
    prediction API, ID conventions, current-team rule, as-of semantics;
  - the new modules.
- **OPEN_QUESTIONS:**
  - **Q-4:** skew is now used by no production position.
  - **Q-5:** marked largely solved, with what's still open.
  - **Q-7:** a note about the prediction API.
  - **Q-8:** QB's unpredicted fields and the leagues they affect.
  - **Q-14:** now the next QB priority, with both measured gaps.
- **STATUS and README:** updated.

## Decisions for you
1. **Merge:** `worktree-stat-pipeline` is a fast-forward from main. From
   `C:\Users\jak3t\Audible IQ`:
   ```
   git merge --ff-only worktree-stat-pipeline
   git push origin main
   ```
   I can't merge or push main from this background session.
2. **Drift reference:** the 2019–2025 fold ranges flag about 60 of 106 coefficients on every
   retrain, because they're narrow and the production window differs. The warnings will be
   noise week to week. Options:
   - build the reference from leave-one-season-out folds *inside the production window*
     (2022+) instead;
   - loosen the warning threshold;
   - keep it as is.
3. **Artifacts in git:** they're committed now, so every weekly retrain changes tracked files.
   Commit them weekly (an audit trail), or git-ignore them and treat `MODEL_METRICS.md` as the
   record?
4. **Early-season QB:** the override makes QB's early-season fold worse (2026 wks 1–3 MAE
   8.559 vs 8.406, Spearman 0.323 vs 0.364). Accepted as part of the override, but worth
   knowing.
5. **Still open (Q-5):**
   - projected *fantasy points* for future games through the engine; today it's
     `calculate_points_vectorized(prediction["stats"], scoring)`;
   - injury/inactive status (a rostered player who's ruled out is still projected);
   - wiring `predict_player_stats` into `tools/` (Q-6/Q-7).
6. **Dependency:** I installed `pytest` 9.1.1 into your Python with pip. The project has no
   requirements file. Want one?
