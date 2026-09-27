# Audible IQ — Open Questions / Deferred Decisions

_Baseline: 2026-09-24 audit. Format: status / cause / resolution path.
Numbering is stable; don't renumber._

## Decisions flagged by the V1 vision (VISION.md)

### Q-1. Sleeper write strategy — OPEN
- **Cause:** Sleeper's public API is read-only, no auth. Add/drop/start/sit
  execution has no first-party path.
- **Options:** (a) recommend-only, user executes in Sleeper; (b) unofficial
  authenticated write (account-security, ToS, reliability risk); (c) browser
  automation.
- **Working assumption:** (a) for V1. (b)/(c) are deferred, named, and must
  not be silently built around (i.e. no code should assume a write path).
- **Resolution path:** confirm (a) explicitly; record (b)/(c) as
  post-V1 with the risk noted. Agent tool contract should expose only
  read/recommend tools.

### Q-2. Meaning of "modularity / fully mutable business logic" — OPEN
- **Cause:** ambiguity between (i) swappable implementations behind a stable
  interface and (ii) end-user-tunable model parameters. (ii) conflicts with
  the no-fitted-parameters stance for expected_points.
- **Working reading:** (i). Concretely: engines are functions
  `DataFrame/dict in -> DataFrame/dict out` behind a named contract; a
  different implementation can be swapped in via config/registry with no
  caller changes. League *settings* are user-configurable (that is data, not
  model tuning).
- **Resolution path:** confirm the reading; write the interface contract
  (see STATUS.md sequence step 3) so "swappable" is testable, not a slogan.

### Q-3. Boom/bust distributional scope — OPEN
- **Cause:** vision asks for P(X+ points) and possibly P(X | boom), i.e. a
  distribution per player-week, not a class label. Much larger than the
  original binary/ternary classifier.
- **Sub-decisions:** (a) threshold definition (bounded multiplier of
  expected points; what floor/cap?); (b) distribution family: empirical
  residual quantiles by position x projection bucket (no fitted parameter,
  explainable) vs. parametric (e.g. gamma / zero-inflated) fit per position;
  (c) is P(X | boom) in V1 or deferred; (d) how a distribution is returned
  to the agent (see Q-6).
- **Resolution path:** a design spike using the *corrected* pipeline's
  residuals (after Q-5) before any more RB/WR/TE PBP feature work — the
  chosen approach determines whether `labels.py` / `model.py` exist at all.

## Found in the 2026-09-24 audit

### Q-4. Are the README/CLAUDE.md skew conclusions still valid? — PARTIALLY ANSWERED: no; docs rewritten 2026-09-27, k grid still not re-run
- **Update 2026-09-27:** README "Model Evaluation" now holds the final LOSO table (2019-2025,
  continuous windows). opponent_skew does not beat rolling-average-alone at any position
  (QB 7.923 vs 7.902 MAE, WR 4.706 vs 4.694, TE 4.342 vs 4.303). Still open: re-run the k grid
  (and consider dropping skew for QB/WR/TE) on the continuous-window pipeline; retune or
  redefine confidence tiers (calibration still non-monotonic every fold).
- **Update 2026-09-24:** B-1..B-4 fixed; test.py re-run at the existing k=16 (no retuning),
  2024/25, 12,153 scored player-weeks. Projection vs baseline MAE / R²: QB 8.102 vs 8.102 /
  0.177 vs 0.174; RB 4.455 vs 4.407 / 0.420 vs 0.422; WR 4.462 vs 4.440 / 0.336 vs 0.340;
  TE 4.186 vs 4.148 / 0.328 vs 0.329. Spearman: QB 0.437, RB 0.730, WR 0.644, TE 0.611.
  Conclusion: skew does not help any position on corrected data (QB was previously the
  one reliable win, now a tie); the calibration inversion persists (high 5.142 > medium
  4.710 > low 4.497). Still open: re-run the k grid search on the corrected pipeline, decide
  whether to retune confidence thresholds/definition, then rewrite the README table and CLAUDE.md "Grid search finding".
- **Cause:** the audit reproduced three pipeline defects (see STATUS.md,
  "Broken", B-1..B-3): duplicated skew keys fan out ~8% of 2024 prediction
  rows (19,357 predictions vs 17,896 stats rows); the current-season
  rolling average is null until `window` games so the documented early-season
  blend is effectively bypassed; skew's `.shift(1)` operates over player rows,
  not games. The k=16 grid search, "skew harmless for RB/WR/TE", and the
  calibration non-monotonicity finding were all produced on this pipeline.
- **Resolution path:** fix B-1..B-3, re-run `test.py` metrics, and either
  reaffirm or rewrite the README's Model Evaluation section and the CLAUDE.md
  "Grid search finding" paragraph. CLAUDE.md's "don't relitigate" note covers
  design decisions; this is a correctness re-baseline, not relitigating.

### Q-5. Inference-shaped pipeline — OPEN (explicitly NOT addressed by the 2026-09-27 RB merge)
- **Note 2026-09-27:** the RB stat vector and the position registry are a backtest-validated
  path only. Stat-vector models are fit offline per LOSO fold and passed into the engine;
  there is still no as-of feature builder for an upcoming week and no persisted production
  model. Continuous windows make the as-of computation simpler (last N games regardless of
  season), but none of it is built.
- **Cause:** every feature table is one row per *played* player-game
  (backtest-shaped). Projecting an upcoming week has no stats row to hang
  rolling features or opponent skew on. CLAUDE.md asserts train/serve parity
  for trailing volume; the code has no serve path.
- **Resolution path:** as-of feature functions
  `(season, week) -> features for every rostered player's upcoming game`,
  built from schedule + latest cumulative skew, sharing code with the
  backtest path.

### Q-6. ToolResult v2 for distributional outputs — OPEN
- **Fixed 2026-09-27 (explanation only):** `tools/expected_points.py` now builds its
  explanation from the row's `projection_method` (rolling_plus_skew vs stat_vector) and
  reports `projection_method` / `rolling_window` in metadata. The rest of Q-6 (typed value
  payload, per-tool confidence semantics) is still open.
- **Cause:** `ToolResult` = scalar `value` (Any) + a single 4-tier
  `confidence` + free-text `explanation` + untyped `metadata`. Confidence's
  meaning (games played) is expected-points-specific; boom/bust has a
  different notion (distribution sharpness, sample of residuals).
- **Resolution path:** decide whether `value` becomes a typed
  per-tool payload (dict, JSON-serializable, with a documented schema) and
  whether `confidence` moves inside it. Must keep the rule that the UI reads
  `.value` only.

### Q-7. Tool dispatch / dependency injection — OPEN
- **Cause:** `TOOL_REGISTRY["expected_points"]["func"]` takes
  `(args, projections_df)` but the registry entry carries no way to supply
  `projections_df`; a generic agent loop can't call it.
- **Resolution path:** resolve with a context object or closures bound at
  startup (ties into Q-2).

### Q-8. Scope of scoring-settings fidelity — OPEN
- **Cause:** any nonzero `ScoringSettings` field not in
  `SCORING_TO_STAT_COLUMN` (e.g. `rec_0_4`..`rec_30_39` bucketed reception
  points, `bonus_rush_rec_yd_100/200`, `bonus_fd_*`, `pass_int_td`, `fum`)
  contributes 0 silently. Documented only for long-play bonuses, not these.
  Vision says "highly customizable based on league settings".
- **Resolution path:** at minimum, surface unsupported-nonzero fields as a
  warning/known-limitation per league; decide which to actually support.
- **Extension 2026-09-26 (RB stat-vector projection, `projections/expected_points/stat_vector/rb.py`):**
  the stat vector predicts only carries, rushing yards/TDs, targets, receptions, receiving
  yards/TDs. **Not predicted, contribute 0** through the same mechanism (the column is absent,
  so `calculate_points_vectorized` skips it): 2-pt conversions (`rush_2pt`, `rec_2pt`), first
  downs (`rush_fd`, `rec_fd`), fumbles (`fum_lost`), and threshold bonuses
  (`bonus_rush_yd_100/200`, `bonus_rush_att_20`). The bonuses can't just be thresholded from
  the prediction: E[1{yards ≥ 100}] ≠ 1{E[yards] ≥ 100}. They would need a distribution
  (ties into Q-3). Measured cost on actual 2024/25 RB stats, tested league: mean(actual −
  recomposed) = −0.053 points/game, mean |gap| = 0.108, 94.9% of rows exact.

### Q-9. Trigger-based automation — OPEN (scope unclear)
- **Cause:** referenced in the audit request but absent from the pasted
  vision text (possibly in the truncated portion). Interacts with Q-1.
- **Resolution path:** confirm whether it's in V1 and what triggers (e.g.
  weekly refresh, lineup-lock reminders, waiver deadlines).

### Q-11. Promote the RB stat-vector projection? — RESOLVED 2026-09-27: promoted
- **Resolution:** RB is served by `stat_vector` through the position registry in
  `engine/expected_points.py`. (a) attempt-weighted rates are the default. (b) fitted
  coefficients were explicitly accepted for RB by the author, with conditions recorded in
  CLAUDE.md "Projection formula (per position)": a position switches only after LOSO
  validation including bias, and coefficients are fixed and reported per fold. Final RB
  LOSO: MAE 4.717 / R² 0.378 / Spearman 0.685 vs current formula 4.769 / 0.346 / 0.663 and
  rolling alone 4.733 / 0.349 / 0.668 (README). The metadata fan-out bug noted below is
  fixed (`load_player_metadata` de-duplicates + `assert_unique_key`).
- History:
- **Cause:** portable (league-agnostic) expected points via predicted raw stats +
  unmodified scoring. Held-out RB at n=8 roughly ties `rolling_avg_prior + opponent_skew`
  (MAE 4.430 vs 4.513 on 2024→2025; 4.444 vs 4.396 on 2025→2024; Spearman 0.743/0.721 vs
  0.732/0.727). See README "Model Evaluation" and `rb_stat_vector_eval.py` (since replaced by `stat_vector_eval.py RB`).
- **Sub-decisions:** (a) OLS vs attempt-weighted rate models: unweighted under-projects
  0.2-0.5 pts/game (per-game ypc 4.18 vs pooled 4.39; rush TD rate 0.027 vs 0.033), weighted
  is unbiased but MAE +0.055; (b) this adds fitted coefficients to expected points, which
  conflicts with CLAUDE.md's "no fitted parameter in the projection" stance. That stance
  needs an explicit revisit before promotion, not a silent change; (c) QB/WR/TE follow only
  if (a)/(b) are resolved.
- **Also found:** `epa_allowed_rush` on carries flips sign between fits (+15.1 p<0.001 vs
  −1.0 n.s.); the only matchup term stable in both directions is weighted ypc ×
  `epa_allowed_rush` (+3.7 / +5.7, p ≤ 0.03).
- **Update 2026-09-27 (iteration 2):** epa removed from volume models (no accuracy change);
  attempt-weighted rates made the default, answering (a). Evaluated leave-one-season-out over
  2019-2025. The stat vector has the highest R² in 6/7 folds and the best Spearman in 5/7
  vs both the current formula and rolling-only. MAE is worse than rolling-only in 5/7
  folds (it's mean-unbiased on right-skewed points). 2024 is the only season where it
  loses every metric. Constant rates lost in all 7 folds (league-average +0.032 MAE,
  season-to-date +0.09). On the partial 2026 fold (weeks 1-3, 172 rows) it loses clearly
  (MAE 4.765 vs 4.376): the early-season case needs a look before any promotion. (b) is
  still open.
- **Latent bug found:** `load_player_metadata` fans out rows for players listed twice in the
  ID crosswalk (Justin Hamilton, Corey Moore: defensive players, only when loading pre-2023
  seasons; 48 duplicate player-week keys over 2018-2026). No effect on QB/RB/WR/TE today;
  the eval driver asserts RB keys are unique. Loader not changed.

### Q-12. PBP count reconciliation — ACCEPTED (2026-09-26)
- Pass plays = `play_type == 'pass' & sack == 0` (nflfastR `pass_attempt` includes sacks,
  so it can't be used). REG 2024: 17,839 vs 17,811 official = +99 two-point tries − 71 spikes.
- Rush plays = `play_type == 'run'`: 14,317 vs 14,687 official = −405 kneels
  (`play_type == 'qb_kneel'`) + 36 two-point runs (off by 1). Accepted as-is; excluding
  2-pt tries from both would be a further filter change, not done.
- (Superseded 2026-09-27) The effective-n proposal for epa shrinkage is retired. Skew and
  epa_allowed now use continuous 17-game windows, so week 1 already has prior-season games
  in the window and n = games actually in the window.

### Q-13. Continuous windows cost the incumbent formula accuracy — PARTIALLY RESOLVED (2026-09-27)
- **Update:** per-position window grid (`rolling_window_eval.py`, {8,10,12,14,16,20}, LOSO
  mean MAE) set QB 12, WR 10, TE 12 in `POSITION_ROLLING_WINDOWS`. QB now ties the old blend
  (MAE 7.894 vs 7.893) with better R² (0.216 vs 0.202) and Spearman (0.483 vs 0.473). WR
  (4.700 vs 4.661) and TE (4.321 vs 4.282) remain ~0.04 MAE behind. WR's curve is flat
  (10/12/14 within 0.007). WR then moved to the stat vector (MAE 4.689 / R² 0.344 /
  Spearman 0.651 vs tuned current formula 4.697 / 0.321 / 0.635), so only TE is still
  affected. Remaining options for TE: accept, or validate a TE stat vector.
- **Cause:** replacing the season-partitioned window + prior-season blend with one continuous
  trailing window (per the 2026-09-27 instruction) made `rolling_avg_prior + opponent_skew`
  worse on LOSO 2019-2025: MAE +0.03 QB, +0.06 RB, +0.05 WR, +0.06 TE; weeks 1-3 +0.08 to
  +0.16; R² −0.017 to −0.018 for RB/WR/TE (QB R² +0.006). Measured on identical rows.
- **Why (likely):** for a returning player the old blend leaned on last season's full-season
  average (~17 games), a bigger sample than the last 8 games.
- **Checked:** a 12-game continuous player window recovers QB (7.895 MAE vs 7.893 blend, better
  R²/Spearman) but WR/TE stay ~+0.04 MAE worse. Window left at 8 (not tuned).
- **Resolution path:** accept (one rule everywhere), tune the player window per position, or
  move WR/TE to the stat vector (Part 2 of the same instruction starts WR). RB is unaffected
  in production because it no longer uses this formula.

### Q-10. Long-lived carry-overs (from CLAUDE.md / README) — DEFERRED
- Confidence-tier thresholds are provisional; calibration was non-monotonic
  at every k tested (partly explained by B-2, see STATUS.md).
- Skew for RB/WR/TE: revisit with finer-grained matchup features (pending Q-4).
- Rookies resolve to `None` projection by design; the UI/agent must handle it.
- Flat-JSON storage per user: not implemented.
- Kicker/DEF/IDP and long-play bonuses: out of scope by design.
