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

### Q-4. Are the README/CLAUDE.md skew conclusions still valid? — PARTIALLY ANSWERED (2026-09-24): no; docs not yet rewritten
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

### Q-5. Inference-shaped pipeline — OPEN
- **Cause:** every feature table is one row per *played* player-game
  (backtest-shaped). Projecting an upcoming week has no stats row to hang
  rolling features or opponent skew on. CLAUDE.md asserts train/serve parity
  for trailing volume; the code has no serve path.
- **Resolution path:** as-of feature functions
  `(season, week) -> features for every rostered player's upcoming game`,
  built from schedule + latest cumulative skew, sharing code with the
  backtest path.

### Q-6. ToolResult v2 for distributional outputs — OPEN
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

### Q-9. Trigger-based automation — OPEN (scope unclear)
- **Cause:** referenced in the audit request but absent from the pasted
  vision text (possibly in the truncated portion). Interacts with Q-1.
- **Resolution path:** confirm whether it's in V1 and what triggers (e.g.
  weekly refresh, lineup-lock reminders, waiver deadlines).

### Q-10. Long-lived carry-overs (from CLAUDE.md / README) — DEFERRED
- Confidence-tier thresholds are provisional; calibration was non-monotonic
  at every k tested (partly explained by B-2, see STATUS.md).
- Skew for RB/WR/TE: revisit with finer-grained matchup features (pending Q-4).
- Rookies resolve to `None` projection by design; the UI/agent must handle it.
- Flat-JSON storage per user: not implemented.
- Kicker/DEF/IDP and long-play bonuses: out of scope by design.
