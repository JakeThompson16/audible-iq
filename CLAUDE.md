# Audible IQ — Project Context

Fantasy football decision-support tool. Pulls live league data from Sleeper,
computes league-accurate fantasy scoring, and builds matchup-adjusted
projections and boom/bust probabilities to power start/sit recommendations.

Solo project, ~2.5 week build window, portfolio/learning-focused following a
summer internship building agentic AI systems at Loomis Sayles. Prioritizes
explainable, deliberately-scoped engineering over marginal accuracy gains.

## Architecture

Layered to isolate external API/data-source dependencies from core logic:

- `domain/` — core objects: `Player`, `Team`, `League`, `ScoringSettings`.
  Platform-agnostic dataclasses. Never reference Sleeper/nflreadpy shapes
  directly. Identity-based `__eq__`/`__hash__` (e.g. Player compares on
  sleeper_id + gsis_id, not full state).
- `clients/` — raw API/data-source wrappers.
  - `clients/sleeper_client.py` — raw Sleeper API calls (leagues, rosters,
    users, scoring settings). No business logic.
  - `clients/nflreadpy/` — player metadata (`player_data.py`) and
    team/schedule data (`team_data.py`) via nflreadpy/nflverse. The
    dynastyprocess crosswalk URL patch lives in `player_data.py`.
- `common/frames.py` — `assert_unique_key(df, keys, name)` (use on the right
  side of every one-to-one join) and `assert_no_fanout(before, after, name)`.
  Shared; don't write local duplicate checks.
- `adapters/` (Sleeper) — maps raw external JSON into domain schema
  (e.g. `ScoringSettings.from_dict()`).
- `engine/` — pure computation across domain objects/dataframes.
  - `engine/scoring.py` — `calculate_points_vectorized()`: dot product of
    stats and league ScoringSettings via `SCORING_TO_STAT_COLUMN` registry
    (maps ScoringSettings field names -> stats_df column names, since they
    don't align 1:1).
- `projections/rolling_window.py` — the ONE continuous trailing-window
  utility (`trailing_mean`, `trailing_sum`, `trailing_count`) used by every
  rolling feature. See "Continuous rolling windows" below.
- `projections/expected_points/features/` — feature engineering for the
  projection model.
  - `player_rolling.py` — trailing averages per player over a per-position
    window (`POSITION_ROLLING_WINDOWS`), spanning seasons:
    `rolling_avg_prior` (fantasy points), `n_games_in_window`, `rolling_window`,
    `trailing_opportunities_avg` (carries + targets combined — see below),
    `trailing_targets_avg`, `trailing_attempts_avg`.
  - `opponent_skew.py` — Adjusted Points Allowed (APA): opponent-adjusted
    matchup residual per defense/position over the defense's trailing
    `SKEW_WINDOW` (17) games.
  - `epa_allowed.py` — `epa_allowed_pass` / `epa_allowed_rush` per
    (defteam, season, week): opponent EPA/play allowed minus that week's
    league EPA/play, trailing 17-game window like opponent skew, shrunk toward
    0 by n / (n + k) with n = games in window. Pass =
    `play_type == 'pass' & sack == 0`, run = `play_type == 'run'` (shared with
    `aggregate_pbp.py`).
  - `stat_rolling.py` — per-stat trailing features for the stat vector:
    `roll_<volume>`, `roll_<rate>` (ratio of rolling sums), `delta_<volume>`
    (3-game minus 8-game mean), `stat_games_in_window`.
- `projections/expected_points/stat_vector/` — portable expected points:
  predict the raw stat line, score it with the unmodified
  `calculate_points_vectorized`. Never predicts fantasy points directly (that
  locks the model to one league's scoring).
  - `core.py` — shared machinery: `StatVectorSpec` (per-position model
    spec), OLS/WLS fitting, `fit_stat_vector()` (offline fit ->
    `StatVectorModel`), `StatVectorModel.predict_points()`.
  - `rb.py`, `wr.py`, `te.py`, `qb.py` — the four specs, all in production
    (QB by owner override, see "Projection formula"). `specs.py` —
    `STAT_VECTOR_SPECS`.
    Each spec carries its own feature `window` and a `derivations` chain
    (`rush_receive_derivations(...)` for RB/WR/TE, `PASS_RUSH_DERIVATIONS`
    for QB) that turns predicted volumes x rates into the stat line. A spec's
    `candidate_terms` are evaluated-but-off features, re-testable with
    `stat_vector_eval.py <POS> --candidates`.
- `engine/expected_points.py` — `calculate_expected_points(stats_df,
  context, position_implementations=None)`. Joins opponent skew for every
  row (week/season/opponent_team/position), then dispatches each position to
  an implementation via `POSITION_IMPLEMENTATIONS` (see "Swappable
  implementations" below): all four positions -> `stat_vector`.
  `rolling_plus_skew` stays registered as the alternative implementation and
  evaluation baseline. Adds `projection`, `projection_method`, and a
  `confidence` tier per row. `ExpectedPointsContext` carries skew_df,
  epa_df, scoring, fitted stat-vector models (fitting happens offline), and
  `rolling_windows` (position -> window, default `POSITION_ROLLING_WINDOWS`);
  the engine rejects stats whose `rolling_window` doesn't match.
- `evaluation/backtest.py` — offline harness: `load_inputs()`,
  `loso_folds()` (leave-one-season-out over 2019-2025 + partial 2026),
  `predict_fold()` (fit stat-vector models on train seasons, run the engine).
  Drivers: `test.py` (all positions, production registry),
  `stat_vector_eval.py <POS>` (new model vs current formula vs rolling alone),
  `rolling_window_eval.py` (per-position window grid for rolling_plus_skew).
- `engine/metrics.py` — `evaluate_projections(predictions_df, actuals_df)`:
  offline evaluation only (not used at inference time). MAE/RMSE/mean
  error/R², reported for the full projection AND for the
  `rolling_avg_prior`-only baseline side by side (does opponent_skew earn
  its complexity?), plus Spearman rank correlation per
  (season, week, position) — weighted most heavily, since start/sit is a
  ranking problem, not a point-estimate problem. Everything segmented by
  position (QB/RB/WR/TE); an aggregate number across positions is close to
  meaningless given the scale differences. MAPE deliberately omitted —
  undefined/unstable near zero-point fantasy outcomes. Includes a
  calibration check (MAE bucketed by confidence tier) — see below.
- `tools/` — agent-facing tool contract.
  - `domain/tool_result.py` — `ToolResult` (value, confidence, explanation,
    metadata). `confidence` and `explanation` are agent-facing reasoning
    aids only — never leak them into a user-facing projection display; a UI
    adapter should read only `.value`.
  - `tools/expected_points.py` — `ExpectedPointsArgs` (gsis_id, week, season)
    + `get_expected_points()`, a thin lookup wrapper around a precomputed
    `calculate_expected_points()` output, returning `ToolResult`. The
    explanation is chosen by the row's `projection_method`
    (`_METHOD_EXPLANATIONS`, same keys as engine `IMPLEMENTATIONS`); add an
    entry when adding an implementation.
  - `tools/registry.py` — `TOOL_REGISTRY` dict, tool name -> {args_model,
    func, description}.
- `pipeline/` — production packaging (see "Production pipeline" below).
  - `context.py` — `FeatureContext`, loaded once per process (`get_context()`,
    `refresh_context()`): stats, weekly EPA residuals, team schedule with
    completion flags, weekly rosters, latest teams, `data_through`.
  - `train.py` — `train_all()` + `python -m pipeline.train retrain`.
  - `artifacts.py` — JSON artifact format, spec-compatibility check,
    validation, safe overwrite.
  - `predict.py` — `predict_player_stats()` / `predict_many()`.
  - `reference_ranges.py` / `reference_ranges.json` — 2019-2025 LOSO
    coefficient ranges used by retrain validation.
- `search/` — no UI code; used by the UI, and later the agent and the Sleeper roster flow.
  - `player_search.py` — `build_player_index()` (cached; `refresh_player_index()`),
    `PlayerIndex.suggest(query, size=8)` / `.get(gsis_id)`. fast-autocomplete
    over current-season rostered QB/RB/WR/TE, keyed by gsis_id.
  - `projection_service.py` — `project_player(gsis_id, season=None, week=None)`:
    everything a projection card needs, as a thin layer over
    `predict_player_stats` (see "Projection service" below).
- `webapp/main.py` — NiceGUI page (presentation only): `python webapp/main.py`
  -> http://localhost:8090 (`--port N` to change; 8080 is often taken).
- `requirements.txt` — pinned dependencies.
- `artifacts/` — `stat_vector/<POS>.json` production models (committed),
  `MODEL_METRICS.md` (regenerated on every retrain). `*.json.prev` is the
  local previous generation (git-ignored).
- `tests/` — pytest (`python -m pytest`): parity with the backtest harness,
  round-trip scoring, ID resolution, status paths, artifact safety,
  determinism. Downloads real data; takes several minutes.
- `config.py` — `PLAYER_METADATA` is the single canonical list of player
  identity/metadata fields; all player construction depends on it.

## Key design decisions (with rationale — don't relitigate without reason)

**Projection formula (per position)**: all four positions use the stat vector
(fitted OLS volume + attempt-weighted rate models, recomposed through league
scoring). `rolling_avg_prior + opponent_skew` (simple, unfitted) remains
registered as the alternative implementation and the evaluation baseline.

**QB is on the stat vector by explicit OWNER OVERRIDE (2026-09-27), not
because it passed the selection policy.** The QB stat vector missed the
Spearman-first bar: LOSO 2019-2025 mean Spearman 0.483 vs 0.492 for
rolling+skew at window 14 (MAE 7.911 vs 7.915, R² 0.238 vs 0.217; partial 2026
weeks 1-3 worse: 8.559 / 0.154 / 0.323 vs 8.406 / 0.160 / 0.364). Reasons for the
override: one architecture and one output (a predicted stat line) for all
four positions, portability across scoring settings, and the stat-prediction
pipeline needs it. Q-14 (the QB rush component) is the next QB priority.

Adopting fitted coefficients for
RB was an explicit decision on 2026-09-27 (Q-11b), made because the stat vector
beat the incumbent on held-out seasons AND is league-portable. The original
reason for staying unfitted still applies: the projection is boom/bust's
reference point, so bias in it propagates into what "boom"/"bust" mean. So a
position switches only after a LOSO validation that includes bias, and the
fitted coefficients are fixed, inspectable, and reported per fold, not tuned
per user.

**Leakage prevention (critical, applies everywhere)**: all rolling
features use `.shift(1)` before the window (built into
`projections/rolling_window.py`), so week W's value only ever reflects games
strictly before W. This applies to player rolling averages, opponent skew,
epa_allowed, and every stat-vector feature. Never remove a `.shift(1)`
without understanding this.

**Opponent skew (APA)**: `residual = fantasy_points - rolling_avg_prior`,
aggregated to one value per (defense, position, week), then averaged over the
defense's trailing 17 games (`.shift(1)` first, same leakage rule). Additive, not multiplicative — ratios
blow up near small denominators and would overweight noise from low-baseline
players. `min_games` floor filters out low-sample rows; QB may need a higher
floor than RB/WR/TE since QB fantasy points are structurally higher-variance
per game (confirmed by inspecting which position dominated the extreme
tails).

Originally capped at a hard ±8 after inspecting real output showed extreme
values (-16 to +20) universally clustered at low `n_games` (3-5). Replaced
with sample-size-weighted shrinkage: `skew_shrunk = skew_raw * (n / (n +
k))`, `n` = `n_games` backing that row's estimate — low-n rows get pulled
hard toward 0, well-supported rows are trusted close to their raw value. A
hard clip either lets noisy low-n rows through unshrunk (below the
threshold) or flattens well-supported high-magnitude rows to the same cap
as a barely-qualified one (above it); shrinkage scales continuously with
the actual evidence behind each row instead. `k=16` was chosen by grid
search (k in {2, 4, 8, 16}) against `engine/metrics.py`'s
`evaluate_projections()` output on 2024/25 held-out data — not fit via
optimization, deliberately, to keep expected_points free of any fitted
parameter (the projection is boom/bust's reference point; a fitted
shrinkage constant would blur that same explainability boundary the
additive-not-multiplicative and no-fitted-model decisions above already
protect). A wide `sanity_clip` (±12, `calculate_all_position_skews`)
remains afterward as a defense-in-depth guardrail only (e.g. n=0 edge
cases) — it is NOT the flattening mechanism; shrinkage is.

> **WARNING (2026-09-24): the grid-search finding below, the k=16 choice, and the README
> "Model Evaluation" numbers were produced on a pipeline with three defects, now fixed as of
> 2026-09-24: duplicated skew keys fanning out predictions (B-1), a null-until-window rolling
> average that bypassed the early-season blend (B-2), and a row-level (not game-level)
> skew shift that leaked same-week data and miscounted `n_games` (B-3). Re-run at the
> existing k=16 shows skew no longer helps even QB (see STATUS.md / OPEN_QUESTIONS.md Q-4).
> Treat the paragraph below as historical until the grid search is re-run.**

Grid search finding: shrinkage brought RB/WR/TE to roughly break-even with
the `rolling_avg_prior`-only baseline (previously net-negative at every
tested k below 16), but an n-segmented breakdown at k=16 showed the R²
delta (projection vs. baseline) hovering near zero uniformly across every
`n_games` bucket (3-5, 6-8, 9-11, 12+), not concentrated at low n. That
means the original "skew hurts" effect wasn't primarily low-n noise that
shrinkage could clean up — opponent matchup signal for RB/WR/TE appears
genuinely weak at the current feature granularity, at any sample size.
Shrinkage mostly neutralized the harm rather than unlocking real signal.
QB is the exception: skew improves MAE/R² over baseline consistently
across all tested k. Treat opponent_skew as reliably additive for QB;
for RB/WR/TE it is now harmless rather than helpful, worth revisiting
(e.g. finer-grained matchup features) rather than trusting as-is.

**Volume inclusion filter for skew calc**: uses *trailing* rolling volume
(not same-week volume) as the inclusion threshold, specifically because it's
computable identically at training time and inference time — same-week
volume would cause train/serve skew since you don't know this week's
targets/carries when projecting. RB/WR use combined `opportunities`
(carries + targets) rather than position-typical stat alone, to correctly
capture dual-usage players (e.g. Deebo Samuel, Curtis Samuel) who'd be
undercounted by targets-only or carries-only filtering.

**RB stat-vector matchup inputs** (`stat_vector/rb.py`): matchup terms live
in the rate models only. Rush-side rates (ypc, rush_td_rate) use
`epa_allowed_rush`; receiving-side rates (catch_rate, ypr, rec_td_rate) use
`epa_allowed_pass`. This resolves the "should RB also get epa_allowed_pass?"
question: yes, but only on the receiving-work models, since each model's
matchup term should describe the play type that produces that stat. Volume
models (carries, targets) have no matchup term. It flipped sign between
seasons (+15.1 vs −1.0 on carries), and removing it changed LOSO mean MAE by
0.001. With six training seasons, all five rate-model epa coefficients are
positive in every fold (ypc p < 0.001).

**WR stat vector** (`stat_vector/wr.py`, switched on 2026-09-27): same
structure as RB (targets + carries volume, five rates). Validated finding:
weaker pass defenses raise WR catch rate (`epa_allowed_pass` on catch_rate
+0.16 to +0.24, significant in all 7 LOSO folds). Unlike RB, WR receiving
efficiency persists (trailing catch rate / ypr / rec-TD-rate coefficients
0.13-0.29, all p < 1e-8). Off by default and NOT validated:
`epa_allowed_pass` in the targets volume model (re-tested for WR, sign flips,
p >= 0.48) and in ypr / rec_td_rate (significant in <= 1 of 7 folds); they are
`candidate_terms`. Rushing-side WR rates keep `epa_allowed_rush` as
specified but carry no signal (WR carries are tiny).

**TE stat vector** (`stat_vector/te.py`, switched on 2026-09-27): WR's
structure refit on TE data, window 10. Validated matchup findings on TE:
`epa_allowed_pass` on ypr (+3.7 to +5.7) and rec_td_rate (+0.10 to +0.13)
positive and significant in 7/7 folds, which is unlike WR, where neither
cleared. On catch_rate it's positive 7/7 and significant 5/7. Off (candidate):
targets volume EPA, positive 7/7 but significant only 3/7.

**QB stat vector** (`stat_vector/qb.py`, in production by owner override): volume
= pass attempts + rush attempts (scrambles included via play_type == 'run');
rates completion_rate / int_rate per attempt, yards_per_completion /
pass_td_rate per completion, ypc / rush_td_rate per rush attempt; completions,
incompletions, yards, TDs and INTs derived by multiplication. **int_rate's
epa_allowed_pass coefficient is NEGATIVE by design** (a softer pass defense
forces fewer interceptions): -0.027 to -0.034, significant in 7/7 folds. That
is the expected, correct sign; don't "fix" it. Completion rate (+, 7/7),
yards/completion and pass TD rate (+, 6/7) are the other validated passing
terms; rushing EPA terms and volume-model EPA were not significant.

**Stat-vector rate models are attempt-weighted by default** (`fit_models(weight_rates=True)`).
Unweighted fits under-project RB by +0.31 pts/game on average (LOSO bias);
weighted bias is −0.03, at a cost of +0.046 MAE. Replacing fitted rates with
constants was tested for RB and lost in every fold (league-average rates +0.032
MAE, season-to-date rates +0.09 MAE); both variants were removed from the code
(results in README / git history). Rate models fit only on rows where the rate
is defined (denominator > 0). A player with history but no attempts of a kind in
the window gets the pooled training rate for that `roll_<rate>`. Rookies with no
history stay None.

**Selection policy (stated rule, 2026-09-27)**: start/sit is a pairwise
ranking decision, so model and parameter choices prioritize mean LOSO
**Spearman, then R²**, with MAE as a guardrail rather than the target.
- Switching a position's implementation: adopt the candidate if Spearman and
  R² are comparable-or-better AND MAE is comparable-or-better. A strict MAE
  win is not required (WR's switch was the first case: best R²/Spearman,
  MAE won 3/7 folds but lowest mean).
- Parameter grids (e.g. a stat-vector window): `evaluation.backtest.select_by_policy`
  ranks by mean Spearman, then mean R² (ties within 0.001), then lower MAE.
- Report MAE and bias every time; a candidate that ranks better but is
  materially biased does not pass (boom/bust measures deviation from it).
- The `POSITION_ROLLING_WINDOWS` grid predates this rule and was chosen on
  MAE (QB 12, WR 10, TE 12). QB was reselected to 14 under this rule. WR/TE
  stay 10/12: they're on the stat vector, so their rolling average only feeds
  skew and baselines.

**Swappable implementations** (`engine/expected_points.py`, VISION.md / Q-2):
`IMPLEMENTATIONS` maps a name to a function `(rows, ExpectedPointsContext) ->
KEY + projection`, and `POSITION_IMPLEMENTATIONS` maps position -> name. Never
add an inline `if position == ...` branch to pick a method. To move a position
to the stat vector: add its `StatVectorSpec` to `STAT_VECTOR_SPECS`, validate
with `stat_vector_eval.py <POS>` (LOSO, report bias and coefficient stability),
then flip its entry in `POSITION_IMPLEMENTATIONS`. Per-position parameters
of an implementation (e.g. `POSITION_ROLLING_WINDOWS` for rolling_plus_skew)
are config next to the registry and carried in the context, never branches.
Per-call overrides
(`position_implementations={"RB": "rolling_plus_skew"}`) are how comparisons
score the old formula. The engine never fits; fitted models arrive in the
context. This is a backtest-validated path only: there is no inference
(upcoming-week) path yet (Q-5).

**Confidence tiers** (`engine/expected_points.py`): derived *solely* from
player-side `games_this_season` — not opponent_skew's `n_games`, not outcome
volatility (volatility is boom/bust's job, not expected points'). Tiers:
`insufficient_data` (0 games), `low` (<4), `medium` (<8), `high` (8+).
Thresholds are provisional. Calibration is judged on NORMALIZED MAE (MAE /
mean actual points per tier), which is monotonically decreasing for every
position (verified 2026-09-27, Q-15). Raw MAE is inverted only because
higher tiers hold higher-scoring players; don't read that as miscalibration.
`engine/metrics.py`'s calibration check reports both;
retune the thresholds if it doesn't.

Two known implications of this design, both deliberate — don't special-case
either:
- **Mid-season injury returns show conservative confidence.** A player
  returning from injury in, say, week 10 will show `low`/`medium`
  confidence even though the opponent's skew data is fully mature by then.
  This is because confidence is player-side-only by design — defensive
  scheme/personnel are more stable year-over-year than any single player's
  availability, so the two shouldn't be conflated into one tier. Net effect:
  confidence is conservative in this case, never overconfident, which is the
  acceptable failure direction.
- **Skew confidence assumes defensive scheme continuity year-over-year.**
  opponent_skew and epa_allowed windows span the season boundary (see
  below) without adjustment for defensive coordinator or personnel turnover.
  A defense that overhauled its scheme in the offseason keeps last year's
  games in its 17-game window until current-season games displace them.

**Continuous rolling windows (early-season handling)**: every rolling
feature is a trailing window over the entity's last N games, partitioned by
entity only, so it spans the season boundary freely. Week 1 uses the tail of
last season. There is no blend-weight formula; the evidence behind a value is
the actual count of games in the window (`n_games_in_window`, skew/epa
`n_games`), which also drives skew/epa shrinkage n / (n + k) and the
`min_games` floor. Windows: player rolling features per position
(`POSITION_ROLLING_WINDOWS`: QB 14, WR 10, TE 12, RB 8; LOSO grid over
{8, 10, 12, 14, 16, 20}, lowest mean MAE), stat-vector features 8, opponent
skew and epa_allowed 17 (one season of defense games, not tuned).
- History (retired 2026-09-27): season-partitioned windows blended with last
  season's full-season average at weight `min(games / 9, 0.9)`, and a
  proposed effective-n = n_games + (1 − w) · prior_games for epa shrinkage.
  Both were replaced by the continuous window for one consistent rule.
- Measured cost for the incumbent `rolling_avg_prior + opponent_skew`
  formula (LOSO means 2019-2025) at a shared 8-game window: MAE worse by +0.03
  (QB), +0.06 (RB), +0.05 (WR), +0.06 (TE) vs the retired blend. After the
  per-position window tuning: QB ties the blend (7.894 vs 7.893 MAE, better R²
  and Spearman); WR and TE then moved to the stat vector, which beats the old
  blend's numbers on R²/Spearman. Q-13 resolved.
- Confidence tiers are unaffected (verified identical on all 141,434 rows).
- True rookies with no prior games: `rolling_avg_prior`
  resolves to `None`, NOT a fabricated positional-average fallback. This is
  deliberate — a fake baseline would look like real data but isn't
  well-grounded. Downstream (agent/UI) must handle `None` as an explicit
  "insufficient data to project" state, not silently produce a bad number.

**Scoring settings scope**: `ScoringSettings` covers offensive skill
positions (QB/RB/WR/TE) only. Kicker, team defense/IDP, and return/special-
teams categories are intentionally excluded (out of product scope). Long-play
bonus categories (40+/50+ yard TD bonuses, etc.) are represented in the
dataclass for completeness but always evaluate to 0 contribution — computing
them would require play-by-play level aggregation (`load_pbp()`), not
available in the weekly stats pipeline, and no tested league actually uses
them.

**ID crosswalk**: `nflreadpy.load_ff_playerids()` is the authoritative
source for cross-platform IDs (sleeper_id, gsis_id, etc.) — solved the
Sleeper<->nflreadpy identity problem for free, no fuzzy name-matching needed.

**Storage**: flat JSON files for user data (just a Sleeper username per
user), not a database — deliberately minimal for this scope. Storage is
kept behind a thin interface so it could be swapped for SQLite later without
touching the rest of the app.

**Scoring validated**: `calculate_points_vectorized()` cross-checked against
real Sleeper league data (Trey McBride, full season, two different league
scoring configs including a TE-premium league) — exact matches.

## Production pipeline (training, artifacts, prediction)

**Retraining**: `python -m pipeline.train retrain [--start 2022] [--end YYYY]
[--dry-run]` or `train_all(start_season=2022, end_season=None, save=True)`.
Fits every spec in `STAT_VECTOR_SPECS` with `fit_stat_vector` unchanged, on
target rows from `start_season` through `data_through` = the last week whose
scheduled games are ALL final (an in-progress week is excluded). Two seasons
before `start_season` are loaded as history only (trailing windows, epa).
Idempotent: same data -> same artifacts except `fit_timestamp`. Safe to run
weekly; schedule it Tuesday morning (after Monday Night Football is final),
e.g. Windows Task Scheduler / cron running the CLI from the repo root. It
exits non-zero on a validation error and leaves the old artifacts in place.
Scheduling itself is not set up.

**Artifact format** (`artifacts/stat_vector/<POS>.json`, JSON, never pickle):
`schema_version`, `position`, `spec_name`, `spec_hash` (sha256 of the spec
config: sub-models with ordered features and denominators, derivations,
window, weighting), `spec_config`, `window`, `weight_rates`, `rate_priors`,
`sub_models` (per target: kind, denominator, weighted, ordered `features`,
`intercept`, `coefficients`, `std_errors`, `p_values`, `n`, `r2_in_sample`),
`trained_on` (seasons, weeks), `data_through`, `row_counts`, `fit_timestamp`.
Floats are rounded to 12 significant digits so retrains are byte-identical
(multi-threaded aggregation can flip the last bit). The loader
(`artifacts.check_compatible`) compares the artifact to the in-code spec and
raises `ArtifactSpecMismatch` on any difference: never apply mismatched
coefficients; retrain instead.

**Safe overwrite**: artifacts and `MODEL_METRICS.md` are written to a temp
dir, validated, then swapped in, keeping `<POS>.json.prev`. Validation
errors (nothing written): NaN/inf, a sub-model below its row floor (volume
1000, rate 50), a coefficient GROSSLY outside its 2019-2025 LOSO range
(beyond it by more than the range's width), or a coefficient that was
p < 0.01 last time moving > 5 SE and > 50%. Warnings only: outside the LOSO
range but not gross (common: those fold ranges are narrow because folds share
5/6 of their data, and production trains on a different window), > 25% move
vs the previous artifact.

**Prediction API** (`pipeline/predict.py`):
`predict_player_stats(player_id, gsis_id=False, season=None, week=None)` and
`predict_many(ids, ...)` (the single call is a thin wrapper). Returns a
JSON-serializable dict: `player` {gsis_id, sleeper_id, name, position, team,
team_source}, `target` {season, week, opponent}, `status`, `reason`, `stats`
(exact `calculate_points_vectorized` column names, only predicted categories),
`volume_rate_detail`, `games_this_season`, `confidence` (engine
`confidence_tier`, unchanged), `projection_method`, `model_version`
{fit_timestamp, data_through}, `unpredicted`. Status: ok | bye |
no_game_scheduled | no_history | unknown_player | unsupported_position; never
zeros for missing data. Pass `models=` to use in-memory models instead of the
production artifacts (the parity tests do this).

**ID conventions**: all ids are compared as `str`; `predict.normalize_id`
turns 4984 / 4984.0 / "4984.0" into "4984" at the boundary. Sleeper -> GSIS
and GSIS -> info maps come from `load_player_metadata` (its dedupe +
`assert_unique_key`), cached with `lru_cache`; `refresh_player_maps()` clears
them. Position comes from the player's stats rows (what the backtest uses),
falling back to the crosswalk. QB/RB/WR/TE only.

**Current team**: nflverse weekly rosters at the latest week <= the target
(pre-game information), else nflverse `latest_team` (current season), else
the player's last stats row. Checked 2020-2026: roster team equals the
stats-row team in every played QB/RB/WR/TE week (0 mismatches). For players
who haven't played yet this season, the last stats row is often stale
(85 of 174 in 2026 were on a new roster team). Never use the ff_playerids
`team` field: different abbreviations (KCC, LAR, JAC) and stale.

**As-of semantics**: only data strictly before the target game. The target
defaults to the team's next unplayed game (schedule `completed` flag); an
explicit (season, week) works for past weeks too. A placeholder row for the
target game is appended to the player's history and run through the SAME
`build_features` the backtest uses (`.shift(1)` ignores the placeholder's own
stats). `epa_allowed` for a defense-week not yet played is the defense's value
after its last game (`epa_allowed_from_weekly(..., as_of_keys=...)`, identical
to what a played week gets). The context is loaded once per process;
per-player calls don't touch nflreadpy (`refresh_context()` after new data).

**Prediction sanity bounds** (`stat_vector/core.py`, shared by backtest,
engine and prediction): before derivation, volumes are floored at 0,
per-attempt probabilities (catch, completion, all TD and INT rates) are
clipped to [0, 1], and yardage rates (ypc, ypr, yards/completion) are floored
at 0. `check_stat_line()` then asserts no negative stat, receptions <=
targets, completions <= attempts (raises `StatVectorBoundsError`).
Audit 2026-09-27 (raw predictions before bounds, 2019-2025 LOSO, all
positions): the only violations were TE rush_td_rate < 0 in 22 of 8,548 rows
and TE ypc < 0 in 2 of 8,548 (the latter gave negative rushing yards before the
yardage floor existed). Live (all 783 indexed players): the bounds changed 2 TE
rush_td_rate predictions; 0 derived violations. test.py and the parity tests
were unchanged. Predictions report `bounds_applied` (sub-models a bound changed).

**Player search** (`search/player_search.py`): membership = on an nflverse
weekly roster for the current season (same source the pipeline uses for
current team), at a position in `POSITION_IMPLEMENTATIONS`, and present in the
`load_player_metadata` crosswalk (so every suggestion resolves in
`predict_player_stats`; ~140 rostered players are excluded for not being in
the crosswalk). Names are normalized (accents folded, apostrophes and periods
removed, hyphens to spaces, Jr./Sr./II/III/IV/V dropped) and indexed as the full
name plus every trailing part (last names, incl. "st brown"). fast-autocomplete
0.9.0 gives prefix matches (sorted by `count` = games played over the last two
seasons) and typo matches (edit distance < 3 once 3+ characters are
unmatched); a fallback handles "first-prefix last-prefix" ("pat mah"). Names
collide, so keys map to sets of gsis_ids and a selection is always the
gsis_id, never re-resolved from the display string ("Name · POS · TEAM").

**Search API (gsis-keyed)**: `build_player_index(season=None)` -> cached
`PlayerIndex`; `refresh_player_index(reload_data=True)` rebuilds (and reloads
rosters + id maps). `PlayerIndex.suggest(query, size=8)` -> list of
`{gsis_id, sleeper_id, name, display, position, team}`; `.get(gsis_id)` -> that
dict or None. Everything downstream (UI selection, agent, Sleeper roster flow)
passes the gsis_id; never look a player up again by display string or name.

**Projection service** (`search/projection_service.py`): `project_player` returns
`predict_player_stats`' dict (player, target, status, reason, stats,
volume_rate_detail, bounds_applied, games_this_season, confidence,
projection_method, model_version, unpredicted, in_training_window) plus
`target.home_away` / `gameday` / `game_final` and, for a final game,
`actual` (the player's real stats, same keys as `stats`; None with
`actual_note` if he has no stats row). Default target = the team's next
unplayed game; a bye week has no schedule row, so the following week is
returned, always with its week number. `in_training_window` (added to the
prediction output) is True when the target season is a trained season and the
week is <= the artifact's data_through: a comparison with actuals there is
in-sample. No per-call nflreadpy loads (process-wide caches).

**Web UI** (`webapp/main.py`, NiceGUI 3.17): no business logic. Startup loads
the context + index in a worker thread (`run.io_bound`) behind a loading
state; the search box calls `PlayerIndex.suggest` (debounced 0.2 s) and a
selection calls `project_player` with the gsis_id. Non-ok statuses render as
labeled messages; a static note says injuries/inactives aren't modeled. Dev
mode (off by default) shows confidence, projection_method,
in_training_window, model version, volume/rate detail, a week override, and
for a final game the actuals next to the projection with an "in-sample"
badge when in_training_window. The footer shows data_through and fit time.
Tested with NiceGUI's `User` fixture (`tests/test_app.py`, plugin enabled in
pytest.ini).

## Known bugs already hit once — don't reintroduce

- Python string literals with a missing comma between them silently
  concatenate (caused a real bug in `RAW_STAT_COLUMNS`). Watch for this in
  any multi-line list of string constants.
- `pl.Expr.clip(lower, upper)` — argument order is (min, max). Reversing it
  silently clamps nearly everything to one boundary value rather than
  erroring. If a distribution looks suspiciously pinned to one value, check
  clip() argument order first.
- Joining two dataframes without every relevant key column (e.g. joining
  stats to opponent skew without `position` in the join key) causes silent
  row fan-out (each row matches multiple rows on the other side) rather than
  an error. Symptom: row count balloons, and/or a `_right`-suffixed
  duplicate column appears for a column that exists on both sides.
- The dynastyprocess ID crosswalk (`load_ff_playerids`) maps ~10 gsis_ids
  to two rows (sometimes two different people). An inner join on gsis_id then
  duplicates every stats row for those players. `load_player_metadata`
  now de-duplicates and asserts. Always `assert_unique_key` the right side of
  a join that should be one-to-one.
- Always sanity-check aggregation output (sort, print head/tail, check
  n_games / sample sizes, check percentiles) rather than trusting that code
  which runs without error is correct. Several real bugs in this project
  were caught this way, not by code review.

## Response output convention

Write every response to the user in `response.md` at the repo root, fully
overwriting the file each time (no appending, no history). The chat message
should only point to the file.

- Applies to **every** response: answers to questions, clarifying questions,
  status updates, and end-of-task/job reports. Not just long reports.
- `response.md` holds the full text of that response, the same content that
  would otherwise have gone in chat. Don't write a shorter summary there.
- When working in a git worktree, "repo root" means the worktree root. Commit
  `response.md` with the work so it survives worktree cleanup.
- The chat message stays minimal (a pointer to the file, plus any one-line
  status marker the harness requires, e.g. `result:`).

## Style/scope conventions

- Prefer simple, explainable logic over marginal-accuracy complexity,
  especially where the simpler version feeds into something else that
  depends on it being interpretable (e.g. the projection baseline).
- Every deliberate scope-out (kicker/DEF, long-play bonuses, rookie nulls,
  etc.) should be a one-line, explicit README/known-limitations note, not a
  silent gap.
- polars, not pandas.
- Domain classes (`domain/`) are validated-construction dataclasses with
  identity-based equality/hashing, kept free of computation logic and free
  of any external-API-shaped data.