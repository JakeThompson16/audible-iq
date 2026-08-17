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
    team/schedule data (`team_data.py`) via nflreadpy/nflverse.
- `adapters/` (Sleeper) — maps raw external JSON into domain schema
  (e.g. `ScoringSettings.from_dict()`).
- `engine/` — pure computation across domain objects/dataframes.
  - `engine/scoring.py` — `calculate_points_vectorized()`: dot product of
    stats and league ScoringSettings via `SCORING_TO_STAT_COLUMN` registry
    (maps ScoringSettings field names -> stats_df column names, since they
    don't align 1:1).
- `projections/expected_points/features/` — feature engineering for the
  projection model.
  - `player_rolling.py` — trailing (prior-games-only) rolling averages per
    player: `rolling_avg_prior` (fantasy points), `trailing_opportunities_avg`
    (carries + targets combined — see below), `trailing_targets_avg`,
    `trailing_attempts_avg`.
  - `opponent_skew.py` — Adjusted Points Allowed (APA): opponent-adjusted
    matchup scoring per defense/position, cumulative across the season.
- `engine/expected_points.py` — `calculate_expected_points(stats_df, skew_df)`:
  joins rolling stats to opponent skew (same validated join keys as
  `test.py` — week/season/opponent_team/position) and computes
  `projection = rolling_avg_prior + opponent_skew` plus a `confidence` tier
  per row. See "Confidence tiers" below for the derivation and its known
  limitations.
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
    `calculate_expected_points()` output, returning `ToolResult`.
  - `tools/registry.py` — `TOOL_REGISTRY` dict, tool name -> {args_model,
    func, description}.
- `config.py` — `PLAYER_METADATA` is the single canonical list of player
  identity/metadata fields; all player construction depends on it.

## Key design decisions (with rationale — don't relitigate without reason)

**Projection formula**: `projection = rolling_avg_prior + opponent_skew`.
Deliberately simple/explainable (recent form + matchup adjustment) rather
than a fitted model, because the projection is the reference point the
boom/bust classifier measures deviation against — bias in the baseline
would silently propagate into what "boom"/"bust" mean.

**Leakage prevention (critical, applies everywhere)**: all rolling/cumulative
features use `.shift(1)` before `.rolling_mean()` / `.cum_sum()` /
`.cum_count()`, so week W's value only ever reflects games strictly before
W. This applies to player rolling averages AND opponent skew. Never remove
a `.shift(1)` without understanding this.

**Opponent skew (APA)**: `residual = fantasy_points - rolling_avg_prior`,
averaged cumulatively per (defense, position) across the season, using
`.shift(1)` first (same leakage rule). Additive, not multiplicative — ratios
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

**Confidence tiers** (`engine/expected_points.py`): derived *solely* from
player-side `games_this_season` — not opponent_skew's `n_games`, not outcome
volatility (volatility is boom/bust's job, not expected points'). Tiers:
`insufficient_data` (0 games), `low` (<4), `medium` (<8), `high` (8+).
Thresholds are provisional, not yet calibrated against actual MAE per
bucket — `engine/metrics.py`'s calibration check confirms whether MAE comes
out monotonically decreasing (high < medium < low) against real output;
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
  opponent_skew blends in prior-season data early in the season (see below)
  without adjustment for defensive coordinator or personnel turnover. A
  defense that overhauled its scheme in the offseason will still get
  partial credit from last year's skew profile until enough current-season
  games accumulate to dilute it out.

**Early-season handling**: current-season weight = `min(games_played / 9, 0.9)`;
last season's full-season average inherits the rest — so it always retains
at least 0.1 weight even once the current season has plenty of games.
- Opponent skew: blends current-season cumulative skew with prior-season
  full-season average skew per (defense, position), weighted by `n_games`
  this season (full trust ~week 10, never fully to 1.0).
- Player rolling average: same blend pattern — returning veterans blend
  current-season rolling avg with last-season's full-season average,
  weighted by `games_this_season`.
- True rookies with no current- or prior-season data: `rolling_avg_prior`
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
- Always sanity-check aggregation output (sort, print head/tail, check
  n_games / sample sizes, check percentiles) rather than trusting that code
  which runs without error is correct. Several real bugs in this project
  were caught this way, not by code review.

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