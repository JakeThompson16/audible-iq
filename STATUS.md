# Audible IQ — Status

_Baseline: 2026-09-24 audit against VISION.md. Supersedes the untracked
`current_state.md` (2026-09-13), which is left in place but stale.
Findings marked **[verified]** were reproduced by running the code
(polars 1.39.0, 2023–2024 data, window=8 as in test.py); the rest are from
reading the code and were not executed._

## One-line summary
The expected-points backtest pipeline and scoring engine exist, but the
pipeline has three verified correctness defects that invalidate the README's
model-evaluation conclusions. There is no inference (upcoming-week) path, no
boom/bust output, no agent, no UI, and no rosters/leaguemates loading beyond
a raw Sleeper call.

## Done
- Sleeper read client: user, leagues, league users, rosters (raw JSON) —
  `clients/sleeper_client.py`
- `ScoringSettings` + `from_dict` — `domain/scoring.py`
- League-accurate scoring: `calculate_points_vectorized` — `engine/scoring.py`
  (validated exact vs. Sleeper on two configs; see Q-8 for unsupported fields)
- Sleeper<->nflreadpy ID bridge and player stats loading —
  `clients/nflreadpy/player_data.py`
- Schedule -> per-team opponent table — `clients/nflreadpy/team_data.py`
- Evaluation harness — `engine/metrics.py` (logic sound; inputs are suspect, B-1)

## Update 2026-09-27: RB stat vector merged into the engine
- Expected points — `engine/expected_points.py`: position -> implementation registry.
  RB = stat vector (`projections/expected_points/stat_vector/`), QB/WR/TE =
  rolling_avg_prior + opponent_skew. Backtest-validated only (Q-5 still open).
- Continuous rolling windows everywhere (`projections/rolling_window.py`): player
  features, opponent skew, epa_allowed, stat-vector features. Per-position player
  windows (QB 12, WR 10, TE 12, RB 8) recover QB fully; WR/TE ~0.04 MAE behind the
  old blend (Q-13).
- Tool explanation reads `projection_method` (Q-6 explanation part fixed).
- Shared `common/frames.assert_unique_key`, applied at `load_player_metadata`
  (crosswalk duplicates fixed), `load_player_stats`, games/skew/epa/actuals joins.
- LOSO harness `evaluation/backtest.py`; drivers `test.py`, `stat_vector_eval.py`.
- Tools layer (`tools/`) intentionally untouched (Q-6/Q-7).

## In progress
- Rolling features — `projections/expected_points/features/player_rolling.py`
- Opponent skew — `projections/expected_points/features/opponent_skew.py`
  (no better than rolling alone at any position on corrected data; Q-4)
- Tool contract slice — `tools/expected_points.py`, `tools/registry.py`,
  `domain/tool_result.py`
- PBP scaffolding — `clients/nflreadpy/pbp_data.py`,
  `projections/boom_bust/features/aggregate_pbp.py` (returns nothing; pass
  filter now excludes sacks, 2026-09-26)
- RB stat vector: DONE 2026-09-27 (see update above; Q-11 resolved). LOSO mean MAE 4.717 /
  R² 0.378 / Spearman 0.685 vs rolling alone 4.733 / 0.349 / 0.668.

## Not started
NiceGUI app; boom/bust threshold + distribution; inference path; leaguemate
roster loading + adapters; non-Sleeper league config; agent loop; storage;
trigger-based automation (scope unclear, Q-9); write strategy (Q-1).

---

# Gap analysis vs. VISION.md

## 1. ALIGNED & SOLID
| Component | Where | Why it fits |
|---|---|---|
| Sleeper read client | `clients/sleeper_client.py` (`get_user`, `get_user_leagues`, `get_league_users`, `get_league_rosters`) | Thin, no business logic; matches "leverage Sleeper API" and the recommend-only default. |
| Scoring engine | `engine/scoring.py::calculate_points_vectorized` | polars in/out, driven entirely by league settings, table-driven via `SCORING_TO_STAT_COLUMN`. This *is* the "customizable by league settings" core. (Caveat: Q-8.) |
| ScoringSettings dataclass | `domain/scoring.py` | Platform-agnostic fields, validated construction. |
| ID crosswalk | `clients/nflreadpy/player_data.py::load_player_metadata` | Bridge already built, as the vision states. |
| Team schedule derivation | `clients/nflreadpy/team_data.py::pull_team_games` | Clean polars boundary; the natural basis for upcoming-opponent lookup. |
| Leakage discipline | `.shift(1)` convention in features | Kept as a rule (but see B-3 for where it's not actually holding). |
| Eval harness design | `engine/metrics.py::evaluate_projections` | Position-segmented, Spearman-weighted, dict output; reusable for the distributional layer (extend with calibration/pinball metrics). |
| Explainability stance of the projection formula | `engine/expected_points.py` | `rolling_avg_prior + opponent_skew` matches the "explainability over accuracy" goal. |

## 2. NEEDS REFACTOR TO MATCH VISION
| Component | Specific conflict |
|---|---|
| `domain/scoring.py::ScoringSettings.from_dict` | Docstring/behavior is Sleeper-payload-shaped, so the platform-agnostic domain knows Sleeper's schema. CLAUDE.md's `adapters/` layer doesn't exist; move the mapping there so a manual (non-Sleeper) config builds the same object. |
| `clients/sleeper_client.py` | Imports `domain.scoring` at module top (only used by the `__main__` demo) — client depends on domain. Also: no timeouts, no error/retry handling, no caching. Missing endpoints the vision needs: NFL state (current week), players map, matchups, transactions. |
| `domain/player.py::Player` | (a) Holds computation (`reset_cache`) against columns that don't exist in the pipeline (`rolling_avg`, `boom_prob`, `bust_prob`; pipeline column is `rolling_avg_prior`); (b) mutable scalar `boom_prob`/`bust_prob` can't hold a distribution; (c) never used by any pipeline. Conflicts with "domain free of computation logic" and with Q-3. |
| `domain/team.py`, `domain/league.py` | Never constructed from Sleeper data (no adapter). Needed for roster + leaguemate loading. |
| `tools/registry.py` / `tools/expected_points.py` | `func(args, projections_df)` can't be invoked generically from the registry (Q-7); tool is hard-wired to one precomputed DataFrame shape. Blocks "swappable implementation behind a stable interface". |
| `domain/tool_result.py::ToolResult` | `value: Any`, one 4-tier `confidence`, untyped `metadata`: see B-6 / Q-6. |
| Hard-coded constants in code | Window (6 default, 8 in test.py), 9-game blend, volume thresholds 6/4/2/6, `k=16`, tier thresholds 4/8, ±12 clip live in function bodies/defaults. Under Q-2's reading, they should be named config behind the engine interface (still hand-chosen, not user-tunable). |
| `engine/expected_points.py::calculate_expected_points` | Signature is `(stats_df, skew_df)`: tied to the backtest-shaped table; no stable input contract for "player + upcoming week" (Q-5). |
| Entry points | `test.py` is the real integration driver: hardcoded username, `leagues[1]`, and the nflreadpy URL monkey-patch lives *there*, not in `clients/`. Anything else that calls `load_ff_playerids()` without importing test.py (e.g. `v.py`, `pull_pbp_features`) hits the 404 the patch works around. Move the patch into `clients/nflreadpy/`. |
| Storage | Flat-JSON per-user store specified in CLAUDE.md, not built; needed by the web app for settings/league config. |

## 3. MISSING ENTIRELY
| Component | Depends on (exists) | Clean-slate |
|---|---|---|
| **NiceGUI app** (server + localhost modes) | `ToolResult.value`-only display rule; engines | Everything: app shell, pages, session state, deployment config. Depends on the tool/engine contract being stable first. |
| **Inference path** for upcoming weeks (Q-5) | schedule table, rolling logic, scoring | As-of feature builder + as-of skew lookup. **Prerequisite for everything user-facing.** Size: M. |
| **Boom/bust threshold** (bounded multiplier/decay of expected points with floors) | expected points output | Threshold function, floor/cap rules, tests on low-baseline players. Size: S–M. |
| **Probability-distribution layer**: P(X+ points), P(X | boom) | Corrected expected points + historical residuals; `engine/metrics.py` scaffolding | Distribution model (recommend empirical, position x projection-bucket residual quantiles first: explainable, no fitted parameter), calibration metrics (reliability, pinball/CRPS, coverage), monotonicity/tail handling for P(23+), handling of the zero-floor and TD-driven right skew. **Sized honestly: L.** Roughly 3–5x the original classifier because it needs a calibrated *distribution*, its own evaluation harness, and a new tool/UI contract, not just labels + a model. P(X\|boom) is an additional M on top. |
| **Leaguemate roster loading** | `get_league_rosters` (raw), `get_league_users` | Sleeper `player_id` -> gsis mapping of every roster, Team/League adapters, cache for the large players map. Size: S–M. |
| **Non-Sleeper league config** | `ScoringSettings` dataclass | Manual-entry form/JSON loader + validation; `adapters/` layer. Size: S–M. |
| **Agent loop** | `TOOL_REGISTRY` (1 tool) | Orchestration, additional tools (variance, on-demand nflreadpy stats, boom/bust, roster), prompt/grounding, guardrails against unsupported claims. Size: L. |
| **Additional agent tools** (past variance, on-demand stats) | nflreadpy loaders | Tool wrappers + args models. Size: M. |
| **Persistence** | none | Flat-JSON store behind an interface. Size: S. |
| **Trigger-based automation** | none | Scope undefined (Q-9); Sleeper write path is Q-1-gated. |
| **Write execution (add/drop/start/sit)** | none | Deliberately absent; recommend-only per Q-1. |

## 4. BROKEN OR SUBOPTIMAL

### FIXED 2026-09-24: B-1, B-2, B-3, B-4 (details below kept as the original diagnosis)
- B-1: skew aggregated to one row per key + uniqueness assert (`opponent_skew.py`) and
  post-join row-count assert (`engine/expected_points.py`).
- B-2: `rolling_mean(..., min_samples=1)` on all five rolling columns (`player_rolling.py`).
- B-3: residuals aggregated to (defense, season, week) before shift/cumulate; `n_games`
  now counts games, so `min_games=3` / `k=16` mean what CLAUDE.md says.
- B-4: `insufficient_data` is a scored calibration tier (`engine/metrics.py`); it does carry
  projections (2,643 of 3,446 rows, 2024/25) — returning vets get last season's average.
- **Corrected 2024/25 metrics, existing k=16, no retuning** (`test.py`, 12,153 scored
  player-weeks; projection vs `rolling_avg_prior`-only baseline):

  | Pos | Proj MAE | Base MAE | Proj R² | Base R² | Skew helps MAE? | Spearman |
  |-----|---------:|---------:|--------:|--------:|:---:|---:|
  | QB | 8.102 | 8.102 | 0.177 | 0.174 | No (tie; R² +0.003) | 0.437 |
  | RB | 4.455 | 4.407 | 0.420 | 0.422 | No | 0.730 |
  | WR | 4.462 | 4.440 | 0.336 | 0.340 | No | 0.644 |
  | TE | 4.186 | 4.148 | 0.328 | 0.329 | No | 0.611 |

  Calibration (MAE by tier, all positions): high 5.142 (n=4,727), medium 4.710 (3,383),
  low 4.497 (3,147), insufficient_data 4.525 (896). **The inversion persists**
  (`monotonic_decreasing_mae` False for every position and overall); it is not explained by B-2/B-4.
- Net effect vs. the pre-fix README table (k=16): the baseline itself improved materially
  (B-2: R² e.g. QB 0.101 -> 0.174, RB 0.392 -> 0.422; Spearman up everywhere) and the
  "QB skew reliably helps" finding disappeared (QB MAE tie), i.e. it was largely an
  artifact of the row-level leakage. RB/WR/TE skew remains slightly negative/neutral.
  k has NOT been re-tuned; the grid search should be re-run on the corrected pipeline.

### Verified bugs in the existing pipeline (high priority) — all four now fixed, see above
- **B-1 [verified] Skew rows duplicate, fanning out predictions.**
  `_calculate_position_skew` emits one row per *qualifying player-week*, not
  per (defense, week, position). Of 1,384 (defense, week, season, position)
  keys, **602 are duplicated (max 4 rows each)**. The left join in
  `calculate_expected_points` then fans out: **19,357 prediction rows vs.
  17,896 stats rows for 2024 (+8.2%)**. This is the fan-out class CLAUDE.md
  warns about, arriving via the *right* side's non-unique key. `evaluate_projections`
  scores the inflated frame. Fix: aggregate skew to one row per key.
- **B-2 [verified] The early-season blend is effectively bypassed.**
  `rolling_mean(window_size=window)` after `shift(1)` yields null until
  `window` prior games exist. So `_current_season_rolling_avg` is null for the
  first `window` weeks, and the blend branch (both non-null) never fires
  early. Example: Travis Kelce 2024, weeks 1–9 all have
  `rolling_avg_prior = 16.07` (last season's average alone); the blend begins
  week 10 with current-season weight already 8/9. The documented
  `min(games/9, 0.9)` ramp from CLAUDE.md never operates in its intended
  range. Same nulls make `trailing_*_avg` null for the first `window` games,
  so those player-weeks are excluded from the skew volume filter. Fix:
  `rolling_mean(..., min_samples=1)` (or expanding mean until the window
  fills).
- **B-3 [structural, follows from B-1; not separately measured] Skew
  `.shift(1)` leaks / mis-counts.** Rows are sorted by (opponent, season,
  week) and shifted by one *row*, but each week has several player rows per
  defense. The second player-row of week W sees the first player-row's
  week-W residual (same-week leakage), and `n_games` counts player-rows, not
  games (so `min_games=3` and `k=16` mean something different than
  documented). Fix: aggregate residuals to (defense, position, week) first,
  then shift/cumulate over weeks.
- **B-4 [verified] `confidence == "insufficient_data"` is not what the
  comments say.** `engine/metrics.py` asserts that tier "never has a non-null
  projection", but 1,319 of 1,735 such 2024 rows have one (returning
  veterans in week 1 get last season's average). Those rows are excluded from
  every calibration tier (tiers list omits it) yet counted in "scoreable".
  Combined with B-2 (tiers `low` 1–3 and `medium` 4–7 games are the same
  last-season-average-only projection), this plausibly explains the
  documented non-monotonic calibration; that link is a hypothesis, not tested.
- **B-5 Consequence for documentation.** README "Model Evaluation" table,
  the k=16 choice, and CLAUDE.md's "Grid search finding" were all produced
  through B-1..B-3. They should be treated as unverified until re-run
  (Q-4). The additive-not-multiplicative and no-fitted-parameter *design*
  decisions are unaffected.

### `aggregate_pbp.py` (checked as requested)
- **Double `.otherwise()` in `rush_value`/`pass_value`: NOT present** in the
  current file — each has a single `.when().then().otherwise()`. (A stray
  blank line + trailing comma sits after `rush_value`; harmless.) Real
  issue there: `10 / yardline_100` is undefined at `yardline_100 == 0`.
- **`deep_pass_completion` [confirmed]** is identical to `deep_pass_attempt`
  (only checks `air_yards >= 20`); it never checks `complete_pass == 1`.
- **`redzone_td` [confirmed-by-reading]** uses `touchdown == 1`, which also
  fires on defensive TDs (pick-sixes); should use `pass_touchdown` /
  `rush_touchdown`.
- **QB concat / role handling [confirmed-by-reading]:** `pl.concat(how=
  "diagonal")` of pass and rush rows leaves `designed_run` null on pass rows
  and there is no explicit role column (`actor_role`: pass/rush/target) —
  downstream code must infer it from `play_type` and null patterns. For
  RB/WR/TE the frame named `receptions` is actually *targets*
  (filter on `receiver_player_id` includes incompletions).
- Additional: `play_type == "pass"` includes sacks; 2-pt attempts
  (`yardline_100 == 2`) count as red-zone opportunities; penalty-nullified
  plays aren't filtered. `pull_pbp_features` has no `return` and its
  annotation says DataFrame while `_aggregate_pbp_data` returns a dict.
  `_build_cross_walk` is unused. The module isn't callable from anywhere,
  and it would 404 on the ID crosswalk unless test.py's patch was applied.

### Architectural suboptimalities relative to the new vision
- **B-6 `ToolResult` won't hold a distribution.** `confidence` (one of four
  tiers derived from games played) has no meaning for boom/bust; the
  distribution's sharpness/sample support is a different quantity. `value`
  is `Any`, so consumers can't rely on shape. `explanation` is a string built
  per tool. Needed: a documented, JSON-serializable per-tool value schema
  (e.g. `{"expected": .., "boom_threshold": .., "p_ge": {"18": .., "23": ..}}`)
  and per-tool confidence semantics. Keep the UI-reads-`.value`-only rule.
- **B-7 Silent fallbacks contradict "explainable".** `opponent_skew` nulls
  are `fill_null(0.0)` (a missing matchup looks like a neutral one);
  unmapped nonzero scoring fields contribute 0 (Q-8); no warnings surface
  either to the agent.
- **B-8 `confidence` is opponent-blind by design** (documented and
  deliberate; not a bug), but the agent will want it as an input to
  "how much should I trust this", so B-6 matters more than it did.
- **B-9 Season/timing:** today is 2026-09-24 (2026 season live). Sleeper
  demo uses season "2026" but the pipeline has only ever been evaluated on
  2024/25; nothing has run on in-season data. Expect nflreadpy freshness/
  schema issues; unverified.

---

# Proposed sequence (dependency-ordered)

0. **Stop-the-bleeding fixes (small, no design decisions needed):** B-1
   (aggregate skew per key + assert unique key), B-2 (`min_samples`), B-3
   (game-level shift), B-4 (confidence semantics/calibration tier set), move
   the nflreadpy URL patch into `clients/nflreadpy/`. Add asserts that make
   the CLAUDE.md "fan-out" and "n_games" checks automatic. **Re-run
   `test.py`, then update README/CLAUDE.md numbers (Q-4).** This precedes
   everything else because boom/bust is *defined* relative to the projection;
   a defective baseline corrupts every distribution built on it.
1. **Decide Q-1, Q-2, Q-3** (cheap conversations; they gate the contract
   design). Also confirm Q-9 scope.
2. **Design spike for boom/bust (Q-3/Q-6)** using the corrected residuals:
   threshold rule, distribution family, output schema, evaluation metrics.
   **Do this before more RB/WR/TE PBP feature work**, and before touching
   `aggregate_pbp.py`: the spike determines whether PBP-derived features are
   even inputs to the distribution, and what `labels.py`/`model.py`
   (planned, not yet existing) must look like. Park PBP scaffolding as-is
   (fixing the confirmed bugs above is trivial but not urgent).
3. **Engine/tool contract (Q-2, Q-6, Q-7):** ToolResult v2, registry with
   bound context, engine interface (`DataFrame/dict -> DataFrame/dict`).
4. **Inference path (Q-5):** as-of feature builder, shared with backtest.
5. **Adapters + Sleeper client hardening:** `adapters/` layer, Team/League
   construction, own roster + leaguemate rosters, current-week state, players
   map caching, timeouts; manual league-config loader; storage interface.
6. **Boom/bust implementation** per step 2, with its own eval harness.
7. **Agent loop + additional tools** (variance, on-demand stats, roster).
8. **NiceGUI app** (localhost first, then server-side), reading `.value` only.
9. Deferred/named: write execution (Q-1 b/c), trigger automation (Q-9),
   V2 trade engine, skew feature granularity for RB/WR/TE (post-Q-4).

Steps 3 and 4 can run in parallel; 5 can start any time after step 0.
