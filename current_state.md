# Audible IQ — Current State

_Snapshot as of 2026-09-13. Not tracked in git (local reference only)._

## One-line summary

Sleeper-integrated fantasy football decision-support tool. The expected-points
pipeline (rolling average + opponent-matchup adjustment, confidence-tiered,
offline-evaluated) is built and validated end to end. Boom/bust
classification — the actual start/sit differentiator — is in early feature
scaffolding, not yet computing anything. No agent/UI layer wired up yet
despite a tool contract existing for one.

## What actually works today

- **Sleeper integration** (`clients/sleeper_client.py`, `v.py`/`test.py` as
  drivers): resolve a username → user_id → leagues → league scoring settings.
  Live-tested against a real account/league.
- **Scoring settings adapter** (`domain/scoring.py`): raw Sleeper JSON →
  `ScoringSettings` dataclass. QB/RB/WR/TE only, by design (see CLAUDE.md).
- **League-accurate scoring engine** (`engine/scoring.py`,
  `calculate_points_vectorized`): validated exact-match against real Sleeper
  output across two league configs (standard + TE-premium), using Trey
  McBride's full season as the check.
- **Player metadata / ID crosswalk** (`clients/nflreadpy/player_data.py`):
  `nflreadpy.load_ff_playerids()` + `load_players()` join solves
  Sleeper↔gsis_id↔pfr_id identity without fuzzy matching. Also derives
  `opportunities` (targets + carries) and various boolean/threshold stat
  columns (`over_300_passing_yards`, `te_receptions`, etc.).
- **Team/schedule data** (`clients/nflreadpy/team_data.py`): used to build
  opponent join keys for skew.
- **Rolling player features**
  (`projections/expected_points/features/player_rolling.py`):
  `rolling_avg_prior`, `trailing_opportunities_avg`, `trailing_targets_avg`,
  `trailing_attempts_avg` — all leakage-safe (`.shift(1)` before rolling),
  with the early-season current/prior-season blend described in CLAUDE.md.
- **Opponent skew (APA)**
  (`projections/expected_points/features/opponent_skew.py`,
  `calculate_all_position_skews`): cumulative, sample-size-weighted-shrinkage
  (`k=16`, grid-searched, not fitted) adjustment per (defense, position).
  Wide ±12 sanity clip as defense-in-depth only.
- **Expected points engine** (`engine/expected_points.py`,
  `calculate_expected_points`): joins rolling stats to skew on
  week/season/opponent_team/position, computes
  `projection = rolling_avg_prior + opponent_skew` plus a confidence tier
  (`insufficient_data`/`low`/`medium`/`high`, player-side games-played only).
  True rookies with no data correctly resolve to `None`, not a fabricated
  fallback.
- **Offline evaluation harness** (`engine/metrics.py`,
  `evaluate_projections`): MAE/RMSE/mean-error/R² for projection vs.
  rolling-avg-only baseline, Spearman rank correlation per
  (season, week, position), and a confidence-tier calibration check —
  all segmented by position. MAPE deliberately omitted. Run end-to-end in
  `test.py` against 2024/2025 held-out data (2023 loaded only to seed
  blending, never scored).
  - **Known result from this harness** (per CLAUDE.md): opponent_skew is
    reliably additive for QB; for RB/WR/TE it's currently net-neutral
    (shrinkage removed the harm but didn't unlock real signal) — flagged as
    a target for finer-grained matchup features, not yet acted on.
- **Agent-facing tool contract, thin slice** (`tools/expected_points.py`,
  `tools/registry.py`, `domain/tool_result.py`): `ExpectedPointsArgs` +
  `get_expected_points()` wraps a precomputed `calculate_expected_points()`
  output as a `ToolResult`, registered in `TOOL_REGISTRY`. This is the only
  tool implemented so far, and nothing currently calls into the registry
  (no agent loop, no orchestration).

## In progress / scaffolding only (not yet producing output)

- **Play-by-play ingestion** (`clients/nflreadpy/pbp_data.py`,
  `load_pbp_data`): thin wrapper around `nflreadpy.load_pbp()`. New,
  uncommitted.
- **Boom/bust PBP feature aggregation**
  (`projections/boom_bust/features/aggregate_pbp.py`): builds
  position-segmented play-level dataframes (QB/RB/WR/TE) with derived
  columns — redzone opportunity/TD, big-rush flag, deep-pass flag,
  yardline-weighted "value" of a rush/pass attempt. `pull_pbp_features()`
  currently computes `positional_data` and **returns nothing** — the
  function has no `return` statement yet, so this module is not usable from
  anywhere else in the codebase yet. New, uncommitted, not integrated into
  `engine/` or `projections/expected_points/`.
- **`domain/player.py`, `domain/team.py`, `domain/league.py`**: exist per
  the architecture doc but not exercised by `test.py`/`v.py` — the working
  pipeline currently flows through raw stats dataframes and
  `ScoringSettings` only, not through `Player`/`Team`/`League` objects.

## Not started

- Boom/bust classifier itself (the aggregated PBP features have no
  downstream model or probability output yet).
- Any agent orchestration loop that actually calls `TOOL_REGISTRY`.
- Any UI/display layer (CLAUDE.md notes a UI adapter should read only
  `ToolResult.value` — no such adapter exists yet).
- Persistent storage (CLAUDE.md specifies flat JSON per user; not yet
  implemented — `v.py`/`test.py` take username as a hardcoded string).
- Kicker/DEF/IDP, long-play TD bonuses: explicitly out of scope (not gaps).

## Loose ends worth knowing about

- `v.py` is a scratch/debug script (dumps `load_ff_playerids()` columns),
  not part of the pipeline.
- `test.py` is currently the de facto integration entry point (hardcoded
  Sleeper username, picks `leagues[1]`) rather than a real test file despite
  the name.
- `aggregate_pbp.py`'s missing `return` is the most immediate correctness
  gap if boom/bust work resumes — it silently discards
  `_aggregate_pbp_data()`'s output.
