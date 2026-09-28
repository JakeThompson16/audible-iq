# Sanity bounds, player search, projection service (UI blocked: prompt cut off)

**Summary:**
- **`prompt.txt` is truncated.** It ends mid-sentence in Part 2 ("…home/away if the schedule
  has it. If") and has no NiceGUI section at all (68 lines).
- **Done:** Part 0, Part 1, and the specified part of Part 2.
- **Not built:** the NiceGUI shell. I need the rest of the prompt rather than guess the UI.
- **Branch:** `worktree-player-search` (commit `915e2fd`, pushed, not merged). It's built on
  `worktree-stat-pipeline`, which isn't merged either.
- **Tests:** 33 pass, 1 skipped. Parity with the backtest is still exact (48 player-weeks, 396
  values, 0 mismatches). `test.py` output is unchanged.

## Part 0: prediction sanity bounds
**Backtest audit.** Raw sub-model predictions before any clipping, all 2019–2025
leave-one-season-out folds:

| Pos | Sub-model | Rule | Violations | Worst value |
|---|---|---|---:|---:|
| TE | rush_td_rate | outside [0,1] | 22 / 8,548 (0.26%) | −0.043 |
| TE | ypc | < 0 | 2 / 8,548 (0.02%) | −0.67 → **negative rushing yards** (ypc wasn't clipped) |

- **All other positions and sub-models: 0 violations.** That covers every volume, catch and
  completion rate, TD/INT rate and yardage rate for QB, RB and WR, and TE's other models.
- **Derived checks:** 0 violations (incompletions ≥ 0, receptions ≤ targets, completions ≤
  attempts) apart from those 2 TE rushing-yard rows.

**Live audit** (all 783 players in the search index, next game, production artifacts):
- 669 `ok`, 114 `no_history`, 0 `unknown_player`.
- The bounds changed 2 predictions, both TE rush-TD rate.
- 0 derived violations.

**Bounds added** in the shared derivation path (`stat_vector/core.py`, used by the backtest,
engine and prediction):
- **Clipped to [0,1]:** all per-attempt probabilities (catch, completion, rush/rec/pass TD
  rates, INT rate). Before, TD/INT rates only had a ≥0 floor.
- **Floor at 0:** yardage rates (ypc, ypr, yards per completion). New.
- **Floor at 0:** volumes. Already there.
- **`check_stat_line()`:** asserts no negative stat, receptions ≤ targets and completions ≤
  attempts, and raises `StatVectorBoundsError` if the clipping ever fails to guarantee them.
  It has its own unit test.
- **`bounds_applied`:** predictions now report which sub-models a bound changed (dev detail).

**Metric change: none.** `test.py` output is identical, because the bounds only touch 2 TE
rows' rushing yards in the backtest. The parity tests pass unchanged. Documented in CLAUDE.md,
"Prediction sanity bounds".

## Part 1: `search/player_search.py` (no UI code)
**fast-autocomplete 0.9.0,** checked against the installed source, not memory:
- **Constructor:** `AutoComplete(words, synonyms=None, full_stop_words=None,
  valid_chars_for_string=None, ...)`.
- **Words:** a dict `{key: context}`; the count comes from `context["count"]`.
- **Search:** `search(word, max_cost=2, size=5)` returns a list of token lists and is
  LFU-cached.
- **Normalization:** it lowercases; keeps only a–z, digits, space, `-`, `:` and `_`; turns
  `-` into a space; drops everything else, *including accented letters* (so "José" would
  become "jos"); caps at 40 characters.
- **Multi-word names:** handled as graph paths.
- **Typos:** fuzzy Levenshtein matching (distance < `max_cost`) runs only once 3 or more
  characters are unmatched.
- **Backend:** it needs a Levenshtein library; I use the `[levenshtein]` C extra, which the
  package itself recommends.

**What I built:**
- **Membership:**
  - on a current-season nflverse weekly roster (the same source the pipeline uses for team);
  - at a position from `POSITION_IMPLEMENTATIONS`;
  - in the `load_player_metadata` crosswalk.
  - Result: **783 players.**
- **Excluded:**
  - **142 rostered players because they aren't in the ID crosswalk;**
  - 10 because their stats-row position isn't QB/RB/WR/TE.
- **Keyed by gsis_id.** Names are normalized: accents folded, apostrophes and periods removed,
  hyphens become spaces, Jr./Sr./II/III/IV/V dropped. Each is indexed as the full name plus
  every trailing part (last names, including "st brown").
- **Shared names:** keys map to *sets* of gsis_ids, and a selection is always the gsis_id,
  never the display string ("Name · POS · TEAM").
- **Relevance:** the autocomplete `count` is games played over the last two seasons.
- **Fallback:** handles "first-prefix last-prefix" queries like "pat mah".
- **API:**
  - `build_player_index(season=None)`, cached;
  - `refresh_player_index()`, which also reloads rosters and the ID maps;
  - `PlayerIndex.suggest(query, size=8)` and `.get(gsis_id)`, returning `{gsis_id,
    sleeper_id, name, display, position, team}`.
- **Tested examples:** prefix ("mahom"), last name ("mahomes", "allen", "njigba", "st
  brown"), typo ("patrik mahomes", "maccaffrey", "jamar chase"), "pat mah", hyphen
  ("amon-ra", "smith-njigba"), apostrophe ("jamarr"), suffix ("kenneth walker" → Kenneth
  Walker III), periods ("a.j. brown").
- **Ordering:** "josh" ranks by games played (Allen, Downs, Jacobs, …). "patrick" puts Tim
  Patrick first, since an exact last-name match beats a first-name prefix, then Mahomes.

**Latency** (1,005 queries):

| | Latency |
|---|---:|
| Median, first call | **0.14 ms** |
| p95, first call | 1.03 ms |
| Max, first call (fuzzy / fallback path) | 7.5 ms |
| Median, repeat (LFU-cached) | 0.06 ms |
| Index build (context already loaded) | ~1 s |
| Cold, including loading the context | ~10 s |

## Part 2: `search/projection_service.py` (as far as the prompt goes)
- **What it does:** `project_player(gsis_id, season=None, week=None)` is a thin layer over
  `predict_player_stats` (same statuses, stats keys and artifacts).
  - The default target is the team's next unplayed game.
  - It adds `home_away` ("home"/"away") and `gameday` to `target`. The team schedule now
    carries `home` and `gameday`.
- **Example:** Mahomes → 2026 week 4 **at LV**, 2026-10-04, `ok`: 31.4 attempts, 228 passing
  yards, 1.4 passing TDs, 19.5 rushing yards.
- **Tested:** a played week returns home/away; a bye returns `status="bye"` with `home_away`
  None and no stats.
- **Everything after "If" wasn't specified,** so nothing more is built. My guess at the next
  clause (behaviour when there's no scheduled game) is already covered by the passed-through
  statuses.

## Also
- **`requirements.txt` (new, pinned):** polars 1.39.0, numpy 2.4.2, scipy 1.17.1, nflreadpy
  0.1.5, requests 2.32.5, pydantic 2.13.4, fast-autocomplete[levenshtein] 0.9.0,
  python-Levenshtein 0.27.5, pytest 9.1.1. I installed fast-autocomplete and
  python-Levenshtein into your Python.
- **Warning silenced:** the polars join_asof sortedness warning in the `epa_allowed` as-of
  path (both sides are already sorted per defense).
- **Docs:** CLAUDE.md (search, service, sanity bounds, requirements), README, STATUS.

## What I need
1. **The rest of `prompt.txt`:** Part 2 after "If", and the whole NiceGUI section (layout,
   pages, what it shows). Then I'll build the UI on top of `PlayerIndex.suggest` and
   `project_player`.
2. **Merge when ready:** `worktree-stat-pipeline` and then `worktree-player-search` are both
   fast-forwards from main (`2fbc712`). `git merge --ff-only worktree-player-search` brings in
   both. I can't merge or push main from this background session.
