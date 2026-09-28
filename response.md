# Player search, projection service, NiceGUI page

**Where the prompt was cut off:** Part 2, line 69, at "…home/away if the schedule has it. If".
Parts 0 and 1 and the start of Part 2 were already done (commit `915e2fd`). This round
finishes Part 2 from "If the team is on bye…", then Parts 3 and 4.

**Status:**
- **Branch:** `worktree-player-search` (commit `467cf5b`, pushed, **not merged**). It sits on
  `worktree-stat-pipeline`, which is also unmerged; both are fast-forwards from main.
- **Tests:** 43 pass, 1 skipped. Parity with the backtest is still exact (48 player-weeks,
  396 values, 0 mismatches).
- **Run the app:** **`python webapp/main.py` → http://localhost:8090.** Checked: the server
  starts and serves the page.

## Part 0: sanity-bound violation frequencies (recap)
Raw predictions before bounds, 2019–2025 leave-one-season-out:

| Pos | Sub-model | Violation | Frequency |
|---|---|---|---:|
| TE | rush_td_rate | < 0 | 22 / 8,548 (0.26%) |
| TE | ypc | < 0 (gave negative rushing yards) | 2 / 8,548 (0.02%) |
| all others | all sub-models and derived checks | – | 0 |

- **Live check** (all 783 indexed players): bounds changed 2 predictions (TE rush-TD rate);
  0 derived violations.
- **Bounds now in the shared path:**
  - all per-attempt probabilities clipped to [0,1];
  - yardage rates floored at 0;
  - volumes floored at 0;
  - `check_stat_line()` asserts no negative stats, receptions ≤ targets and completions ≤
    attempts.
- **`test.py` output is identical** with the bounds.

## Part 1: search index
**Index size per position** (players on a current-season (2026) weekly roster):

| QB | RB | WR | TE | Total |
|---:|---:|---:|---:|---:|
| 114 | 184 | 317 | 168 | **783** |

**Exclusions:**

| Reason | Count |
|---|---:|
| On a 2026 roster at QB/RB/WR/TE but **not in the ID crosswalk** | **142** |
| Stats-row position isn't QB/RB/WR/TE | 10 |
| **Not rostered:** played QB/RB/WR/TE in 2025 and in the crosswalk, but on no 2026 roster (e.g. D'Ernest Johnson, Cam Akers, Raheem Mostert, Miles Sanders) | 63 |

- **Every index entry resolves** through `predict_player_stats` (0 `unknown_player`). That's
  tested.

**Suggest latency** (1,005 queries):

| | Latency |
|---|---:|
| Median, first call | **0.14 ms** |
| p95, first call | 1.0 ms |
| Max, first call (typo / fallback path) | 7.5 ms |
| Median, repeat (cached) | 0.06 ms |

**Sample suggestions:**
- **Common name:** "williams" → Kyren Williams · RB · LA, Caleb Williams · QB · CHI, Jameson
  Williams · WR · DET, Kyle Williams · WR · NE, Javonte Williams · RB · DAL
- **Common first name:** "josh" → Josh Allen · QB · BUF, Josh Downs · WR · IND, Josh Jacobs ·
  RB · GB, Josh Oliver · TE · MIN, Josh Palmer · WR · BUF (ranked by games played)
- **Suffixed:** "brian thomas" → Brian Thomas Jr. · WR · JAX; "kenneth walker" → Kenneth Walker
  III · RB · KC
- **Typo:** "mahomez" → Patrick Mahomes · QB · KC; "jamar chase" → Ja'Marr Chase · WR · CIN
- **Punctuation:** "st brown" / "amon-ra" → Amon-Ra St. Brown; "a.j. brown" → A.J. Brown;
  "pat mah" → Patrick Mahomes

## Part 2: `project_player` (finished)
- **Default target:** the team's next unplayed game. A bye has no schedule row, so the
  following week is returned, and the week number is always included.
- **Returns everything the card needs:** player, target (season, week, opponent,
  **home_away**, gameday, **game_final**), status, reason, stats, volume/rate detail,
  bounds_applied, games_this_season, confidence, projection_method, model_version,
  unpredicted, and **`in_training_window`**.
  - `in_training_window` is new in the prediction output: true when the target season was
    trained on and the week is at or before the artifact's `data_through`.
- **Final games:** include **`actual`**, the player's real stats under the same keys. If he
  has no stats row, `actual` is None and `actual_note` explains why.
- **Statuses:** predict's statuses pass through with reasons; never zeros.
- **No per-call loads:** it uses the process-wide caches.

**Sample output, one player per position** (all 2026 week 4, `status: ok`, `in_training_window:
false`, confidence "low" = 3 games played so far):

| Player | Game | Projected stat line |
|---|---|---|
| **QB** Josh Allen (BUF) | vs NE, home, 2026-10-04 | 28.3 att, 17.9 cmp, 200.3 pass yds, 1.2 pass TD, 0.7 INT, 7.8 rush att, 35.4 rush yds, 0.58 rush TD |
| **RB** Bijan Robinson (ATL) | at NO, away, 2026-10-05 | 19.3 car, 85.1 rush yds, 0.58 rush TD, 5.3 tgt, 4.1 rec, 29.5 rec yds, 0.16 rec TD |
| **WR** Ja'Marr Chase (CIN) | vs JAX, home, 2026-10-04 | 8.7 tgt, 5.4 rec, 66.1 rec yds, 0.42 rec TD, 0.1 car, 0.6 rush yds |
| **TE** Trey McBride (ARI) | at NYG, away, 2026-10-04 | 9.6 tgt, 7.0 rec, 69.8 rec yds, 0.53 rec TD |

## Part 3: NiceGUI page (`webapp/main.py`)
- **Version:** checked against NiceGUI **3.17.1**'s bundled `llms.md` and source for the
  current APIs:
  - `@ui.page`, `@ui.refreshable`, `ui.timer`, `run.io_bound`;
  - `ui.item(text, on_click=)` inside a `ui.list`;
  - `on_value_change`;
  - the 3-second page-builder timeout (why heavy loading runs in a worker thread).
- **No business logic in the UI:** suggestions come from `PlayerIndex.suggest` (debounced
  0.2 s), and a selection passes its gsis_id to `project_player`. Labels and one-decimal
  rounding are presentation only.
- **Card:**
  - name, position, team;
  - header "Week N (season) vs OPP (home/away)";
  - the position's stat line with readable labels.
- **Non-ok statuses:** labeled messages (bye, no history/rookie, no game scheduled,
  unsupported, unknown).
- **Always shown:** a static note that injury/inactive status isn't modeled.
- **Dev mode** (off by default) adds:
  - confidence, projection_method, in_training_window, games this season, model data_through
    and fit time, team source, bounds_applied, unpredicted categories;
  - volume/rate detail;
  - a season/week override;
  - for a final game, actuals next to the projection, with an **in-sample** badge when
    `in_training_window` is true.
- **Footer:** model `data_through` and fit time from the artifacts.
- **Startup:** a spinner "Loading player data and models…" while the context and index load in
  a worker thread.

## Part 4: tests, deps, docs
- **Search tests:**
  - prefix, last-name, full-name, **mixed-case**, **one-character typo** ("mahomez");
  - Jr./III suffixes, apostrophes, hyphens, periods, accents (normalization unit test);
  - **duplicate names → distinct entries and gsis_ids** (a synthetic index, since no two
    current players share a full name; the real-roster version skips);
  - only QB/RB/WR/TE (derived from `POSITION_IMPLEMENTATIONS`); only current-season rostered;
  - every entry resolves without `unknown_player`; team matches the pipeline;
  - refresh rebuilds; latency.
- **`project_player` tests:**
  - the default target equals the pipeline's next game;
  - a final game includes actuals matching the real stats row, with `in_training_window`
    true;
  - bye, rookie (`no_history`) and unknown player;
  - every output is JSON-serializable.
- **App smoke test:** NiceGUI's `User` fixture opens the page, waits out the loading state,
  types "mahomes", clicks the suggestion, sees the card, header and stat line, then toggles
  dev mode. The plugin and `asyncio_mode` are set in `pytest.ini`.
- **`requirements.txt` (pinned):** polars 1.39.0, numpy 2.4.2, scipy 1.17.1, nflreadpy 0.1.5,
  requests 2.32.5, pydantic 2.13.4, fast-autocomplete[levenshtein] 0.9.0,
  python-Levenshtein 0.27.5, **nicegui 3.17.1**, pytest 9.1.1, **pytest-asyncio 1.4.0**.
- **Docs:**
  - **CLAUDE.md:** search API and the gsis-keyed design, `project_player` fields and
    semantics, the web UI, the run command.
  - **STATUS.md:** updated.
  - **OPEN_QUESTIONS:** new **Q-16** (the 142 crosswalk gaps, with a fix path; injuries; the 63
    unrostered).
  - **README:** run instructions.

## Surprises
1. **Port 8080 is taken on your machine** by another app (process 11464, which serves a page
   titled "Dovetail"). NiceGUI's default port failed to bind. I didn't touch that process; the
   app now defaults to **8090** on 127.0.0.1, with `--port N` to override.
2. **142 current-roster players aren't in the dynastyprocess crosswalk,** so they can't be
   searched or projected yet. Q-16 has a fix path: fall back to nflverse `load_players` for
   identity; those players would have no Sleeper ID.
3. **The 2025 week-10 test was KC's bye:** my first card test hard-coded it. The tests now pick
   weeks from the schedule.
4. **"patrick" ranks Tim Patrick above Patrick Mahomes,** because an exact last-name match beats
   a first-name prefix. "pat mah" and "mahomes" both find Mahomes first. Worth a look if you'd
   rather first-name prefixes win.
5. **Week-4 targets while week 3 isn't final:** those teams have already played week 3; two
   teams hadn't. That's expected, since the target is each team's next unplayed game.
6. **pip reported an unrelated conflict:** `sportsdataverse 0.0.40` (already on your machine,
   not used here) wants polars ≤0.18.15.
7. **Installed into your Python:** nicegui 3.17.1 and pytest-asyncio 1.4.0.

## Next command
- **Try it:** `python webapp/main.py`, then open http://localhost:8090.
- **Merge when ready**, from `C:\Users\jak3t\Audible IQ`: `git merge --ff-only
  worktree-player-search` (this includes the unmerged `worktree-stat-pipeline`), then
  `git push origin main`. I can't merge or push main from this background session.
