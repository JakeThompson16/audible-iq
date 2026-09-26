# Response

My previous reply is below in full. From now on every reply goes in this file, including
questions, status updates and final job reports, and chat will only point here.

- **CLAUDE.md** ("Response output convention"): the rule was already there. I made it
  explicit that it covers every reply, that this file holds the full reply rather than a
  summary, and that inside a git worktree it lives at the worktree root and gets committed.
- **Memory:** saved the same preference so it carries into future sessions.
- **Where this file is:** this session runs in the worktree, so this is
  `.claude/worktrees/rb-stat-vector/response.md` on branch `worktree-rb-stat-vector`. Edits
  to your main checkout are blocked for this session, so `Audible IQ/response.md` on main
  still has the older text until you merge or copy it.
- **The longer report** that was here before (full coefficient tables, sanity checks) is in
  commit `cdc20d3`: `git show cdc20d3:response.md`.

---

## Previous reply

I built the RB-only stat pipeline and compared it with the current formula on held-out
seasons: it roughly ties. The work is pushed to the new branch `worktree-rb-stat-vector`.
The whole run completes and all the data checks pass, but there are no automated tests.

**How I resolved the two prompts:** Prompt 1's sack fix and `epa_allowed` stayed, since
Prompt 2 depends on them. Its fantasy-points regression for all four positions was dropped,
because Prompt 2 forbids predicting points directly and forbids QB/WR/TE.
`opponent_skew.py`, `engine/expected_points.py`, `engine/scoring.py` and `engine/metrics.py`
are unchanged.

**What doesn't match the prompts:**
- **Not "already built":** Prompt 2 said `epa_allowed` and the delta features existed. They
  didn't, so I built them.
- **Effective-n shrinkage:** the fix Prompt 1 refers to isn't in the code. I defined it for
  `epa_allowed` only, so prior-season games count as partial evidence.
- **Extra variant:** plain OLS under-projects RB points by 0.2–0.5 per game. The rate models
  give a 1-carry game the same weight as a 25-carry game. I added a version weighted by
  attempts, reported separately; plain OLS is still the main result.
- **Your uncommitted work:** the B-1..B-4 fixes were uncommitted on main, so I copied them
  into the branch as the first commit (`ea60f13`). Your own checkout wasn't touched.

**Pass filter:** it now excludes sacks, which gives 17,839 pass attempts for REG 2024
against 17,811 official. The +28 is exactly 99 two-point tries minus 71 spikes. The
370-carry rushing gap is also fully explained: 405 kneels minus 36 two-point runs. I left
both gaps as accepted.

**Results (window n=8, same rows for every method):**

| Method | Train 2024 → test 2025 (MAE / R² / Spearman) | Train 2025 → test 2024 (MAE / R² / Spearman) |
|---|---|---|
| New stat-based model | **4.430** / 0.425 / **0.743** | 4.444 / 0.413 / 0.721 |
| Current rolling average + skew | 4.513 / 0.407 / 0.732 | 4.396 / 0.434 / 0.727 |
| Rolling average only | 4.441 / 0.408 / 0.735 | **4.373** / **0.436** / 0.728 |

- **Portability is free, but there's no accuracy gain yet:** the new model wins one
  direction and loses the other. The 8-game window was best of 4, 6 and 8.
- **Almost all the signal is in carries and targets:** recent usage explains about 57% of
  the variance in carries. Per-game efficiency barely carries over; the five rate models
  each explain 1.5% or less, and every VIF is about 1.
- **Matchup effects:** opponent rush-EPA on carries flips sign between the two fits (+15.1
  vs −1.0). The only matchup effect significant in both directions is weaker run defenses
  raising yards per carry (+3.7 and +5.7), in the attempt-weighted fit.
- **Leaving out 2-point conversions, first downs, fumbles and yardage bonuses costs about
  0.11 points per game:** they now score 0. That's logged in `OPEN_QUESTIONS.md` under Q-8,
  with a new Q-11 (promotion) and Q-12 (count reconciliation).

**Decisions for you:**
1. Promote, iterate or park it? Promoting adds fitted coefficients, which goes against the
   "no fitted model" rule in `CLAUDE.md`.
2. Plain or attempt-weighted rate models? Weighted removes the bias but makes MAE about
   0.055 worse.
3. QB/WR/TE only after those two.

To reproduce: `python rb_stat_vector_eval.py` from the worktree.
