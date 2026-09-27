# TE and QB stat vectors

**Summary:** **TE switches to the stat vector. QB narrowly misses the bar and stays on
`rolling_avg_prior + opponent_skew`.**

- **Branch:** `worktree-qb-te-stat-vector` (commit `6fb4ed2`, pushed, **not merged**).
- **Base:** it's built on `worktree-wr-stat-vector-tuned`, which also isn't merged yet, so it
  includes the WR switch. Main has the RB merge (`fac0d4b`).
- **Registry now:** RB, WR and TE use `stat_vector`; QB uses `rolling_plus_skew`.

## What changed to support QB/TE
- **Specs:**
  - `stat_vector/te.py`
  - `stat_vector/qb.py`
  - `specs.py` registers all four.
- **`StatVectorSpec`** now carries:
  - its own feature `window`;
  - a `derivations` chain that turns predicted volumes × rates into the stat line. RB, WR
    and TE share `rush_receive_derivations`; QB has `PASS_RUSH_DERIVATIONS`.
- **QB derivation:**
  - completions = attempts × completion rate
  - incompletions = attempts − completions
  - passing yards / TDs = completions × rate
  - interceptions = attempts × INT rate
  - rushing yards / TDs = rush attempts × rate
  - `calculate_points_vectorized` applies the league's negative `pass_int` weight
    unchanged.
- **`stat_rolling.py`** adds pass-attempt volume and the four passing rates
  (completion_rate, yards_per_completion, pass_td_rate, int_rate).
- **Rushing:** QB rush attempts use the existing `play_type == 'run'` counting, so scrambles
  are included. That's unchanged.
- **RB and WR are unaffected by the refactor:** the full-engine backtest reproduces RB
  4.717 / 0.378 / 0.686 and WR 4.690 / 0.344 / 0.651 exactly.

## Selection policy (now written into CLAUDE.md)
- **Ranking first:** start/sit is a pairwise ranking decision, so choices go by mean LOSO
  **Spearman, then R²**. MAE is a guardrail, and bias is always reported.
- **Switching a position:** requires Spearman and R² comparable-or-better at
  comparable-or-better MAE. A strict MAE win isn't needed.
- **Parameter grids:** `select_by_policy` ranks by Spearman, then R² (ties within 0.001),
  then MAE.
- **Flag:** the earlier rolling-window grid (QB 12 / WR 10 / TE 12) was chosen on MAE,
  before this rule existed. Under this rule it would pick QB 14, WR 14, TE 12. I noted that
  in CLAUDE.md but didn't re-select.

## Window grid for each position's own stat vector
LOSO means over the 7 full seasons for the new model. Each cell is MAE / R² / Spearman.

| Window | TE | QB |
|---:|---|---|
| 8 | 4.244 / 0.322 / 0.589 | 7.928 / 0.237 / 0.481 |
| 10 | **4.236 / 0.325 / 0.590** | 7.911 / 0.239 / 0.480 |
| 12 | 4.246 / 0.325 / 0.588 | **7.911 / 0.238 / 0.483** |
| 14 | 4.249 / 0.324 / 0.588 | 7.928 / 0.235 / 0.482 |
| 16 | 4.259 / 0.322 / 0.585 | 7.942 / 0.233 / 0.481 |
| 20 | 4.275 / 0.318 / 0.582 | 7.963 / 0.228 / 0.479 |

The policy chose **TE 10** and **QB 12**. Both curves are flat, within about 0.01 Spearman.

## TE: switches
Each cell is MAE / R² / Spearman.

| Test season | TE stat vector | Current formula (w=12) | Rolling alone |
|---|---|---|---|
| 2019 | **4.292 / 0.302 / 0.547** | 4.401 / 0.251 / 0.531 | 4.343 / 0.261 / 0.543 |
| 2020 | **4.539 / 0.283 / 0.544** | 4.633 / 0.256 / 0.536 | 4.580 / 0.267 / 0.542 |
| 2021 | **4.186 / 0.332 / 0.541** | 4.247 / 0.314 / 0.530 | 4.225 / 0.315 / 0.530 |
| 2022 | **4.156 / 0.318 / 0.606** | 4.248 / 0.284 / 0.564 | 4.226 / 0.290 / 0.578 |
| 2023 | **4.084 / 0.369 / 0.639** | 4.179 / 0.326 / 0.612 | 4.180 / 0.334 / 0.617 |
| 2024 | 4.256 / **0.336** / **0.632** | 4.274 / 0.320 / 0.610 | **4.254** / 0.326 / 0.620 |
| 2025 | **4.117 / 0.341 / 0.621** | 4.244 / 0.316 / 0.606 | 4.197 / 0.314 / 0.611 |
| **Mean** | **4.233 / 0.326 / 0.590** | 4.318 / 0.295 / 0.570 | 4.286 / 0.301 / 0.577 |
| 2026 wk 1–4 (189 rows) | **4.539** / 0.246 / 0.561 | 4.587 / **0.260** / **0.576** | 4.573 / 0.242 / 0.564 |

- **Folds won:** all three metrics in 7 of 7 against the current formula. Against
  rolling-alone, R² and Spearman in 7 of 7, MAE in 6 of 7.
- **Bias:** +0.02.
- **Clears the bar easily. `"TE": "stat_vector"` is flipped.**
- **Early season:** the partial 2026 fold is mixed; better MAE, slightly lower R² and
  Spearman on 189 rows.

### TE coefficient stability (7 folds)

| Model | Term | Range | Positive in | p < .05 in |
|---|---|---|---:|---:|
| targets | trailing avg | 0.870–0.885 | 7 | 7 |
| targets | delta | 0.231–0.281 | 7 | 7 |
| carries | trailing avg | 0.91–0.97 | 7 | 7 |
| catch_rate | trailing rate | 0.133–0.172 | 7 | 7 |
| catch_rate | epa_allowed_pass | +0.083 to +0.178 | 7 | 5 |
| ypr | trailing rate | 0.223–0.261 | 7 | 7 |
| ypr | **epa_allowed_pass** | **+3.68 to +5.74** | 7 | **7** |
| rec_td_rate | trailing rate | 0.098–0.136 | 7 | 7 |
| rec_td_rate | **epa_allowed_pass** | **+0.101 to +0.126** | 7 | **7** |
| ypc / rush_td_rate | all terms | noise (TE carries are tiny) | – | ≤ 2 |

- **Matchup terms re-tested on TE, not inherited from WR:**
  - catch rate EPA is positive in 7 of 7 folds and significant in 5 of 7, so it stays ON;
  - **yards-per-reception and TD-rate EPA clear significance in 7 of 7 folds, so they're
    ON for TE**, unlike WR, where neither cleared.
- **Targets volume EPA, re-tested:** +0.81 to +1.10, positive in 7 of 7 but significant in
  only 3 of 7 (p 0.03–0.13), so it stays a candidate (off). Turning it on doesn't change the
  means (4.233 / 0.326 / 0.590 either way).

## QB: evaluated, stays on rolling + skew
Each cell is MAE / R² / Spearman.

| Test season | QB stat vector | Current formula (w=12) | Rolling alone |
|---|---|---|---|
| 2019 | **8.063 / 0.191** / 0.424 | 8.166 / 0.156 / 0.424 | 8.197 / 0.146 / 0.396 |
| 2020 | 8.129 / 0.275 / 0.519 | 7.869 / 0.299 / 0.542 | **7.825 / 0.304 / 0.543** |
| 2021 | 8.089 / 0.247 / 0.498 | 7.938 / 0.237 / **0.528** | **7.850 / 0.251** / 0.524 |
| 2022 | **7.257 / 0.258** / 0.502 | 7.324 / 0.234 / **0.518** | 7.358 / 0.227 / 0.509 |
| 2023 | 7.641 / **0.226** / 0.487 | 7.623 / 0.218 / **0.492** | **7.571** / 0.220 / 0.485 |
| 2024 | 7.917 / **0.260 / 0.517** | 7.929 / 0.238 / 0.496 | **7.874** / 0.244 / 0.498 |
| 2025 | **8.279 / 0.207 / 0.437** | 8.416 / 0.155 / 0.412 | 8.410 / 0.152 / 0.405 |
| **Mean** | 7.911 / **0.238** / 0.483 | 7.895 / 0.220 / **0.487** | **7.869** / 0.221 / 0.480 |
| 2026 wk 1–4 (96 rows) | 8.575 / 0.152 / 0.305 | **8.284** / 0.162 / **0.317** | 8.291 / **0.165** / 0.316 |

- **Against the current formula:** R² wins 6 of 7 folds (+0.018 mean), and bias is much
  smaller (−0.10 vs −0.31).
- **Against the primary metric:** Spearman is slightly worse (0.483 vs 0.487, winning only
  2 of 7 folds). MAE is +0.016, and the early-season fold is worse.
- **Verdict:** under the Spearman-first rule it doesn't clear, so QB isn't switched. It's
  close, and logged as Q-14.

### QB coefficient stability (7 folds)

| Model | Term | Range | Positive in | p < .05 in |
|---|---|---|---:|---:|
| attempts | trailing avg | 0.733–0.750 | 7 | 7 |
| attempts | delta | 0.452–0.491 | 7 | 7 |
| carries | trailing avg | 0.824–0.859 | 7 | 7 |
| carries | delta | 0.149–0.196 | 7 | 7 |
| completion_rate | trailing rate | 0.281–0.330 | 7 | 7 |
| completion_rate | epa_allowed_pass | +0.161 to +0.233 | 7 | 7 |
| yards_per_completion | trailing rate | 0.326–0.363 | 7 | 7 |
| yards_per_completion | epa_allowed_pass | +1.15 to +3.56 | 7 | 6 |
| pass_td_rate | trailing rate | 0.242–0.273 | 7 | 7 |
| pass_td_rate | epa_allowed_pass | +0.035 to +0.063 | 7 | 6 |
| int_rate | trailing rate | 0.117–0.176 | 7 | 7 |
| **int_rate** | **epa_allowed_pass** | **−0.027 to −0.034** | **0** | **7** |
| ypc | trailing rate | 0.543–0.574 | 7 | 7 |
| ypc | epa_allowed_rush | +0.80 to +2.85 | 7 | 0 |
| rush_td_rate | trailing rate | 0.152–0.262 | 7 | 7 |
| rush_td_rate | epa_allowed_rush | +0.04 to +0.11 | 7 | 1 |

- **The int_rate sign is negative in every fold and significant in every fold.** That's the
  expected, correct direction: a softer pass defense forces fewer interceptions. It's
  flagged in CLAUDE.md as "don't fix".
- **Volume-model EPA, re-tested, doesn't hold for QB either:**
  - pass attempts: −0.17 to +4.28, p ≥ 0.25 in every fold
  - rush attempts: +0.69 to +1.98, p ≥ 0.09
- **Rushing EPA terms:** positive but not significant.
- **Candidate refinements (Q-14):**
  - **Scrambles vs designed runs.** The combined rush model isn't obviously weak (trailing
    YPC 0.55, TD rate stable). But in one 2025 week-10 sample the stat vector's top QBs were
    Herbert and Nix, where rolling had Jackson and Allen, which hints that rushing QBs may be
    under-projected. That's unverified; I logged it and didn't split anything.
  - **Missing categories.** QBs lose more to unpredicted categories than other positions.
    Recomposing actual stats without fumbles and 2-point conversions costs mean |gap| 0.47
    points/game in this league, against 0.04–0.11 for RB/WR/TE. Logged under Q-8.

## Confidence calibration recheck (all four positions, final registry)
MAE by tier over pooled held-out predictions, 2019–2025:

| Pos | high | medium | low | insufficient | monotonic? |
|---|---:|---:|---:|---:|---|
| QB | 7.857 | 7.756 | 7.923 | 8.292 | No |
| RB | 5.093 | 4.507 | 4.430 | 4.543 | No |
| WR | 4.926 | 4.561 | 4.450 | 4.678 | No |
| TE | 4.690 | 4.106 | 3.881 | 3.899 | No |
| All | 5.232 | 4.773 | 4.748 | 4.978 | No |

- **Still inverted after B-2 and the continuous-window fix.** The "high" tier has the
  largest error at RB, WR and TE.
- **Likely cause:** error scale, not wrong tiers. High-tier rows are regulars with bigger
  point totals, so their absolute errors are bigger. MAE-by-tier can't validate a
  games-played tier.
- **Logged as Q-15,** with the fix path: a scale-free calibration metric or a redefined
  confidence. **The agent layer shouldn't rely on the current tiers as-is.**

## Tool explanation check
I ran real engine rows through `tools/expected_points.py`:
- **QB (rolling):** "projection = rolling_avg_prior (30.12, last 12 games) + opponent_skew
  (-0.44) vs MIN." (Lamar Jackson, 2025 wk 10)
- **TE (stat vector):** "projection = this league's scoring applied to a predicted stat line
  (stat vector: predicted volume — pass attempts, targets, carries — from recent usage, times
  per-attempt rates adjusted for SEA's EPA allowed). opponent_skew is not part of this
  projection." (Trey McBride)

One wording change: the stat-vector text used to say "predicted carries/targets", which
would read wrong for a QB. It now says "predicted volume — pass attempts, targets, carries".
Everything else reads correctly, including QB rows run through the stat vector.

## Other notes
- **nflreadpy added 2026 week 4** since the last run, so the partial fold is now weeks 1–4
  (RB 226 rows, was 175). Partial-fold numbers aren't comparable to earlier reports.
- **Docs updated:**
  - **CLAUDE.md:** selection policy, TE/QB specs, int_rate sign note, registry.
  - **README:** TE and QB leave-one-season-out tables, production table, calibration.
  - **OPEN_QUESTIONS:**
    - Q-13 resolved (no production position carries the continuous-window cost any more);
    - Q-4 narrowed (skew now only affects QB);
    - new Q-14 (QB) and Q-15 (calibration);
    - Q-8 extended for QB.
  - **STATUS:** updated.

## For review (nothing merged)
1. **Merge order:** `worktree-wr-stat-vector-tuned` first, then `worktree-qb-te-stat-vector`.
   Both are fast-forwards from main (`fac0d4b`).
2. **QB:** accept "stays on rolling + skew", or tell me to try the scramble split or
   fumble/2-point prediction first.
3. **Q-15:** confidence tiers need a scale-free redefinition before the agent layer uses
   them.
4. **Rolling windows:** optionally re-select them under the new Spearman-first policy (QB 14,
   WR 14, TE 12 instead of 12/10/12). Only QB's matters now, since only QB uses the rolling
   formula in production.
