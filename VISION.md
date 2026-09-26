# Audible IQ — V1 Vision

_Re-established 2026-09-24. Source: author's own statement of intent, lightly
organized, not reinterpreted. Anything under "Open decisions" is explicitly
NOT settled. The original text appeared truncated in two places (start and
the last bullet of the boom/bust list); those are marked below rather than
guessed at._

## Guiding principle

**#1 architectural principle is modularity.** The business logic should be
fully mutable. Input and return formats should be consistent throughout the
app: polars DataFrames, dicts, and JSON.

## The product

- A **NiceGUI web app**, deployable as a server-side web app or run on
  localhost.
- A **fantasy football decision engine**, highly customizable based on
  league settings.
- **Primary connector is Sleeper.** Users can configure league settings by
  hand if they don't use Sleeper, but the goal is to lean on the Sleeper API
  for minimal setup and full use of what it exposes.
- Loads **your roster** from Sleeper, as well as **leaguemates' rosters**
  (groundwork for a potential V2 trade engine).
- The Sleeper <-> nflreadpy identity bridge is already built.

## Engines

### Expected points
Primary goal is **explainability**. Willing to sacrifice perfect accuracy to
limit hidden biases that affect downstream dependencies.

### Boom/bust classifier (primary inference product)
A sophisticated classifier including:

1. **A boom/bust threshold definition** — possibly a set multiplier / decay
   module applied to expected points, bounded by minimums so that a player
   expected for 3 points can't "bust" and 4 points wouldn't count as a
   "boom".
2. **A probability of X+ points.** Based on P(boom) and the boom amount: if
   boom for a player is 18 points, P(18+) is the boom probability, and
   P(23+) would follow a probability-distribution model anchored on that
   boom threshold.
3. **Potentially P(X points | boom).** E.g. P(18+ | boom) = 1.00, with the
   higher thresholds following from the distribution.
   _[Original sentence cut off here: "...and the following would"]_

### Agent (next major V1 deliverable)
Uses the engines above to make sophisticated, explainable suggestions given
expected points, boom probability, past variance, and other stats that can
be computed on demand, derived from data pulled via nflreadpy.

---

## Open decisions (NOT settled — see OPEN_QUESTIONS.md)

**Q-1. Sleeper write strategy.** Sleeper's public API is read-only with no
auth (per Sleeper's own docs). Executing add/drop/start/sit needs an
explicit strategy: (a) **recommend-only** — user executes in Sleeper's app;
(b) **unofficial authenticated write** — real account-security, ToS, and
reliability risk; (c) **browser automation**. *Working assumption for V1:
recommend-only. (b) and (c) are deferred and named, not silently built
around.*

**Q-2. What "modularity / fully mutable business logic" means.** Working
reading: *swappable implementations behind a stable interface* — NOT
end-user-tunable model parameters. This must not contradict the existing
"no fitted parameters, hand-chosen and evaluated" stance in expected points
(see CLAUDE.md, projection formula / shrinkage `k`).

**Q-3. Boom/bust distributional scope.** P(X+ points) and P(X points | boom)
as full distributional outputs are a meaningfully larger lift than the
binary/ternary classifier originally scoped. Sized honestly in
STATUS.md's gap analysis; scope needs an explicit decision.

**Q-4 (design note, not a question).** A boom/bust threshold as a *bounded
multiplier/decay of expected points* is consistent with the additive-over-
multiplicative lesson from opponent_skew: multiplicative forms distort near
small baselines, and the floor bound here does the same job the additive
design did there.
