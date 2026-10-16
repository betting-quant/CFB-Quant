# Source-backed AI matchup audit — spreads/totals only

**Do not provide projected spreads, totals, API model predictions, or market odds to the AI auditor before it finishes its blind football research.** All numerical scenario adjustments must be isolated, not automatically added to model probabilities.

For a specified matchup and kickoff cutoff, use ONLY verifiable sources published at or before `as_of_utc`. Return:

| Field | Requirement |
|---|---|
| Game | Team names, location, kickoff, source |
| Starting QB | Name, status, source, published/updated timestamp |
| OL and DL | Known absences, verified starters, opposing tactical effects |
| Secondary and skill players | Known absences, role changes |
| Scheme matchup | How team A's run/pass concepts interact with team B's defense, and vice versa |
| Pace and plays | Evidence-supported tempo estimates; note uncertainty |
| Scoring tails | Explosive plays, pick-six risk, turnovers, red zone, defensive touchdowns, late-game scoring |
| Weather | Forecast *issued before kickoff*; source and timestamp |
| Strength of schedule | Recent comparisons against actual opponents (no stale rankings) |
| Counterargument | Most plausible reason the expected game script fails |

Each finding must have `status` (`verified` / `disputed` / `unknown`), `source_url`, `source_published_utc`, and `football_effect`. If unknown, write UNKNOWN, not a guess.

Do not assign arbitrary numeric point swings or claim an 85% cover probability. Give signed research conclusions suitable for review against model assumptions. Produce a separate **scenario register** identifying variables that could matter if externally quantified: QB change, pace shift, weather, missing offensive linemen, coverage matchup, explosive variance.

**Audit complete only when all significant findings have source timestamps and the main red-team counterargument is documented.**
