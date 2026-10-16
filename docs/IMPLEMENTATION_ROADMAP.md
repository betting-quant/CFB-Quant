# CFB-Quant: staged plan for the spreads/totals engine

## Stage 0: preservations, code audit — COMPLETE for uploaded code-only archive

- Back up existing project; don't discard player-prop files, only defer them.
- Preserve model comparison, historical feature engineering, team live report logic.
- Current limitations remain: saved data, production artifacts, and market history were not provided, so numerical accuracy, EV and profitability cannot be asserted.

## Stage 1: chronological point OOF + bootstrap distribution — ADD-ON DELIVERED

- Generate Ridge point forecasts fit strictly through prior seasons.
- Pair by game and preserve residual home/away dependence.
- Fit on residual data only before forecast season.
- Produce range estimates, game-level Brier / interval coverage and CRPS.
- Push-aware settlement, EV arithmetic, price and game time guards.
- Validate with actual locally saved historical data (not yet done by assistant).

**Gate:** Analyze generated validation CSV and compare vs training-only baselines. If marginal gains disappear, fix model before new complexity.

## Stage 2: data integrity + opponent adjustment — TO BUILD

- Audit original CFBD ingestion source files, timestamp provenance, FBS-only filters and team IDs.
- Enforce `game_start_utc`, `retrieved_at_utc`, and `as_of_utc` everywhere; block same-week future-information leaks.
- Add point-in-time verified play-by-play features: EPA/PPA per play (including pass vs run), early/passing downs, success rate, standard-down and passing-down performance, explosive-play rates, drive efficiency, field position, red zone, takeaways and possessions.
- Create strength-of-schedule / opponent adjustments learned ONLY from games known prior to that week's cutoff.
- Apply shrinkage for low-sample early-season teams; validate newly added groups via ablation.
- Make live production trainer date-aware, not hardcoded to 2026 Week 3. Save model provenance / hashes.

**Gate:** Predictive metrics improve relative to stage 1 and a market benchmark across multiple untouched later-date validation windows.

## Stage 3: realistic joint scoring simulation — TO BUILD

- Compare and test a conditional 2-team residual model and a pace/drive-score simulation.
- Explicitly handle NCAA overtime, correlated scoring, turnovers, defensive scoring and extreme tails.
- Calibrate probability and coverage separately for margins, totals and relevant alternate lines.
- Evaluate interval coverage by favorite size, total level, season, injury status and volatility class.
- Measure simulation stability (Monte Carlo precision does not imply model calibration).

**Gate:** Held-out CRPS, log/Brier scores, coverage and tail reliability beat simple baselines; don't overfit the historical tail.

## Stage 4: AI matchup audit — TO BUILD

- Independent, market-blind source-backed personnel/scheme/OL/DL, QB, coaching, weather investigation.
- As-of timestamped sources; verification status unknown/verified/disputed.
- Feed findings to numerical *scenario* inputs with externally supported effect sizes; never let AI overwrite the probability engine arbitrarily.
- Red-team most important alternate explanation.

**Gate:** Measured added value vs the non-AI model, using frozen pregame reports.

## Stage 5: real price benchmark and paper-trading — TO BUILD

- Timestamp bookmaker quotes and closing lines; log market availability and staleness.
- Evaluate model-vs-market forecasts and EV after vig, limits, pushes and realistic quote availability.
- Never tune staking or decision thresholds on the same seasons used to declare success.
- Keep live prospective paper-trading samples; record every PASS and REVIEW, not just winners.

**Gate:** Enough forward-tested data to assess whether any edge is statistically supported. Even then, there is no guarantee of profit or a daily income target.
