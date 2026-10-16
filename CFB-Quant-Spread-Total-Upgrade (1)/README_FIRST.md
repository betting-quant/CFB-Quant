# CFB-Quant Spread/Total Upgrade — installation

**This ZIP is a non-destructive ADD-ON for the EXISTING `CFB-Quant` project.** It does NOT replace your Python models, player-prop files, production artifacts, historical data, or `.env`. Do not start a second project.

## What this upgrade DOES

1. Trains **historical OOF team-point Ridge forecasts** using the same CFB-Quant point-model family and your existing matchup features, with an earlier-season-only training rule.
2. Pairs home and away teams by game; retains their historical prediction errors together (avoids treating their scoring as independent).
3. Creates bootstrap scoring distributions from **earlier-season residuals only**.
4. Evaluates historical totals, margins, interval coverage, Brier errors and empirical CRPS using strict season-blocked validation.
5. Produces research-only spread/total win/push/lose rates and American-odds EV arithmetic from timestamped CSV sportsbook quotes.
6. Requires an explicit as-of timestamp and refuses games already underway or market quotes obtained after the cutoff.
7. Provides 10 isolated unit tests and a source-verification AI scouting template.

## What it DOES NOT do yet

- It is **not a calibrated professional betting model** and doesn't demonstrate long-run betting profitability.
- This first probability distribution is an empirical joint residual bootstrap, NOT a calibrated possession-by-possession stochastic model. It may misestimate tails or simulate scores not possible under NCAA overtime rules. Do not treat extreme alt-line probabilities as reliable until validated.
- It does not yet model verified injury, QB/OL, weather, play-by-play EPA, drive scoring or opponent adjustments beyond those already present in your original matchups features.
- It does not fetch sportsbook markets or CFBD API data; your existing project feeds it.
- It does not retrain live 2026 production models; your original trainer contains a hardcoded 2026 Week-3 guard and must be upgraded separately.
- Historical OOF seasons do **not** automatically constitute an untouched model-selection evaluation. The existing project's 2025 holdout has already influenced prior model decisions.
- Pregame FBS-only filters, same-week sequencing, model-input point-in-time availability, quality of source data, model artifact lineage and executable odds are **not yet verified**.

## Windows / VS Code: exact first steps

### 1. Protect your existing project

Save unsaved VS Code editors. Back up `C:\Users\DC\Documents\Quant-Models\CFB-Quant` before making changes. Do **not** delete player-prop files; just stop running them.

### 2. Extract ZIP into your existing `CFB-Quant` root

In Windows File Explorer, open `CFB-Quant-Spread-Total-Upgrade.zip` and copy its `src`, `tests`, `docs`, and `inputs` folders **into the existing CFB-Quant folder**. Merge the folders. The new Python files live exclusively inside the NEW path `src/cfb_quant/spread_total/`, plus `tests/test_spread_total_upgrade.py`.

There is no need to copy your data into this ZIP or share any API key.

### 3. VS Code terminal in CFB-Quant

```powershell
python -m pip install -e .
python -m pytest -q tests
```

Expect **10 new tests** plus your existing tests. The addon passed 17 tests combined in the developer's isolated environment (7 original + 10 new). Your local environment and datasets can vary. Don't run unscoped `pytest` yet because your old root-level `test_fast_history_equivalence.py` references a missing historical backup script.

### 4. Check you have the original historical feature inputs

```powershell
Test-Path "data/processed/cfb_matchup_features_v0_1.parquet"
Test-Path "reports/cfb_matchup_feature_manifest_v0_1.json"
```

Both need to show `True`. If either is `False`, **stop and report which one**. We need to repair data-building / manifest inputs, not invent replacement data.

### 5. Generate historical team-points forecasts (NO API KEY NEEDED)

```powershell
python -m cfb_quant.spread_total.oof_points --first-year 2015 --last-year 2025
```

Default output: `data/processed/cfb_spread_total_points_oof_v1.parquet`. This may take time, because it fits a Ridge model for each prediction season. These are **team-point** forecasts only, not player props or seven other targets.

### 6. Evaluate spread/total uncertainty chronologically

```powershell
python -m cfb_quant.spread_total.cli validate --start-year 2019 --samples 5000
```

Output: `reports/spread_total_historical_validation_v1.csv`. You will see historical margin/total MAE, CRPS, 80% interval coverage, and Brier scores on pre-specified hypothetical markets. These are **prediction quality metrics, not ROI**.

**STOP here and send me your terminal output and the historical validation CSV.** Next we will audit scores, distributions, FBS inclusion, residual stability, calibration, and then upgrade the outdated 2026 live-training pipeline.

## Only after we repair the live training pipeline

Your pregame report will be in the form `reports/matchup_games_YYYY_week_N.csv`, with `game_id`, `start_date`, `home_team`, `away_team`, `home_points`, `away_points`. Verify the CSV was generated using a pregame-valid, current model.

Example for a FUTURE matchup (replace season, file path, and cutoff with your real values):

```powershell
python -m cfb_quant.spread_total.cli forecast --season 2026 --games "reports/matchup_games_2026_week_N.csv" --as-of "2026-10-10T08:00:00-06:00"
```

Optional sportsbook prices file (CSV columns: `game_id,market,selection,line,american_odds,bookmaker,quoted_at_utc`):

```powershell
python -m cfb_quant.spread_total.cli forecast --season 2026 --games "reports/matchup_games_2026_week_N.csv" --as-of "2026-10-10T08:00:00-06:00" --markets "inputs/your_actual_lines.csv"
```

For spreads, `selection` is `home` or `away`, and the `line` is the spread **for that selected team**, e.g. `away,+17.5`. For totals, `selection` is `under` or `over`, with line equal to the total points threshold. Odds are American, e.g. `-110`, and timestamps must be pregame and before the as-of cutoff. `inputs/sportsbook_lines_EXAMPLE.csv` contains **fictional** lines; do not use them as actual markets.

No wager is placed by this program.

## Roadmap

See `docs/IMPLEMENTATION_ROADMAP.md` for stage gates; see `docs/AI_MATCHUP_AUDIT_SPREAD_TOTAL.md` for the scouting protocol.
