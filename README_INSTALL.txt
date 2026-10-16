CFB-Quant — Spread/Total Validation Diagnostics v2

PURPOSE: Read-only analysis of the 5,990-row historical validation CSV.
No CFBD key, new API calls, model training, overwrite of your forecasts,
or betting automation required. This is an audit, not a new trained model.

INSTALL:
1. Back up your original CFB-Quant folder.
2. Extract this ZIP into the ROOT of the existing CFB-Quant project.
   The new module must be at src/cfb_quant/spread_total/diagnostics.py
3. In your active VS Code PowerShell virtual environment, run:

   python -m pytest -q tests/test_spread_total_diagnostics.py
   python -m cfb_quant.spread_total.diagnostics

4. The terminal displays the all-season weighted metrics.
5. Output folder: reports/spread_total_diagnostics_v2/
   - validation_overall_v2.csv
   - validation_by_season_v2.csv
   - validation_by_total_v2.csv
   - validation_by_mismatch_v2.csv
   - validation_by_season_period_v2.csv
   - validation_home_win_calibration_v2.csv

Upload these small reports, especially by_total and by_mismatch.

Interpretation:
- The reported 80% prediction intervals should cover ~80% of outcomes.
- Lower/upper tail frequencies should be near 10% each overall.
- Average error by projected-total and mismatch groups can reveal
  weaknesses hidden by the overall mean.
- Home win reliability is only a diagnostic of joint score distributions;
  it is NOT a substitute for calibration of betting spread and total lines.
- The climatology Brier baseline is not a sportsbook baseline.
- Full data leakage, FBS opponent verification, odds benchmarking,
  personnel adjustments, and unseen prospective validation remain to do.

SAFETY: Does not touch original .env, saved models, or processed parquet.
