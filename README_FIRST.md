# CFB-Quant v5.7 historical training year hotfix

Fixes the error `ValueError: Historical training must be exclusively 2014-2025` when a historical feature Parquet contains extra-season rows. Training now **selects only actual scored rows from 2014–2025**, verifies that all 12 training seasons are present, and explicitly reports any discarded rows. Rows dated 2026 and later are NEVER fit to the model. All snapshot protections remain in force.

## Install

```powershell
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\CFB-Quant-v57-Training-Year-Hotfix.zip" -DestinationPath . -Force
python -m pytest -q tests/test_predict_v57.py
python -m cfb_quant.live.predict_v57 --snapshot "reports/live_feature_bridge_v56/2026-10-10_20261010T083748Z" --samples 10000
```

This patch replaces only `src/cfb_quant/live/predict_v57.py` and its tests. Does not edit Parquet source data, historical manifests, snapshots, API data, or other model modules.

If a different validation fails, do not bypass it. Paste full error output so the source data can be diagnosed.
