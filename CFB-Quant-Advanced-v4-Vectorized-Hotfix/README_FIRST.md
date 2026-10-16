# CFBD Advanced v4 vectorized feature builder hotfix

Your 2014-2025 advanced JSON caches downloaded successfully. You do **not** need to fetch again.

This hotfix replaces the repeated `pandas.groupby(...).transform(lambda expanding...)` with vectorized group operations for the 144 advanced features. It leaves the API fetcher, raw data, original v3 baseline, and previous trained forecasts unchanged.

## Installation (PowerShell in your existing CFB-Quant project)

1. Back up the CFB-Quant project; extract this ZIP into the CFB-Quant root.
2. `python CFB-Quant-Advanced-v4-Vectorized-Hotfix/install_hotfix.py --check`
3. If PASS, `python CFB-Quant-Advanced-v4-Vectorized-Hotfix/install_hotfix.py --apply`
4. `python -m pytest -q tests/test_advanced_football_v4.py CFB-Quant-Advanced-v4-Vectorized-Hotfix/tests/test_advanced_v4_vectorized.py`
5. `python -m cfb_quant.spread_total.advanced_v4 build --max-year 2025`

The installer first makes `src/cfb_quant/spread_total/advanced_v4.py.before_vectorized_v4.bak`. It refuses to modify a mismatched source. The `build` operation does **not** overwrite any existing v4 output files. If interrupted builds left only partial artifacts, inspect them before removing or renaming.

Do not run the OOF training until the builder completes and the feature coverage report has been reviewed. Raw advanced API records are postgame and must never be used unshifted as model inputs. This hotfix changes implementation speed and preserves feature semantics; it does **not** prove data availability before kickoff or model profitability.
