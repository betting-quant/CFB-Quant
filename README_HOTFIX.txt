CFB-Quant spread/total pairing hotfix.

Corrects validation failure when original _target_mask excludes one team for a game.
This patch: (1) audits all OOF game pairs to reports/spread_total_pair_audit_v1.csv;
(2) excludes only incomplete one-sided games; (3) rejects duplicate or inconsistent
metadata; (4) aborts if under 70% of games have both predictions.

After backing up original core.py and cli.py, extract ZIP into CFB-Quant project
root. Your existing OOF parquet is not touched and does NOT need regeneration.

python -m pytest -q tests/test_spread_total_upgrade.py tests/test_pairing_hotfix.py
python -m cfb_quant.spread_total.cli validate --start-year 2019 --samples 5000

Do not interpret historical bootstrap probabilities as betting-ready EV until
independent point-in-time feature and outcome-distribution calibration audits.
