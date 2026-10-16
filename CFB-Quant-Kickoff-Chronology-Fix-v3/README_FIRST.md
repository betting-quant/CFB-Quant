# CFB-Quant Kickoff Chronology Fix v3

This archive includes a replacement **for exactly one existing Python file**, plus a source-hash checking installer and four regression tests. Nothing is installed automatically. Your original data/model artifacts are unchanged by the installer.

1. Back up your whole CFB-Quant folder. Make sure VS Code files are saved.
2. Extract this ZIP to your **CFB-Quant folder**. It will create a `CFB-Quant-Kickoff-Chronology-Fix-v3` subfolder.
3. Verify `reports/spread_total_schedule_chronology_v2/team_game_kickoff_map.csv` exists.
4. In a VS Code terminal at CFB-Quant root, run `python CFB-Quant-Kickoff-Chronology-Fix-v3/install_chronology_fix_v3.py --check`.
5. Only if PASS: run `python CFB-Quant-Kickoff-Chronology-Fix-v3/install_chronology_fix_v3.py --apply`.
6. Run `python -m pytest -q CFB-Quant-Kickoff-Chronology-Fix-v3/tests/test_kickoff_chronology_fix_v3.py`.

The installer refuses to overwrite a different version of `src/cfb_quant/features/matchup_engine.py`. It creates a backup named `matchup_engine.py.before_kickoff_fix.bak` when applying.

**Do not rebuild or overwrite original feature / prediction parquet files yet.** We will first check the tests, then run the updated feature builder with new output paths, verify zero ordering inversions, and retrain on those outputs. The verified kickoff mapping fixes ordering, but original CFBD source snapshots are still not point-in-time validated, so betting profitability remains unproven.
