# CFB-Quant Phase Volatility v5.3 — Research Add-on

This add-on compares the existing **FBS-only hybrid residual distribution** against an **experimental phase-conditioned residual mixture**, using historical errors from **previous seasons only**. It addresses the v5.2 finding that early-season margin coverage was 75.5% against an 80% nominal target.

No Python scripts, model weights, API caches, historical forecasts, or v5.2 outputs are overwritten. The file is placed at `src/cfb_quant/spread_total/phase_volatility_v53.py`.

## Installation

1. Extract this ZIP **into your existing `CFB-Quant` project folder** (merge `src` and `tests`).
2. Run:

```powershell
python -m pytest -q tests/test_phase_volatility_v53.py
```

3. Run:

```powershell
python -m cfb_quant.spread_total.phase_volatility_v53 --samples 1200
```

4. Package these small reports for review:

```powershell
Compress-Archive -Path "reports\spread_total_phase_volatility_v53\overall.csv","reports\spread_total_phase_volatility_v53\by_season.csv","reports\spread_total_phase_volatility_v53\by_phase.csv","reports\spread_total_phase_volatility_v53\phase_pool_audit.csv","reports\spread_total_phase_volatility_v53\calibration_bins.csv" -DestinationPath "CFB-v53-Results.zip" -Force
```

Upload `CFB-v53-Results.zip` into the conversation.

## How it works

For each game in 2020–2024, the existing hybrid model supplies its fair margin and total. The tool tests two uncertainty models:

- `pooled`: sample *paired* home/away residuals from all preceding FBS-versus-FBS seasons (current method).
- `phase_mixed`: 70% residuals from earlier FBS-versus-FBS games in the **same phase** (regular season weeks 1–4; regular weeks 5+; postseason) and 30% from the full prior-season residual pool. If fewer than 40 previous phase games are available, use pooled sampling instead.

The 70% split is a **preselected experiment**, not a statistically optimized weight. This is a distribution comparison, not a new team forecast. The model preserves home/away error correlation and never uses scores from the evaluation season in its residual history.

## How to interpret

Inspect `by_phase.csv` first: early-season **margin interval coverage**, margin CRPS and Brier scores matter more than the already-identical point MAE. Check that any improvement is not bought at the expense of much poorer later-season or postseason probabilities. Do **not** promote the variant based on these repeatedly examined development seasons; use a truly future locked evaluation later.

## Known limitations

- Both underlying prediction models can still have FCS games in their historical training data. This test filters FBS-vs-FBS **evaluation games and residuals**, not the original fitted scoring model.
- No timestamped sportsbook odds, model-driven EV validation, or personnel information are used.
- Upstream CFBD point-in-time availability has not been fully certified.
- A residual bootstrap with score truncation is not a fully realistic play-by-play or drive-scoring simulation.
