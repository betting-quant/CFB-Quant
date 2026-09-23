# CFB QUANT MODEL CHECKPOINT
## 2026-09-23

Repository:
CFB-Quant

Current architecture has been extensively leakage-safe backtested.

---

# CURRENT MODEL STATE

## 1. Market-Blind Matchup Engine

Historical leakage-safe team/game feature engine:

- 20,738 team-game rows
- 10,369 games
- Seasons 2014-2025
- 267 leakage-safe football features

Current production architecture originally predicts:

- points
- pass attempts
- rush attempts
- total plays
- passing yards
- rushing yards
- possession seconds

Historical chronological OOF predictions were successfully generated.

---

# 2. GAME-LEVEL CALIBRATION

Historical OOF game calibration completed.

Development:
2017-2024

Temporal test:
2025

Residual calibration artifact created.

Main conclusion:

Early-season spread uncertainty is materially larger than later-season
spread uncertainty.

Raw residual probabilities must NOT be interpreted as sportsbook betting
probabilities without market testing.

---

# 3. HISTORICAL SPORTSBOOK DATA

CFBD historical betting-line data downloaded for:

2017-2025

Historical providers include:

- consensus
- Bovada
- ESPN Bet
- DraftKings
- Caesars
- William Hill
- others

Historical canonical sportsbook market was constructed.

Prediction sites such as TeamRankings and NumberFire were excluded from
the sportsbook composite.

---

# 4. RAW PYTHON VS MARKET BACKTEST

The original market-blind Python fair spread did NOT produce a stable
ATS betting edge.

Important conclusion:

DO NOT use raw Python spread disagreement as an automatic ATS betting
signal.

Large Python-vs-market spread disagreement was not reliably predictive.

Spread model remains useful as:

- independent football projection
- matchup context
- game environment
- player-prop handoff input

but NOT as a standalone ATS bet generator.

---

# 5. DIRECT GAME MODELS

Direct margin and total models were tested.

Margin development winner:
Extra Trees

Direct Extra Trees margin was slightly better than the original
derived-points margin on football prediction accuracy.

However, it still did NOT reliably beat sportsbook spread prices.

Therefore:

Direct Extra Trees margin = useful fair-line challenger
NOT automatic ATS betting model.

For totals, the existing derived-points Ridge approach remained the
preferred development baseline.

---

# 6. OPENING-LINE + CLV TEST

Spread models were tested against opening lines.

Conclusion:

Spread Python signals did not consistently generate:

- positive ATS ROI
- positive CLV
- reliable open-to-close prediction

SPREAD BETTING SIGNAL IS FROZEN / DISABLED.

Do not continue threshold mining spreads with this architecture.

---

# 7. TOTAL OPENING-LINE TEST

Raw total disagreement showed some historical Under performance and
positive CLV.

However:

ATS performance was unstable in 2025.

Therefore:

Raw Python total edge is NOT currently an automatic betting trigger.

Interesting finding:

The total model appeared more useful for predicting MARKET MOVEMENT than
for directly predicting game-total betting outcomes.

---

# 8. MARKET RESIDUAL MODEL

Built:

src/cfb_quant/models/market_residual_tournament.py

The model attempted to predict sportsbook opening-line pricing error
using:

- football features
- independent Python projection
- opening sportsbook market
- Python/market disagreement
- provider information

Results:

It did NOT beat the sportsbook opening number on final-game outcome
accuracy.

Conclusion:

Do not use market-residual outcome model as betting engine.

---

# 9. MARKET MOVEMENT MODEL

Built:

src/cfb_quant/models/market_movement_tournament.py

Targets:

SPREAD:
closing home margin - opening home margin

TOTAL:
closing total - opening total

Closing lines are TARGETS ONLY.

Closing lines are NOT included as model inputs.

---

# 10. SPREAD MOVEMENT RESULT

Early-season Extra Trees spread movement signal was inconsistent.

Development Weeks 1-4:

- MAE gain slightly negative
- RMSE gain slightly positive
- modest movement correlation

2025 improved, but prior years were inconsistent.

Conclusion:

DO NOT promote spread movement model.

Spread remains informational only.

---

# 11. TOTAL MOVEMENT RESULT — FIRST PROMISING GAME SIGNAL

Extra Trees TOTAL open-to-close movement is the first game-level signal
worth advancing.

Development 2022-2024, Weeks 1-4:

Games:
909

Zero-move baseline MAE:
1.8721

Extra Trees MAE:
1.8384

MAE gain:
+0.0337

Zero-move baseline RMSE:
2.4824

Extra Trees RMSE:
2.4231

RMSE gain:
+0.0593

Movement correlation:
0.1934

Direction accuracy for >= 0.5 point moves:
58.85%

Direction accuracy for >= 1 point moves:
60.25%

---

# 12. 2025 TEMPORAL TEST — TOTAL MOVEMENT

Games:
357

Zero-move MAE:
1.6618

Extra Trees MAE:
1.6183

MAE gain:
+0.0435

Zero-move RMSE:
2.1837

Extra Trees RMSE:
2.0893

RMSE gain:
+0.0944

Movement correlation:
0.3022

Direction accuracy >= 0.5 movement:
59.04%

Direction accuracy >= 1 point movement:
62.66%

---

# 13. TOTAL MOVEMENT YEAR-BY-YEAR

Extra Trees produced positive total-movement MAE/RMSE information in:

2022
2023
2024
2025

This is currently the strongest validated game-level use case.

---

# CURRENT PRODUCTION DECISIONS

SPREAD FAIR LINE
Keep as independent information.

SPREAD ATS BETTING
DISABLED.

SPREAD MOVEMENT MODEL
NOT PROMOTED.

RAW TOTAL BETTING EDGE
NOT PROMOTED.

MARKET RESIDUAL OUTCOME MODEL
NOT PROMOTED.

TOTAL OPEN -> CLOSE MOVEMENT MODEL
PROMISING / ACTIVE VALIDATION CANDIDATE.

Model:
Extra Trees

Use case:
Early-season CFB total line movement.

---

# EXACT NEXT STEP

DO NOT retrain production yet.

Next test is:

CFB TOTAL MOVEMENT SIGNAL CALIBRATION v0.1

Purpose:

Falsify whether Extra Trees is actually learning market movement or
merely exploiting the historical tendency for college football totals
to move downward.

The next audit must compare:

1. Extra Trees movement signal
2. Naive always-Under strategy

Required checks:

- model average CLV
- always-Under average CLV
- model CLV advantage vs always-Under
- positive CLV rate
- direction accuracy for >=0.5 moves
- direction accuracy for >=1.0 moves
- Pearson movement correlation
- Spearman movement correlation
- predicted movement magnitude thresholds
- predicted Over performance
- predicted Under performance
- year-by-year stability
- 2025 temporal validation

Important thresholds to test:

0.00
0.10
0.20
0.25
0.35
0.50
0.75
1.00
1.25
1.50 predicted movement points

Also test predicted OVER and UNDER directions separately.

The critical question:

Does Extra Trees outperform simply betting early Unders?

If YES:
Promote Total Movement model to production and create a 2026 Week 4
live total-movement board:

current/open total
-> predicted closing total
-> expected CLV
-> BET NOW / WAIT / PASS

If NO:
Stop trying to manufacture game-level betting signals from this
architecture and retain sportsbook game lines as the baseline.

---

# IMPORTANT EXISTING OUTPUTS

reports/cfb_game_oof_calibration_v0_1.csv

reports/cfb_game_probability_curve_v0_1.csv

reports/cfb_game_interval_validation_v0_1.csv

reports/cfb_historical_market_composite_v0_1.csv

reports/cfb_game_market_backtest_v0_1.csv

reports/cfb_game_market_backtest_summary_v0_1.csv

reports/cfb_game_market_robustness_v0_2.csv

reports/cfb_market_incremental_value_v0_3.csv

reports/cfb_direct_game_model_oof_v0_1.parquet

reports/cfb_direct_game_model_summary_v0_1.csv

reports/cfb_direct_margin_market_test_v0_1.csv

reports/cfb_direct_margin_market_summary_v0_1.csv

reports/cfb_historical_market_open_close_v0_2.csv

reports/cfb_total_opening_line_clv_audit_v0_1.csv

reports/cfb_total_opening_line_clv_summary_v0_1.csv

reports/cfb_market_residual_oof_v0_1.parquet

reports/cfb_market_residual_summary_v0_1.csv

reports/cfb_market_movement_oof_v0_1.parquet

reports/cfb_market_movement_summary_v0_1.csv

---

# IMPORTANT SOURCE MODULES

src/cfb_quant/models/game_direct_tournament.py

src/cfb_quant/models/market_residual_tournament.py

src/cfb_quant/models/market_movement_tournament.py

---

# SECURITY

CFBD API key must remain an environment variable.

DO NOT commit API keys, .env secrets, or authentication tokens.

Windows environment variable:

CFBD_API_KEY

---

# RESUME INSTRUCTION

After pulling this repository onto the laptop:

1. Activate .venv.
2. Ensure CFBD_API_KEY exists as a User environment variable.
3. Read this checkpoint.
4. Continue with TOTAL MOVEMENT SIGNAL CALIBRATION v0.1.
5. Do NOT redo previous threshold/backtest work.
