# CFB Matchup Model Tournament v0.1

## Protocol

- Feature count: 267
- One observation is one team-game.
- Model selection uses chronological walk-forward validation only.
- Validation folds:
  - train 2014-2020 -> validate 2021
  - train 2014-2021 -> validate 2022
  - train 2014-2022 -> validate 2023
  - train 2014-2023 -> validate 2024
- 2025 is excluded from all model selection.
- 2026 is excluded from validation and holdout evaluation.
- After winners are selected, each winner is refit on 2014-2024 and evaluated once on 2025.
- Only shifted pregame offense and opponent-defense history is used as model input.

## Winners from 2021-2024 validation

| Target | Winner | Validation MAE | Improvement vs recent baseline |
|---|---|---:|---:|
| points | ridge | 9.4799 | 20.20% |
| pass_attempts | extra_trees | 6.7377 | 8.84% |
| rush_attempts | hist_gradient_boosting | 6.4236 | 11.15% |
| total_plays | hist_gradient_boosting | 7.4272 | 13.00% |
| passing_yards | hist_gradient_boosting | 65.2236 | 10.34% |
| rushing_yards | ridge | 57.4233 | 14.73% |
| possession_seconds | hist_gradient_boosting | 209.9765 | 12.84% |

## 2025 untouched holdout

| Target | Winner | Rows | MAE | RMSE | Bias | Corr |
|---|---|---:|---:|---:|---:|---:|
| points | ridge | 1868 | 9.1532 | 11.5465 | -0.0396 | 0.6039 |
| pass_attempts | extra_trees | 1868 | 6.3907 | 7.9917 | 0.3688 | 0.4970 |
| rush_attempts | hist_gradient_boosting | 1868 | 6.1360 | 7.6619 | 0.0464 | 0.5070 |
| total_plays | hist_gradient_boosting | 1868 | 7.2476 | 8.9972 | 0.0301 | 0.3655 |
| passing_yards | hist_gradient_boosting | 1868 | 62.3795 | 77.9359 | 3.9046 | 0.4749 |
| rushing_yards | ridge | 1868 | 54.1428 | 68.8311 | -2.8559 | 0.5305 |
| possession_seconds | hist_gradient_boosting | 1868 | 203.7136 | 255.3848 | -2.8319 | 0.3300 |

## Controls

- No random train/test split.
- 2025 is not used for model or hyperparameter selection.
- 2026 is not used in tournament scoring.
- Current-game matchup targets are not in the manifest feature list.
- Predictions are clipped only to fixed predeclared bounds.
- Tournament results establish predictive validation only; they do not establish sportsbook profitability.
