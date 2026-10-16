# CFB Model Tournament v0.1

## Protocol

- Feature count: 465
- Model selection uses chronological walk-forward validation only.
- Validation folds:
  - train 2014-2020 -> validate 2021
  - train 2014-2021 -> validate 2022
  - train 2014-2022 -> validate 2023
  - train 2014-2023 -> validate 2024
- 2025 is excluded from all model selection.
- After winners are selected, each winner is refit on 2014-2024 and evaluated once on 2025.
- Eligibility uses pregame historical role signals only; current-game role flags are not model features or eligibility inputs.

## Winners from 2021-2024 validation

| Target | Winner | Validation MAE | Improvement vs recent baseline |
|---|---|---:|---:|
| pass_attempts | hist_gradient_boosting | 5.0284 | 7.99% |
| completions | hist_gradient_boosting | 3.2718 | 5.73% |
| passing_yards | extra_trees | 43.0133 | 5.19% |
| rush_attempts | hist_gradient_boosting | 2.4562 | 5.16% |
| rushing_yards | hist_gradient_boosting | 16.6190 | 7.16% |
| receptions | hist_gradient_boosting | 1.1835 | 6.19% |
| receiving_yards | hist_gradient_boosting | 17.5824 | 7.08% |

## 2025 untouched holdout

| Target | Winner | Rows | MAE | RMSE | Bias | Corr |
|---|---|---:|---:|---:|---:|---:|
| pass_attempts | hist_gradient_boosting | 4158 | 4.8762 | 7.6307 | 0.3523 | 0.8587 |
| completions | hist_gradient_boosting | 4158 | 3.2073 | 5.0609 | -0.0329 | 0.8480 |
| passing_yards | extra_trees | 4158 | 41.6252 | 64.5851 | 3.0292 | 0.8288 |
| rush_attempts | hist_gradient_boosting | 12733 | 2.3821 | 3.6770 | -0.0248 | 0.7803 |
| rushing_yards | hist_gradient_boosting | 12733 | 16.1208 | 26.2660 | -0.2855 | 0.6679 |
| receptions | hist_gradient_boosting | 16979 | 1.1877 | 1.6198 | 0.0471 | 0.5988 |
| receiving_yards | hist_gradient_boosting | 16979 | 17.3711 | 25.0284 | 1.3382 | 0.5600 |

## Controls

- No random train/test split.
- 2025 is not used for model or hyperparameter selection.
- Current-game target columns are not in the manifest feature list.
- Predictions are clipped only to fixed, predeclared target bounds.
- Tournament results establish predictive validation only; they do not establish sportsbook profitability.
