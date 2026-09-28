# Stat-vector model metrics

_Regenerated in full by `python -m pipeline.train retrain`; do not edit by hand._

- Generated: 2026-09-28T01:43:53+00:00
- Data through: 2026 week 2 (last week whose scheduled games are all final)
- Training seasons: 2022-2026; history-only seasons loaded: 2020, 2021
- Stat-vector windows: RB 8, WR 8, TE 10, QB 12; rolling-alone baseline windows: QB 14, RB 8, WR 10, TE 12
- Rate models attempt-weighted: True

## Coefficients (production fit)

### RB  (6635 player-weeks)

| Sub-model | Term | Coef | SE | p | n |
|---|---|---:|---:|---:|---:|
| carries | intercept | 0.81299 | 0.0983 | 1.66e-16 | 6487 |
|  | roll_carries | 0.88872 | 0.0101 | 0 |  |
|  | delta_carries | 0.43737 | 0.0253 | 1.57e-65 |  |
| targets | intercept | 0.37151 | 0.0359 | 7e-25 | 6487 |
|  | roll_targets | 0.80584 | 0.014 | 0 |  |
|  | delta_targets | 0.23498 | 0.0274 | 1.32e-17 |  |
| ypc | intercept | 3.4557 | 0.12 | 1.04e-170 | 5430 |
|  | roll_ypc | 0.20386 | 0.0266 | 2.01e-14 |  |
|  | delta_carries | 0.0011264 | 0.00983 | 0.909 |  |
|  | epa_allowed_rush | 4.1926 | 0.825 | 3.8e-07 |  |
| rush_td_rate | intercept | 0.027226 | 0.00109 | 1.74e-130 | 5430 |
|  | roll_rush_td_rate | 0.11586 | 0.0262 | 1.02e-05 |  |
|  | delta_carries | 0.00018022 | 0.000259 | 0.486 |  |
|  | epa_allowed_rush | 0.019011 | 0.0218 | 0.383 |  |
| catch_rate | intercept | 0.71323 | 0.0245 | 5.81e-171 | 4452 |
|  | roll_catch_rate | 0.083325 | 0.0309 | 0.00708 |  |
|  | delta_targets | 0.0010057 | 0.00379 | 0.791 |  |
|  | epa_allowed_pass | 0.176 | 0.0835 | 0.0352 |  |
| ypr | intercept | 6.9589 | 0.251 | 8.36e-155 | 4060 |
|  | roll_ypr | 0.050576 | 0.0321 | 0.115 |  |
|  | delta_targets | 0.12005 | 0.0816 | 0.141 |  |
|  | epa_allowed_pass | 1.1201 | 1.8 | 0.534 |  |
| rec_td_rate | intercept | 0.034765 | 0.00225 | 2.57e-52 | 4060 |
|  | roll_rec_td_rate | 0.083468 | 0.0315 | 0.00818 |  |
|  | delta_targets | 0.00081586 | 0.00192 | 0.671 |  |
|  | epa_allowed_pass | 0.056272 | 0.0424 | 0.184 |  |

### WR  (10604 player-weeks)

| Sub-model | Term | Coef | SE | p | n |
|---|---|---:|---:|---:|---:|
| targets | intercept | 0.43299 | 0.0448 | 5.04e-22 | 10380 |
|  | roll_targets | 0.87717 | 0.00873 | 0 |  |
|  | delta_targets | 0.25905 | 0.0214 | 1.51e-33 |  |
| carries | intercept | 0.049974 | 0.00516 | 4.66e-22 | 10380 |
|  | roll_carries | 0.67282 | 0.0119 | 0 |  |
|  | delta_carries | 0.15979 | 0.0215 | 1.26e-13 |  |
| catch_rate | intercept | 0.46327 | 0.0139 | 1.85e-228 | 8969 |
|  | roll_catch_rate | 0.26079 | 0.0216 | 2.67e-33 |  |
|  | delta_targets | 0.0012461 | 0.0018 | 0.49 |  |
|  | epa_allowed_pass | 0.26666 | 0.0544 | 9.52e-07 |  |
| ypr | intercept | 8.9273 | 0.266 | 1.45e-230 | 8031 |
|  | roll_ypr | 0.29237 | 0.0204 | 3.22e-46 |  |
|  | delta_targets | -0.01673 | 0.0506 | 0.741 |  |
| rec_td_rate | intercept | 0.066445 | 0.00237 | 5.31e-165 | 8031 |
|  | roll_rec_td_rate | 0.1282 | 0.0223 | 8.93e-09 |  |
|  | delta_targets | 0.0011836 | 0.00118 | 0.316 |  |
| ypc | intercept | 5.7588 | 0.322 | 1.85e-64 | 1356 |
|  | roll_ypc | -0.027951 | 0.0419 | 0.504 |  |
|  | delta_carries | -0.14384 | 0.403 | 0.721 |  |
|  | epa_allowed_rush | 3.2746 | 6.12 | 0.592 |  |
| rush_td_rate | intercept | 0.041015 | 0.00522 | 7.66e-15 | 1356 |
|  | roll_rush_td_rate | 0.022907 | 0.0435 | 0.599 |  |
|  | delta_carries | -0.010541 | 0.00992 | 0.288 |  |
|  | epa_allowed_rush | 0.1143 | 0.151 | 0.448 |  |

### TE  (5269 player-weeks)

| Sub-model | Term | Coef | SE | p | n |
|---|---|---:|---:|---:|---:|
| targets | intercept | 0.35711 | 0.0516 | 5.04e-12 | 5160 |
|  | roll_targets | 0.87839 | 0.0135 | 0 |  |
|  | delta_targets | 0.19728 | 0.0309 | 1.9e-10 |  |
| carries | intercept | 0.0055216 | 0.00565 | 0.329 | 5160 |
|  | roll_carries | 0.87814 | 0.00999 | 0 |  |
|  | delta_carries | -0.21658 | 0.0303 | 1.08e-12 |  |
| catch_rate | intercept | 0.61332 | 0.025 | 3.72e-125 | 4594 |
|  | roll_catch_rate | 0.13858 | 0.0346 | 6.34e-05 |  |
|  | delta_targets | -0.0029551 | 0.00326 | 0.365 |  |
|  | epa_allowed_pass | 0.18406 | 0.0811 | 0.0233 |  |
| ypr | intercept | 7.7495 | 0.359 | 8.94e-98 | 4104 |
|  | roll_ypr | 0.24009 | 0.0338 | 1.45e-12 |  |
|  | delta_targets | 0.059659 | 0.0702 | 0.395 |  |
|  | epa_allowed_pass | 3.0854 | 1.75 | 0.0773 |  |
| rec_td_rate | intercept | 0.066907 | 0.00353 | 9.02e-77 | 4104 |
|  | roll_rec_td_rate | 0.080351 | 0.0337 | 0.0173 |  |
|  | delta_targets | 0.00086098 | 0.00224 | 0.701 |  |
|  | epa_allowed_pass | 0.11516 | 0.0559 | 0.0394 |  |
| ypc | intercept | 2.4195 | 0.734 | 0.00117 | 189 |
|  | roll_ypc | 0.32537 | 0.144 | 0.0255 |  |
|  | delta_carries | -0.44672 | 0.248 | 0.0738 |  |
|  | epa_allowed_rush | -11.223 | 9.85 | 0.256 |  |
| rush_td_rate | intercept | 0.083361 | 0.0185 | 1.23e-05 | 189 |
|  | roll_rush_td_rate | -0.21693 | 0.169 | 0.2 |  |
|  | delta_carries | -0.011861 | 0.0107 | 0.268 |  |
|  | epa_allowed_rush | 0.084713 | 0.424 | 0.842 |  |

### QB  (2817 player-weeks)

| Sub-model | Term | Coef | SE | p | n |
|---|---|---:|---:|---:|---:|
| attempts | intercept | 7.3225 | 0.702 | 5.43e-25 | 2768 |
|  | roll_attempts | 0.73132 | 0.0241 | 9.99e-175 |  |
|  | delta_attempts | 0.48539 | 0.0398 | 2.1e-33 |  |
| carries | intercept | 0.56924 | 0.0858 | 3.94e-11 | 2768 |
|  | roll_carries | 0.83682 | 0.0202 | 3.4e-291 |  |
|  | delta_carries | 0.18074 | 0.0435 | 3.36e-05 |  |
| completion_rate | intercept | 0.44205 | 0.024 | 1.57e-71 | 2651 |
|  | roll_completion_rate | 0.31418 | 0.037 | 3.14e-17 |  |
|  | delta_attempts | -0.0002071 | 0.00038 | 0.586 |  |
|  | epa_allowed_pass | 0.23838 | 0.0413 | 8.42e-09 |  |
| yards_per_completion | intercept | 6.9465 | 0.435 | 6.5e-55 | 2606 |
|  | roll_yards_per_completion | 0.36435 | 0.0394 | 4.93e-20 |  |
|  | delta_attempts | 0.010745 | 0.00985 | 0.275 |  |
|  | epa_allowed_pass | 0.68169 | 1.07 | 0.523 |  |
| pass_td_rate | intercept | 0.050553 | 0.0027 | 2.97e-73 | 2606 |
|  | roll_pass_td_rate | 0.25179 | 0.0366 | 7.08e-12 |  |
|  | delta_attempts | 1.0321e-05 | 0.000213 | 0.961 |  |
|  | epa_allowed_pass | 0.033393 | 0.0231 | 0.148 |  |
| int_rate | intercept | 0.020864 | 0.00115 | 3.21e-69 | 2651 |
|  | roll_int_rate | 0.069416 | 0.0455 | 0.127 |  |
|  | delta_attempts | -5.611e-05 | 0.000117 | 0.632 |  |
|  | epa_allowed_pass | -0.02892 | 0.0127 | 0.0232 |  |
| ypc | intercept | 1.9519 | 0.175 | 4.31e-28 | 2443 |
|  | roll_ypc | 0.55757 | 0.0363 | 8.63e-51 |  |
|  | delta_carries | -0.01308 | 0.0514 | 0.799 |  |
|  | epa_allowed_rush | 2.7332 | 1.98 | 0.167 |  |
| rush_td_rate | intercept | 0.032234 | 0.0029 | 5.87e-28 | 2443 |
|  | roll_rush_td_rate | 0.29264 | 0.0447 | 7.4e-11 |  |
|  | delta_carries | -0.0001106 | 0.00159 | 0.944 |  |
|  | epa_allowed_rush | 0.0095798 | 0.061 | 0.875 |  |

## In-sample fit (in-sample, not an accuracy claim)

| Pos | In-sample R² per sub-model (rate models: weighted R²) |
|---|---|
| RB | carries 0.551, targets 0.343, ypc 0.016, rush_td_rate 0.004, catch_rate 0.003, ypr 0.001, rec_td_rate 0.002 |
| WR | targets 0.496, carries 0.234, catch_rate 0.019, ypr 0.025, rec_td_rate 0.004, ypc 0.001, rush_td_rate 0.002 |
| TE | targets 0.453, carries 0.608, catch_rate 0.005, ypr 0.013, rec_td_rate 0.002, ypc 0.053, rush_td_rate 0.015 |
| QB | attempts 0.278, carries 0.383, completion_rate 0.038, yards_per_completion 0.032, pass_td_rate 0.019, int_rate 0.003, ypc 0.089, rush_td_rate 0.017 |

## Out-of-sample accuracy (leave-one-season-out inside the training window)

Each full season (2022, 2023, 2024, 2025) is scored by models fit on the window's other full seasons; 2026 (partial) is test-only, fit on all of them. Same harness and league scoring as the README; fewer, shorter folds than the README's 2019-2025 table, so numbers are not identical. Cells: MAE / R² / Spearman / bias (actual − predicted).

| Pos | Fold | n | Stat vector | Rolling alone |
|---|---|---:|---|---|
| QB | 2022 | 652 | 7.209 / 0.261 / 0.505 / 0.14 | 7.287 / 0.241 / 0.525 / 0.21 |
| QB | 2023 | 678 | 7.597 / 0.234 / 0.493 / -0.58 | 7.529 / 0.230 / 0.492 / -0.26 |
| QB | 2024 | 686 | 7.909 / 0.259 / 0.514 / 0.04 | 7.943 / 0.235 / 0.494 / -0.15 |
| QB | 2025 | 678 | 8.269 / 0.207 / 0.435 / -0.08 | 8.411 / 0.155 / 0.408 / -0.62 |
| QB | 2026 partial (through wk 2) | 74 | 8.757 / 0.176 / 0.323 / 0.86 | 8.550 / 0.206 / 0.409 / -0.38 |
| **QB** | **mean of full seasons** | | **7.746 / 0.240 / 0.487 / -0.12** | 7.793 / 0.215 / 0.480 / -0.20 |
| RB | 2022 | 1618 | 4.636 / 0.376 / 0.650 / -0.22 | 4.770 / 0.317 / 0.619 / -0.33 |
| RB | 2023 | 1510 | 4.545 / 0.381 / 0.705 / -0.14 | 4.525 / 0.349 / 0.683 / 0.05 |
| RB | 2024 | 1578 | 4.486 / 0.423 / 0.720 / 0.15 | 4.412 / 0.425 / 0.719 / -0.08 |
| RB | 2025 | 1611 | 4.465 / 0.424 / 0.744 / 0.06 | 4.504 / 0.394 / 0.723 / -0.15 |
| RB | 2026 partial (through wk 2) | 170 | 4.655 / 0.428 / 0.684 / -0.48 | 4.611 / 0.427 / 0.684 / -0.61 |
| **RB** | **mean of full seasons** | | **4.533 / 0.401 / 0.705 / -0.04** | 4.553 / 0.371 / 0.686 / -0.13 |
| WR | 2022 | 2457 | 4.719 / 0.336 / 0.628 / -0.11 | 4.741 / 0.313 / 0.622 / -0.35 |
| WR | 2023 | 2552 | 4.523 / 0.367 / 0.638 / 0.03 | 4.574 / 0.354 / 0.625 / -0.23 |
| WR | 2024 | 2508 | 4.579 / 0.346 / 0.660 / 0.33 | 4.543 / 0.338 / 0.646 / 0.02 |
| WR | 2025 | 2587 | 4.310 / 0.362 / 0.660 / -0.18 | 4.358 / 0.333 / 0.637 / -0.41 |
| WR | 2026 partial (through wk 2) | 276 | 4.670 / 0.323 / 0.617 / -0.08 | 4.780 / 0.275 / 0.582 / -0.32 |
| **WR** | **mean of full seasons** | | **4.533 / 0.353 / 0.647 / 0.02** | 4.554 / 0.334 / 0.633 / -0.24 |
| TE | 2022 | 1239 | 4.151 / 0.319 / 0.605 / -0.07 | 4.226 / 0.289 / 0.579 / -0.14 |
| TE | 2023 | 1211 | 4.060 / 0.371 / 0.644 / 0.03 | 4.179 / 0.335 / 0.617 / -0.00 |
| TE | 2024 | 1249 | 4.250 / 0.335 / 0.632 / 0.07 | 4.248 / 0.327 / 0.622 / 0.03 |
| TE | 2025 | 1320 | 4.099 / 0.342 / 0.621 / -0.03 | 4.197 / 0.314 / 0.611 / -0.17 |
| TE | 2026 partial (through wk 2) | 141 | 4.600 / 0.250 / 0.542 / -0.19 | 4.629 / 0.253 / 0.560 / -0.66 |
| **TE** | **mean of full seasons** | | **4.140 / 0.341 / 0.625 / -0.00** | 4.212 / 0.316 / 0.607 / -0.07 |

### Confidence calibration (pooled full-season folds)

Per tier: MAE / mean actual points / normalized MAE. Judge on normalized MAE (Q-15).

| Pos | high | medium | low | insufficient | normalized monotonic |
|---|---|---|---|---|---|
| QB | 7.760 / 20.15 / 0.385 | 7.644 / 17.89 / 0.427 | 7.591 / 13.86 / 0.548 | 8.458 / 11.23 / 0.753 | True |
| RB | 4.859 / 9.07 / 0.536 | 4.294 / 7.37 / 0.583 | 4.320 / 6.68 / 0.647 | 4.349 / 6.18 / 0.704 | True |
| WR | 4.790 / 8.18 / 0.585 | 4.389 / 7.13 / 0.616 | 4.299 / 6.43 / 0.668 | 4.433 / 5.95 / 0.745 | True |
| TE | 4.609 / 8.30 / 0.555 | 4.100 / 6.76 / 0.606 | 3.701 / 5.21 / 0.711 | 3.680 / 4.54 / 0.810 | True |
| ALL | 5.068 / 9.63 / 0.526 | 4.643 / 8.24 / 0.563 | 4.595 / 7.18 / 0.640 | 4.830 / 6.46 / 0.747 | True |

## Coefficient drift vs 2019-2025 leave-one-season-out ranges

46 of 106 coefficients inside their 2019-2025 fold range. Flagged:

| Pos | Sub-model | Term | Value | 2019-2025 range | Status |
|---|---|---|---:|---|---|
| RB | carries | intercept | 0.813 | 0.8484 … 0.9525 | outside range |
| RB | carries | roll_carries | 0.8887 | 0.8745 … 0.8874 | outside range |
| RB | targets | intercept | 0.3715 | 0.3965 … 0.4495 | outside range |
| RB | targets | roll_targets | 0.8058 | 0.7814 … 0.8053 | outside range |
| RB | ypc | intercept | 3.456 | 3.561 … 3.796 | outside range |
| RB | ypc | roll_ypc | 0.2039 | 0.1337 … 0.1774 | outside range |
| RB | ypc | epa_allowed_rush | 4.193 | 4.345 … 5.242 | outside range |
| RB | rush_td_rate | delta_carries | 0.0001802 | -0.0004098 … 8.369e-06 | outside range |
| RB | catch_rate | intercept | 0.7132 | 0.7153 … 0.7236 | outside range |
| RB | catch_rate | roll_catch_rate | 0.08332 | 0.06354 … 0.07672 | outside range |
| RB | catch_rate | epa_allowed_pass | 0.176 | 0.09169 … 0.1505 | outside range |
| RB | ypr | intercept | 6.959 | 7.043 … 7.159 | outside range |
| RB | ypr | delta_targets | 0.1201 | 0.06596 … 0.104 | outside range |
| RB | ypr | epa_allowed_pass | 1.12 | 1.379 … 2.876 | outside range |
| RB | rec_td_rate | intercept | 0.03477 | 0.03483 … 0.03675 | outside range |
| RB | rec_td_rate | roll_rec_td_rate | 0.08347 | 0.03377 … 0.06265 | outside range |
| WR | targets | intercept | 0.433 | 0.4753 … 0.5102 | outside range |
| WR | targets | roll_targets | 0.8772 | 0.8724 … 0.8764 | outside range |
| WR | targets | delta_targets | 0.259 | 0.228 … 0.2528 | outside range |
| WR | carries | intercept | 0.04997 | 0.0351 … 0.04709 | outside range |
| WR | carries | roll_carries | 0.6728 | 0.7002 … 0.7803 | outside range |
| WR | catch_rate | intercept | 0.4633 | 0.4674 … 0.474 | outside range |
| WR | catch_rate | roll_catch_rate | 0.2608 | 0.2451 … 0.2572 | outside range |
| WR | catch_rate | delta_targets | 0.001246 | -0.00014 … 0.001231 | outside range |
| WR | catch_rate | epa_allowed_pass | 0.2667 | 0.1638 … 0.2368 | outside range |
| WR | ypr | delta_targets | -0.01673 | -0.0755 … -0.03321 | outside range |
| WR | rec_td_rate | delta_targets | 0.001184 | -0.001233 … 0.000177 | outside range |
| WR | ypc | intercept | 5.759 | 5.887 … 6.241 | outside range |
| WR | ypc | epa_allowed_rush | 3.275 | -4.286 … 1.083 | outside range |
| WR | rush_td_rate | roll_rush_td_rate | 0.02291 | 0.02403 … 0.06281 | outside range |
| WR | rush_td_rate | delta_carries | -0.01054 | -0.001785 … 0.009499 | outside range |
| TE | targets | intercept | 0.3571 | 0.367 … 0.4193 | outside range |
| TE | targets | delta_targets | 0.1973 | 0.2308 … 0.2814 | outside range |
| TE | carries | roll_carries | 0.8781 | 0.9111 … 0.9691 | outside range |
| TE | carries | delta_carries | -0.2166 | 0.08775 … 0.423 | outside range |
| TE | catch_rate | intercept | 0.6133 | 0.5782 … 0.6089 | outside range |
| TE | catch_rate | epa_allowed_pass | 0.1841 | 0.08349 … 0.1779 | outside range |
| TE | ypr | epa_allowed_pass | 3.085 | 3.676 … 5.739 | outside range |
| TE | rec_td_rate | roll_rec_td_rate | 0.08035 | 0.09829 … 0.1365 | outside range |
| TE | rec_td_rate | delta_targets | 0.000861 | -0.0008622 … 0.0004375 | outside range |
| TE | ypc | intercept | 2.419 | 2.758 … 4.093 | outside range |
| TE | ypc | roll_ypc | 0.3254 | 0.1466 … 0.2989 | outside range |
| TE | ypc | delta_carries | -0.4467 | -0.3958 … -0.1983 | outside range |
| TE | rush_td_rate | roll_rush_td_rate | -0.2169 | -0.109 … 0.06312 | outside range |
| TE | rush_td_rate | epa_allowed_rush | 0.08471 | 0.1086 … 0.4177 | outside range |
| QB | attempts | roll_attempts | 0.7313 | 0.7327 … 0.7496 | outside range |
| QB | completion_rate | delta_attempts | -0.0002071 | -7.768e-05 … 6.426e-05 | outside range |
| QB | completion_rate | epa_allowed_pass | 0.2384 | 0.1609 … 0.2329 | outside range |
| QB | yards_per_completion | intercept | 6.946 | 6.998 … 7.417 | outside range |
| QB | yards_per_completion | roll_yards_per_completion | 0.3643 | 0.3256 … 0.3633 | outside range |
| QB | yards_per_completion | delta_attempts | 0.01074 | -0.001227 … 0.005135 | outside range |
| QB | yards_per_completion | epa_allowed_pass | 0.6817 | 1.151 … 3.562 | outside range |
| QB | pass_td_rate | epa_allowed_pass | 0.03339 | 0.03525 … 0.06252 | outside range |
| QB | int_rate | intercept | 0.02086 | 0.01861 … 0.01993 | outside range |
| QB | int_rate | roll_int_rate | 0.06942 | 0.1172 … 0.176 | outside range |
| QB | int_rate | delta_attempts | -5.611e-05 | -5.062e-05 … 4.042e-05 | outside range |
| QB | ypc | delta_carries | -0.01308 | -0.008303 … 0.08242 | outside range |
| QB | rush_td_rate | intercept | 0.03223 | 0.03558 … 0.04031 | outside range |
| QB | rush_td_rate | roll_rush_td_rate | 0.2926 | 0.152 … 0.2619 | outside range |
| QB | rush_td_rate | epa_allowed_rush | 0.00958 | 0.03805 … 0.11 | outside range |

## Warnings

### Feature null rates, latest completed week (2026 wk 2)

Nulls before rate-prior filling; a null volume feature means no projection (no history).

- RB (95 rows): delta_carries 3.2%, delta_targets 3.2%, roll_carries 3.2%, roll_catch_rate 12.6%, roll_rec_td_rate 12.6%, roll_rush_td_rate 7.4%, roll_targets 3.2%, roll_ypc 7.4%, roll_ypr 12.6%
- WR (152 rows): delta_carries 3.9%, delta_targets 3.9%, roll_carries 3.9%, roll_catch_rate 7.2%, roll_rec_td_rate 8.6%, roll_rush_td_rate 63.2%, roll_targets 3.9%, roll_ypc 63.2%, roll_ypr 8.6%
- TE (73 rows): delta_carries 2.7%, delta_targets 2.7%, roll_carries 2.7%, roll_catch_rate 5.5%, roll_rec_td_rate 8.2%, roll_rush_td_rate 76.7%, roll_targets 2.7%, roll_ypc 76.7%, roll_ypr 8.2%
- QB (39 rows): delta_attempts 2.6%, delta_carries 2.6%, roll_attempts 2.6%, roll_carries 2.6%, roll_completion_rate 2.6%, roll_int_rate 2.6%, roll_pass_td_rate 2.6%, roll_rush_td_rate 2.6%, roll_yards_per_completion 2.6%, roll_ypc 2.6%

### Unpredicted scoring categories (score 0 in every projection)

- RB: pass_yd, pass_td, pass_int, pass_2pt, pass_att, pass_cmp, pass_fd, pass_inc, pass_sack, bonus_pass_yd_300, bonus_pass_yd_400, bonus_pass_cmp_25, rush_2pt, rush_fd, bonus_rush_yd_100, bonus_rush_yd_200, bonus_rush_att_20, rec_2pt, rec_fd, fum_lost
- WR: pass_yd, pass_td, pass_int, pass_2pt, pass_att, pass_cmp, pass_fd, pass_inc, pass_sack, bonus_pass_yd_300, bonus_pass_yd_400, bonus_pass_cmp_25, rush_2pt, rush_fd, bonus_rush_yd_100, bonus_rush_yd_200, bonus_rush_att_20, rec_2pt, rec_fd, fum_lost
- TE: pass_yd, pass_td, pass_int, pass_2pt, pass_att, pass_cmp, pass_fd, pass_inc, pass_sack, bonus_pass_yd_300, bonus_pass_yd_400, bonus_pass_cmp_25, rush_2pt, rush_fd, bonus_rush_yd_100, bonus_rush_yd_200, bonus_rush_att_20, rec_2pt, rec_fd, fum_lost
- QB: pass_2pt, pass_fd, pass_sack, bonus_pass_yd_300, bonus_pass_yd_400, bonus_pass_cmp_25, rush_2pt, rush_fd, bonus_rush_yd_100, bonus_rush_yd_200, bonus_rush_att_20, rec, rec_yd, rec_td, rec_2pt, rec_fd, fum_lost

### Validation warnings

- RB.carries.intercept = 0.813 outside 2019-2025 range [0.8484, 0.9525]
- RB.carries.roll_carries = 0.8887 outside 2019-2025 range [0.8745, 0.8874]
- RB.targets.intercept = 0.3715 outside 2019-2025 range [0.3965, 0.4495]
- RB.targets.roll_targets = 0.8058 outside 2019-2025 range [0.7814, 0.8053]
- RB.ypc.intercept = 3.456 outside 2019-2025 range [3.561, 3.796]
- RB.ypc.roll_ypc = 0.2039 outside 2019-2025 range [0.1337, 0.1774]
- RB.ypc.epa_allowed_rush = 4.193 outside 2019-2025 range [4.345, 5.242]
- RB.rush_td_rate.delta_carries = 0.0001802 outside 2019-2025 range [-0.0004098, 8.369e-06]
- RB.catch_rate.intercept = 0.7132 outside 2019-2025 range [0.7153, 0.7236]
- RB.catch_rate.roll_catch_rate = 0.08332 outside 2019-2025 range [0.06354, 0.07672]
- RB.catch_rate.epa_allowed_pass = 0.176 outside 2019-2025 range [0.09169, 0.1505]
- RB.ypr.intercept = 6.959 outside 2019-2025 range [7.043, 7.159]
- RB.ypr.delta_targets = 0.1201 outside 2019-2025 range [0.06596, 0.104]
- RB.ypr.epa_allowed_pass = 1.12 outside 2019-2025 range [1.379, 2.876]
- RB.rec_td_rate.intercept = 0.03477 outside 2019-2025 range [0.03483, 0.03675]
- RB.rec_td_rate.roll_rec_td_rate = 0.08347 outside 2019-2025 range [0.03377, 0.06265]
- WR.targets.intercept = 0.433 outside 2019-2025 range [0.4753, 0.5102]
- WR.targets.roll_targets = 0.8772 outside 2019-2025 range [0.8724, 0.8764]
- WR.targets.delta_targets = 0.259 outside 2019-2025 range [0.228, 0.2528]
- WR.carries.intercept = 0.04997 outside 2019-2025 range [0.0351, 0.04709]
- WR.carries.roll_carries = 0.6728 outside 2019-2025 range [0.7002, 0.7803]
- WR.catch_rate.intercept = 0.4633 outside 2019-2025 range [0.4674, 0.474]
- WR.catch_rate.roll_catch_rate = 0.2608 outside 2019-2025 range [0.2451, 0.2572]
- WR.catch_rate.delta_targets = 0.001246 outside 2019-2025 range [-0.00014, 0.001231]
- WR.catch_rate.epa_allowed_pass = 0.2667 outside 2019-2025 range [0.1638, 0.2368]
- WR.ypr.delta_targets = -0.01673 outside 2019-2025 range [-0.0755, -0.03321]
- WR.rec_td_rate.delta_targets = 0.001184 outside 2019-2025 range [-0.001233, 0.000177]
- WR.ypc.intercept = 5.759 outside 2019-2025 range [5.887, 6.241]
- WR.ypc.epa_allowed_rush = 3.275 outside 2019-2025 range [-4.286, 1.083]
- WR.rush_td_rate.roll_rush_td_rate = 0.02291 outside 2019-2025 range [0.02403, 0.06281]
- WR.rush_td_rate.delta_carries = -0.01054 outside 2019-2025 range [-0.001785, 0.009499]
- TE.targets.intercept = 0.3571 outside 2019-2025 range [0.367, 0.4193]
- TE.targets.delta_targets = 0.1973 outside 2019-2025 range [0.2308, 0.2814]
- TE.carries.roll_carries = 0.8781 outside 2019-2025 range [0.9111, 0.9691]
- TE.carries.delta_carries = -0.2166 outside 2019-2025 range [0.08775, 0.423]
- TE.catch_rate.intercept = 0.6133 outside 2019-2025 range [0.5782, 0.6089]
- TE.catch_rate.epa_allowed_pass = 0.1841 outside 2019-2025 range [0.08349, 0.1779]
- TE.ypr.epa_allowed_pass = 3.085 outside 2019-2025 range [3.676, 5.739]
- TE.rec_td_rate.roll_rec_td_rate = 0.08035 outside 2019-2025 range [0.09829, 0.1365]
- TE.rec_td_rate.delta_targets = 0.000861 outside 2019-2025 range [-0.0008622, 0.0004375]
- TE.ypc.intercept = 2.419 outside 2019-2025 range [2.758, 4.093]
- TE.ypc.roll_ypc = 0.3254 outside 2019-2025 range [0.1466, 0.2989]
- TE.ypc.delta_carries = -0.4467 outside 2019-2025 range [-0.3958, -0.1983]
- TE.rush_td_rate.roll_rush_td_rate = -0.2169 outside 2019-2025 range [-0.109, 0.06312]
- TE.rush_td_rate.epa_allowed_rush = 0.08471 outside 2019-2025 range [0.1086, 0.4177]
- QB.attempts.roll_attempts = 0.7313 outside 2019-2025 range [0.7327, 0.7496]
- QB.completion_rate.delta_attempts = -0.0002071 outside 2019-2025 range [-7.768e-05, 6.426e-05]
- QB.completion_rate.epa_allowed_pass = 0.2384 outside 2019-2025 range [0.1609, 0.2329]
- QB.yards_per_completion.intercept = 6.946 outside 2019-2025 range [6.998, 7.417]
- QB.yards_per_completion.roll_yards_per_completion = 0.3643 outside 2019-2025 range [0.3256, 0.3633]
- QB.yards_per_completion.delta_attempts = 0.01074 outside 2019-2025 range [-0.001227, 0.005135]
- QB.yards_per_completion.epa_allowed_pass = 0.6817 outside 2019-2025 range [1.151, 3.562]
- QB.pass_td_rate.epa_allowed_pass = 0.03339 outside 2019-2025 range [0.03525, 0.06252]
- QB.int_rate.intercept = 0.02086 outside 2019-2025 range [0.01861, 0.01993]
- QB.int_rate.roll_int_rate = 0.06942 outside 2019-2025 range [0.1172, 0.176]
- QB.int_rate.delta_attempts = -5.611e-05 outside 2019-2025 range [-5.062e-05, 4.042e-05]
- QB.ypc.delta_carries = -0.01308 outside 2019-2025 range [-0.008303, 0.08242]
- QB.rush_td_rate.intercept = 0.03223 outside 2019-2025 range [0.03558, 0.04031]
- QB.rush_td_rate.roll_rush_td_rate = 0.2926 outside 2019-2025 range [0.152, 0.2619]
- QB.rush_td_rate.epa_allowed_rush = 0.00958 outside 2019-2025 range [0.03805, 0.11]
