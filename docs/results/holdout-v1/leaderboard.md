# CBC screening research benchmark

Research screening benchmark only. The public release has sample identifiers but no patient identifiers or diagnosis dates. Repeated patients and patient overlap across cohorts cannot be verified. Results do not establish early detection, clinical utility, or patient-disjoint generalization. Thresholds are selected on validation records; validation performance is therefore not an unbiased estimate for the selected operating point. External/test evaluation must be reserved until the model configuration and threshold-selection procedure are frozen.

Models use fixed hyperparameters and fit development records only. Median imputation and logistic-regression scaling are learned only from development data. No calibration correction or hyperparameter tuning is performed.

External/test cohorts were evaluated with the explicit holdout flag.

Rows retain the prespecified experiment order; this table does not select a winner.

| Feature set | Model | Cohort | N | Prevalence | AUROC | AP | Brier | Sensitivity | Specificity | Precision | FP / 1,000 negatives |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| basic_cbc | prevalence | validation | 211612 | 0.0369 | 0.5000 | 0.0369 | 0.1234 | 0.0000 | 1.0000 | — | 0.0000 |
| basic_cbc | prevalence | external | 149648 | 0.0387 | 0.5000 | 0.0387 | 0.1240 | 0.0000 | 1.0000 | — | 0.0000 |
| basic_cbc | prevalence | test | 58481 | 0.0392 | 0.5000 | 0.0392 | 0.1242 | 0.0000 | 1.0000 | — | 0.0000 |
| basic_cbc | logistic | validation | 211612 | 0.0369 | 0.9077 | 0.4002 | 0.0718 | 0.6486 | 0.9500 | 0.3323 | 49.9966 |
| basic_cbc | logistic | external | 149648 | 0.0387 | 0.9044 | 0.4359 | 0.0626 | 0.6239 | 0.9555 | 0.3605 | 44.5436 |
| basic_cbc | logistic | test | 58481 | 0.0392 | 0.9041 | 0.4025 | 0.0760 | 0.6328 | 0.9462 | 0.3240 | 53.7987 |
| basic_cbc | rf | validation | 211612 | 0.0369 | 0.9432 | 0.4712 | 0.0586 | 0.7449 | 0.9500 | 0.3637 | 49.9966 |
| basic_cbc | rf | external | 149648 | 0.0387 | 0.9448 | 0.5194 | 0.0548 | 0.7208 | 0.9539 | 0.3864 | 46.0729 |
| basic_cbc | rf | test | 58481 | 0.0392 | 0.9506 | 0.4567 | 0.0635 | 0.7795 | 0.9436 | 0.3603 | 56.3969 |
| basic_cbc | xgboost | validation | 211612 | 0.0369 | 0.9454 | 0.4692 | 0.0570 | 0.7622 | 0.9500 | 0.3690 | 49.9917 |
| basic_cbc | xgboost | external | 149648 | 0.0387 | 0.9465 | 0.5260 | 0.0536 | 0.7305 | 0.9539 | 0.3892 | 46.1285 |
| basic_cbc | xgboost | test | 58481 | 0.0392 | 0.9507 | 0.4947 | 0.0612 | 0.7917 | 0.9455 | 0.3717 | 54.5461 |
| expanded_cbc | prevalence | validation | 211612 | 0.0369 | 0.5000 | 0.0369 | 0.1234 | 0.0000 | 1.0000 | — | 0.0000 |
| expanded_cbc | prevalence | external | 149648 | 0.0387 | 0.5000 | 0.0387 | 0.1240 | 0.0000 | 1.0000 | — | 0.0000 |
| expanded_cbc | prevalence | test | 58481 | 0.0392 | 0.5000 | 0.0392 | 0.1242 | 0.0000 | 1.0000 | — | 0.0000 |
| expanded_cbc | logistic | validation | 211612 | 0.0369 | 0.9188 | 0.3598 | 0.0691 | 0.6768 | 0.9500 | 0.3418 | 49.9966 |
| expanded_cbc | logistic | external | 149648 | 0.0387 | 0.9205 | 0.4225 | 0.0569 | 0.6288 | 0.9590 | 0.3815 | 41.0263 |
| expanded_cbc | logistic | test | 58481 | 0.0392 | 0.9308 | 0.3789 | 0.0641 | 0.6738 | 0.9536 | 0.3720 | 46.3597 |
| expanded_cbc | rf | validation | 211612 | 0.0369 | 0.9628 | 0.5234 | 0.0502 | 0.8189 | 0.9500 | 0.3859 | 49.9966 |
| expanded_cbc | rf | external | 149648 | 0.0387 | 0.9643 | 0.5911 | 0.0450 | 0.7988 | 0.9564 | 0.4243 | 43.6121 |
| expanded_cbc | rf | test | 58481 | 0.0392 | 0.9699 | 0.5130 | 0.0515 | 0.8729 | 0.9453 | 0.3940 | 54.7241 |
| expanded_cbc | xgboost | validation | 211612 | 0.0369 | 0.9602 | 0.5062 | 0.0507 | 0.8063 | 0.9500 | 0.3822 | 49.9966 |
| expanded_cbc | xgboost | external | 149648 | 0.0387 | 0.9648 | 0.5878 | 0.0423 | 0.7908 | 0.9591 | 0.4374 | 40.9359 |
| expanded_cbc | xgboost | test | 58481 | 0.0392 | 0.9689 | 0.5220 | 0.0508 | 0.8716 | 0.9476 | 0.4041 | 52.3749 |

AP = average precision. Undefined metrics are shown as — and saved as null. Specificity is targeted on validation overall and may differ by site or cohort.

## Performance by site

| Feature set | Model | Cohort | Site | N | Sensitivity | Specificity | FP / 1,000 negatives |
|---|---|---|---|---:|---:|---:|---:|
| basic_cbc | prevalence | validation | A | 59986 | 0.0000 | 1.0000 | 0.0000 |
| basic_cbc | prevalence | validation | B | 51228 | 0.0000 | 1.0000 | 0.0000 |
| basic_cbc | prevalence | validation | C | 100398 | 0.0000 | 1.0000 | 0.0000 |
| basic_cbc | prevalence | external | D | 30166 | 0.0000 | 1.0000 | 0.0000 |
| basic_cbc | prevalence | external | E | 94320 | 0.0000 | 1.0000 | 0.0000 |
| basic_cbc | prevalence | external | F | 20179 | 0.0000 | 1.0000 | 0.0000 |
| basic_cbc | prevalence | external | G | 4983 | 0.0000 | 1.0000 | 0.0000 |
| basic_cbc | prevalence | test | real_world | 58481 | 0.0000 | 1.0000 | 0.0000 |
| basic_cbc | logistic | validation | A | 59986 | 0.7343 | 0.9544 | 45.5689 |
| basic_cbc | logistic | validation | B | 51228 | 0.4920 | 0.9644 | 35.5503 |
| basic_cbc | logistic | validation | C | 100398 | 0.6275 | 0.9398 | 60.1797 |
| basic_cbc | logistic | external | D | 30166 | 0.6367 | 0.9395 | 60.5020 |
| basic_cbc | logistic | external | E | 94320 | 0.6218 | 0.9658 | 34.2472 |
| basic_cbc | logistic | external | F | 20179 | 0.5733 | 0.9470 | 53.0485 |
| basic_cbc | logistic | external | G | 4983 | 0.6656 | 0.8761 | 123.8589 |
| basic_cbc | logistic | test | real_world | 58481 | 0.6328 | 0.9462 | 53.7987 |
| basic_cbc | rf | validation | A | 59986 | 0.8122 | 0.9505 | 49.5337 |
| basic_cbc | rf | validation | B | 51228 | 0.6720 | 0.9661 | 33.8801 |
| basic_cbc | rf | validation | C | 100398 | 0.7171 | 0.9413 | 58.6940 |
| basic_cbc | rf | external | D | 30166 | 0.7002 | 0.9409 | 59.0805 |
| basic_cbc | rf | external | E | 94320 | 0.7455 | 0.9637 | 36.3393 |
| basic_cbc | rf | external | F | 20179 | 0.6724 | 0.9485 | 51.4851 |
| basic_cbc | rf | external | G | 4983 | 0.7344 | 0.8517 | 148.2852 |
| basic_cbc | rf | test | real_world | 58481 | 0.7795 | 0.9436 | 56.3969 |
| basic_cbc | xgboost | validation | A | 59986 | 0.8298 | 0.9510 | 48.9747 |
| basic_cbc | xgboost | validation | B | 51228 | 0.6902 | 0.9665 | 33.5222 |
| basic_cbc | xgboost | validation | C | 100398 | 0.7339 | 0.9408 | 59.2031 |
| basic_cbc | xgboost | external | D | 30166 | 0.7130 | 0.9448 | 55.1973 |
| basic_cbc | xgboost | external | E | 94320 | 0.7522 | 0.9620 | 38.0282 |
| basic_cbc | xgboost | external | F | 20179 | 0.6845 | 0.9496 | 50.3908 |
| basic_cbc | xgboost | external | G | 4983 | 0.7452 | 0.8552 | 144.8310 |
| basic_cbc | xgboost | test | real_world | 58481 | 0.7917 | 0.9455 | 54.5461 |
| expanded_cbc | prevalence | validation | A | 59986 | 0.0000 | 1.0000 | 0.0000 |
| expanded_cbc | prevalence | validation | B | 51228 | 0.0000 | 1.0000 | 0.0000 |
| expanded_cbc | prevalence | validation | C | 100398 | 0.0000 | 1.0000 | 0.0000 |
| expanded_cbc | prevalence | external | D | 30166 | 0.0000 | 1.0000 | 0.0000 |
| expanded_cbc | prevalence | external | E | 94320 | 0.0000 | 1.0000 | 0.0000 |
| expanded_cbc | prevalence | external | F | 20179 | 0.0000 | 1.0000 | 0.0000 |
| expanded_cbc | prevalence | external | G | 4983 | 0.0000 | 1.0000 | 0.0000 |
| expanded_cbc | prevalence | test | real_world | 58481 | 0.0000 | 1.0000 | 0.0000 |
| expanded_cbc | logistic | validation | A | 59986 | 0.7269 | 0.9580 | 41.9709 |
| expanded_cbc | logistic | validation | B | 51228 | 0.6913 | 0.9554 | 44.6366 |
| expanded_cbc | logistic | validation | C | 100398 | 0.6405 | 0.9424 | 57.5718 |
| expanded_cbc | logistic | external | D | 30166 | 0.6956 | 0.9384 | 61.6462 |
| expanded_cbc | logistic | external | E | 94320 | 0.5762 | 0.9699 | 30.1393 |
| expanded_cbc | logistic | external | F | 20179 | 0.6087 | 0.9547 | 45.3361 |
| expanded_cbc | logistic | external | G | 4983 | 0.6989 | 0.8796 | 120.4046 |
| expanded_cbc | logistic | test | real_world | 58481 | 0.6738 | 0.9536 | 46.3597 |
| expanded_cbc | rf | validation | A | 59986 | 0.8752 | 0.9514 | 48.6254 |
| expanded_cbc | rf | validation | B | 51228 | 0.7889 | 0.9642 | 35.7888 |
| expanded_cbc | rf | validation | C | 100398 | 0.7886 | 0.9418 | 58.2368 |
| expanded_cbc | rf | external | D | 30166 | 0.7878 | 0.9428 | 57.2429 |
| expanded_cbc | rf | external | E | 94320 | 0.8036 | 0.9665 | 33.5498 |
| expanded_cbc | rf | external | F | 20179 | 0.7735 | 0.9495 | 50.4950 |
| expanded_cbc | rf | external | G | 4983 | 0.8280 | 0.8581 | 141.8702 |
| expanded_cbc | rf | test | real_world | 58481 | 0.8729 | 0.9453 | 54.7241 |
| expanded_cbc | xgboost | validation | A | 59986 | 0.8609 | 0.9520 | 48.0141 |
| expanded_cbc | xgboost | validation | B | 51228 | 0.7974 | 0.9648 | 35.2122 |
| expanded_cbc | xgboost | validation | C | 100398 | 0.7725 | 0.9411 | 58.9018 |
| expanded_cbc | xgboost | external | D | 30166 | 0.7832 | 0.9457 | 54.2611 |
| expanded_cbc | xgboost | external | E | 94320 | 0.7918 | 0.9693 | 30.7059 |
| expanded_cbc | xgboost | external | F | 20179 | 0.7755 | 0.9513 | 48.7233 |
| expanded_cbc | xgboost | external | G | 4983 | 0.8151 | 0.8591 | 140.8833 |
| expanded_cbc | xgboost | test | real_world | 58481 | 0.8716 | 0.9476 | 52.3749 |

Full ranking metrics, confusion counts, and ten-bin reliability data for every site are in `metrics.json`. No individual predictions are exported.
