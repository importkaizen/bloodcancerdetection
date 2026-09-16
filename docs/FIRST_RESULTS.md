# First local benchmark results

The first fixed experiment completed on 2026-09-13 using LeukoAlert v1. All eight prespecified combinations were evaluated: prevalence-only, logistic regression, Random Forest, and XGBoost with five and fifteen CBC features. This is a new baseline comparison, not a reproduction of the source paper's model.

Models fit 26,922 development records. Thresholds targeted 95% specificity using 211,612 validation records. Frozen models and thresholds were then evaluated on 149,648 external records and 58,481 records from the released real-world test file. No tuning followed these results.

## Main observations

Expanded CBC features improved the tree models' ranking and sensitivity in this experiment. For example, Random Forest's external AUROC increased from 0.9448 to 0.9643 and sensitivity from 72.08% to 79.88%. Test AUROC increased from 0.9506 to 0.9699 and sensitivity from 77.95% to 87.29%. These are descriptive comparisons; statistical significance has not been established.

| Fifteen-feature model | External AUROC | External sensitivity | External specificity | Test AUROC | Test sensitivity | Test specificity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.9205 | 62.88% | 95.90% | 0.9308 | 67.38% | 95.36% |
| Random Forest | 0.9643 | 79.88% | 95.64% | 0.9699 | 87.29% | 94.53% |
| XGBoost | 0.9648 | 79.08% | 95.91% | 0.9689 | 87.16% | 94.76% |

Hospital variation is substantial. With expanded XGBoost, specificity was 96.93% at external site E but 85.91% at site G. That corresponds to about 31 versus 141 false positives per 1,000 negative records. A pooled threshold does not guarantee the same operating characteristics at every hospital.

Calibration remains a research problem. At the test prevalence of 3.92%, expanded XGBoost had precision 40.41% and Brier score 0.0508. A constant probability equal to that test prevalence would have a Brier score of about 0.0377; that is a descriptive reference using observed test prevalence, not a deployable fitted comparator. The prespecified development-prevalence baseline scored 0.1242. High AUROC alone does not establish reliable individual probabilities.

## Evidence and interpretation

The complete [comparison by cohort and hospital](results/holdout-v1/leaderboard.md), [aggregate metrics and reliability bins](results/holdout-v1/metrics.json), and [run manifest](results/holdout-v1/manifest.json) are preserved with the source. The manifest includes dataset and preparation checksums, source-file hashes, package versions, parameters, seed, and cohort counts. The local run also retains eight fitted bundles under `runs/holdout-v1/models/`.

The source contains sample IDs but no patient linkage or diagnosis dates. These are record-level screening results; patient overlap cannot be excluded and advance prediction cannot be evaluated. The preparation audit preserves the 105-record discrepancy between released binary-task files and the paper's stated total. See [the research guide](RESEARCH.md) for source attribution and limitations.

The next study should investigate hospital differences and probability calibration using development/validation data, with a new untouched evaluation cohort for subsequent model claims. The external and test outcomes above have now been inspected and must not be treated as untouched for future model selection.

## Build verification

54 automated tests passed on local Windows with Python 3.11. They exercise preparation, cohort boundaries, development-only preprocessing, threshold selection, aggregate reporting, longitudinal feature windows, artifact version checks, and saved tabular prediction with missing values. The complete eight-model benchmark also ran successfully.

The Docker application stack and optional PyTorch LSTM/federated routines were not run. This build keeps benchmark models offline. Longitudinal feature version v2 requires matching retrained artifacts before worker inference.
