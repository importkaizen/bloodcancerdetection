# First CBC screening benchmark protocol

Specified before evaluating the external/test model results for this build.

Question: How does performance vary across the released hospital cohorts when using five common CBC measurements versus an expanded fifteen-feature CBC set?

Data: checksum-pinned LeukoAlert v1. Preserve original files and cohort assignments. Use the released leukemia-versus-healthy label without inferring incident disease or patient histories. Investigate the documented row-count and feature-count discrepancies separately.

Comparisons: prevalence-only classifier, logistic regression, Random Forest, and XGBoost for each feature set. Fixed hyperparameters in `hemasight/research/benchmark.py`, seed 42. No hyperparameter tuning or selection based on external/test outcomes. Median imputation on development data; scaling for logistic regression on development data.

Operating point: choose the smallest representable threshold achieving at least 95% specificity on pooled validation negatives, handling ties conservatively. Freeze this threshold before evaluating external/test records. Report each model and feature set, not only the best result.

Evaluation: ranking (AUROC, average precision), calibration (Brier score, reliability bins), operating-point sensitivity, specificity, precision, and false positives per 1,000 negative records. Report counts/prevalence and per-site results. No naive confidence intervals assuming repeat records are independent. No patient-disjoint or clinical efficacy claims.

Model bundles remain offline. This study does not change the dashboard's clinical interpretation, set medical decision thresholds, or deploy a new screening service. Any subsequent model revision requires a new documented experiment and a defensible untouched evaluation cohort.
