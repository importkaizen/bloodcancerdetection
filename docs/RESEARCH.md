# Reproducible CBC screening research

HemaSight now has an offline research workflow alongside its original API and dashboard. The workflow compares ordinary CBC measurements across the released LeukoAlert cohorts. It does not create synthetic patient timelines or automatically publish a model to the API.

## Explore the saved benchmarks

With the local stack running, open [Research benchmarks](http://127.0.0.1:3000/research). Select the evaluation cohort and either five or fifteen CBC measurements to compare all four planned models. The detailed model selector updates hospital sensitivity/specificity, confusion counts, and probability calibration. The page also shows cohort sizes, the validation-selected operating threshold, source provenance, and the study limitations. These views read saved aggregates; they do not train models or repeat holdout evaluation.

The API serves `GET /research/runs` and `GET /research/runs/{run_id}`. Add `?download=true` to export the aggregate report. In the dashboard these routes are proxied beneath `/api`. Completed reports live in `docs/results/{run_id}/manifest.json` and `metrics.json`; `RESEARCH_REPORTS_DIR` can override the directory. The API container includes the saved reports at build time, so rebuild it to include a newly added report.

Reports are checked for required cohorts and experiments, count/metric consistency, site totals, reliability-bin totals, and fixed thresholds. Invalid reports are omitted from the catalog with an unavailable count, and direct retrieval returns a generic error. Only declared aggregate fields are returned; raw records, artifact paths, and fitted models are not served. A download is an aggregate view, not a replacement for the complete offline experiment manifest.

## Dataset comparison

Open [Dataset comparison](http://127.0.0.1:3000/research?view=dataset) to inspect the prepared measurements behind the saved benchmark. Select any of the 15 CBC measurements and compare pooled cohorts or individual hospitals. The table and spread chart show the median, middle 50%, and 5th–95th percentiles of observed values, alongside missingness and label prevalence. The displayed intervals are descriptive distributions, not clinical reference ranges or uncertainty intervals.

Generate an immutable profile for a completed run:

```sh
python -m hemasight.research.profile data/processed/leukoalert-v1/features.csv --run-directory docs/results/holdout-v1
```

The supplied run already has `profile.json`, so the command will refuse to overwrite it. For another completed run, use its directory. The generator verifies the prepared CSV checksum and cohort counts against that run. It summarizes only development and previously evaluated cohorts, without fitting, predicting, imputing, scaling, or changing labels. Quantiles use linear interpolation on observed values; entirely missing groups retain null quantiles. No record identifiers are exported.

The profile API is `GET /research/runs/{run_id}/profile`, with `?download=true` for an attachment. The API verifies units, ordered quantiles, pooled/site counts, allowed cohorts, and dataset identity. Missing profiles leave the original model-results view usable. Rebuild the API image after adding a profile to `docs/results`.

The saved release profile includes 15 pooled/site groups and 225 feature distributions. There are no missing values across the 15 selected measurements in the prepared release; this does not establish completeness of the original clinical records. WBC medians are 6.01 (development), 7.15 (validation), 6.50 (external), and 6.99 (test) in 10^9/L. Such differences describe the released groups and cannot identify why model performance differs.

## Development cross-validation and sanity checks

Open [Development checks](http://127.0.0.1:3000/research?view=development) for a five-fold, seed-42 diagnostic using only the 26,922 development records. This follow-up was added after inspecting the original benchmark; it is exploratory and does not replace its external/test results. All four fixed models and both measurement sets use the same shuffled stratified record splits. Imputation and logistic-regression scaling are fitted independently within each training fold. No operating thresholds, tuning, model selection, or deployment are performed.

```sh
python -m hemasight.research.cross_validation data/processed/leukoalert-v1/features.csv --run-directory docs/results/holdout-v1 --folds 5 --seed 42
```

The supplied run already contains `cross_validation.json`; the command refuses to overwrite it. For another completed run, use its directory. Dataset identity and development counts must match its manifest. Every development record is evaluated once per experiment; validation, external, and test records never reach the splitter, model fitting, or prediction. Only aggregate fold scores and source/dependency fingerprints are saved.

The control fits logistic regression after shuffling each training fold's labels, then scores the unchanged evaluation labels. It uses one shuffle per fold, with seed equal to base seed plus fold number, also used as the estimator seed. This is a descriptive negative control, not a formal permutation test. No p-values or pass/fail decisions are produced.

The dashboard reports equally weighted fold means and observed fold ranges. Training sets overlap, stratification can hide variation, and absent patient identifiers prevent checking repeated-patient leakage. Ranges are not confidence intervals. See the [scikit-learn cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html) and [permutation-test documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html) for the distinction from formal permutation inference.

| Features | Model | Mean development AUROC | Fold range |
|---|---|---:|---:|
| 5 | Logistic regression | 0.8989 | 0.8941–0.9046 |
| 5 | Random Forest | 0.9494 | 0.9450–0.9537 |
| 5 | XGBoost | 0.9454 | 0.9421–0.9508 |
| 15 | Logistic regression | 0.9231 | 0.9180–0.9251 |
| 15 | Random Forest | 0.9748 | 0.9724–0.9779 |
| 15 | XGBoost | 0.9688 | 0.9661–0.9720 |

Prevalence baselines have AUROC 0.5 in each fold. Shuffled-label logistic controls have mean AUROC 0.3718 (five features; range 0.2667–0.4757) and 0.3724 (fifteen features; range 0.2315–0.5188). These below-chance averages are retained exactly as observed; a single training-label shuffle per fold is too limited to characterize a null distribution or certify leakage absence. The seeds and result files were not changed to make the control look closer to chance.

The read-only endpoint is `GET /research/runs/{run_id}/cross-validation`; add `?download=true` for aggregate JSON. The API verifies the expected experiments, fold sizes/class counts, metric bounds, dataset identity, and development-only scope. Rebuild the API image after adding a new report. All 50 fold evaluations are available in the saved file and dashboard.

## Install and verify

Python 3.11 is the tested environment. The pinned `requirements.lock` was resolved on Windows; the CI workflow uses Windows and Python 3.11. The package also declares compatible minimum dependencies for other platforms, but those combinations have not been validated here.

```sh
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python -m pytest -q
```

With uv, use `uv venv --python 3.11`, `uv pip install -r requirements.lock`, and `uv pip install --no-deps -e .`. PyTorch is optional: install `.[deep-learning]` only for the experimental LSTM/federated paths.

## Prepare the data

Source: Liu, Shilong (2026), [LeukoAlert-project, version 1](https://data.mendeley.com/datasets/vc7kwnyppz/1), DOI `10.17632/vc7kwnyppz.1`, CC BY 4.0. See the [accompanying paper](https://www.nature.com/articles/s41746-026-03016-3). Cite the source when publishing analyses.

```sh
python -m hemasight.research.leukoalert download --output data/raw/LeukoAlert-project.zip
python -m hemasight.research.leukoalert prepare --archive data/raw/LeukoAlert-project.zip --output data/processed/leukoalert-v1
```

The download and preparation commands verify the archive SHA-256. Preparation reads only explicitly named CSV entries from the ZIP; it never extracts or runs the supplied research code. Existing dataset directories are not overwritten.

Outputs:

- `features.csv`: 15 canonical CBC columns, binary `target`, sample provenance, site, and cohort.
- `manifest.json`: source checksum, prepared-data checksum, units, label counts, missing/invalid numeric counts, duplicate sample-ID counts, and interpretation limits.
- `audit.md`: readable source/cohort summary.

The five-feature subset is WBC, RBC, platelets, hemoglobin, and lymphocyte percentage. The expanded subset adds hematocrit, MCV, MCH, MCHC, RDW-CV, and absolute neutrophil, monocyte, eosinophil, basophil, and lymphocyte counts. Hemoglobin and MCHC are converted from g/L to g/dL. Lymphocyte percentage and absolute count remain different features. Missing or nonnumeric entries remain missing, with counts recorded; no imputation or clinical range filtering occurs during preparation.

| Original files | Canonical cohort | Role |
| --- | --- | --- |
| training, sites A–C | development | Fit preprocessing and models |
| validation, sites A–C | validation | Choose the operating threshold; assess development performance |
| validation, sites D–G | external | Evaluate only after the experiment is fixed |
| test_site_true_world | test | Final evaluation, using the released file |

The canonical `sample_id` namespaces the original sample value by file for provenance. It is never a patient identifier. `real_world` is a source-file label, not a newly identified hospital.

## Run the benchmark

```sh
python -m hemasight.research.benchmark data/processed/leukoalert-v1/features.csv --output runs/development-v1 --models logistic rf xgboost
```

Every run includes a prevalence-only baseline plus the requested models, using fixed hyperparameters and seed 42 by default. Both feature sets run by default. Median imputation is fitted on development data only; logistic regression also scales using development data. No feature selection, oversampling, probability recalibration, or hyperparameter search is performed.

The threshold targets 95% specificity in the pooled validation cohort. Tied negative scores are handled conservatively. The same selected threshold is applied at every evaluation site. Validation is used to select that threshold, so its operating-point performance is not an unbiased estimate of the selected procedure.

After freezing the experiment, explicitly evaluate external and test cohorts:

```sh
python -m hemasight.research.benchmark data/processed/leukoalert-v1/features.csv --output runs/holdout-v1 --models logistic rf xgboost --evaluate-holdouts
```

An ordinary development run does not evaluate either holdout. The explicit holdout command does not enforce a global one-use lock: researchers must retain the experiment history and avoid iteratively tuning to those results.

Each new output directory contains `manifest.json`, `metrics.json`, `leaderboard.md`, and separate model bundles with their fitted preprocessing, feature order, and threshold. The manifest records dataset identity, source-code identity, software versions, seed, and parameters. Previous runs are never overwritten. Reports contain aggregate results, not individual predictions.

Metrics include AUROC, average precision, Brier score, sensitivity, specificity, precision, false positives per 1,000 negative records, and ten-bin reliability summaries, overall and by site. Single-class cohorts have null ranking metrics; undefined denominators are null. These are record-level descriptive results. Patient-cluster confidence intervals are deliberately absent because the release does not supply patient linkage.

## Interpretation limits and source discrepancies

- The release contains sample IDs but no explicit patient IDs, measurement dates, or diagnosis dates. Repeated patients and overlap across cohorts cannot be verified. Do not call these patient-disjoint results or evidence of months-ahead detection.
- The inspected binary-task files contain 446,663 records; the article reports 446,558. The 105-record discrepancy is retained in the audit.
- The binary files have 76 columns, while the paper describes a 72-feature model. This implementation is an explicitly different 5/15-feature benchmark, not an exact reproduction of that model.
- The negative string label is `healthy`; it does not independently prove disease absence or describe the diagnostic verification process.
- The training files are enriched for the positive label. Ranking performance, probability calibration, and predictive value must be considered separately when evaluating other cohorts.
- The public dataset's cohort design and diagnoses require further study-documentation review before making clinical claims. [TRIPOD+AI](https://www.bmj.com/content/bmj/385/bmj-2023-078378.full.pdf) is a useful reporting reference.

## Longitudinal feature migration

Worker feature version `v2` uses the latest five eligible visits, including the target, with stable ordering by measurement date and record ID. Trends are changes per elapsed day. Missing values keep their original dates; fewer than two distinct usable times yields a missing slope. Rolling means and variance use the same window.

Existing `v1` feature rows and trained models are not numerically compatible with these semantics. Rebuild the cohort and retrain with the matching feature-version setting before enabling worker inference. The ten-row example CSV has legacy illustrative feature values and remains a smoke-test fixture only.

The worker reconstructs history from records currently available in the database. An older test arriving after a newer test was processed does not automatically rebuild every affected later result. Same-day ordering uses an ID tie-breaker because the existing API stores dates. Neither mechanism establishes real historical information availability. Optional ingestion retry keys and worker replay guards are now implemented; availability timestamps, deliberate cohort reconstruction, and automatic backfill orchestration remain future work. See [application verification](APPLICATION_TESTING.md) for the tested scope.

Tabular training now saves its fitted imputation/scaling together and supports explicit patient grouping. LSTM artifacts use their own scaler/config paths. Experimental federated/LSTM routines remain outside this benchmark; their training fit is not a validated research result.
