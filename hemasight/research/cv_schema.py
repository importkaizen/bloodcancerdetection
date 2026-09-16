"""Public aggregate contract for development-only cross-validation."""
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from .schema import FEATURE_SETS

Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
ModelName = Literal["prevalence", "logistic", "rf", "xgboost"]


class FoldResult(BaseModel):
    feature_set: Literal["basic_cbc", "expanded_cbc"]
    model: ModelName
    control: Literal["original", "shuffled_training_labels"]
    fold: int = Field(ge=1)
    train_n: int = Field(gt=0)
    train_positive: int = Field(gt=0)
    evaluation_n: int = Field(gt=0)
    evaluation_positive: int = Field(gt=0)
    auroc: Probability
    average_precision: Probability
    brier: Probability


class CrossValidationReport(BaseModel):
    schema_version: Literal["development_cv_v1"]
    status: Literal["completed"]
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    development_n: int = Field(gt=0)
    development_positive: int = Field(gt=0)
    started_at_utc: str
    completed_at_utc: str
    source_sha256: dict[str, str]
    versions: dict[str, str]
    seed: int = Field(ge=0, le=2**32-100)
    n_splits: int = Field(ge=2, le=10)
    split_strategy: Literal["stratified_kfold_shuffled_records"]
    fit_cohort: Literal["development"]
    evaluation_cohort: Literal["development"]
    shuffled_control: Literal["logistic_training_labels_per_fold"]
    models: list[ModelName] = Field(min_length=1)
    feature_sets: dict[str, list[str]] = Field(min_length=1)
    results: list[FoldResult]

    @model_validator(mode="after")
    def complete_paired_folds(self):
        if self.development_positive >= self.development_n:
            raise ValueError("Both classes are required")
        if len(set(self.models)) != len(self.models) or "logistic" not in self.models:
            raise ValueError("Distinct models including logistic are required")
        if any(name not in FEATURE_SETS or columns != FEATURE_SETS[name] for name, columns in self.feature_sets.items()):
            raise ValueError("Unexpected feature set")
        expected = {(features, model, "original", fold)
                    for features in self.feature_sets for model in self.models
                    for fold in range(1, self.n_splits + 1)}
        expected |= {(features, "logistic", "shuffled_training_labels", fold)
                     for features in self.feature_sets for fold in range(1, self.n_splits + 1)}
        keys = [(r.feature_set, r.model, r.control, r.fold) for r in self.results]
        if len(set(keys)) != len(keys) or set(keys) != expected:
            raise ValueError("Missing or duplicate planned fold results")
        counts = {}
        for row in self.results:
            if row.train_n + row.evaluation_n != self.development_n:
                raise ValueError("Fold sizes do not add up")
            if row.train_positive + row.evaluation_positive != self.development_positive:
                raise ValueError("Fold label counts do not add up")
            if row.train_positive >= row.train_n or row.evaluation_positive >= row.evaluation_n:
                raise ValueError("Each fold must contain both classes")
            pair = (row.evaluation_n, row.evaluation_positive)
            if row.fold in counts and counts[row.fold] != pair:
                raise ValueError("Experiments use inconsistent folds")
            counts[row.fold] = pair
        if sum(n for n, _ in counts.values()) != self.development_n or sum(p for _, p in counts.values()) != self.development_positive:
            raise ValueError("Evaluation folds do not partition development records")
        return self
