"""Aggregate-only dataset profile contract, shared by the CLI and API."""
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from .schema import COLUMN_MAP

Finite = Annotated[float, Field(allow_inf_nan=False)]


class Distribution(BaseModel):
    observed: int = Field(ge=0)
    missing: int = Field(ge=0)
    p05: Finite | None
    q1: Finite | None
    median: Finite | None
    q3: Finite | None
    p95: Finite | None

    @model_validator(mode="after")
    def valid_quantiles(self):
        values = [self.p05, self.q1, self.median, self.q3, self.p95]
        if self.observed == 0:
            if any(v is not None for v in values):
                raise ValueError("Empty distributions must have null quantiles")
        elif any(v is None for v in values) or values != sorted(values):
            raise ValueError("Quantiles must be finite and ordered")
        return self


class GroupProfile(BaseModel):
    cohort: Literal["development", "validation", "external", "test"]
    scope: Literal["overall", "site"]
    site: str | None = Field(max_length=80)
    n: int = Field(gt=0)
    n_positive: int = Field(ge=0)
    features: dict[str, Distribution]

    @model_validator(mode="after")
    def consistent_group(self):
        if self.n_positive > self.n or (self.scope == "overall" and self.site is not None) or (self.scope == "site" and not self.site):
            raise ValueError("Invalid group counts or scope")
        if set(self.features) != set(COLUMN_MAP):
            raise ValueError("Profile must contain the canonical CBC features")
        if any(d.observed + d.missing != self.n for d in self.features.values()):
            raise ValueError("Feature counts must add up to group size")
        return self


class DatasetProfile(BaseModel):
    schema_version: Literal["cbc_profile_v1"]
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_at_utc: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    units: dict[str, str]
    groups: list[GroupProfile] = Field(min_length=1)

    @model_validator(mode="after")
    def consistent_groups(self):
        if self.units != {name: spec[2] for name, spec in COLUMN_MAP.items()}:
            raise ValueError("Measurement units differ from the canonical schema")
        keys = [(g.cohort, g.scope, g.site) for g in self.groups]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate profile group")
        for cohort in {g.cohort for g in self.groups}:
            overall = [g for g in self.groups if g.cohort == cohort and g.scope == "overall"]
            sites = [g for g in self.groups if g.cohort == cohort and g.scope == "site"]
            if len(overall) != 1 or not sites:
                raise ValueError("Missing overall or site profiles")
            total = overall[0]
            if sum(s.n for s in sites) != total.n or sum(s.n_positive for s in sites) != total.n_positive:
                raise ValueError("Site totals differ from pooled counts")
            for feature in COLUMN_MAP:
                if sum(s.features[feature].observed for s in sites) != total.features[feature].observed:
                    raise ValueError("Site observed counts differ from pooled counts")
        return self
