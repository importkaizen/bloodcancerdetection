"""Database-independent longitudinal CBC features.

Version 2 uses the latest five visits, including the target test. Slopes are
measured per elapsed day; missing values do not compress the time axis.
"""
from datetime import datetime
from math import isfinite
from typing import Iterable, Optional, Protocol, Sequence

import numpy as np

FEATURE_VERSION = "v2"
WINDOW_SIZE = 5


class BloodTestRecord(Protocol):
    id: int
    patient_id: int
    date: datetime
    wbc: Optional[float]
    rbc: Optional[float]
    platelets: Optional[float]
    hemoglobin: Optional[float]
    lymphocytes: Optional[float]


def _finite_value(value: Optional[float]) -> Optional[float]:
    if value is None or not isfinite(value):
        return None
    return float(value)


def select_history_window(
    target: BloodTestRecord,
    history: Iterable[BloodTestRecord],
    window_size: int = WINDOW_SIZE,
) -> list[BloodTestRecord]:
    """Return chronological visits available at the target, with stable ties.

    IDs break ties only for equal timestamps. The target itself is always
    included, even if the caller supplies only prior history.
    """
    if window_size < 1:
        raise ValueError("window_size must be positive")
    cutoff = (target.date, target.id)
    eligible = {
        record.id: record
        for record in history
        if record.patient_id == target.patient_id
        and (record.date, record.id) <= cutoff
    }
    eligible[target.id] = target
    return sorted(eligible.values(), key=lambda record: (record.date, record.id))[-window_size:]


def trend_slope(
    values: Sequence[Optional[float]], dates: Sequence[datetime]
) -> Optional[float]:
    """Least-squares slope per day over finite observations and their dates."""
    if len(values) != len(dates):
        raise ValueError("Each value must have a corresponding date")
    observations = [
        (date, clean)
        for value, date in zip(values, dates)
        if (clean := _finite_value(value)) is not None
    ]
    if len(observations) < 2:
        return None
    origin = observations[0][0]
    elapsed = np.array(
        [(date - origin).total_seconds() / 86400.0 for date, _ in observations],
        dtype=float,
    )
    centered_time = elapsed - elapsed.mean()
    denominator = float(np.dot(centered_time, centered_time))
    if denominator == 0.0:
        return None
    measurements = np.array([value for _, value in observations], dtype=float)
    slope = np.dot(centered_time, measurements - measurements.mean()) / denominator
    return _finite_value(slope)


def _variance(values: Sequence[Optional[float]]) -> Optional[float]:
    clean = [clean for value in values if (clean := _finite_value(value)) is not None]
    return float(np.var(clean)) if len(clean) >= 2 else None


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    clean = [clean for value in values if (clean := _finite_value(value)) is not None]
    return float(np.mean(clean)) if clean else None


def compute_longitudinal_features(
    target: BloodTestRecord,
    history: Iterable[BloodTestRecord],
    window_size: int = WINDOW_SIZE,
) -> dict[str, Optional[float]]:
    """Compute every aggregate from the same target-inclusive visit window.

    Variance is population variance (ddof=0). Means use available values within
    the visit window; missing tests never cause older visits to be substituted.
    """
    window = select_history_window(target, history, window_size)
    dates = [record.date for record in window]
    columns = {
        name: [getattr(record, name) for record in window]
        for name in ("wbc", "rbc", "platelets", "hemoglobin", "lymphocytes")
    }
    hemoglobin_slope = trend_slope(columns["hemoglobin"], dates)
    result = {
        "wbc_trend": trend_slope(columns["wbc"], dates),
        "platelet_var": _variance(columns["platelets"]),
        "hemoglobin_drop_rate": -hemoglobin_slope if hemoglobin_slope is not None else None,
        "lymphocyte_spike": trend_slope(columns["lymphocytes"], dates),
        **{
            f"{name}_rolling_avg": _mean(columns[name])
            for name in ("wbc", "rbc", "platelets", "hemoglobin")
        },
        **{name: _finite_value(getattr(target, name)) for name in columns},
    }
    return result
