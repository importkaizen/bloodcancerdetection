"""Regression coverage for visit windows and elapsed-time feature semantics."""
import importlib
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from hemasight.db.models import Base, BloodTest, Patient
from hemasight.ml.features import (
    FEATURE_VERSION,
    compute_longitudinal_features,
    select_history_window,
    trend_slope,
)


START = datetime(2024, 1, 1)


def record(test_id, day, **overrides):
    values = dict(
        id=test_id,
        patient_id=1,
        date=START + timedelta(days=day),
        wbc=10.0 + 2.0 * day,
        rbc=5.0,
        platelets=100.0 + day,
        hemoglobin=15.0 - 0.1 * day,
        lymphocytes=1.0 + 0.05 * day,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_latest_five_visits_include_target_and_all_metrics_share_window():
    history = [record(index + 1, index) for index in range(12)]
    # An old outlier must not affect variance or any other recent aggregate.
    history[0].platelets = 10000.0
    target = history[-1]
    assert [row.id for row in select_history_window(target, history)] == [8, 9, 10, 11, 12]

    result = compute_longitudinal_features(target, list(reversed(history)))
    assert result["wbc_rolling_avg"] == pytest.approx(28.0)
    assert result["rbc_rolling_avg"] == pytest.approx(5.0)
    assert result["platelets_rolling_avg"] == pytest.approx(109.0)
    assert result["hemoglobin_rolling_avg"] == pytest.approx(14.1)
    assert result["platelet_var"] == pytest.approx(2.0)
    assert result["wbc_trend"] == pytest.approx(2.0)
    assert result["hemoglobin_drop_rate"] == pytest.approx(0.1)
    assert result["lymphocyte_spike"] == pytest.approx(0.05)


def test_slopes_use_irregular_elapsed_days_including_partial_days():
    history = [record(index + 1, day) for index, day in enumerate([0, 0.5, 3, 20, 60])]
    result = compute_longitudinal_features(history[-1], history)
    assert result["wbc_trend"] == pytest.approx(2.0)
    assert result["hemoglobin_drop_rate"] == pytest.approx(0.1)
    assert result["lymphocyte_spike"] == pytest.approx(0.05)


@pytest.mark.parametrize("missing", [None, float("nan"), float("inf")])
def test_missing_observations_keep_their_original_dates(missing):
    dates = [START + timedelta(days=day) for day in [0, 1, 10]]
    assert trend_slope([2.0, missing, 22.0], dates) == pytest.approx(2.0)


def test_missing_values_do_not_pull_older_observations_into_window():
    history = [record(index + 1, index, wbc=1000.0) for index in range(7)]
    for row, value in zip(history[-5:], [None, float("nan"), 6.0, None, 10.0]):
        row.wbc = value
        row.platelets = None
    result = compute_longitudinal_features(history[-1], history)
    assert result["wbc_rolling_avg"] == pytest.approx(8.0)
    assert result["wbc_trend"] == pytest.approx(2.0)
    assert result["platelet_var"] is None
    assert result["platelets_rolling_avg"] is None


@pytest.mark.parametrize(
    "values, days",
    [
        ([], []),
        ([5.0], [0]),
        ([None, 5.0], [0, 1]),
        ([5.0, 6.0], [1, 1]),
        ([5.0, 6.0, None], [1, 1, 2]),
    ],
)
def test_trend_requires_two_finite_observations_at_distinct_times(values, days):
    assert trend_slope(values, [START + timedelta(days=day) for day in days]) is None


def test_equal_time_ids_and_future_tests_are_excluded_deterministically():
    target = record(10, 5)
    history = [
        record(1, 6, wbc=1000.0),  # A smaller ID does not make a later test eligible.
        record(11, 5, wbc=1000.0),
        record(9, 5),
        record(8, 4),
        record(99, 4, patient_id=2, wbc=1000.0),
    ]
    assert [row.id for row in select_history_window(target, history)] == [8, 9, 10]
    assert compute_longitudinal_features(target, history)["wbc_trend"] == pytest.approx(2.0)


def test_missing_target_raw_values_are_returned_as_none():
    target = record(1, 0, wbc=float("nan"), rbc=None)
    result = compute_longitudinal_features(target, [])
    assert result["wbc"] is None
    assert result["rbc"] is None
    assert result["wbc_rolling_avg"] is None
    assert result["wbc_trend"] is None


def test_sqlite_query_uses_recent_window_and_matches_pure_selection():
    from hemasight.workers.feature_worker import compute_features_for_blood_test, load_history_window

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    history = [record(index + 1, index) for index in range(12)]
    history.extend([record(13, 11), record(14, 12), record(15, 10, patient_id=2)])
    with Session(engine) as db:
        db.add_all([Patient(id=1, external_id="one"), Patient(id=2, external_id="two")])
        db.add_all([BloodTest(**vars(row)) for row in history])
        db.commit()
        target = db.get(BloodTest, 12)
        window = load_history_window(db, target)
        assert [row.id for row in window] == [8, 9, 10, 11, 12]
        assert [row.id for row in window] == [row.id for row in select_history_window(target, history)]
        same_time_target = db.get(BloodTest, 13)
        assert [row.id for row in load_history_window(db, same_time_target)] == [9, 10, 11, 12, 13]
        feature = compute_features_for_blood_test(db, target, window)
        db.add(feature)
        db.commit()
        assert feature.feature_version == FEATURE_VERSION == "v2"
        assert feature.blood_test_id == 12
        assert feature.wbc_rolling_avg == pytest.approx(28.0)
        assert feature.platelet_var == pytest.approx(2.0)
    engine.dispose()


def test_worker_import_does_not_initialize_database(monkeypatch):
    import hemasight.db.models as models
    import hemasight.workers.feature_worker as worker

    def fail_if_engine_created(*args, **kwargs):
        raise AssertionError("Importing the worker must not initialize the database")

    monkeypatch.setattr(models, "get_engine", fail_if_engine_created)
    try:
        importlib.reload(worker)
    finally:
        monkeypatch.undo()
        importlib.reload(worker)
