import pandas as pd
import pytest

from backend import dataset
from backend.dataset import DataError, add_record, calculate_kwh, load_data, validate, write_data


def test_calculate_kwh():
    assert calculate_kwh(1000, 10, 30) == pytest.approx(300.0)


def test_validate_rejects_out_of_range_rows():
    frame = pd.DataFrame([
        {"wattage": 1000, "hours": 8, "days": 30, "efficiency": 5, "cost": 500},   # good
        {"wattage": 1000, "hours": 30, "days": 30, "efficiency": 5, "cost": 500},  # hours > 24
        {"wattage": -5, "hours": 8, "days": 30, "efficiency": 5, "cost": 500},     # negative watts
        {"wattage": 1000, "hours": 8, "days": 30, "efficiency": 9, "cost": 500},   # bad label
    ])
    report = validate(frame)
    assert len(report.frame) == 1
    assert report.dropped == 3
    assert report.reasons


def test_validate_coerces_text_numbers_and_drops_junk():
    frame = pd.DataFrame([
        {"wattage": "1000", "hours": "8", "days": "30", "efficiency": "5", "cost": "500"},
        {"wattage": "oops", "hours": 8, "days": 30, "efficiency": 5, "cost": 500},
    ])
    report = validate(frame)
    assert len(report.frame) == 1
    assert report.frame["wattage"].iloc[0] == 1000


def test_validate_drops_duplicates():
    row = {"wattage": 1000, "hours": 8, "days": 30, "efficiency": 5, "cost": 500}
    assert len(validate(pd.DataFrame([row, row, row])).frame) == 1


def test_validate_requires_columns():
    with pytest.raises(DataError):
        validate(pd.DataFrame([{"watts": 1}]))


def test_strict_mode_raises():
    bad = pd.DataFrame([{"wattage": 1000, "hours": 99, "days": 30, "efficiency": 5, "cost": 1}])
    with pytest.raises(DataError):
        validate(bad, strict=True)


def test_add_features_computes_effective_rate():
    frame = pd.DataFrame([{"wattage": 1000, "hours": 10, "days": 30, "efficiency": 5, "cost": 1500}])
    feat = dataset.add_features(frame)
    assert feat["kwh"].iloc[0] == pytest.approx(300)
    assert feat["rate"].iloc[0] == pytest.approx(5.0)


def test_synthesize_is_deterministic_and_valid():
    a, b = dataset.synthesize(40, seed=7), dataset.synthesize(40, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert validate(a).dropped == 0
    assert (a["cost"] > 0).all()


def test_round_trip_and_append(tmp_path):
    path = tmp_path / "d.csv"
    write_data(dataset.synthesize(20), path)
    before = load_data(path)
    after = add_record(1500, 3.0, 20, 4, 420.0, path=path)
    assert len(after) == len(before) + 1
    assert len(load_data(path)) == len(after)


def test_add_record_rejects_impossible_values(tmp_path):
    path = tmp_path / "d.csv"
    write_data(dataset.synthesize(20), path)
    with pytest.raises(DataError):
        add_record(1000, 99, 30, 5, 100.0, path=path)


def test_write_is_atomic_leaving_no_temp_files(tmp_path):
    path = tmp_path / "d.csv"
    write_data(dataset.synthesize(10), path)
    assert list(tmp_path.glob("*.tmp")) == []


def test_load_data_tolerates_an_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    assert load_data(path, seed_if_missing=False).empty
