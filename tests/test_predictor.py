import pytest

from backend.config import TARIFF_RESIDENTIAL_LARGE as T
from backend.dataset import calculate_kwh, synthesize
from backend.predictor import (
    MIN_ROWS,
    ModelError,
    load_history,
    load_model,
    predict,
    train,
)
from backend.tariff import marginal_cost


@pytest.fixture(scope="module")
def dataset():
    return synthesize(160, seed=3)


@pytest.fixture(scope="module")
def bundle(dataset, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("model")
    return train(dataset, tariff=T, model_path=tmp / "m.joblib", history_path=tmp / "h.json")


def test_refuses_to_train_on_too_little_data():
    with pytest.raises(ModelError):
        train(synthesize(MIN_ROWS - 1))


def test_metrics_are_out_of_fold_not_in_sample(bundle):
    m = bundle.metrics["model"]
    assert 0.0 < m["r2"] <= 1.0
    # An in-sample forest would score ~0.999; held-out folds must leave real error.
    assert m["mae"] > 0
    assert bundle.metrics["n_splits"] >= 2


def test_beats_a_forest_fitted_directly_on_cost(bundle):
    ours = bundle.metrics["model"]["mae"]
    naive = bundle.metrics["baseline_forest_on_cost"]["mae"]
    assert ours < naive


def test_beats_the_pure_tariff_formula(bundle):
    assert bundle.metrics["model"]["mae"] < bundle.metrics["baseline_tariff_only"]["mae"]


def test_prediction_scales_with_usage(bundle):
    low = predict(bundle, 1000, 2, 30, 3)["cost"]
    high = predict(bundle, 1000, 8, 30, 3)["cost"]
    assert high > low * 2


def test_better_efficiency_never_costs_more(bundle):
    costs = [predict(bundle, 1200, 8, 30, e)["cost"] for e in (1, 2, 3, 4, 5)]
    assert costs[0] > costs[-1]


def test_extrapolates_beyond_the_training_range(dataset, tmp_path):
    """The headline fix: a forest fitted on cost caps out; this engine does not."""
    cheap = dataset[dataset["cost"] < dataset["cost"].quantile(0.75)]
    b = train(cheap, tariff=T, model_path=tmp_path / "m.joblib", history_path=tmp_path / "h.json")

    got = predict(b, 7000, 4, 30, 5)["cost"]
    truth = marginal_cost(calculate_kwh(7000, 4, 30) * 0.8, T, 250)["total"]

    assert got > cheap["cost"].max()          # not capped by the training maximum
    assert abs(got - truth) / truth < 0.20    # and in the right neighbourhood


def test_uncertainty_band_brackets_the_estimate(bundle):
    p = predict(bundle, 1200, 8, 30, 3)
    assert p["cost_low"] <= p["cost"] <= p["cost_high"]
    assert p["cost_high"] > p["cost_low"]


def test_zero_usage_costs_nothing(bundle):
    assert predict(bundle, 1000, 0, 30, 5)["cost"] == pytest.approx(0.0)


def test_persisted_model_round_trips(dataset, tmp_path):
    path = tmp_path / "m.joblib"
    original = train(dataset, tariff=T, model_path=path, history_path=tmp_path / "h.json")
    reloaded = load_model(path)
    assert reloaded is not None
    assert reloaded.n_samples == original.n_samples
    a = predict(original, 1200, 8, 30, 4)["cost"]
    b = predict(reloaded, 1200, 8, 30, 4)["cost"]
    assert a == pytest.approx(b)


def test_stale_schema_is_treated_as_missing(dataset, tmp_path):
    import joblib
    path = tmp_path / "m.joblib"
    train(dataset, model_path=path, history_path=tmp_path / "h.json")
    payload = joblib.load(path)
    payload["schema_version"] = 0
    joblib.dump(payload, path)
    assert load_model(path) is None


def test_corrupt_model_file_is_treated_as_missing(tmp_path):
    path = tmp_path / "m.joblib"
    path.write_bytes(b"not a model")
    assert load_model(path) is None


def test_missing_model_file_returns_none(tmp_path):
    assert load_model(tmp_path / "nope.joblib") is None


def test_history_accumulates(dataset, tmp_path):
    hist = tmp_path / "h.json"
    for _ in range(3):
        train(dataset, model_path=tmp_path / "m.joblib", history_path=hist)
    assert len(load_history(hist)) == 3
