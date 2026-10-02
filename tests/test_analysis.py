import pytest

from backend import analysis
from backend.config import CO2_KG_PER_KWH, TARIFF_RESIDENTIAL_LARGE as T


def flat_predictor(rate=5.0):
    return lambda w, h, d, e: (w * h * d / 1000.0) * rate * (1.2 - 0.1 * (e - 1))


def test_carbon_scales_linearly():
    c = analysis.carbon_footprint(100)
    assert c["kg_per_month"] == pytest.approx(100 * CO2_KG_PER_KWH)
    assert c["kg_per_year"] == pytest.approx(c["kg_per_month"] * 12)
    assert c["trees_to_offset"] > 0


def test_cost_horizons():
    h = analysis.cost_horizons(300, 30)
    assert h["per_day"] == pytest.approx(10)
    assert h["per_year"] == pytest.approx(3600)


def test_recommendations_are_positive_and_sorted():
    recs = analysis.recommendations(flat_predictor(), 1200, 8, 30, 3)
    assert recs
    assert all(r.monthly_saving > 0 for r in recs)
    assert [r.monthly_saving for r in recs] == sorted(
        (r.monthly_saving for r in recs), reverse=True
    )
    assert all(r.yearly_saving == pytest.approx(r.monthly_saving * 12) for r in recs)


def test_no_efficiency_upgrade_suggested_at_label_5():
    recs = analysis.recommendations(flat_predictor(), 1200, 8, 30, 5)
    assert not any("label-5" in r.title for r in recs)


def test_idle_appliance_yields_no_advice():
    assert analysis.recommendations(flat_predictor(), 10, 0, 1, 5) == []


def test_efficiency_curve_is_monotonic():
    curve = analysis.efficiency_curve(flat_predictor(), 1200, 8, 30)
    costs = [c["cost"] for c in curve]
    assert len(curve) == 5
    assert costs == sorted(costs, reverse=True)


def test_household_rollup_totals_match_parts():
    items = [
        {"name": "AC", "wattage": 1200, "hours": 8, "days": 30, "efficiency": 4, "quantity": 2},
        {"name": "Fridge", "wattage": 150, "hours": 24, "days": 30, "efficiency": 5, "quantity": 1},
    ]
    roll = analysis.household_rollup(items, T, 100)
    assert roll["total_kwh"] == pytest.approx(576 + 108)
    assert roll["total_cost"] == pytest.approx(sum(a["cost"] for a in roll["appliances"]))
    assert roll["final_household_kwh"] == pytest.approx(100 + roll["total_kwh"])


def test_household_rollup_stacks_without_double_counting_the_first_block():
    """Two appliances together must cost what one combined load of the same size costs."""
    from backend.tariff import marginal_cost
    items = [
        {"name": "a", "wattage": 1000, "hours": 5, "days": 30, "efficiency": 3, "quantity": 1},
        {"name": "b", "wattage": 500, "hours": 5, "days": 30, "efficiency": 3, "quantity": 1},
    ]
    roll = analysis.household_rollup(items, T, 80)
    combined = marginal_cost(225, T, 80)["total"]
    assert roll["total_cost"] == pytest.approx(combined)


def test_empty_household():
    roll = analysis.household_rollup([], T, 200)
    assert roll["total_cost"] == 0
    assert roll["total_kwh"] == 0
