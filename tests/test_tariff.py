import math

import pytest

from backend.config import TARIFF_FLAT_LEGACY, TARIFF_RESIDENTIAL_LARGE as T
from backend.tariff import bill, energy_charge, marginal_cost, marginal_rate, tier_segments


def test_energy_charge_matches_published_blocks():
    # 150 @ 3.2484 + 250 @ 4.2218 + 100 @ 4.4217
    expected = 150 * 3.2484 + 250 * 4.2218 + 100 * 4.4217
    assert energy_charge(500, T) == pytest.approx(expected)


def test_first_block_only():
    assert energy_charge(100, T) == pytest.approx(100 * 3.2484)


def test_bill_components_sum_to_total():
    b = bill(420, T)
    assert b["subtotal"] == pytest.approx(b["energy"] + b["ft"] + b["service"])
    assert b["total"] == pytest.approx(b["subtotal"] * (1 + T.vat_rate))


def test_zero_consumption_still_owes_service_charge():
    assert bill(0, T)["total"] == pytest.approx(T.service_charge * 1.07)
    # but the appliance's marginal cost is nil
    assert marginal_cost(0, T, 300)["total"] == 0.0


def test_marginal_cost_is_progressive():
    """The same appliance costs more in a heavier-consuming household."""
    light = marginal_cost(100, T, 0)["total"]
    heavy = marginal_cost(100, T, 500)["total"]
    assert heavy > light


def test_marginal_cost_excludes_service_charge():
    assert marginal_cost(100, T, 0)["service"] == 0.0


def test_marginal_cost_is_additive():
    """Splitting a load in two must not change what it costs."""
    whole = marginal_cost(200, T, 250)["total"]
    split = marginal_cost(120, T, 250)["total"] + marginal_cost(80, T, 370)["total"]
    assert whole == pytest.approx(split)


def test_segments_account_for_all_kwh():
    segs = tier_segments(620, T, start_kwh=90)
    assert sum(s["kwh"] for s in segs) == pytest.approx(620)
    assert all(s["charge"] == pytest.approx(s["kwh"] * s["rate"]) for s in segs)


def test_segments_start_in_the_right_block():
    """A household already past 400 kWh buys only top-block energy."""
    segs = tier_segments(50, T, start_kwh=450)
    assert len(segs) == 1
    assert segs[0]["rate"] == pytest.approx(4.4217)


def test_flat_legacy_tariff_reproduces_old_behaviour():
    assert bill(100, TARIFF_FLAT_LEGACY)["total"] == pytest.approx(450.0)


def test_marginal_rate_rises_with_baseline():
    rates = [marginal_rate(T, b) for b in (0, 200, 500)]
    assert rates == sorted(rates)
    assert all(math.isfinite(r) for r in rates)
