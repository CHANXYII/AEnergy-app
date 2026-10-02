"""End-to-end smoke tests: every page must render without raising."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
APPS = ROOT / "apps"
TIMEOUT = 300

PAGE_SCRIPT = """
import sys; sys.path.insert(0, {apps!r})
import streamlit as st
st.set_page_config(layout="wide")
from frontend import theme; theme.inject()
from frontend.pages import {module} as page
page.render()
"""


def _page(module: str, **state) -> AppTest:
    at = AppTest.from_string(PAGE_SCRIPT.format(apps=str(APPS), module=module),
                             default_timeout=TIMEOUT)
    for key, value in state.items():
        at.session_state[key] = value
    return at.run()


def _assert_clean(at: AppTest) -> AppTest:
    assert not at.exception, "\n".join(str(e.value) for e in at.exception)
    return at


def tile_values(at: AppTest) -> list[str]:
    return [
        m.group(1)
        for block in at.markdown
        for m in [re.search(r'ae-tile-value">([^<]+)<', str(block.value))]
        if m
    ]


@pytest.mark.parametrize(
    "module", ["estimator", "household", "insights", "control_center"]
)
def test_every_page_renders(module):
    _assert_clean(_page(module))


def test_entry_point_renders_and_navigates():
    at = _assert_clean(AppTest.from_file(str(ROOT / "main.py"), default_timeout=TIMEOUT).run())
    assert at.sidebar.selectbox[0].label == "Tariff"


def test_estimator_reacts_to_a_preset_change():
    at = _assert_clean(_page("estimator"))
    at.selectbox[0].set_value("EV home charger (7 kW)").run()
    _assert_clean(at)
    assert at.number_input[0].value == 7000


def test_estimator_shows_a_cost_and_an_uncertainty_band():
    at = _assert_clean(_page("estimator"))
    # Match the rendered div, not the stylesheet that also names the class.
    rendered = [
        m for block in at.markdown
        for m in [re.search(r'ae-hero-value">(฿[\d,]+)<.*?ae-hero-band\'>([^<]+)<',
                            str(block.value), re.S)]
        if m
    ]
    assert rendered, "no hero cost card rendered"
    cost, band = rendered[0].groups()
    assert float(cost.lstrip("฿").replace(",", "")) > 0
    assert "likely range" in band


def test_emptying_the_household_does_not_crash():
    """Regression: an empty list produced a column-less DataFrame."""
    _assert_clean(_page("household", household=[]))


def test_household_totals_track_the_appliance_list():
    fan = _page("household", household=[
        {"name": "Fan", "icon": "🌬️", "wattage": 50.0, "hours": 10.0,
         "days": 30, "efficiency": 4, "quantity": 1},
    ])
    ev = _page("household", household=[
        {"name": "EV charger", "icon": "🔌", "wattage": 7000.0, "hours": 3.0,
         "days": 20, "efficiency": 5, "quantity": 1},
    ])
    _assert_clean(fan)
    _assert_clean(ev)

    def baht(at):
        return float(tile_values(at)[0].lstrip("฿").replace(",", ""))

    assert baht(ev) > baht(fan) * 10


def test_control_center_rejects_an_impossible_record(tmp_path, monkeypatch):
    at = _assert_clean(_page("control_center"))
    at.number_input[0].set_value(1000)   # watts
    at.number_input[1].set_value(0.0)    # hours — zero energy
    at.number_input[2].set_value(30)     # days
    at.number_input[3].set_value(100.0)  # cost
    at.button[0].click().run()
    _assert_clean(at)
    assert at.error, "expected a validation error, record was accepted"
