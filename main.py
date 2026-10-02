"""AEnergy — entry point.

Run with:  streamlit run main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `apps/` importable without requiring `pip install -e .`, so a fresh
# clone runs with nothing but `streamlit run main.py`. An editable install
# works too — this just becomes a no-op.
APPS = Path(__file__).resolve().parent / "apps"
if str(APPS) not in sys.path:
    sys.path.insert(0, str(APPS))

import streamlit as st

from backend.config import (
    DEFAULT_HOUSEHOLD_BASELINE_KWH,
    DEFAULT_TARIFF_NAME,
    TARIFFS,
)
from frontend import state, theme
from frontend.pages import control_center, estimator, household, insights

st.set_page_config(
    page_title="AEnergy — Precision Energy Platform",
    page_icon="⚡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

theme.inject()


def _sidebar() -> None:
    accent = theme.palette()["accent"]
    muted = theme.palette()["muted"]
    st.markdown(
        f"""<div style="text-align:center;padding:14px 0 6px;">
              <div style="font-size:2.6rem;line-height:1;">⚡️</div>
              <h2 style="color:{accent} !important;margin:2px 0 0;">AEnergy</h2>
              <p style="color:{muted} !important;font-size:.82rem;margin:0;">
                Precision Energy Platform</p>
            </div>""",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("##### Billing assumptions")
    st.selectbox(
        "Tariff", list(TARIFFS), key="tariff_name",
        index=list(TARIFFS).index(st.session_state.get("tariff_name", DEFAULT_TARIFF_NAME)),
        help="Thai residential tariffs are block-progressive — the rate rises with consumption.",
    )
    st.slider(
        "Rest of household (kWh/month)", 0, 1500,
        int(st.session_state.get("baseline_kwh", DEFAULT_HOUSEHOLD_BASELINE_KWH)),
        step=10, key="baseline_kwh",
        help="Everything else your home already draws. Because the tariff is progressive, "
             "an appliance's true cost depends on which block its kWh land in.",
    )

    tariff = state.selected_tariff()
    baseline = state.household_baseline()
    from backend.tariff import marginal_rate

    st.caption(f"Your next kWh costs **฿{marginal_rate(tariff, baseline):.3f}** all-in "
               f"(Ft ฿{tariff.ft_rate:.4f}, VAT {tariff.vat_rate:.0%}).")

    st.divider()
    bundle = state.current_model()
    if bundle is None:
        st.warning("Engine not trained", icon="⚠️")
    else:
        st.caption(
            f"🧠 Engine ready — {bundle.n_samples:,} records, "
            f"out-of-fold R² **{bundle.metrics['model']['r2']:.3f}**, "
            f"mean error **฿{bundle.metrics['model']['mae']:,.0f}**"
        )
    st.caption("Theme follows your Streamlit setting (⋮ → Settings).")


with st.sidebar:
    _sidebar()

pages = [
    st.Page(estimator.render, title="Estimator", icon="⚡", url_path="estimator", default=True),
    st.Page(household.render, title="Household", icon="🏠", url_path="household"),
    st.Page(insights.render, title="Insights", icon="📊", url_path="insights"),
    st.Page(control_center.render, title="Control Center", icon="⚙️", url_path="control"),
]

st.navigation(pages).run()

st.divider()
st.markdown(
    "<div class='ae-footer'>© 2026 AEnergy · costs modelled on MEA/PEA residential tariffs · "
    "estimates only, not a billing statement</div>",
    unsafe_allow_html=True,
)
