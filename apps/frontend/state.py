"""Streamlit-aware caching in front of the pure-Python core.

The ``backend`` package stays free of Streamlit imports so it can be
unit-tested and reused headlessly; all memoisation lives here.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from backend import dataset, predictor
from backend.config import (
    DEFAULT_HOUSEHOLD_BASELINE_KWH,
    DEFAULT_TARIFF_NAME,
    TARIFFS,
    TariffConfig,
)


@st.cache_data(show_spinner="Loading dataset…")
def get_data(_version: int = 0) -> pd.DataFrame:
    return dataset.load_data()


@st.cache_resource(show_spinner="Loading AI engine…")
def get_model(_version: int = 0):
    """The trained engine, loaded once per process rather than per prediction.

    The old code re-read and un-pickled model.pkl on every widget interaction.
    If no usable artefact exists yet, one is trained on first use so a fresh
    clone is immediately functional.
    """
    bundle = predictor.load_model()
    if bundle is not None:
        return bundle
    frame = dataset.load_data()
    if len(frame) < predictor.MIN_ROWS:
        return None
    try:
        with st.spinner("First run — training the engine…"):
            return predictor.train(frame)
    except Exception:
        return None


def bump(key: str) -> None:
    """Invalidate a cached resource after a write."""
    st.session_state[key] = st.session_state.get(key, 0) + 1


def data_version() -> int:
    return st.session_state.get("data_version", 0)


def model_version() -> int:
    return st.session_state.get("model_version", 0)


def current_dataset() -> pd.DataFrame:
    return get_data(data_version())


def current_model():
    return get_model(model_version())


def selected_tariff() -> TariffConfig:
    return TARIFFS[st.session_state.get("tariff_name", DEFAULT_TARIFF_NAME)]


def household_baseline() -> float:
    return float(st.session_state.get("baseline_kwh", DEFAULT_HOUSEHOLD_BASELINE_KWH))


def cost_function(bundle):
    """A plain ``f(w, h, d, e) -> cost`` closure for the analysis helpers."""
    if bundle is None:
        return None
    return lambda w, h, d, e: predictor.predict(bundle, w, h, d, e)["cost"]
