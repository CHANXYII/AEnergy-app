"""Small presentational helpers shared across pages."""
from __future__ import annotations

import html

import streamlit as st


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f"<h1>{html.escape(title)}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='ae-sub'>{html.escape(subtitle)}</p>", unsafe_allow_html=True)
    st.write("")


def hero(value: str, label: str, band: str | None = None) -> None:
    band_html = f"<div class='ae-hero-band'>{html.escape(band)}</div>" if band else ""
    st.markdown(
        f"""<div class="ae-hero">
              <div class="ae-hero-label">{html.escape(label)}</div>
              <div class="ae-hero-value">{html.escape(value)}</div>
              {band_html}
            </div>""",
        unsafe_allow_html=True,
    )


def tile(label: str, value: str, hint: str = "") -> None:
    hint_html = f"<div class='ae-tile-hint'>{html.escape(hint)}</div>" if hint else ""
    st.markdown(
        f"""<div class="ae-tile">
              <div class="ae-tile-label">{html.escape(label)}</div>
              <div class="ae-tile-value">{html.escape(value)}</div>
              {hint_html}
            </div>""",
        unsafe_allow_html=True,
    )


def tiles(items: list[tuple[str, str, str]]) -> None:
    """Render a responsive row of stat tiles from (label, value, hint) triples."""
    cols = st.columns(len(items), gap="small")
    for col, (label, value, hint) in zip(cols, items):
        with col:
            tile(label, value, hint)


def recommendation(icon: str, title: str, detail: str, monthly: float, yearly: float) -> None:
    st.markdown(
        f"""<div class="ae-rec">
              <div class="ae-rec-icon">{html.escape(icon)}</div>
              <div>
                <div class="ae-rec-title">{html.escape(title)}</div>
                <div class="ae-rec-detail">{html.escape(detail)}</div>
              </div>
              <div class="ae-rec-save">฿{monthly:,.0f}<small>฿{yearly:,.0f} / year</small></div>
            </div>""",
        unsafe_allow_html=True,
    )


def baht(amount: float, decimals: int = 2) -> str:
    return f"฿{amount:,.{decimals}f}"
