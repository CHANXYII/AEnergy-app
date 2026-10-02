"""Visual system: CSS variables, glassmorphism shell, and a matching Plotly template."""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

#: Sequential pink ramp used for the efficiency label (1 = worst, 5 = best).
EFFICIENCY_COLORS = {
    1: "#FDA4AF",
    2: "#FB7185",
    3: "#F43F5E",
    4: "#BE185D",
    5: "#831843",
}

#: Categorical palette for everything that is not an efficiency label.
CATEGORICAL = ["#E11D48", "#6366F1", "#0EA5E9", "#F59E0B", "#10B981", "#8B5CF6", "#EC4899", "#14B8A6"]

LIGHT = {
    "bg_gradient": "linear-gradient(120deg,#FFE4E6 0%,#FAFAFA 25%,#E0E7FF 50%,#FFF0F5 75%,#FEF3C7 100%)",
    "surface": "rgba(255,255,255,0.58)",
    "surface_solid": "#FFFFFF",
    "border": "rgba(255,255,255,0.9)",
    "text": "#1E293B",
    "muted": "#64748B",
    "accent": "#BE185D",
    "accent_hover": "#9D174D",
    "grid": "rgba(100,116,139,0.18)",
    "shadow": "0 8px 32px rgba(225,29,72,0.08),0 4px 12px rgba(99,102,241,0.05)",
    "shadow_hover": "0 12px 40px rgba(225,29,72,0.15),0 6px 16px rgba(99,102,241,0.10)",
}

DARK = {
    "bg_gradient": "linear-gradient(120deg,#1B1021 0%,#0F172A 30%,#15203B 55%,#241326 80%,#1A1326 100%)",
    "surface": "rgba(30,41,59,0.55)",
    "surface_solid": "#1E293B",
    "border": "rgba(148,163,184,0.22)",
    "text": "#F1F5F9",
    "muted": "#94A3B8",
    "accent": "#F472B6",
    "accent_hover": "#F9A8D4",
    "grid": "rgba(148,163,184,0.16)",
    "shadow": "0 8px 32px rgba(0,0,0,0.45),0 4px 12px rgba(236,72,153,0.10)",
    "shadow_hover": "0 12px 40px rgba(0,0,0,0.55),0 6px 20px rgba(236,72,153,0.18)",
}


def active_theme() -> str:
    """'dark' or 'light', following whatever the viewer picked in Streamlit."""
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def palette() -> dict:
    return DARK if active_theme() == "dark" else LIGHT


_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {{
  --ae-surface: {surface};
  --ae-surface-solid: {surface_solid};
  --ae-border: {border};
  --ae-text: {text};
  --ae-muted: {muted};
  --ae-accent: {accent};
  --ae-accent-hover: {accent_hover};
  --ae-shadow: {shadow};
  --ae-shadow-hover: {shadow_hover};
}}

.stApp {{
  background: {bg_gradient};
  background-size: 200% 200%;
  background-attachment: fixed;
  animation: aeAurora 18s ease infinite;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  color: var(--ae-text);
}}

@keyframes aeAurora {{
  0%   {{ background-position: 0% 50%; }}
  50%  {{ background-position: 100% 50%; }}
  100% {{ background-position: 0% 50%; }}
}}

@media (prefers-reduced-motion: reduce) {{
  .stApp {{ animation: none; }}
  div[data-testid="stVerticalBlockBorderWrapper"] {{ transition: none !important; }}
}}

h1, h2, h3, h4, h5, h6 {{
  color: var(--ae-text) !important;
  font-weight: 700 !important;
  letter-spacing: -0.025em;
}}
h1 {{ font-weight: 800 !important; }}

p, label, span, li {{ color: var(--ae-text); }}
.ae-sub {{ color: var(--ae-muted) !important; font-size: 1.05rem; margin-top: -0.4rem; }}

section[data-testid="stSidebar"] {{
  background: var(--ae-surface) !important;
  backdrop-filter: blur(24px) saturate(160%);
  -webkit-backdrop-filter: blur(24px) saturate(160%);
  border-right: 1px solid var(--ae-border) !important;
}}

/* Frosted glass containers */
div[data-testid="stVerticalBlockBorderWrapper"] {{
  background: var(--ae-surface) !important;
  backdrop-filter: blur(20px) saturate(160%) !important;
  -webkit-backdrop-filter: blur(20px) saturate(160%) !important;
  border: 1px solid var(--ae-border) !important;
  border-radius: 16px !important;
  box-shadow: var(--ae-shadow) !important;
  padding: 1.1rem !important;
  transition: box-shadow .3s ease, transform .3s ease;
}}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
  box-shadow: var(--ae-shadow-hover) !important;
  transform: translateY(-2px);
}}

/* Hero cost card */
.ae-hero {{
  background: linear-gradient(135deg,#E11D48,#BE185D 55%,#7C3AED);
  padding: 30px 28px; border-radius: 18px; text-align: center;
  box-shadow: 0 12px 28px -8px rgba(225,29,72,.45);
  margin-bottom: 18px;
}}
.ae-hero .ae-hero-label {{
  color:#FFE4E6 !important; text-transform:uppercase; letter-spacing:1.6px;
  font-size:.82rem; font-weight:700; margin-bottom:6px;
}}
.ae-hero .ae-hero-value {{
  color:#fff !important; font-size:3.4rem; font-weight:800; line-height:1.1;
  text-shadow:0 2px 8px rgba(0,0,0,.18); font-variant-numeric: tabular-nums;
}}
.ae-hero .ae-hero-band {{ color:#FBCFE8 !important; font-size:.86rem; margin-top:8px; font-weight:500; }}

/* Stat tiles */
.ae-tile {{
  background: var(--ae-surface); border:1px solid var(--ae-border); border-radius:14px;
  padding:14px 16px; height:100%;
}}
.ae-tile .ae-tile-label {{
  color:var(--ae-muted) !important; font-size:.76rem; font-weight:700;
  text-transform:uppercase; letter-spacing:.09em;
}}
.ae-tile .ae-tile-value {{
  color:var(--ae-text) !important; font-size:1.55rem; font-weight:700;
  line-height:1.25; font-variant-numeric: tabular-nums;
}}
.ae-tile .ae-tile-hint {{ color:var(--ae-muted) !important; font-size:.78rem; }}

/* Recommendation rows */
.ae-rec {{
  display:flex; gap:14px; align-items:flex-start; padding:12px 14px; border-radius:12px;
  background: var(--ae-surface); border:1px solid var(--ae-border); margin-bottom:9px;
}}
.ae-rec-icon {{ font-size:1.5rem; line-height:1; }}
.ae-rec-title {{ color:var(--ae-text) !important; font-weight:700; font-size:.97rem; }}
.ae-rec-detail {{ color:var(--ae-muted) !important; font-size:.84rem; line-height:1.45; }}
.ae-rec-save {{
  margin-left:auto; text-align:right; white-space:nowrap; color:#10B981 !important;
  font-weight:800; font-size:1.02rem; font-variant-numeric: tabular-nums;
}}
.ae-rec-save small {{ display:block; color:var(--ae-muted) !important; font-weight:600; font-size:.72rem; }}

.stButton>button {{
  background: var(--ae-accent); color:#fff !important; border:none; border-radius:9px;
  padding:10px 16px; font-weight:600; transition:all .2s ease;
}}
.stButton>button * {{ color:#fff !important; }}
.stButton>button:hover {{ background: var(--ae-accent-hover); box-shadow:0 4px 10px -2px rgba(157,23,77,.45); }}
.stButton>button:focus-visible {{ outline:3px solid var(--ae-accent); outline-offset:2px; }}

.stTabs [data-baseweb="tab-list"] {{ background: var(--ae-surface); border-radius:10px; padding:4px; }}
.stTabs [data-baseweb="tab"] {{ color: var(--ae-muted); font-weight:600; border-radius:7px; }}
.stTabs [aria-selected="true"] {{ background: var(--ae-surface-solid) !important; color: var(--ae-accent) !important; }}

div[data-testid="stMetricValue"] {{ color: var(--ae-text) !important; font-variant-numeric: tabular-nums; }}
div[data-testid="stMetricLabel"] {{ color: var(--ae-muted) !important; }}

.ae-footer {{ text-align:center; color:var(--ae-muted) !important; font-size:.84rem; padding:10px 0 4px; }}

@media (max-width: 640px) {{
  .ae-hero .ae-hero-value {{ font-size:2.4rem; }}
  .ae-rec {{ flex-wrap: wrap; }}
  .ae-rec-save {{ margin-left:0; text-align:left; }}
}}
</style>
"""


def inject() -> None:
    """Install the stylesheet and register the matching Plotly template."""
    st.markdown(_CSS.format(**palette()), unsafe_allow_html=True)
    _register_plotly_template()


def _register_plotly_template() -> None:
    p = palette()
    pio.templates["aenergy"] = go.layout.Template(
        layout=go.Layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color=p["text"], size=13),
            colorway=CATEGORICAL,
            margin=dict(l=12, r=12, t=48, b=12),
            xaxis=dict(gridcolor=p["grid"], zerolinecolor=p["grid"], linecolor=p["grid"]),
            yaxis=dict(gridcolor=p["grid"], zerolinecolor=p["grid"], linecolor=p["grid"]),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=p["muted"])),
            hoverlabel=dict(font=dict(family="Inter, sans-serif")),
            title=dict(font=dict(size=15, color=p["text"]), x=0, xanchor="left"),
        )
    )


def style(fig, height: int | None = None, title: str | None = None):
    """Apply the AEnergy template to a figure built anywhere in the app."""
    fig.update_layout(template="aenergy")
    if height:
        fig.update_layout(height=height)
    if title:
        fig.update_layout(title=title)
    return fig
