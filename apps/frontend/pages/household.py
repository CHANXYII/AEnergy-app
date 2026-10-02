"""Household builder: stack several appliances into one monthly bill."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from backend import analysis, predictor
from backend.config import APPLIANCE_PRESETS, PRESETS_BY_NAME
from backend.tariff import bill
from frontend import state
from frontend.components import baht, page_header, tiles
from frontend.theme import style

STARTER = [
    {"name": "Air conditioner (12,000 BTU)", "icon": "❄️", "wattage": 1200,
     "hours": 8.0, "days": 30, "efficiency": 4, "quantity": 1},
    {"name": "Refrigerator (2-door)", "icon": "🧊", "wattage": 150,
     "hours": 24.0, "days": 30, "efficiency": 5, "quantity": 1},
    {"name": "LED lighting (whole home)", "icon": "💡", "wattage": 120,
     "hours": 6.0, "days": 30, "efficiency": 5, "quantity": 1},
]


def _items() -> list[dict]:
    if "household" not in st.session_state:
        st.session_state["household"] = [dict(a) for a in STARTER]
    return st.session_state["household"]


@st.dialog("Add an appliance")
def _add_dialog() -> None:
    names = [a.name for a in APPLIANCE_PRESETS if a.name != "Custom"]
    choice = st.selectbox("Preset", names, format_func=lambda n: f"{PRESETS_BY_NAME[n].icon}  {n}")
    preset = PRESETS_BY_NAME[choice]
    c1, c2 = st.columns(2)
    wattage = c1.number_input("Watts", 1, 50_000, preset.wattage, step=25)
    quantity = c2.number_input("How many", 1, 50, 1)
    c3, c4 = st.columns(2)
    hours = c3.slider("Hours/day", 0.0, 24.0, preset.hours_per_day, 0.25)
    days = c4.slider("Days/month", 1, 31, preset.days_per_month)
    efficiency = st.select_slider("Efficiency label", [1, 2, 3, 4, 5], preset.efficiency)

    if st.button("Add to household", width="stretch"):
        _items().append({
            "name": choice, "icon": preset.icon, "wattage": float(wattage),
            "hours": float(hours), "days": int(days),
            "efficiency": int(efficiency), "quantity": int(quantity),
        })
        # Drop the editor's accumulated edit-deltas; they are keyed by row
        # position and would otherwise be replayed onto the new, longer frame.
        st.session_state.pop("household_editor", None)
        st.rerun()


EDITOR_COLUMNS = ["icon", "name", "quantity", "wattage", "hours", "days", "efficiency"]


def _editor(items: list[dict]) -> list[dict]:
    # Build explicitly so an emptied household still yields a typed, columned
    # frame for the editor rather than a shapeless empty DataFrame.
    frame = pd.DataFrame(items, columns=EDITOR_COLUMNS)
    edited = st.data_editor(
        frame,
        hide_index=True,
        num_rows="dynamic",
        width="stretch",
        key="household_editor",
        column_config={
            "icon": st.column_config.TextColumn("", width="small", default="🔌"),
            "name": st.column_config.TextColumn("Appliance", width="medium", required=True),
            "quantity": st.column_config.NumberColumn("Qty", min_value=1, max_value=50, step=1, default=1),
            "wattage": st.column_config.NumberColumn("Watts", min_value=1, max_value=50_000, step=25, default=100),
            "hours": st.column_config.NumberColumn("Hrs/day", min_value=0.0, max_value=24.0, step=0.25, format="%.2f", default=4.0),
            "days": st.column_config.NumberColumn("Days/mo", min_value=1, max_value=31, step=1, default=30),
            "efficiency": st.column_config.NumberColumn("Label", min_value=1, max_value=5, step=1, default=3),
        },
    )
    cleaned = edited.dropna(subset=["name", "wattage", "hours", "days"]).to_dict("records")
    for row in cleaned:
        row["quantity"] = int(row.get("quantity") or 1)
        row["days"] = int(row["days"])
        row["efficiency"] = int(row.get("efficiency") or 3)
        row["icon"] = row.get("icon") or "🔌"
    return cleaned


def render() -> None:
    page_header(
        "Household Builder",
        "Stack every appliance you own into a single, tier-aware monthly bill.",
    )

    tariff = state.selected_tariff()
    baseline = state.household_baseline()
    bundle = state.current_model()
    items = _items()

    with st.container(border=True):
        head, action = st.columns([3, 1])
        head.markdown("### Your appliances")
        head.caption("Edit any cell directly, or use the ➕ row at the bottom of the table.")
        if action.button("➕ Add preset", width="stretch"):
            _add_dialog()
        items = _editor(items)
        st.session_state["household"] = items

    if not items:
        st.info("Add at least one appliance to see the household rollup.")
        return

    roll = analysis.household_rollup([dict(i) for i in items], tariff, baseline)
    frame = pd.DataFrame(roll["appliances"])
    frame["display"] = frame["icon"] + " " + frame["name"]

    # Price the same appliances through the AI engine for a second opinion.
    if bundle is not None:
        frame["ai_cost"] = [
            predictor.predict(bundle, r.wattage, r.hours, r.days, r.efficiency)["cost"] * r.quantity
            for r in frame.itertuples()
        ]
    else:
        frame["ai_cost"] = frame["cost"]

    full = bill(baseline + roll["total_kwh"], tariff)
    st.write("")
    tiles([
        ("Appliance total", baht(roll["total_cost"], 0), f"{roll['total_kwh']:,.0f} kWh/month"),
        ("AI second opinion", baht(frame["ai_cost"].sum(), 0), "modelled from real bills"),
        ("Whole-home bill", baht(full["total"], 0), f"{roll['final_household_kwh']:,.0f} kWh incl. baseline"),
        ("CO₂ per year", f"{roll['carbon']['kg_per_year']:,.0f} kg",
         f"≈{roll['carbon']['trees_to_offset']:,.0f} trees to offset"),
    ])

    st.write("")
    left, right = st.columns([1.3, 1], gap="large")

    with left:
        with st.container(border=True):
            ranked = frame.sort_values("cost", ascending=True)
            fig = px.bar(
                ranked, x="cost", y="display", orientation="h", text="cost",
                color="effective_rate", color_continuous_scale=["#FDA4AF", "#831843"],
                labels={"cost": "Monthly cost (THB)", "display": "", "effective_rate": "฿/kWh"},
            )
            fig.update_traces(texttemplate="฿%{text:,.0f}", textposition="outside", cliponaxis=False)
            st.plotly_chart(
                style(fig, height=max(280, 46 * len(frame)), title="Cost by appliance"),
                key="compare_bar",
            )

    with right:
        with st.container(border=True):
            fig2 = px.pie(frame, values="kwh", names="display", hole=0.55)
            fig2.update_traces(textposition="inside", textinfo="percent")
            st.plotly_chart(style(fig2, height=max(280, 46 * len(frame)),
                                  title="Share of consumption"), key="compare_pie")

    biggest = frame.loc[frame["cost"].idxmax()]
    share = biggest["cost"] / roll["total_cost"] if roll["total_cost"] else 0
    st.info(
        f"**{biggest['icon']} {biggest['name']}** is your largest line item at "
        f"{baht(biggest['cost'], 0)}/month — {share:.0%} of the appliance bill. "
        f"Target it first: a 20% cut there saves {baht(biggest['cost'] * 0.2 * 12, 0)} a year."
    )

    with st.expander("Per-appliance detail"):
        table = frame[["display", "quantity", "wattage", "hours", "days", "efficiency",
                       "kwh", "effective_rate", "cost", "ai_cost"]]
        st.dataframe(
            table, hide_index=True, width="stretch",
            column_config={
                "display": "Appliance", "quantity": "Qty", "wattage": "W",
                "hours": st.column_config.NumberColumn("Hrs/day", format="%.2f"),
                "days": "Days", "efficiency": "Label",
                "kwh": st.column_config.NumberColumn("kWh/mo", format="%.1f"),
                "effective_rate": st.column_config.NumberColumn("฿/kWh", format="฿%.3f"),
                "cost": st.column_config.NumberColumn("Tariff cost", format="฿%.0f"),
                "ai_cost": st.column_config.NumberColumn("AI estimate", format="฿%.0f"),
            },
        )
        st.caption(
            "Appliances are charged largest-first on top of your household baseline, so each one "
            "sits in the tariff block it genuinely occupies rather than every device being priced "
            "as if it were the household's first kWh."
        )
