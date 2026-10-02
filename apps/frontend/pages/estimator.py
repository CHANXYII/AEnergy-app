"""Estimator: configure one appliance and see what it really costs."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backend import analysis, predictor
from backend.config import APPLIANCE_PRESETS, EFFICIENCY_LABELS, PRESETS_BY_NAME
from backend.dataset import calculate_kwh
from backend.tariff import marginal_cost
from frontend import state
from frontend.components import baht, hero, page_header, recommendation, tiles
from frontend.theme import EFFICIENCY_COLORS, palette, style


def _controls() -> tuple[float, float, int, int]:
    preset_names = [a.name for a in APPLIANCE_PRESETS]
    choice = st.selectbox(
        "Appliance preset",
        preset_names,
        format_func=lambda n: f"{PRESETS_BY_NAME[n].icon}  {n}",
        key="preset_choice",
        help="Picking a preset loads typical values you can then fine-tune.",
    )
    preset = PRESETS_BY_NAME[choice]

    # Re-seed the sliders whenever the preset changes, but leave the user's own
    # adjustments alone on every other rerun.
    if st.session_state.get("_applied_preset") != choice:
        st.session_state["_applied_preset"] = choice
        st.session_state["p_w"] = preset.wattage
        st.session_state["p_h"] = preset.hours_per_day
        st.session_state["p_d"] = preset.days_per_month
        st.session_state["p_e"] = preset.efficiency

    wattage = st.number_input("Power draw (W)", 1, 50_000, step=25, key="p_w")
    hours = st.slider("Hours per day", 0.0, 24.0, step=0.25, key="p_h")
    days = st.slider("Active days per month", 1, 31, key="p_d")
    efficiency = st.select_slider(
        "Efficiency label", options=[1, 2, 3, 4, 5], key="p_e",
        help="Thai energy label, 1 (poor) to 5 (best in class).",
    )
    st.caption(EFFICIENCY_LABELS[efficiency])
    return float(wattage), float(hours), int(days), int(efficiency)


def _gauge(predicted: float, tariff_cost: float, low: float, high: float):
    top = max(predicted, tariff_cost, high) * 1.35 or 1.0
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=predicted,
            number={"prefix": "฿", "valueformat": ",.0f"},
            delta={
                "reference": tariff_cost,
                "valueformat": ",.0f",
                "increasing": {"color": "#EF4444"},
                "decreasing": {"color": "#10B981"},
            },
            gauge={
                "axis": {"range": [0, top], "tickformat": ",.0f"},
                "bar": {"color": "#E11D48", "thickness": 0.72},
                "bgcolor": "rgba(148,163,184,0.12)",
                "borderwidth": 0,
                "steps": [{"range": [low, high], "color": "rgba(225,29,72,0.18)"}],
                "threshold": {
                    "line": {"color": palette()["text"], "width": 4},
                    "thickness": 0.82,
                    "value": tariff_cost,
                },
            },
        )
    )
    return style(fig, height=230)


def _tier_chart(breakdown: dict):
    segs = breakdown["segments"]
    if not segs:
        return None
    frame = pd.DataFrame(segs)
    frame["text"] = frame.apply(lambda r: f"{r['kwh']:,.0f} kWh @ ฿{r['rate']:.4f}", axis=1)
    fig = px.bar(
        frame, x="charge", y="label", orientation="h", text="text",
        labels={"charge": "Energy charge (THB)", "label": "Tariff block"},
        color="rate", color_continuous_scale=["#FDA4AF", "#831843"],
    )
    fig.update_traces(textposition="inside", insidetextanchor="start", cliponaxis=False)
    fig.update_layout(coloraxis_showscale=False, yaxis={"autorange": "reversed"})
    return style(fig, height=max(150, 60 * len(segs)), title="Where each kWh lands on the tariff")


def _efficiency_chart(curve: list[dict], current: int):
    frame = pd.DataFrame(curve)
    fig = px.bar(
        frame, x="label", y="cost", text="cost",
        color="efficiency",
        color_discrete_map={e: EFFICIENCY_COLORS[e] for e in EFFICIENCY_COLORS},
        labels={"label": "", "cost": "Monthly cost (THB)"},
    )
    fig.update_traces(texttemplate="฿%{text:,.0f}", textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    fig.add_annotation(
        x=f"Label {current}", y=frame.loc[frame.efficiency == current, "cost"].iloc[0],
        text="you are here", showarrow=True, arrowhead=2, ay=-38,
        font=dict(size=11, color=palette()["muted"]),
    )
    return style(fig, height=300, title="Same appliance at every efficiency label")


def render() -> None:
    page_header(
        "Energy Cost Estimator",
        "Price a single appliance against Thailand's progressive residential tariff.",
    )

    bundle = state.current_model()
    tariff = state.selected_tariff()
    baseline = state.household_baseline()

    left, right = st.columns([1, 1.25], gap="large")

    with left:
        with st.container(border=True):
            st.markdown("### Device configuration")
            wattage, hours, days, efficiency = _controls()

    kwh = calculate_kwh(wattage, hours, days)
    tariff_view = marginal_cost(kwh, tariff, baseline)
    tariff_cost = tariff_view["total"]

    if bundle is None:
        with right:
            st.warning(
                "No trained engine found. Open **Control Center → Model Engine** and train one "
                "to unlock AI predictions. Tariff-based costing below still works."
            )
            hero(baht(tariff_cost, 0), "Tariff-based monthly cost",
                 f"{kwh:,.1f} kWh at ฿{tariff_view['effective_rate']:.3f}/kWh")
        return

    pred = predictor.predict(bundle, wattage, hours, days, efficiency)
    cost = pred["cost"]
    carbon = analysis.carbon_footprint(kwh)
    horizons = analysis.cost_horizons(cost, days)

    with right:
        hero(
            baht(cost, 0),
            "AI-estimated monthly cost",
            f"likely range {baht(pred['cost_low'], 0)} – {baht(pred['cost_high'], 0)}",
        )
        tiles([
            ("Consumption", f"{kwh:,.1f} kWh", f"{kwh / max(days, 1):,.2f} kWh per active day"),
            ("Effective rate", f"฿{pred['rate']:.2f}", "all-in, per kWh"),
            ("Per year", baht(horizons["per_year"], 0), f"{baht(horizons['per_day'], 1)} per active day"),
        ])
        st.write("")
        with st.container(border=True):
            st.markdown(
                f"The needle is the AI estimate; the dark marker is the pure tariff calculation "
                f"(**{baht(tariff_cost, 0)}**). A needle to the **left** means the engine has "
                f"learned this appliance behaves better than its nameplate suggests."
            )
            st.plotly_chart(_gauge(cost, tariff_cost, pred["cost_low"], pred["cost_high"]),
                            key="gauge")

    st.write("")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Daily", baht(horizons["per_day"], 1))
    with c2:
        st.metric("Monthly", baht(horizons["per_month"], 0))
    with c3:
        st.metric("Yearly", baht(horizons["per_year"], 0))
    with c4:
        st.metric("CO₂ per year", f"{carbon['kg_per_year']:,.0f} kg",
                  help=f"About {carbon['trees_to_offset']:,.0f} mature trees' worth of annual absorption.")

    st.write("")
    tab_save, tab_eff, tab_tier = st.tabs(["💡 Ways to save", "🏷️ Efficiency impact", "🧾 Tariff breakdown"])

    with tab_save:
        predict_fn = state.cost_function(bundle)
        recs = analysis.recommendations(predict_fn, wattage, hours, days, efficiency)
        if not recs:
            st.success("This configuration is already close to optimal — no change saves more than ฿0.50/month.")
        else:
            st.caption("Each figure is a real what-if run back through the engine, not a rule of thumb.")
            for rec in recs:
                recommendation(rec.icon, rec.title, rec.detail, rec.monthly_saving, rec.yearly_saving)
            total = sum(r.yearly_saving for r in recs)
            st.info(f"Applying everything above is worth roughly **{baht(total, 0)} per year**, "
                    "though the savings overlap rather than stacking perfectly.")

    with tab_eff:
        curve = analysis.efficiency_curve(state.cost_function(bundle), wattage, hours, days)
        st.plotly_chart(_efficiency_chart(curve, efficiency), key="eff")
        best, worst = curve[-1]["cost"], curve[0]["cost"]
        st.caption(f"Across the full label range this appliance swings "
                   f"{baht(worst - best, 0)} per month ({baht((worst - best) * 12, 0)} per year).")

    with tab_tier:
        st.caption(
            f"Your household already draws **{baseline:,.0f} kWh/month**, so this appliance's "
            f"{kwh:,.1f} kWh are priced at the margin — in the tariff blocks sitting above that baseline."
        )
        chart = _tier_chart(tariff_view)
        if chart is not None:
            st.plotly_chart(chart, key="tiers")
        breakdown = pd.DataFrame([
            {"Component": "Energy charge", "THB": tariff_view["energy"]},
            {"Component": f"Ft fuel adjustment ({tariff.ft_rate:.4f}/kWh)", "THB": tariff_view["ft"]},
            {"Component": f"VAT {tariff.vat_rate:.0%}", "THB": tariff_view["vat"]},
            {"Component": "Total (tariff only)", "THB": tariff_view["total"]},
        ])
        st.dataframe(
            breakdown, hide_index=True,
            column_config={"THB": st.column_config.NumberColumn("THB", format="฿%.2f")},
        )
        st.caption("The fixed monthly service charge is excluded: the household pays it whether "
                   "or not this appliance is plugged in.")
