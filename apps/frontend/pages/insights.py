"""Dataset and model diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from backend.dataset import add_features, summary_stats
from frontend import state
from frontend.components import baht, page_header, tiles
from frontend.theme import EFFICIENCY_COLORS, style

LABEL_COLORS = {f"Label {k}": v for k, v in EFFICIENCY_COLORS.items()}


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    feat = add_features(df)
    feat["Efficiency Label"] = "Label " + feat["efficiency"].astype(str)
    return feat


def _tab_consumption(feat: pd.DataFrame) -> None:
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            opts = dict(
                x="kwh", y="cost", color="Efficiency Label",
                color_discrete_map=LABEL_COLORS,
                category_orders={"Efficiency Label": sorted(LABEL_COLORS)},
                hover_data={"wattage": True, "hours": ":.1f", "days": True},
                labels={"kwh": "Consumption (kWh/month)", "cost": "Cost (THB)"},
            )
            try:  # the OLS trendline needs statsmodels, which is optional
                fig = px.scatter(feat, trendline="ols", trendline_scope="overall",
                                 trendline_color_override="rgba(148,163,184,0.7)", **opts)
            except (ImportError, ModuleNotFoundError):
                fig = px.scatter(feat, **opts)
            fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color="rgba(255,255,255,.6)")))
            st.plotly_chart(style(fig, height=380, title="Cost vs consumption"), key="ins_scatter")
            st.caption("Darker points are more efficient appliances. They sit below the overall "
                       "trend line: the same kWh costs them less.")

    with right:
        with st.container(border=True):
            avg = (
                feat.groupby("Efficiency Label", as_index=False)
                .agg(rate=("rate", "mean"), n=("rate", "size"))
                .sort_values("Efficiency Label")
            )
            fig = px.bar(
                avg, x="Efficiency Label", y="rate", text="rate",
                color="Efficiency Label", color_discrete_map=LABEL_COLORS,
                hover_data={"n": True},
                labels={"rate": "Effective rate (THB/kWh)", "Efficiency Label": ""},
            )
            fig.update_traces(texttemplate="฿%{text:.2f}", textposition="outside", cliponaxis=False)
            fig.update_layout(showlegend=False)
            st.plotly_chart(style(fig, height=380, title="Effective rate paid by efficiency label"),
                            key="ins_bar")
            st.caption("Comparing *rate* rather than raw cost removes appliance size from the "
                       "picture, isolating the effect of the efficiency label itself.")

    with st.container(border=True):
        fig = px.density_heatmap(
            feat, x="hours", y="wattage", z="cost", histfunc="avg",
            nbinsx=12, nbinsy=10, color_continuous_scale=["#FFF1F2", "#FB7185", "#831843"],
            labels={"hours": "Hours per day", "wattage": "Power draw (W)", "color": "Avg cost"},
        )
        st.plotly_chart(style(fig, height=330, title="Where cost concentrates"), key="ins_heat")


def _tab_model(df: pd.DataFrame) -> None:
    bundle = state.current_model()
    if bundle is None:
        st.warning("No trained engine yet — train one in the Control Center to see diagnostics.")
        return

    metrics = bundle.metrics
    model_m = metrics.get("model", {})
    naive_m = metrics.get("baseline_forest_on_cost", {})

    tiles([
        ("Out-of-fold R²", f"{model_m.get('r2', 0):.3f}", f"{metrics.get('n_splits', 0)}-fold CV"),
        ("Mean abs. error", baht(model_m.get("mae", 0), 0), "average miss per appliance"),
        ("Typical error", f"{model_m.get('mape', 0):.1%}", "relative to actual bill"),
        ("Trained on", f"{bundle.n_samples:,} rows", bundle.trained_at.replace("T", " ")[:16] + " UTC"),
    ])
    st.caption("Every figure is measured on held-out folds the model never saw during fitting.")

    st.write("")
    left, right = st.columns(2, gap="large")

    with left:
        with st.container(border=True):
            comparison = pd.DataFrame([
                {"Approach": "AEnergy engine (rate x kWh)", "MAE": model_m.get("mae", 0)},
                {"Approach": "Forest fitted on cost", "MAE": naive_m.get("mae", 0)},
                {"Approach": "Tariff formula only",
                 "MAE": metrics.get("baseline_tariff_only", {}).get("mae", 0)},
            ])
            comparison = comparison[comparison["MAE"] > 0]
            fig = px.bar(comparison, x="MAE", y="Approach", orientation="h", text="MAE",
                         color="Approach",
                         color_discrete_sequence=["#831843", "#FB7185", "#94A3B8"])
            fig.update_traces(texttemplate="฿%{text:,.0f}", textposition="outside", cliponaxis=False)
            fig.update_layout(showlegend=False, yaxis={"autorange": "reversed"})
            st.plotly_chart(style(fig, height=300, title="Mean error vs baselines (lower is better)"),
                            key="ins_cmp")
            st.caption("The engine predicts an effective ฿/kWh rate and multiplies by physically "
                       "computed energy, so it extrapolates past the largest appliance it has seen. "
                       "A forest fitted straight onto cost structurally cannot.")

    with right:
        with st.container(border=True):
            imp = bundle.importances
            fig = px.bar(imp.sort_values("importance"), x="importance", y="feature",
                         orientation="h", color="importance",
                         color_continuous_scale=["#FDA4AF", "#831843"],
                         labels={"importance": "Relative importance", "feature": ""})
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style(fig, height=300, title="What the engine actually uses"),
                            key="ins_imp")

    oof = metrics.get("oof_predictions")
    if oof and len(oof) == len(df):
        with st.container(border=True):
            feat = add_features(df)
            resid = pd.DataFrame({
                "actual": feat["cost"].to_numpy(),
                "predicted": np.asarray(oof),
                "efficiency": "Label " + feat["efficiency"].astype(str),
            })
            lim = float(max(resid["actual"].max(), resid["predicted"].max())) * 1.05
            fig = px.scatter(resid, x="actual", y="predicted", color="efficiency",
                             color_discrete_map=LABEL_COLORS,
                             category_orders={"efficiency": sorted(LABEL_COLORS)},
                             labels={"actual": "Actual cost (THB)", "predicted": "Predicted (THB)"})
            fig.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim,
                          line=dict(color="rgba(148,163,184,.8)", dash="dash"))
            fig.update_traces(marker=dict(size=8))
            st.plotly_chart(style(fig, height=380, title="Predicted vs actual (held-out folds)"),
                            key="ins_resid")
            st.caption("Points on the dashed line are perfect predictions. Systematic drift above "
                       "or below it would mean the engine is biased for that efficiency class.")


def _tab_data(df: pd.DataFrame, feat: pd.DataFrame) -> None:
    stats = summary_stats(df)
    tiles([
        ("Records", f"{stats['records']:,}", "after validation"),
        ("Median consumption", f"{stats['median_kwh']:,.0f} kWh", "per month"),
        ("Median rate", f"฿{stats['median_rate']:.2f}", "per kWh, all-in"),
        ("Labels covered", "".join(str(e) for e in stats["efficiency_coverage"]) or "—",
         "efficiency classes present"),
    ])
    missing = sorted(set(range(1, 6)) - set(stats["efficiency_coverage"]))
    if missing:
        st.warning(f"No records for efficiency label(s) {', '.join(map(str, missing))}. "
                   "Predictions there are extrapolation — add real bills in the Control Center.")
    st.write("")
    st.dataframe(
        feat[["wattage", "hours", "days", "efficiency", "kwh", "cost", "rate"]],
        width="stretch", height=420,
        column_config={
            "wattage": "Watts",
            "hours": st.column_config.NumberColumn("Hrs/day", format="%.2f"),
            "days": "Days/mo",
            "efficiency": st.column_config.NumberColumn("Label", format="%d"),
            "kwh": st.column_config.NumberColumn("kWh", format="%.1f"),
            "cost": st.column_config.NumberColumn("Cost", format="฿%.2f"),
            "rate": st.column_config.NumberColumn("฿/kWh", format="฿%.3f"),
        },
    )


def render() -> None:
    page_header("Data & Model Insights",
                "What the dataset shows, and how well the engine actually performs on it.")
    df = state.current_dataset()
    if df.empty:
        st.info("No data yet. Add records in the Control Center.")
        return
    feat = _prepare(df)

    t1, t2, t3 = st.tabs(["📈 Consumption patterns", "🧠 Model performance", "🗂️ Raw dataset"])
    with t1:
        _tab_consumption(feat)
    with t2:
        _tab_model(df)
    with t3:
        _tab_data(df, feat)
