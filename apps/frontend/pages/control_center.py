"""Control Center: seed real data, manage the dataset, retrain the engine."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from backend import dataset
from backend import predictor
from backend.config import CSV_FILE, EFFICIENCY_LABELS, MODEL_FILE
from backend.dataset import DataError, calculate_kwh, validate
from backend.predictor import MIN_ROWS, ModelError
from frontend import state
from frontend.components import baht, page_header, tiles
from frontend.theme import style


def _tab_collect() -> None:
    st.markdown("### Record a real electricity bill")
    st.caption("Real observations are what move the engine off its synthetic seed data. "
               "Enter the appliance's settings and the amount it actually added to your bill.")

    with st.form("record_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        wattage = c1.number_input("Power draw (W)", 1, 50_000, 1000, step=25)
        hours = c2.number_input("Hours per day", 0.0, 24.0, 8.0, step=0.25)
        days = c3.number_input("Active days per month", 1, 31, 30)

        c4, c5 = st.columns([1, 2])
        efficiency = c4.selectbox("Efficiency label", [1, 2, 3, 4, 5], index=4,
                                  format_func=lambda e: f"{e} — {EFFICIENCY_LABELS[e].split('—')[1].strip()}")
        cost = c5.number_input("Actual cost on the bill (THB)", 0.0, 1_000_000.0, 0.0, step=10.0)

        kwh = calculate_kwh(wattage, hours, days)
        implied = cost / kwh if kwh > 0 else 0.0
        st.caption(f"That works out to **{kwh:,.1f} kWh/month** "
                   + (f"at an implied **฿{implied:.2f}/kWh**." if implied else "— enter the cost to see the implied rate."))
        if implied and not (1.0 <= implied <= 15.0):
            st.warning(f"An implied rate of ฿{implied:.2f}/kWh is outside the plausible "
                       "฿1–฿15 band. Double-check the wattage, hours or cost before saving.")

        if st.form_submit_button("💾 Save record", width="stretch"):
            try:
                dataset.add_record(wattage, hours, days, efficiency, cost)
            except DataError as exc:
                st.error(str(exc))
            else:
                state.bump("data_version")
                st.success("Record saved. Retrain the engine to fold it into predictions.")

    st.divider()
    st.markdown("### Bulk import")
    upload = st.file_uploader(
        "Upload a CSV with columns: wattage, hours, days, efficiency, cost", type="csv"
    )
    if upload is not None:
        try:
            incoming = pd.read_csv(upload)
            report = validate(incoming)
        except Exception as exc:
            st.error(f"Could not read that file: {exc}")
        else:
            st.write(f"**{len(report.frame)}** usable row(s) found."
                     + (f" {report.dropped} rejected." if report.dropped else ""))
            for reason in report.reasons:
                st.caption(f"• {reason}")
            st.dataframe(report.frame.head(20), hide_index=True, width="stretch")
            mode = st.radio("Import mode", ["Append to dataset", "Replace dataset"], horizontal=True)
            if st.button("Import", disabled=report.frame.empty):
                existing = state.current_dataset()
                merged = (
                    validate(pd.concat([existing, report.frame], ignore_index=True)).frame
                    if mode.startswith("Append")
                    else report.frame
                )
                dataset.write_data(merged)
                state.bump("data_version")
                st.success(f"Dataset now holds {len(merged):,} records.")
                st.rerun()


def _tab_engine(df: pd.DataFrame) -> None:
    bundle = state.current_model()
    tariff = state.selected_tariff()

    # Surfaced after the post-training rerun, which would otherwise discard it.
    if flash := st.session_state.pop("training_flash", None):
        st.success(flash)

    status = "Trained" if bundle else "Not trained"
    tiles([
        ("Engine status", status, bundle.trained_at.replace("T", " ")[:16] + " UTC" if bundle else "train to enable predictions"),
        ("Dataset size", f"{len(df):,} rows", f"minimum {MIN_ROWS} to train"),
        ("Out-of-fold R²", f"{bundle.metrics['model']['r2']:.3f}" if bundle else "—", "on unseen folds"),
        ("Mean abs. error", baht(bundle.metrics["model"]["mae"], 0) if bundle else "—", "per appliance"),
    ])

    st.write("")
    left, right = st.columns([1.4, 1], gap="large")

    with left:
        with st.container(border=True):
            st.markdown("### Algorithm")
            st.markdown(
                "**RandomForestRegressor** (400 trees) predicting the *effective tariff rate* "
                "in ฿/kWh, multiplied back by physically computed energy:\n\n"
                "```\ncost = kwh(watts, hours, days) × model(features)\n```\n"
                "Learning the rate rather than the cost keeps the model inside a bounded, "
                "scale-free target while the kWh term carries extrapolation exactly — so a "
                "7 kW EV charger is priced correctly even if the dataset tops out at 1 kW."
            )
            if bundle:
                st.caption("Features: " + ", ".join(f"`{f}`" for f in bundle.features))

    with right:
        with st.container(border=True):
            st.markdown("### Retrain")
            st.caption("Runs k-fold cross-validation first, so the reported accuracy reflects "
                       "unseen data rather than the rows it just memorised.")
            if st.button("🔄 Retrain engine", width="stretch", disabled=len(df) < MIN_ROWS):
                with st.spinner("Cross-validating and fitting…"):
                    try:
                        fresh = predictor.train(df, tariff=tariff)
                    except (ModelError, ValueError) as exc:
                        st.error(f"Training failed: {exc}")
                    else:
                        state.get_model.clear()
                        state.bump("model_version")
                        m = fresh.metrics["model"]
                        st.session_state["training_flash"] = (
                            f"Trained on {fresh.n_samples:,} records — out-of-fold R² "
                            f"{m['r2']:.3f}, mean error {baht(m['mae'], 0)} ({m['mape']:.1%})."
                        )
                        st.balloons()
                        st.rerun()
            if len(df) < MIN_ROWS:
                st.caption(f"Need {MIN_ROWS - len(df)} more record(s) before training.")

    history = predictor.load_history()
    if len(history) > 1:
        with st.container(border=True):
            hist = pd.DataFrame(history)
            hist["trained_at"] = pd.to_datetime(hist["trained_at"], format="mixed", utc=True)
            fig = px.line(hist, x="trained_at", y="mae", markers=True,
                          hover_data={"n_samples": True, "r2": ":.3f"},
                          labels={"trained_at": "", "mae": "Mean abs. error (THB)"})
            fig.update_traces(line=dict(color="#BE185D", width=2.5))
            st.plotly_chart(style(fig, height=260, title="Accuracy over successive retrains"),
                            key="ctl_hist")
            st.caption("Error should fall as you add real bills. A rise means the new records "
                       "disagree with the old ones — worth investigating before trusting it.")


def _tab_dataset(df: pd.DataFrame) -> None:
    st.markdown("### Manage the dataset")
    st.caption(f"Stored at `{CSV_FILE.relative_to(CSV_FILE.parent.parent)}` — "
               "a single canonical location, independent of where you launched the app from.")

    edited = st.data_editor(
        df, num_rows="dynamic", width="stretch", height=380, key="dataset_editor",
        column_config={
            "wattage": st.column_config.NumberColumn("Watts", min_value=1, max_value=50_000),
            "hours": st.column_config.NumberColumn("Hrs/day", min_value=0.0, max_value=24.0, format="%.2f"),
            "days": st.column_config.NumberColumn("Days/mo", min_value=1, max_value=31),
            "efficiency": st.column_config.NumberColumn("Label", min_value=1, max_value=5),
            "cost": st.column_config.NumberColumn("Cost (THB)", min_value=0.0, format="%.2f"),
        },
    )

    c1, c2, c3 = st.columns(3)
    if c1.button("💾 Save edits", width="stretch"):
        try:
            report = validate(edited)
        except DataError as exc:
            st.error(str(exc))
        else:
            dataset.write_data(report.frame)
            state.bump("data_version")
            st.success(f"Saved {len(report.frame):,} records."
                       + (f" {report.dropped} invalid row(s) dropped." if report.dropped else ""))
            st.rerun()

    c2.download_button("⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="aenergy_data.csv", mime="text/csv", width="stretch")

    with c3.popover("♻️ Reset to seed data", width="stretch"):
        st.warning("This permanently replaces every record with a freshly generated "
                   "synthetic dataset. Download a copy first if you need it.")
        if st.button("Yes, reset the dataset"):
            dataset.write_data(dataset.synthesize())
            MODEL_FILE.unlink(missing_ok=True)
            state.get_model.clear()
            state.bump("data_version")
            state.bump("model_version")
            st.rerun()


def render() -> None:
    page_header("Control Center",
                "Seed real-world data, curate the dataset, and retrain the prediction engine.")
    df = state.current_dataset()
    t1, t2, t3 = st.tabs(["📝 Collect data", "🧠 Model engine", "🗃️ Dataset"])
    with t1:
        _tab_collect()
    with t2:
        _tab_engine(df)
    with t3:
        _tab_dataset(df)
