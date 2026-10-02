"""Dataset loading, validation and durable appends."""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config import (
    CSV_FILE,
    DATA_DIR,
    LEGACY_CSV_FILES,
    TARIFF_RESIDENTIAL_LARGE,
)
from backend.tariff import marginal_cost

COLUMNS = ["wattage", "hours", "days", "efficiency", "cost"]

#: Physically sensible bounds. Rows outside these are rejected rather than
#: silently poisoning the model.
BOUNDS = {
    "wattage": (1.0, 50_000.0),
    "hours": (0.0, 24.0),
    "days": (1.0, 31.0),
    "efficiency": (1.0, 5.0),
    "cost": (0.0, 1_000_000.0),
}


class DataError(ValueError):
    """Raised when a record or file cannot be used."""


@dataclass
class ValidationReport:
    frame: pd.DataFrame
    dropped: int
    reasons: list[str]

    @property
    def ok(self) -> bool:
        return self.dropped == 0


def calculate_kwh(wattage, hours, days):
    """Monthly energy in kWh. Vectorised-safe (works on scalars and Series)."""
    return (wattage * hours * days) / 1000.0


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach derived columns used by both the model and the charts."""
    out = df.copy()
    out["kwh"] = calculate_kwh(out["wattage"], out["hours"], out["days"])
    out["kwh_per_day"] = out["kwh"] / out["days"].clip(lower=1)
    out["duty_cycle"] = (out["hours"] / 24.0) * (out["days"] / 31.0)
    # Effective all-in rate actually paid, in THB/kWh — the model's target.
    out["rate"] = np.where(out["kwh"] > 0, out["cost"] / out["kwh"].replace(0, np.nan), np.nan)
    return out


def validate(df: pd.DataFrame, *, strict: bool = False) -> ValidationReport:
    """Coerce types and drop unusable rows, reporting what went and why."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"Dataset is missing required column(s): {', '.join(missing)}")

    out = df.loc[:, COLUMNS].copy()
    reasons: list[str] = []
    before = len(out)

    for col in COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    nan_rows = int(out.isna().any(axis=1).sum())
    if nan_rows:
        reasons.append(f"{nan_rows} row(s) with non-numeric or empty values")
    out = out.dropna()

    for col, (lo, hi) in BOUNDS.items():
        bad = int(((out[col] < lo) | (out[col] > hi)).sum())
        if bad:
            reasons.append(f"{bad} row(s) with {col} outside {lo:g}–{hi:g}")
        out = out[(out[col] >= lo) & (out[col] <= hi)]

    zero_energy = int((calculate_kwh(out["wattage"], out["hours"], out["days"]) <= 0).sum())
    if zero_energy:
        reasons.append(f"{zero_energy} row(s) consuming zero energy")
    out = out[calculate_kwh(out["wattage"], out["hours"], out["days"]) > 0]

    dupes = int(out.duplicated().sum())
    if dupes:
        reasons.append(f"{dupes} exact duplicate row(s)")
    out = out.drop_duplicates().reset_index(drop=True)

    out["efficiency"] = out["efficiency"].round().astype(int)
    out["days"] = out["days"].round().astype(int)

    report = ValidationReport(frame=out, dropped=before - len(out), reasons=reasons)
    if strict and not report.ok:
        raise DataError("; ".join(report.reasons))
    return report


def synthesize(n_samples: int = 240, seed: int = 42) -> pd.DataFrame:
    """Generate a physically grounded seed dataset.

    Costs are produced from the real progressive tariff rather than a flat
    ฿4.50/kWh, with an efficiency-driven consumption factor and measurement
    noise, so a model trained on it learns something closer to reality.
    """
    rng = np.random.default_rng(seed)
    wattage = rng.choice(
        [9, 50, 65, 80, 120, 150, 300, 400, 500, 700, 1000, 1200, 1500, 1800, 2500, 3500, 7000],
        n_samples,
    ).astype(float)
    hours = np.round(rng.uniform(0.2, 24.0, n_samples), 1)
    days = rng.integers(1, 32, n_samples)
    efficiency = rng.integers(1, 6, n_samples)
    household_baseline = rng.uniform(80, 600, n_samples)

    nominal_kwh = calculate_kwh(wattage, hours, days)
    # A better efficiency label means the appliance draws less than its plate
    # rating to do the same work: label 5 ≈ 0.80x, label 1 ≈ 1.20x.
    actual_kwh = nominal_kwh * (1.20 - 0.10 * (efficiency - 1))

    tariff = TARIFF_RESIDENTIAL_LARGE
    costs = np.array(
        [marginal_cost(k, tariff, b)["total"] for k, b in zip(actual_kwh, household_baseline)]
    )
    costs *= rng.normal(1.0, 0.04, n_samples)  # metering / billing-period noise

    return pd.DataFrame(
        {
            "wattage": wattage.astype(int),
            "hours": hours,
            "days": days.astype(int),
            "efficiency": efficiency.astype(int),
            "cost": np.round(np.clip(costs, 0.01, None), 2),
        }
    )


def _migrate_legacy_csv() -> Path | None:
    """One-time move of a pre-refactor CSV into the canonical data directory."""
    for legacy in LEGACY_CSV_FILES:
        if legacy.exists() and legacy.resolve() != CSV_FILE.resolve():
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy, CSV_FILE)
            return legacy
    return None


def load_data(path: Path = CSV_FILE, *, seed_if_missing: bool = True) -> pd.DataFrame:
    """Return the validated dataset, creating or migrating one if needed."""
    path = Path(path)
    if not path.exists():
        if _migrate_legacy_csv() is None:
            if not seed_if_missing:
                raise DataError(f"No dataset at {path}")
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            write_data(synthesize(), path)

    try:
        raw = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raw = pd.DataFrame(columns=COLUMNS)
    except Exception as exc:  # malformed file — surface it, never crash blind
        raise DataError(f"Could not read {path.name}: {exc}") from exc

    return validate(raw).frame


def write_data(df: pd.DataFrame, path: Path = CSV_FILE) -> None:
    """Atomically replace the dataset file (never leaves a truncated CSV)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="") as handle:
            df.loc[:, COLUMNS].to_csv(handle, index=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def add_record(wattage, hours, days, efficiency, cost, path: Path = CSV_FILE) -> pd.DataFrame:
    """Validate and append one observation. Returns the updated dataset.

    Raises :class:`DataError` on an unusable record instead of writing garbage
    the next training run would have to cope with.
    """
    candidate = pd.DataFrame(
        [{"wattage": wattage, "hours": hours, "days": days, "efficiency": efficiency, "cost": cost}]
    )
    report = validate(candidate)
    if report.frame.empty:
        raise DataError(
            "Record rejected: " + ("; ".join(report.reasons) or "values out of range")
        )

    existing = load_data(path)
    combined = pd.concat([existing, report.frame], ignore_index=True)
    combined = validate(combined).frame
    write_data(combined, path)
    return combined


def summary_stats(df: pd.DataFrame) -> dict:
    """Headline numbers for the Control Center."""
    feat = add_features(df)
    return {
        "records": len(df),
        "median_kwh": float(feat["kwh"].median()) if len(feat) else 0.0,
        "median_rate": float(feat["rate"].median()) if len(feat) else 0.0,
        "rate_spread": float(feat["rate"].quantile(0.9) - feat["rate"].quantile(0.1))
        if len(feat)
        else 0.0,
        "efficiency_coverage": sorted(int(e) for e in df["efficiency"].unique()) if len(df) else [],
    }
