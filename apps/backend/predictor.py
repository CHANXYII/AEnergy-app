"""The AEnergy prediction engine.

Design note — why this is not "RandomForest on cost"
----------------------------------------------------
A tree ensemble can only ever predict a weighted average of training targets,
so a forest fitted directly on monthly cost is structurally incapable of
predicting a value above the most expensive row it has ever seen. Ask the old
model about a 7 kW EV charger when the dataset topped out at ฿6,000 and it
confidently returns ฿6,000.

This engine instead learns the **effective tariff rate** (THB per kWh) and
multiplies it back by physically computed energy:

    cost = kwh(wattage, hours, days) x model(features)

The rate is bounded and roughly scale-free, which trees handle well, while the
kWh term carries the extrapolation exactly. The forest's job is reduced to the
part that genuinely needs learning: how efficiency label, duty cycle and tier
position bend the rate actually paid.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from backend.config import METRICS_FILE, MODEL_FILE, TariffConfig
from backend.dataset import add_features, calculate_kwh
from backend.tariff import marginal_cost

SCHEMA_VERSION = 2
FEATURES = ["wattage", "hours", "days", "efficiency", "kwh", "log_kwh", "kwh_per_day", "duty_cycle"]

#: Any all-in rate outside this band is a data artefact, not a tariff.
RATE_BOUNDS = (1.0, 15.0)

MIN_ROWS = 8


class ModelError(RuntimeError):
    """Raised when the engine cannot train or load."""


@dataclass
class ModelBundle:
    estimator: RandomForestRegressor
    features: list[str]
    metrics: dict
    trained_at: str
    n_samples: int
    schema_version: int = SCHEMA_VERSION

    @property
    def importances(self) -> pd.DataFrame:
        return (
            pd.DataFrame(
                {"feature": self.features, "importance": self.estimator.feature_importances_}
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


def _design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    feat = add_features(df)
    feat["log_kwh"] = np.log1p(feat["kwh"])
    return feat.loc[:, FEATURES]


def _new_estimator(n_samples: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=1 if n_samples < 60 else 2,
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    )


def _cost_metrics(y_true_cost, y_pred_cost) -> dict:
    y_true = np.asarray(y_true_cost, dtype=float)
    y_pred = np.asarray(y_pred_cost, dtype=float)
    err = y_pred - y_true
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    denom = np.maximum(np.abs(y_true), 1e-9)
    return {
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": float(np.mean(np.abs(err) / denom)),
    }


def evaluate(df: pd.DataFrame, tariff: TariffConfig | None = None) -> dict:
    """Honest out-of-fold evaluation, in baht, against two baselines.

    The previous implementation scored R² on the very rows it had just fitted,
    which is why it always reported ~99%. Every number here comes from data the
    fold's model never saw.
    """
    feat = add_features(df)
    X = _design_matrix(df)
    y_rate = feat["rate"].to_numpy()
    y_cost = feat["cost"].to_numpy()
    kwh = feat["kwh"].to_numpy()

    n_splits = int(np.clip(len(df) // 4, 2, 5))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_cost = np.zeros(len(df))
    oof_naive = np.zeros(len(df))
    for train_idx, test_idx in kf.split(X):
        est = _new_estimator(len(train_idx))
        est.fit(X.iloc[train_idx], y_rate[train_idx])
        rate = np.clip(est.predict(X.iloc[test_idx]), *RATE_BOUNDS)
        oof_cost[test_idx] = rate * kwh[test_idx]

        # Baseline: the old approach — a forest fitted straight onto cost.
        naive = _new_estimator(len(train_idx))
        naive.fit(X.iloc[train_idx], y_cost[train_idx])
        oof_naive[test_idx] = np.clip(naive.predict(X.iloc[test_idx]), 0, None)

    results = {
        "n_samples": int(len(df)),
        "n_splits": n_splits,
        "model": _cost_metrics(y_cost, oof_cost),
        "baseline_forest_on_cost": _cost_metrics(y_cost, oof_naive),
    }

    if tariff is not None:
        # Pure physics + tariff, no learning at all.
        tariff_cost = np.array([marginal_cost(k, tariff)["total"] for k in kwh])
        results["baseline_tariff_only"] = _cost_metrics(y_cost, tariff_cost)

    results["oof_predictions"] = oof_cost.tolist()
    return results


def train(
    df: pd.DataFrame,
    *,
    tariff: TariffConfig | None = None,
    model_path: Path = MODEL_FILE,
    history_path: Path = METRICS_FILE,
) -> ModelBundle:
    """Evaluate, fit on all data, and persist atomically."""
    if len(df) < MIN_ROWS:
        raise ModelError(f"Need at least {MIN_ROWS} valid records to train (have {len(df)}).")

    feat = add_features(df)
    if feat["rate"].isna().any():
        raise ModelError("Dataset contains rows with zero energy; cannot derive a tariff rate.")

    metrics = evaluate(df, tariff=tariff)

    estimator = _new_estimator(len(df))
    estimator.fit(_design_matrix(df), feat["rate"].to_numpy())

    bundle = ModelBundle(
        estimator=estimator,
        features=list(FEATURES),
        metrics=metrics,
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        n_samples=len(df),
    )
    _save(bundle, model_path)
    _append_history(bundle, history_path)
    return bundle


def _save(bundle: ModelBundle, path: Path = MODEL_FILE) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": bundle.schema_version,
        "estimator": bundle.estimator,
        "features": bundle.features,
        "metrics": bundle.metrics,
        "trained_at": bundle.trained_at,
        "n_samples": bundle.n_samples,
    }
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    try:
        joblib.dump(payload, tmp)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def load_model(path: Path = MODEL_FILE) -> ModelBundle | None:
    """Load a persisted engine, or ``None`` if absent/stale/corrupt.

    A stale artefact is treated as missing rather than fatal so the UI can just
    offer to retrain.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        payload = joblib.load(path)
        if payload.get("schema_version") != SCHEMA_VERSION:
            return None
        if list(payload.get("features", [])) != FEATURES:
            return None
        return ModelBundle(
            estimator=payload["estimator"],
            features=list(payload["features"]),
            metrics=payload.get("metrics", {}),
            trained_at=payload.get("trained_at", "unknown"),
            n_samples=int(payload.get("n_samples", 0)),
            schema_version=SCHEMA_VERSION,
        )
    except Exception:
        return None


def predict(
    bundle: ModelBundle,
    wattage: float,
    hours: float,
    days: int,
    efficiency: int,
) -> dict:
    """Predict monthly cost with an uncertainty band.

    The band is the 10th–90th percentile across the forest's individual trees,
    which gives an honest read on how much the engine is guessing in regions
    the dataset barely covers.
    """
    kwh = calculate_kwh(wattage, hours, days)
    row = pd.DataFrame(
        [{"wattage": wattage, "hours": hours, "days": days, "efficiency": efficiency, "cost": 0.0}]
    )
    X = _design_matrix(row)

    per_tree = np.array([tree.predict(X.to_numpy())[0] for tree in bundle.estimator.estimators_])
    per_tree = np.clip(per_tree, *RATE_BOUNDS)
    rate = float(np.clip(bundle.estimator.predict(X)[0], *RATE_BOUNDS))

    return {
        "kwh": float(kwh),
        "rate": rate,
        "cost": float(rate * kwh),
        "cost_low": float(np.percentile(per_tree, 10) * kwh),
        "cost_high": float(np.percentile(per_tree, 90) * kwh),
    }


def _append_history(bundle: ModelBundle, path: Path = METRICS_FILE) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if path.exists():
        try:
            history = json.loads(path.read_text())
        except Exception:
            history = []
    history.append(
        {
            "trained_at": bundle.trained_at,
            "n_samples": bundle.n_samples,
            "r2": bundle.metrics["model"]["r2"],
            "mae": bundle.metrics["model"]["mae"],
            "mape": bundle.metrics["model"]["mape"],
        }
    )
    path.write_text(json.dumps(history[-50:], indent=2))


def load_history(path: Path = METRICS_FILE) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []
