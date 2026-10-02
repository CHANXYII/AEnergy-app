"""Central configuration: filesystem paths, tariff defaults, domain constants."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
#
# Everything is anchored to the repository root rather than the process working
# directory. Previously the app used bare relative names ('collected_data.csv',
# 'model.pkl'), so launching from different directories silently read and wrote
# different datasets and different models.
# ---------------------------------------------------------------------------
# apps/backend/config.py -> apps/backend -> apps -> repository root
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
CSV_FILE = DATA_DIR / "collected_data.csv"
MODEL_FILE = DATA_DIR / "model.joblib"
METRICS_FILE = DATA_DIR / "training_history.json"

# Pre-refactor locations, checked once so an existing checkout's data survives
# the upgrade. Both date from when paths were working-directory relative.
LEGACY_CSV_FILES = (
    ROOT_DIR / "collected_data.csv",
    ROOT_DIR / "front-end" / "collected_data.csv",
)

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

#: Grid emission factor for Thailand, kg CO2e per kWh (TGO, residential grid mix).
CO2_KG_PER_KWH = 0.4999

#: Mature trees needed to absorb one tonne of CO2 per year, used for context.
CO2_KG_ABSORBED_PER_TREE_YEAR = 21.0

#: Flat reference rate (THB/kWh) used by the legacy estimator. Retained purely
#: as a comparison line in the UI, never as the costing engine.
LEGACY_FLAT_RATE = 4.5


@dataclass(frozen=True)
class TariffTier:
    """A single progressive-tariff block."""

    upper_kwh: float  # inclusive upper bound of the block; inf for the last one
    rate: float       # THB per kWh inside the block


@dataclass(frozen=True)
class TariffConfig:
    """Thailand residential electricity tariff.

    Defaults follow MEA/PEA residential type 1.2 (dwellings consuming more than
    150 kWh/month), plus the Ft fuel-adjustment charge, a fixed monthly service
    charge and 7% VAT.
    """

    name: str = "Residential 1.2 (>150 kWh/month)"
    tiers: tuple[TariffTier, ...] = (
        TariffTier(150.0, 3.2484),
        TariffTier(400.0, 4.2218),
        TariffTier(float("inf"), 4.4217),
    )
    service_charge: float = 38.22  # THB/month
    ft_rate: float = 0.3672        # THB/kWh fuel adjustment
    vat_rate: float = 0.07

    def flat_equivalent(self, kwh: float) -> float:
        """Average all-in THB/kWh at the given monthly consumption."""
        if kwh <= 0:
            return 0.0
        from backend.tariff import bill  # local import to avoid a cycle

        return bill(kwh, self)["total"] / kwh


TARIFF_RESIDENTIAL_SMALL = TariffConfig(
    name="Residential 1.1 (<=150 kWh/month)",
    tiers=(
        TariffTier(15.0, 2.3488),
        TariffTier(25.0, 2.9882),
        TariffTier(35.0, 3.2405),
        TariffTier(100.0, 3.6237),
        TariffTier(150.0, 3.7171),
        TariffTier(400.0, 4.2218),
        TariffTier(float("inf"), 4.4217),
    ),
    service_charge=8.19,
)

TARIFF_RESIDENTIAL_LARGE = TariffConfig()

TARIFF_FLAT_LEGACY = TariffConfig(
    name="Flat rate (legacy ฿4.50/kWh)",
    tiers=(TariffTier(float("inf"), LEGACY_FLAT_RATE),),
    service_charge=0.0,
    ft_rate=0.0,
    vat_rate=0.0,
)

TARIFFS: dict[str, TariffConfig] = {
    t.name: t for t in (TARIFF_RESIDENTIAL_LARGE, TARIFF_RESIDENTIAL_SMALL, TARIFF_FLAT_LEGACY)
}

DEFAULT_TARIFF_NAME = TARIFF_RESIDENTIAL_LARGE.name

#: Typical household consumption excluding the appliance under test. Progressive
#: tariffs are marginal, so an appliance's true cost depends on what the rest of
#: the home already draws.
DEFAULT_HOUSEHOLD_BASELINE_KWH = 250.0


@dataclass(frozen=True)
class Appliance:
    """A preset to seed the estimator with realistic starting values."""

    name: str
    icon: str
    wattage: int
    hours_per_day: float
    days_per_month: int
    efficiency: int = 3


APPLIANCE_PRESETS: tuple[Appliance, ...] = (
    Appliance("Custom", "🎛️", 1000, 8.0, 30, 5),
    Appliance("Air conditioner (12,000 BTU)", "❄️", 1200, 8.0, 30, 4),
    Appliance("Air conditioner (18,000 BTU)", "❄️", 1800, 8.0, 30, 4),
    Appliance("Refrigerator (2-door)", "🧊", 150, 24.0, 30, 5),
    Appliance("Electric water heater", "🚿", 3500, 0.5, 30, 3),
    Appliance("Washing machine", "🫧", 500, 1.0, 12, 4),
    Appliance("Clothes dryer", "🌀", 2500, 1.0, 12, 3),
    Appliance("Microwave oven", "🍲", 1000, 0.4, 30, 3),
    Appliance("Rice cooker", "🍚", 700, 0.8, 30, 3),
    Appliance("Electric kettle", "☕", 1800, 0.3, 30, 3),
    Appliance("Air fryer", "🍟", 1500, 0.5, 15, 3),
    Appliance("Electric fan", "🌬️", 50, 10.0, 30, 4),
    Appliance("LED TV (43 inch)", "📺", 80, 5.0, 30, 5),
    Appliance("Desktop PC + monitor", "🖥️", 300, 8.0, 22, 3),
    Appliance("Laptop", "💻", 65, 8.0, 30, 5),
    Appliance("LED lighting (whole home)", "💡", 120, 6.0, 30, 5),
    Appliance("Clothes iron", "👔", 1200, 0.5, 8, 3),
    Appliance("Water pump", "🚰", 400, 1.5, 30, 3),
    Appliance("Hair dryer", "💇", 1500, 0.2, 30, 3),
    Appliance("EV home charger (7 kW)", "🔌", 7000, 3.0, 20, 5),
)

PRESETS_BY_NAME: dict[str, Appliance] = {a.name: a for a in APPLIANCE_PRESETS}

#: Human-readable meaning of the 1-5 efficiency label used throughout the app.
EFFICIENCY_LABELS: dict[int, str] = {
    1: "No. 1 — Poor (legacy, no inverter)",
    2: "No. 2 — Below average",
    3: "No. 3 — Average",
    4: "No. 4 — Good (inverter class)",
    5: "No. 5 — Excellent (best in class)",
}
