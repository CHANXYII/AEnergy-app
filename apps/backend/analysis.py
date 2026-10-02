"""Derived, user-facing analysis: emissions, time horizons, savings advice."""
from __future__ import annotations

from dataclasses import dataclass

from backend.config import (
    CO2_KG_ABSORBED_PER_TREE_YEAR,
    CO2_KG_PER_KWH,
    EFFICIENCY_LABELS,
    TariffConfig,
)
from backend.dataset import calculate_kwh
from backend.tariff import marginal_cost


def carbon_footprint(kwh_per_month: float) -> dict:
    """Emissions attributable to the appliance, with a tangible equivalent."""
    kg_month = kwh_per_month * CO2_KG_PER_KWH
    kg_year = kg_month * 12
    return {
        "kg_per_month": kg_month,
        "kg_per_year": kg_year,
        "trees_to_offset": kg_year / CO2_KG_ABSORBED_PER_TREE_YEAR,
    }


def cost_horizons(monthly_cost: float, days_active: int) -> dict:
    """Spread a monthly figure across the horizons people actually budget on."""
    per_day = monthly_cost / max(days_active, 1)
    return {
        "per_day": per_day,
        "per_month": monthly_cost,
        "per_year": monthly_cost * 12,
        "per_5_years": monthly_cost * 60,
    }


@dataclass
class Recommendation:
    icon: str
    title: str
    detail: str
    monthly_saving: float

    @property
    def yearly_saving(self) -> float:
        return self.monthly_saving * 12


def _cost_at(predictor, wattage, hours, days, efficiency) -> float:
    return predictor(wattage, hours, days, efficiency)


def recommendations(
    predictor,
    wattage: float,
    hours: float,
    days: int,
    efficiency: int,
    *,
    limit: int = 4,
) -> list[Recommendation]:
    """Counterfactual savings, priced by the engine itself.

    Each suggestion is a genuine what-if run back through the model rather than
    a hard-coded percentage, so the numbers stay consistent with the headline
    prediction the user is looking at.
    """
    current = _cost_at(predictor, wattage, hours, days, efficiency)
    ideas: list[Recommendation] = []

    if efficiency < 5:
        upgraded = _cost_at(predictor, wattage, hours, days, 5)
        ideas.append(
            Recommendation(
                "🏆",
                f"Upgrade to a label-5 unit (currently label {efficiency})",
                f"Replacing this with a best-in-class model moves you from "
                f"“{EFFICIENCY_LABELS[efficiency].split('—')[1].strip()}” to "
                f"“{EFFICIENCY_LABELS[5].split('—')[1].strip()}”.",
                current - upgraded,
            )
        )

    if hours >= 1.0:
        trimmed = max(0.0, hours - 1.0)
        ideas.append(
            Recommendation(
                "⏱️",
                "Run it one hour less each day",
                f"{hours:.1f} h/day → {trimmed:.1f} h/day. Timers and smart plugs "
                f"make this the cheapest change available.",
                current - _cost_at(predictor, wattage, trimmed, days, efficiency),
            )
        )

    if hours > 4:
        cut = hours * 0.8
        ideas.append(
            Recommendation(
                "📉",
                "Cut runtime by 20%",
                f"{hours:.1f} h/day → {cut:.1f} h/day, e.g. by pre-cooling and "
                f"letting the room coast, or scheduling around occupancy.",
                current - _cost_at(predictor, wattage, cut, days, efficiency),
            )
        )

    if days >= 28:
        ideas.append(
            Recommendation(
                "📅",
                "Give it two days off per month",
                f"{days} → {days - 2} active days. Useful for optional loads such "
                f"as dryers, pumps and secondary appliances.",
                current - _cost_at(predictor, wattage, hours, days - 2, efficiency),
            )
        )

    if wattage > 300:
        derated = wattage * 0.9
        ideas.append(
            Recommendation(
                "🔧",
                "Reclaim 10% of draw through maintenance",
                "Clean filters and coils, reseal ducts, and clear airflow. Neglected "
                "units routinely drift this far above their rated draw.",
                current - _cost_at(predictor, derated, hours, days, efficiency),
            )
        )

    meaningful = [i for i in ideas if i.monthly_saving > 0.5]
    meaningful.sort(key=lambda i: i.monthly_saving, reverse=True)
    return meaningful[:limit]


def efficiency_curve(predictor, wattage: float, hours: float, days: int) -> list[dict]:
    """Predicted cost at every efficiency label, for the comparison chart."""
    return [
        {
            "efficiency": e,
            "label": f"Label {e}",
            "description": EFFICIENCY_LABELS[e],
            "cost": _cost_at(predictor, wattage, hours, days, e),
        }
        for e in (1, 2, 3, 4, 5)
    ]


def household_rollup(appliances: list[dict], tariff: TariffConfig, baseline_kwh: float) -> dict:
    """Aggregate a list of configured appliances into one household bill.

    Appliances are priced in descending order of consumption and stacked onto
    the household baseline, so each one is charged at the tariff block it
    genuinely occupies instead of every device pretending to be the first kWh.
    """
    enriched = []
    for item in appliances:
        kwh = calculate_kwh(item["wattage"], item["hours"], item["days"]) * item.get("quantity", 1)
        enriched.append({**item, "kwh": kwh})
    enriched.sort(key=lambda a: a["kwh"], reverse=True)

    running = baseline_kwh
    total_cost = 0.0
    total_kwh = 0.0
    for item in enriched:
        charge = marginal_cost(item["kwh"], tariff, running)
        item["cost"] = charge["total"]
        item["effective_rate"] = charge["effective_rate"]
        running += item["kwh"]
        total_cost += charge["total"]
        total_kwh += item["kwh"]

    return {
        "appliances": enriched,
        "total_kwh": total_kwh,
        "total_cost": total_cost,
        "baseline_kwh": baseline_kwh,
        "final_household_kwh": running,
        "carbon": carbon_footprint(total_kwh),
    }
