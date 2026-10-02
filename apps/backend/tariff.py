"""Progressive electricity tariff maths.

Thai residential tariffs are block-progressive: the last kWh you consume costs
more than the first. That makes the cost of running one appliance depend on
what the rest of the household already draws, so this module exposes both a
standalone bill and a *marginal* cost on top of a household baseline.
"""
from __future__ import annotations

from backend.config import TariffConfig

__all__ = ["energy_charge", "tier_segments", "bill", "marginal_cost", "marginal_rate"]


def energy_charge(kwh: float, tariff: TariffConfig) -> float:
    """Progressive energy charge in THB for ``kwh`` consumed in one month."""
    return sum(seg["kwh"] * seg["rate"] for seg in tier_segments(kwh, tariff))


def tier_segments(kwh: float, tariff: TariffConfig, start_kwh: float = 0.0) -> list[dict]:
    """Split consumption into the tariff blocks it falls across.

    ``start_kwh`` offsets the split so a household already consuming, say, 250
    kWh has its next kWh priced in the correct (higher) block.

    Returns one dict per touched block with ``label``, ``kwh``, ``rate`` and
    ``charge``, suitable for charting the breakdown directly.
    """
    kwh = max(0.0, float(kwh))
    start = max(0.0, float(start_kwh))
    segments: list[dict] = []
    lower = 0.0
    remaining = kwh

    for tier in tariff.tiers:
        if remaining <= 0:
            break
        block_start, block_end = lower, tier.upper_kwh
        lower = tier.upper_kwh

        # Portion of this block that sits above the household baseline.
        usable_from = max(block_start, start)
        usable_to = block_end
        if usable_to <= usable_from:
            continue

        take = min(remaining, usable_to - usable_from)
        if take <= 0:
            continue

        upper_text = "∞" if block_end == float("inf") else f"{block_end:,.0f}"
        segments.append(
            {
                "label": f"{block_start:,.0f}–{upper_text} kWh",
                "kwh": take,
                "rate": tier.rate,
                "charge": take * tier.rate,
            }
        )
        remaining -= take

    return segments


def bill(kwh: float, tariff: TariffConfig, include_service_charge: bool = True) -> dict:
    """Full monthly bill breakdown for ``kwh`` of standalone consumption."""
    kwh = max(0.0, float(kwh))
    energy = energy_charge(kwh, tariff)
    ft = kwh * tariff.ft_rate
    service = tariff.service_charge if include_service_charge else 0.0
    subtotal = energy + ft + service
    vat = subtotal * tariff.vat_rate
    return {
        "kwh": kwh,
        "energy": energy,
        "ft": ft,
        "service": service,
        "subtotal": subtotal,
        "vat": vat,
        "total": subtotal + vat,
    }


def marginal_cost(kwh: float, tariff: TariffConfig, baseline_kwh: float = 0.0) -> dict:
    """What adding ``kwh`` costs a household already consuming ``baseline_kwh``.

    The service charge is excluded: the household pays it whether or not the
    appliance is plugged in, so attributing it to the appliance would overstate
    its cost.
    """
    kwh = max(0.0, float(kwh))
    baseline_kwh = max(0.0, float(baseline_kwh))
    segments = tier_segments(kwh, tariff, start_kwh=baseline_kwh)
    energy = sum(seg["charge"] for seg in segments)
    ft = kwh * tariff.ft_rate
    subtotal = energy + ft
    vat = subtotal * tariff.vat_rate
    return {
        "kwh": kwh,
        "baseline_kwh": baseline_kwh,
        "segments": segments,
        "energy": energy,
        "ft": ft,
        "service": 0.0,
        "subtotal": subtotal,
        "vat": vat,
        "total": subtotal + vat,
        "effective_rate": (subtotal + vat) / kwh if kwh > 0 else 0.0,
    }


def marginal_rate(tariff: TariffConfig, baseline_kwh: float) -> float:
    """All-in THB/kWh for the *next* kWh at the given baseline consumption."""
    return marginal_cost(1.0, tariff, baseline_kwh)["total"]
