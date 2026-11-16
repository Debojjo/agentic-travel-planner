"""Human-readable rendering of a TripPlan (markdown)."""

from __future__ import annotations

from datetime import datetime

from agent.schemas import FlightLeg, TripPlan
from tools.common import format_inr


def _flight_line(label: str, leg: FlightLeg | None) -> str:
    if leg is None:
        return f"- {label}: not available"
    dep = datetime.fromisoformat(leg.departure_time)
    arr = datetime.fromisoformat(leg.arrival_time)
    return (
        f"- {label}: {leg.airline} ({format_inr(leg.price_inr)}) - {leg.origin} to {leg.destination}, "
        f"departs {dep:%H:%M} on {dep.strftime('%b')} {dep.day}, arrives {arr:%H:%M} ({leg.duration})"
    )


def _weather_line(day) -> str:
    when = datetime.fromisoformat(day.date)
    label = f"- Day {day.day_number} ({when.strftime('%b')} {when.day}): "
    if day.weather is None:
        return label + "not available"
    text = f"{day.weather}, {day.temp_min_c:.0f}\u2013{day.temp_max_c:.0f}\u00b0C"
    if day.rain_expected:
        text += ", rain likely"
    if day.weather_source == "historical_estimate":
        text += " (typical, from last year)"
    return label + text


def format_itinerary(plan: TripPlan) -> str:
    """Render the plan in the format shown in the project brief."""
    lines = [f"**{plan.title}**", plan.summary, "", "**Flights:**"]
    lines.append(_flight_line("Outbound", plan.outbound_flight))
    lines.append("- Return: not needed (one-way trip)" if plan.one_way else _flight_line("Return", plan.return_flight))

    lines += ["", "**Hotel:**"]
    if not plan.needs_hotel:
        lines.append("- Not needed (staying with friends or family)")
    elif plan.hotel:
        h = plan.hotel
        rooms = f" \u00d7 {h.rooms} rooms" if h.rooms > 1 else ""
        lines.append(
            f"- {h.name} ({format_inr(h.price_per_night_inr)}/night, {h.stars}-star) - "
            f"{h.nights} night{'s' if h.nights != 1 else ''}{rooms} = {format_inr(h.total_price_inr)}"
        )
        if h.amenities:
            lines.append(f"- Amenities: {', '.join(h.amenities)}")
    else:
        lines.append("- not available")

    lines += ["", "**Weather:**"] + [_weather_line(d) for d in plan.days]

    lines += ["", "**Itinerary:**"]
    for d in plan.days:
        stops = ", ".join(f"{p.name} ({p.type}, {p.rating})" for p in d.places) or "Leisure day"
        note = f" - {d.note}" if d.note else ""
        lines.append(f"- Day {d.day_number}: {stops}{note}")

    b = plan.budget
    lines += [
        "",
        "**Estimated Budget:**",
        f"- Flights: {format_inr(b.flights)}",
        f"- Hotel: {format_inr(b.hotel)}",
        f"- Food & local travel: {format_inr(b.food_and_local_travel)}",
        "-------------------------------------",
        f"**Total Cost: {format_inr(b.total)}**"
        + (f" ({format_inr(b.per_person)} per person)" if plan.travelers > 1 else ""),
    ]
    if b.within_budget is True:
        lines.append(f"Within your {format_inr(b.budget_limit)} budget ({format_inr(b.remaining)} to spare).")
    elif b.within_budget is False:
        lines.append(f"Over your {format_inr(b.budget_limit)} budget by {format_inr(b.over_by)}.")
    lines += [f"_{a}_" for a in b.assumptions]

    if plan.reasoning:
        lines += ["", "**Why we chose this:**"]
        lines += [f"- {key.capitalize()}: {text}" for key, text in plan.reasoning.items()]
    if plan.warnings:
        lines += ["", "**Notes:**"] + [f"- {w}" for w in plan.warnings]
    return "\n".join(lines)
