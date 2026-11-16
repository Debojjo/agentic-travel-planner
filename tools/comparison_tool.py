"""Comparison tools: put hotels, or flights on one route, side by side and state the trade-offs."""

from __future__ import annotations

from datetime import datetime

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from data_loader import DataStore, load_all
from tools.common import ToolInputError, clamp, error_payload, format_duration, format_inr, resolve_city, to_json
from tools.flight_tool import search_flights
from tools.hotel_tool import value_score


def compare_hotels(
    city: str,
    hotel_ids: list[str] | None = None,
    max_hotels: int = 4,
    nights: int | None = None,
    store: DataStore | None = None,
) -> dict:
    """Compare hotels in a city: the given hotel_ids, or the best-value hotels if none are named."""
    store = store or load_all()
    try:
        city_name = resolve_city(store, city)
    except ToolInputError as exc:
        return error_payload(str(exc))

    in_city = [h for h in store.hotels if h.city == city_name]
    if hotel_ids:
        by_id = {h.hotel_id: h for h in in_city}
        unknown = [i for i in dict.fromkeys(hotel_ids) if i not in by_id]
        if unknown:
            return error_payload(
                f"These hotel ids are not in {city_name}: {', '.join(unknown)}.",
                hotel_ids_in_city=sorted(by_id),
            )
        chosen = [by_id[i] for i in dict.fromkeys(hotel_ids)]
    else:
        ranked = sorted(in_city, key=lambda h: (-value_score(h), -h.stars, h.price_per_night))
        chosen = ranked[: clamp(max_hotels, 2, 6)]
    if len(chosen) < 2:
        return error_payload("At least two hotels are needed to compare.")

    cheapest = min(chosen, key=lambda h: (h.price_per_night, -h.stars))
    best_value = max(chosen, key=lambda h: (value_score(h), h.stars))
    most_stars = max(chosen, key=lambda h: (h.stars, -h.price_per_night))
    gap = max(h.price_per_night for h in chosen) - cheapest.price_per_night

    rows = []
    for hotel in chosen:
        row = {
            "hotel_id": hotel.hotel_id,
            "name": hotel.name,
            "stars": hotel.stars,
            "price_per_night_inr": hotel.price_per_night,
            "value_score": value_score(hotel),
            "amenities": hotel.amenities,
        }
        if nights:
            row["nights"] = nights
            row["total_price_inr"] = hotel.price_per_night * nights
        rows.append(row)

    all_amenities = sorted({a for h in chosen for a in h.amenities})
    summary = (
        f"Cheapest: {cheapest.name} at {format_inr(cheapest.price_per_night)}/night. "
        f"Best value: {best_value.name} at {format_inr(best_value.price_per_night)}/night. "
        f"Most stars: {most_stars.name} ({most_stars.stars}-star, {format_inr(most_stars.price_per_night)}/night). "
        f"Price gap between cheapest and priciest: {format_inr(gap)} per night."
    )
    return {
        "city": city_name,
        "hotels": rows,
        "highlights": {
            "cheapest": cheapest.hotel_id,
            "best_value": best_value.hotel_id,
            "most_stars": most_stars.hotel_id,
        },
        "price_gap_inr": gap,
        "amenity_matrix": {a: [h.hotel_id for h in chosen if a in h.amenities] for a in all_amenities},
        "summary": summary,
        "note": "Value score = stars per Rs 1000 per night (higher is better).",
    }


def compare_flights(
    origin: str,
    destination: str,
    travel_date: str | None = None,
    store: DataStore | None = None,
    now: datetime | None = None,
) -> dict:
    """Compare every flight on a route (up to five): price, duration, price per hour, and the speed/cost trade-off."""
    result = search_flights(origin, destination, travel_date, "cheapest", 5, store, now)
    if "error" in result:
        return result

    flights = result["flights"]
    for flight in flights:
        flight["price_per_hour_inr"] = round(flight["price_inr"] / (flight["duration_minutes"] / 60))
    cheapest = flights[0]
    fastest = min(flights, key=lambda f: (f["duration_minutes"], f["price_inr"]))

    if len(flights) == 1:
        tradeoff = "Only one flight operates on this route."
    elif cheapest["flight_id"] == fastest["flight_id"]:
        tradeoff = "The cheapest flight is also the fastest."
    else:
        extra = fastest["price_inr"] - cheapest["price_inr"]
        saved = cheapest["duration_minutes"] - fastest["duration_minutes"]
        cost = f"{format_inr(extra)} more" if extra > 0 else "the same price"
        tradeoff = f"The fastest flight costs {cost} and saves {format_duration(saved)}."

    summary = (
        f"Cheapest: {cheapest['airline']} at {format_inr(cheapest['price_inr'])} ({cheapest['duration']}). "
        f"Fastest: {fastest['airline']} at {format_inr(fastest['price_inr'])} ({fastest['duration']}). {tradeoff}"
    )
    return {
        "route": result["route"],
        "travel_date": result["travel_date"],
        "flights": flights,
        "highlights": {"cheapest": cheapest["flight_id"], "fastest": fastest["flight_id"]},
        "summary": summary,
        "note": result["note"],
    }


class CompareHotelsInput(BaseModel):
    """Arguments for the compare_hotels tool."""

    city: str = Field(description="City, e.g. 'Goa'")
    hotel_ids: list[str] | None = Field(
        default=None, description="Hotel ids to compare (from a hotel search). Omit to compare the best-value hotels."
    )
    max_hotels: int = Field(default=4, ge=2, le=6, description="How many hotels to compare when hotel_ids is omitted")
    nights: int | None = Field(default=None, ge=1, description="Number of nights, to include total prices")


@tool("compare_hotels", args_schema=CompareHotelsInput)
def compare_hotels_tool(
    city: str, hotel_ids: list[str] | None = None, max_hotels: int = 4, nights: int | None = None
) -> str:
    """Compare hotels side by side: price, stars, value score and amenities, with the cheapest and best-value pick."""
    return to_json(compare_hotels(city, hotel_ids, max_hotels, nights))


class CompareFlightsInput(BaseModel):
    """Arguments for the compare_flights tool."""

    origin: str = Field(description="Departure city")
    destination: str = Field(description="Arrival city")
    travel_date: str | None = Field(default=None, description="YYYY-MM-DD. Omit for the recurring daily schedule.")


@tool("compare_flights", args_schema=CompareFlightsInput)
def compare_flights_tool(origin: str, destination: str, travel_date: str | None = None) -> str:
    """Compare all flights on a route: price, duration, price per hour, and what the fastest flight costs extra."""
    return to_json(compare_flights(origin, destination, travel_date))
