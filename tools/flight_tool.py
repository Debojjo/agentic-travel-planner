"""Flight search tool.

The dataset's flights are dated 2025, so they are treated as recurring schedules:
we keep the time of day and duration and re-date them to the requested travel date.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from data_loader import DataStore, Flight, load_all
from tools.common import (
    ToolInputError,
    clamp,
    error_payload,
    format_duration,
    parse_date,
    resolve_city,
    to_json,
)


def rebase_flight(flight: Flight, travel_date) -> dict:
    """Re-date a flight to travel_date, keeping its departure time and duration."""
    departure = datetime.combine(travel_date, flight.departure_time.time())
    arrival = departure + timedelta(minutes=flight.duration_minutes)
    return {
        "flight_id": flight.flight_id,
        "airline": flight.airline,
        "origin": flight.origin,
        "destination": flight.destination,
        "departure_time": departure.isoformat(timespec="minutes"),
        "arrival_time": arrival.isoformat(timespec="minutes"),
        "duration_minutes": flight.duration_minutes,
        "duration": format_duration(flight.duration_minutes),
        "price_inr": flight.price,
    }


def schedule_flight(flight: Flight) -> dict:
    """A flight as a recurring daily schedule (times of day only, no date)."""
    arrival_day_offset = (flight.arrival_time.date() - flight.departure_time.date()).days
    return {
        "flight_id": flight.flight_id,
        "airline": flight.airline,
        "origin": flight.origin,
        "destination": flight.destination,
        "departure_time": flight.departure_time.strftime("%H:%M"),
        "arrival_time": flight.arrival_time.strftime("%H:%M"),
        "arrives_next_day": arrival_day_offset > 0,
        "duration_minutes": flight.duration_minutes,
        "duration": format_duration(flight.duration_minutes),
        "price_inr": flight.price,
    }


def search_flights(
    origin: str,
    destination: str,
    travel_date: str | None = None,
    sort_by: str = "cheapest",
    max_results: int = 3,
    store: DataStore | None = None,
    now: datetime | None = None,
) -> dict:
    """Find one-way flights between two cities. Results are sorted best-first.

    With a travel_date the flights are re-dated to it (and, for today, flights that already left are
    excluded). Without one, the recurring daily schedule is returned.
    """
    store = store or load_all()
    now = now or datetime.now()
    try:
        origin_city = resolve_city(store, origin)
        destination_city = resolve_city(store, destination)
        day = parse_date(travel_date) if travel_date else None
        if origin_city == destination_city:
            raise ToolInputError("Origin and destination must be different cities.")
    except ToolInputError as exc:
        return error_payload(str(exc))

    matches = [f for f in store.flights if f.origin == origin_city and f.destination == destination_city]
    if not matches:
        reachable = sorted({f.destination for f in store.flights if f.origin == origin_city})
        return error_payload(
            f"No flights from {origin_city} to {destination_city} in the dataset.",
            destinations_from_origin=reachable,
        )

    if day is not None and day == now.date():
        matches = [f for f in matches if f.departure_time.time() > now.time()]
        if not matches:
            return error_payload(
                f"No {origin_city} to {destination_city} flights are left today (it is {now:%H:%M}). Try a later date."
            )

    if sort_by == "fastest":
        matches.sort(key=lambda f: (f.duration_minutes, f.price))
    else:
        matches.sort(key=lambda f: (f.price, f.duration_minutes))

    limit = clamp(max_results, 1, 5)
    return {
        "route": f"{origin_city} -> {destination_city}",
        "travel_date": day.isoformat() if day else None,
        "sorted_by": "fastest" if sort_by == "fastest" else "cheapest",
        "total_found": len(matches),
        "flights": [rebase_flight(f, day) if day else schedule_flight(f) for f in matches[:limit]],
        "note": (
            "Schedules are recurring; times are re-dated to the travel date. One-way fares only."
            if day
            else "No date given, so this is the recurring daily schedule (times of day). One-way fares only."
        ),
    }


def list_flight_routes(city: str, direction: str = "from", store: DataStore | None = None) -> dict:
    """Where can you fly from (or to) a city? Lists connected cities with the cheapest fare."""
    store = store or load_all()
    try:
        city_name = resolve_city(store, city)
    except ToolInputError as exc:
        return error_payload(str(exc))

    outbound = direction != "to"
    flights = [f for f in store.flights if (f.origin if outbound else f.destination) == city_name]
    if not flights:
        word = "from" if outbound else "to"
        return error_payload(f"No flights {word} {city_name} in the dataset.")

    by_city: dict[str, list[Flight]] = {}
    for flight in flights:
        by_city.setdefault(flight.destination if outbound else flight.origin, []).append(flight)
    routes = [
        {
            "city": other,
            "cheapest_price_inr": min(f.price for f in group),
            "fastest_duration": format_duration(min(f.duration_minutes for f in group)),
            "flights": len(group),
        }
        for other, group in sorted(by_city.items())
    ]
    return {
        "city": city_name,
        "direction": "from" if outbound else "to",
        "routes": routes,
        "note": "One-way fares. Only these routes exist in the dataset.",
    }


class FlightSearchInput(BaseModel):
    """Arguments for the search_flights tool."""

    origin: str = Field(description="Departure city, e.g. 'Hyderabad'")
    destination: str = Field(description="Arrival city, e.g. 'Goa'")
    travel_date: str | None = Field(
        default=None, description="Travel date YYYY-MM-DD. Omit it to see the recurring daily schedule instead."
    )
    sort_by: Literal["cheapest", "fastest"] = Field(default="cheapest", description="Ranking: 'cheapest' or 'fastest'")
    max_results: int = Field(default=3, ge=1, le=5, description="Number of flights to return")


@tool("search_flights", args_schema=FlightSearchInput)
def flight_search_tool(
    origin: str, destination: str, travel_date: str | None = None, sort_by: str = "cheapest", max_results: int = 3
) -> str:
    """Search one-way flights between two Indian cities, on a date or as the daily schedule if no date is given.
    Returns options sorted best-first. Call again with origin and destination swapped for the return leg."""
    return to_json(search_flights(origin, destination, travel_date, sort_by, max_results))


class FlightRoutesInput(BaseModel):
    """Arguments for the list_flight_routes tool."""

    city: str = Field(description="The city, e.g. 'Delhi'")
    direction: Literal["from", "to"] = Field(
        default="from",
        description="'from' lists where you can fly TO from this city; 'to' lists where flights come from",
    )


@tool("list_flight_routes", args_schema=FlightRoutesInput)
def flight_routes_tool(city: str, direction: str = "from") -> str:
    """List which cities you can fly to from a city (or fly from, to a city), with the cheapest fare for each."""
    return to_json(list_flight_routes(city, direction))
