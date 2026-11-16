"""Pure helper functions for the Streamlit UI (kept separate so they are easy to test)."""

from __future__ import annotations

from datetime import date, timedelta

AMENITY_OPTIONS = ["wifi", "pool", "gym", "spa", "breakfast", "parking"]


def trip_end_date(start: date, length_days: int) -> date:
    """Last day of a trip that starts on `start` and lasts `length_days` days."""
    return start + timedelta(days=length_days - 1)


def build_preferences(
    interests: str = "",
    flight_pref: str = "Cheapest",
    min_stars: int = 1,
    amenities: list[str] | None = None,
) -> str | None:
    """Combine the form's interests and filters into the free-text preferences the agent reads."""
    parts: list[str] = []
    if interests and interests.strip():
        parts.append(f"interests: {interests.strip()}")
    if flight_pref == "Fastest":
        parts.append("prefer the fastest flights")
    if min_stars > 1:
        parts.append(f"hotel must be at least {min_stars} stars")
    if amenities:
        parts.append("hotel must have: " + ", ".join(amenities))
    return "; ".join(parts) or None


def weather_emoji(description: str | None) -> str:
    """Pick an emoji for a weather description."""
    text = (description or "").lower()
    if "thunder" in text:
        return "\u26c8\ufe0f"
    if any(word in text for word in ("rain", "drizzle", "shower")):
        return "\U0001f327\ufe0f"
    if "snow" in text:
        return "\u2744\ufe0f"
    if "fog" in text:
        return "\U0001f32b\ufe0f"
    if "clear" in text:
        return "\u2600\ufe0f"
    if "partly" in text or "mainly" in text:
        return "\u26c5"
    if "cloud" in text or "overcast" in text:
        return "\u2601\ufe0f"
    return "\U0001f321\ufe0f"


# ------------------------------------------------------------ answer tables ----
import pandas as pd  # noqa: E402

from agent.formatter import format_inr  # noqa: E402


def _clock(value: str) -> str:
    """'2026-12-05T07:27' or '07:27' -> '07:27'."""
    return value[-5:]


def _flights_table(data: dict) -> pd.DataFrame:
    rows = []
    for flight in data["flights"]:
        dep, arr = flight["departure_time"], flight["arrival_time"]
        next_day = flight.get("arrives_next_day") or (len(dep) > 5 and dep[:10] != arr[:10])
        rows.append(
            {
                "Airline": flight["airline"],
                "Route": f"{flight['origin']} \u2192 {flight['destination']}",
                "Departs": _clock(dep),
                "Arrives": _clock(arr) + (" (+1 day)" if next_day else ""),
                "Duration": flight["duration"],
                "Price": format_inr(flight["price_inr"]),
            }
        )
        if "price_per_hour_inr" in flight:
            rows[-1]["Per hour"] = format_inr(flight["price_per_hour_inr"])
    return pd.DataFrame(rows)


def _hotels_table(data: dict, with_value: bool = False) -> pd.DataFrame:
    rows = []
    for hotel in data["hotels"]:
        row = {
            "Hotel": hotel["name"],
            "Stars": "\u2605" * hotel["stars"],
            "Per night": format_inr(hotel["price_per_night_inr"]),
            "Amenities": ", ".join(hotel["amenities"]),
        }
        if with_value:
            row["Value score"] = hotel["value_score"]
        if "total_price_inr" in hotel:
            row[f"{hotel['nights']} nights"] = format_inr(hotel["total_price_inr"])
        rows.append(row)
    return pd.DataFrame(rows)


def _places_table(data: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [{"Place": p["name"], "Type": p["type"], "Rating": f"\u2b50 {p['rating']}"} for p in data["places"]]
    )


def _weather_table(data: dict) -> pd.DataFrame:
    rows = []
    for day in data["daily"]:
        if not day.get("available"):
            rows.append({"Date": day["date"], "Weather": "not available"})
            continue
        chance = day.get("rain_probability_percent")
        rows.append(
            {
                "Date": day["date"],
                "Weather": f"{weather_emoji(day['weather'])} {day['weather']}",
                "Temperature": f"{day['temp_min_c']:.0f}\u2013{day['temp_max_c']:.0f}\u00b0C",
                "Rain chance": f"{chance}%" if chance is not None else "-",
                "Source": "Forecast" if day["source"] == "forecast" else "Typical (last year)",
            }
        )
    return pd.DataFrame(rows)


def _amenities_table(data: dict) -> pd.DataFrame:
    """Overview of a city's amenities, or the hotels that offer one amenity."""
    if "amenity" in data:
        return _hotels_table(data)
    return pd.DataFrame(
        [
            {
                "Amenity": row["amenity"],
                "Hotels": row["hotels"],
                "Cheapest": f"{row['cheapest_hotel']} ({format_inr(row['cheapest_price_inr'])}/night)",
            }
            for row in data["amenities"]
        ]
    )


def _routes_table(data: dict) -> pd.DataFrame:
    heading = "To" if data["direction"] == "from" else "From"
    return pd.DataFrame(
        [
            {
                heading: r["city"],
                "Cheapest": format_inr(r["cheapest_price_inr"]),
                "Fastest": r["fastest_duration"],
                "Flights": r["flights"],
            }
            for r in data["routes"]
        ]
    )


def _title(tool: str, args: dict, data: dict) -> str:
    if tool == "search_flights":
        when = f" on {data['travel_date']}" if data.get("travel_date") else " (daily schedule)"
        route = data.get("route", "").replace("->", "\u2192")
        return f"Flights {route}{when}"
    if tool == "search_hotels":
        return f"Hotels in {data.get('city', args.get('city', ''))}"
    if tool == "search_places":
        return f"Places in {data.get('city', args.get('city', ''))}"
    if tool == "get_weather_forecast":
        return f"Weather in {data.get('city', args.get('city', ''))}"
    if tool == "list_flight_routes":
        word = "from" if data.get("direction") == "from" else "to"
        return f"Flight routes {word} {data.get('city', '')}"
    if tool == "compare_hotels":
        return f"Hotel comparison in {data.get('city', args.get('city', ''))}"
    if tool == "compare_flights":
        when = f" on {data['travel_date']}" if data.get("travel_date") else " (daily schedule)"
        return f"Flight comparison {data.get('route', '').replace('->', chr(0x2192))}{when}"
    if tool == "hotel_amenities":
        city = data.get("city", args.get("city", ""))
        return f"Hotels with {data['amenity']} in {city}" if "amenity" in data else f"Amenities in {city}"
    return tool


_BUILDERS = {
    "search_flights": ("flights", _flights_table),
    "search_hotels": ("hotels", _hotels_table),
    "search_places": ("places", _places_table),
    "get_weather_forecast": ("daily", _weather_table),
    "list_flight_routes": ("routes", _routes_table),
    "compare_hotels": ("hotels", lambda data: _hotels_table(data, with_value=True)),
    "compare_flights": ("flights", _flights_table),
    "hotel_amenities": ("amenities", _amenities_table),
}


def results_to_tables(tool_results: list[dict]) -> list[tuple[str, pd.DataFrame | None, str]]:
    """Turn raw tool outputs into (title, table, note) for display. A failed lookup yields (title, None, reason)."""
    tables = []
    for result in tool_results:
        tool, args, data = result["tool"], result.get("args", {}), result["data"]
        if tool not in _BUILDERS:
            continue  # e.g. the budget tool has no table
        title = _title(tool, args, data)
        if "error" in data:
            tables.append((title, None, str(data["error"])))
            continue
        key, builder = _BUILDERS[tool]
        if tool == "hotel_amenities" and "amenity" in data:
            key = "hotels"  # filtered by one amenity: rows live under 'hotels'
        if data.get(key):
            note = " ".join(part for part in (data.get("summary"), data.get("note")) if part)
            shown, total = len(data[key]), data.get("total_found")
            if total and total > shown:
                note = f"Showing {shown} of {total}. " + note
            tables.append((title, builder(data), note))
    return tables
