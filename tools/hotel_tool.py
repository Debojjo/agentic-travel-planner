"""Hotel recommendation tool."""

from __future__ import annotations

from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from data_loader import DataStore, Hotel, load_all
from tools.common import ToolInputError, clamp, error_payload, resolve_city, to_json


def value_score(hotel: Hotel) -> float:
    """Stars per Rs 1000 of nightly price. Higher means better value."""
    return round(hotel.stars / hotel.price_per_night * 1000, 2)


def _to_dict(hotel: Hotel, nights: int | None) -> dict:
    result = {
        "hotel_id": hotel.hotel_id,
        "name": hotel.name,
        "city": hotel.city,
        "stars": hotel.stars,
        "price_per_night_inr": hotel.price_per_night,
        "amenities": hotel.amenities,
        "value_score": value_score(hotel),
    }
    if nights:
        result["nights"] = nights
        result["total_price_inr"] = hotel.price_per_night * nights
    return result


def search_hotels(
    city: str,
    min_stars: int | None = None,
    max_price_per_night: int | None = None,
    required_amenities: list[str] | None = None,
    sort_by: str = "value",
    nights: int | None = None,
    max_results: int = 3,
    store: DataStore | None = None,
) -> dict:
    """Filter and rank hotels in a city. Results are sorted best-first."""
    store = store or load_all()
    try:
        city_name = resolve_city(store, city)
        if min_stars is not None and not 1 <= min_stars <= 5:
            raise ToolInputError("min_stars must be between 1 and 5.")
        if nights is not None and nights < 1:
            raise ToolInputError("nights must be at least 1.")
    except ToolInputError as exc:
        return error_payload(str(exc))

    in_city = [h for h in store.hotels if h.city == city_name]
    wanted = {a.strip().lower() for a in (required_amenities or [])}
    matches = [
        h
        for h in in_city
        if (min_stars is None or h.stars >= min_stars)
        and (max_price_per_night is None or h.price_per_night <= max_price_per_night)
        and wanted <= set(h.amenities)
    ]

    if not matches:
        prices = [h.price_per_night for h in in_city]
        return error_payload(
            f"No hotels in {city_name} match the filters.",
            hotels_in_city=len(in_city),
            price_range_inr=[min(prices), max(prices)] if prices else None,
            max_stars=max((h.stars for h in in_city), default=None),
        )

    if sort_by == "cheapest":
        matches.sort(key=lambda h: (h.price_per_night, -h.stars))
    elif sort_by == "stars":
        matches.sort(key=lambda h: (-h.stars, h.price_per_night))
    else:
        matches.sort(key=lambda h: (-value_score(h), -h.stars, h.price_per_night))

    limit = clamp(max_results, 1, 5)
    return {
        "city": city_name,
        "sorted_by": sort_by if sort_by in ("cheapest", "stars") else "value",
        "total_found": len(matches),
        "hotels": [_to_dict(h, nights) for h in matches[:limit]],
        "note": "Dataset has star ratings only (no guest reviews). value_score = stars per Rs 1000/night.",
    }


class HotelSearchInput(BaseModel):
    """Arguments for the search_hotels tool."""

    city: str = Field(description="City to stay in, e.g. 'Goa'")
    min_stars: int | None = Field(default=None, ge=1, le=5, description="Minimum star rating")
    max_price_per_night: int | None = Field(default=None, gt=0, description="Maximum price per night in INR")
    required_amenities: list[str] | None = Field(
        default=None, description="Amenities that must be present: wifi, pool, gym, spa, breakfast, parking"
    )
    sort_by: Literal["value", "cheapest", "stars"] = Field(default="value", description="Ranking criterion")
    nights: int | None = Field(default=None, ge=1, description="Number of nights, to include total price")
    max_results: int = Field(default=3, ge=1, le=5, description="Number of hotels to return")


@tool("search_hotels", args_schema=HotelSearchInput)
def hotel_search_tool(
    city: str,
    min_stars: int | None = None,
    max_price_per_night: int | None = None,
    required_amenities: list[str] | None = None,
    sort_by: str = "value",
    nights: int | None = None,
    max_results: int = 3,
) -> str:
    """Find and rank hotels in a city by value, price or stars, with optional filters.
    Pass nights to get the total stay price."""
    return to_json(
        search_hotels(city, min_stars, max_price_per_night, required_amenities, sort_by, nights, max_results)
    )
