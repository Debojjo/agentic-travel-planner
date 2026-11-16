"""Places / attractions discovery tool."""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from data_loader import DataStore, load_all
from tools.common import ToolInputError, clamp, error_payload, resolve_city, to_json


def search_places(
    city: str,
    place_type: str | None = None,
    min_rating: float | None = None,
    max_results: int = 5,
    store: DataStore | None = None,
) -> dict:
    """List attractions in a city, highest rated first."""
    store = store or load_all()
    try:
        city_name = resolve_city(store, city)
        if min_rating is not None and not 0 <= min_rating <= 5:
            raise ToolInputError("min_rating must be between 0 and 5.")
    except ToolInputError as exc:
        return error_payload(str(exc))

    in_city = [p for p in store.places if p.city == city_name]
    available_types = sorted({p.type for p in in_city})
    wanted_type = place_type.strip().lower() if place_type else None

    type_note = None
    if wanted_type and wanted_type not in available_types:
        # Do not dead-end: hand back the best-rated places of other types so the agent never has to guess.
        type_note = f"No '{wanted_type}' places in {city_name}; showing the best-rated places of other types instead."
        wanted_type = None

    matches = [
        p
        for p in in_city
        if (wanted_type is None or p.type == wanted_type) and (min_rating is None or p.rating >= min_rating)
    ]
    if not matches:
        return error_payload(f"No places in {city_name} match the filters.", available_types=available_types)

    matches.sort(key=lambda p: (-p.rating, p.name))
    limit = clamp(max_results, 1, 10)
    return {
        "city": city_name,
        "total_found": len(matches),
        "available_types": available_types,
        "places": [
            {
                "place_id": p.place_id,
                "name": p.name,
                "type": p.type,
                "label": f"{p.name} ({p.type})",
                "rating": p.rating,
            }
            for p in matches[:limit]
        ],
        "note": type_note or "Each city has only a handful of places; the 'type' field is the reliable category.",
        "requested_type_unavailable": type_note is not None,
    }


class PlacesSearchInput(BaseModel):
    """Arguments for the search_places tool."""

    city: str = Field(description="City to explore, e.g. 'Goa'")
    place_type: str | None = Field(
        default=None, description="Optional category: beach, temple, museum, park, fort, lake, market, monument"
    )
    min_rating: float | None = Field(default=None, ge=0, le=5, description="Minimum rating out of 5")
    max_results: int = Field(default=5, ge=1, le=10, description="Number of places to return")


@tool("search_places", args_schema=PlacesSearchInput)
def places_search_tool(
    city: str, place_type: str | None = None, min_rating: float | None = None, max_results: int = 5
) -> str:
    """Discover attractions in a city, ranked by rating. Optionally filter by type or minimum rating."""
    return to_json(search_places(city, place_type, min_rating, max_results))
