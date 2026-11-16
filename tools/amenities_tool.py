"""Hotel amenities tool: what a city's hotels offer, and which hotels have a given amenity."""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from data_loader import DataStore, load_all
from tools.common import ToolInputError, error_payload, resolve_city, to_json


def hotel_amenities(city: str, amenity: str | None = None, store: DataStore | None = None) -> dict:
    """Summarise amenities in a city, or list the hotels that have one amenity (cheapest first)."""
    store = store or load_all()
    try:
        city_name = resolve_city(store, city)
    except ToolInputError as exc:
        return error_payload(str(exc))

    in_city = [h for h in store.hotels if h.city == city_name]
    known = sorted({a for h in store.hotels for a in h.amenities})

    if not amenity or not amenity.strip():
        rows = []
        for name in known:
            having = sorted((h for h in in_city if name in h.amenities), key=lambda h: (h.price_per_night, -h.stars))
            if having:
                rows.append(
                    {
                        "amenity": name,
                        "hotels": len(having),
                        "cheapest_hotel": having[0].name,
                        "cheapest_price_inr": having[0].price_per_night,
                        "hotel_names": [h.name for h in having],
                    }
                )
        return {
            "city": city_name,
            "hotels_in_city": len(in_city),
            "amenities": rows,
            "note": "Counts are hotels in this city offering each amenity; prices are per night.",
        }

    wanted = amenity.strip().lower()
    if wanted not in known:
        return error_payload(f"No hotel in the dataset lists '{amenity}'.", available_amenities=known)
    having = sorted((h for h in in_city if wanted in h.amenities), key=lambda h: (h.price_per_night, -h.stars))
    if not having:
        offered = sorted({a for h in in_city for a in h.amenities})
        return error_payload(f"No hotel in {city_name} offers {wanted}.", offered_in_city=offered)
    return {
        "city": city_name,
        "amenity": wanted,
        "count": len(having),
        "hotels": [
            {
                "hotel_id": h.hotel_id,
                "name": h.name,
                "stars": h.stars,
                "price_per_night_inr": h.price_per_night,
                "amenities": h.amenities,
            }
            for h in having
        ],
        "note": "Sorted by price, cheapest first.",
    }


class HotelAmenitiesInput(BaseModel):
    """Arguments for the hotel_amenities tool."""

    city: str = Field(description="City, e.g. 'Goa'")
    amenity: str | None = Field(
        default=None,
        description=(
            "One of wifi, pool, gym, spa, breakfast, parking. Omit it to see every amenity the city's hotels offer."
        ),
    )


@tool("hotel_amenities", args_schema=HotelAmenitiesInput)
def hotel_amenities_tool(city: str, amenity: str | None = None) -> str:
    """Show which amenities a city's hotels offer, or which hotels have a specific amenity (cheapest first)."""
    return to_json(hotel_amenities(city, amenity))
