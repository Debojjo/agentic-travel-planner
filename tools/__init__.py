"""LangChain tools for the travel planner."""

from tools.amenities_tool import hotel_amenities_tool
from tools.budget_tool import budget_estimate_tool
from tools.comparison_tool import compare_flights_tool, compare_hotels_tool
from tools.flight_tool import flight_routes_tool, flight_search_tool
from tools.hotel_tool import hotel_search_tool
from tools.places_tool import places_search_tool
from tools.weather_tool import weather_forecast_tool


def get_tools() -> list:
    """The five tools the trip-planning agent uses."""
    return [
        flight_search_tool,
        hotel_search_tool,
        places_search_tool,
        weather_forecast_tool,
        budget_estimate_tool,
    ]


def get_lookup_tools() -> list:
    """Tools for answering questions: the planning tools plus route discovery, amenities and comparisons."""
    return get_tools() + [flight_routes_tool, hotel_amenities_tool, compare_hotels_tool, compare_flights_tool]
