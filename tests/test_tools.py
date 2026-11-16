"""Tests for the flight, hotel and places tools."""

import json

from tools import get_lookup_tools, get_tools
from tools.flight_tool import flight_search_tool, list_flight_routes, search_flights
from tools.hotel_tool import hotel_search_tool, search_hotels
from tools.places_tool import places_search_tool, search_places


# ---------------------------------------------------------------- flights ----
def test_flights_cheapest_vs_fastest():
    cheapest = search_flights("Bangalore", "Goa", "2026-12-05", "cheapest")
    fastest = search_flights("Bangalore", "Goa", "2026-12-05", "fastest")
    assert cheapest["flights"][0]["flight_id"] == "FL0021"
    assert fastest["flights"][0]["flight_id"] == "FL0020"
    assert cheapest["total_found"] == 2


def test_flights_are_rebased_to_travel_date():
    flight = search_flights("Hyderabad", "Goa", "2026-12-05")["flights"][0]
    assert flight["departure_time"] == "2026-12-05T14:26"
    assert flight["arrival_time"] == "2026-12-05T18:26"
    assert flight["price_inr"] == 7299


def test_overnight_flight_arrives_next_day():
    flight = search_flights("Kolkata", "Jaipur", "2026-12-05", "fastest")["flights"][0]
    assert flight["arrival_time"] > flight["departure_time"]


def test_flights_city_aliases_work():
    assert "flights" in search_flights("bengaluru", "goa", "2026-12-05")


def test_flights_no_route_returns_alternatives():
    result = search_flights("Delhi", "Goa", "2026-12-05")
    assert "error" in result
    assert result["destinations_from_origin"] == ["Kolkata"]


def test_flights_bad_inputs():
    assert "error" in search_flights("Paris", "Goa", "2026-12-05")
    assert "error" in search_flights("Goa", "Goa", "2026-12-05")
    assert "error" in search_flights("Goa", "Bangalore", "05-12-2026")


# ----------------------------------------------------------------- hotels ----
def test_hotels_value_ranking():
    result = search_hotels("Goa")
    assert result["hotels"][0]["hotel_id"] == "HOT0014"
    scores = [h["value_score"] for h in result["hotels"]]
    assert scores == sorted(scores, reverse=True)


def test_hotels_filters_and_total_price():
    result = search_hotels("Goa", min_stars=4, max_price_per_night=2300, nights=3)
    assert {h["hotel_id"] for h in result["hotels"]} == {"HOT0014", "HOT0015"}
    top = result["hotels"][0]
    assert top["total_price_inr"] == top["price_per_night_inr"] * 3


def test_hotels_required_amenities():
    result = search_hotels("Goa", required_amenities=["Spa", "pool"], max_results=5)
    assert result["hotels"]
    assert all({"spa", "pool"} <= set(h["amenities"]) for h in result["hotels"])


def test_hotels_sort_options():
    assert search_hotels("Goa", sort_by="cheapest")["hotels"][0]["price_per_night_inr"] == 1232
    stars = [h["stars"] for h in search_hotels("Goa", sort_by="stars", max_results=5)["hotels"]]
    assert stars == sorted(stars, reverse=True)


def test_hotels_no_match_and_bad_input():
    assert "error" in search_hotels("Goa", max_price_per_night=500)
    assert "error" in search_hotels("Atlantis")
    assert "error" in search_hotels("Goa", min_stars=9)


# ----------------------------------------------------------------- places ----
def test_places_sorted_by_rating():
    result = search_places("Kolkata", max_results=3)
    assert [p["place_id"] for p in result["places"]] == ["PLC0031", "PLC0032", "PLC0033"]


def test_places_type_filter():
    result = search_places("Kolkata", place_type="Temple")
    assert {p["type"] for p in result["places"]} == {"temple"}


def test_unavailable_place_type_falls_back_instead_of_dead_ending():
    result = search_places("Goa", place_type="temple")  # Goa has forts, markets and museums only
    assert "error" not in result and result["requested_type_unavailable"] is True
    assert "No 'temple' places in Goa" in result["note"]
    assert result["places"][0]["place_id"] == "PLC0012"  # best-rated Goa place of any type
    assert search_places("Goa", place_type="fort")["requested_type_unavailable"] is False


def test_places_errors_for_bad_input():
    assert "error" in search_places("Atlantis")
    assert "error" in search_places("Kolkata", min_rating=4.95)


# ------------------------------------------------------ langchain wrappers ----
def test_tool_registry():
    assert {t.name for t in get_tools()} == {
        "search_flights",
        "search_hotels",
        "search_places",
        "get_weather_forecast",
        "estimate_budget",
    }


def test_tools_invoke_and_return_json():
    out = flight_search_tool.invoke({"origin": "Hyderabad", "destination": "Goa", "travel_date": "2026-12-05"})
    assert json.loads(out)["flights"][0]["flight_id"] == "FL0016"
    out = hotel_search_tool.invoke({"city": "Goa", "nights": 2})
    assert json.loads(out)["hotels"][0]["total_price_inr"] == 2464
    out = places_search_tool.invoke({"city": "Goa", "place_type": "fort"})
    assert json.loads(out)["places"][0]["place_id"] == "PLC0014"


# --------------------------------------- quick-question support (no date, routes) ----
def test_flights_without_a_date_return_the_daily_schedule():
    result = search_flights("Bangalore", "Goa")
    assert result["travel_date"] is None and "recurring daily schedule" in result["note"]
    first = result["flights"][0]
    assert first["flight_id"] == "FL0021"
    assert first["departure_time"] == "07:27" and first["arrival_time"] == "09:27"
    assert first["arrives_next_day"] is False
    overnight = search_flights("Kolkata", "Jaipur", sort_by="fastest")["flights"]
    assert any(f["arrives_next_day"] for f in overnight)  # FL0014 lands 01:41 the next day


def test_flight_tool_accepts_a_missing_date():
    out = json.loads(flight_search_tool.invoke({"origin": "Bangalore", "destination": "Goa"}))
    assert out["flights"][0]["departure_time"] == "07:27"


def test_list_flight_routes():
    from_delhi = list_flight_routes("Delhi")
    assert [r["city"] for r in from_delhi["routes"]] == ["Kolkata"]
    assert from_delhi["routes"][0]["cheapest_price_inr"] == 3779
    to_goa = list_flight_routes("goa", "to")
    assert [r["city"] for r in to_goa["routes"]] == ["Bangalore", "Hyderabad", "Mumbai"]
    assert to_goa["direction"] == "to"
    assert "Did you mean 'Bangalore'" in list_flight_routes("Banglore")["error"]


def test_lookup_tools_extend_the_planning_tools():
    planning = {t.name for t in get_tools()}
    lookup = {t.name for t in get_lookup_tools()}
    assert planning == {
        "search_flights",
        "search_hotels",
        "search_places",
        "get_weather_forecast",
        "estimate_budget",
    }  # the verified planning toolset is unchanged
    assert lookup - planning == {"list_flight_routes", "hotel_amenities", "compare_hotels", "compare_flights"}
