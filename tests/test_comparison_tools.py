"""Tests for the amenities and comparison tools, and their tables."""

import json
from dataclasses import replace
from datetime import datetime

import pytest

from data_loader import load_all
from tools.amenities_tool import hotel_amenities, hotel_amenities_tool
from tools.comparison_tool import compare_flights, compare_flights_tool, compare_hotels, compare_hotels_tool
from ui_helpers import results_to_tables


# ---------------------------------------------------------------- amenities ----
def test_amenities_overview_for_goa():
    result = hotel_amenities("Goa")
    rows = {r["amenity"]: r for r in result["amenities"]}
    assert result["hotels_in_city"] == 5
    assert rows["pool"]["hotels"] == 5 and rows["pool"]["cheapest_price_inr"] == 1232
    assert rows["gym"]["hotels"] == 3 and rows["gym"]["cheapest_hotel"] == "Budget Stay Inn"
    assert rows["parking"]["hotels"] == 2
    assert set(rows) == {"breakfast", "gym", "parking", "pool", "spa", "wifi"}


def test_hotels_with_one_amenity_are_listed_cheapest_first():
    result = hotel_amenities("goa", "Gym")  # city alias and case are normalised
    assert result["count"] == 3
    assert [h["hotel_id"] for h in result["hotels"]] == ["HOT0015", "HOT0012", "HOT0011"]
    assert all("gym" in h["amenities"] for h in result["hotels"])


def test_amenity_errors_are_helpful():
    unknown = hotel_amenities("Goa", "helipad")
    assert "error" in unknown and "pool" in unknown["available_amenities"]
    assert "Did you mean 'Bangalore'" in hotel_amenities("Banglore")["error"]


def test_an_amenity_missing_from_a_city_lists_what_it_does_offer():
    # In the real data every city has every amenity, so remove Delhi's spa hotel from a copy of the data.
    store = load_all()
    trimmed = replace(store, hotels=[h for h in store.hotels if not (h.city == "Delhi" and "spa" in h.amenities)])
    result = hotel_amenities("Delhi", "spa", store=trimmed)
    assert "No hotel in Delhi offers spa" in result["error"]
    assert "pool" in result["offered_in_city"] and "spa" not in result["offered_in_city"]


# ------------------------------------------------------------ hotel compare ----
def test_compare_two_named_hotels():
    result = compare_hotels("Goa", ["HOT0011", "HOT0015"], nights=2)
    assert [h["hotel_id"] for h in result["hotels"]] == ["HOT0011", "HOT0015"]
    assert result["highlights"] == {"cheapest": "HOT0015", "best_value": "HOT0015", "most_stars": "HOT0011"}
    assert result["price_gap_inr"] == 580
    assert result["hotels"][0]["total_price_inr"] == 2828 * 2
    assert result["amenity_matrix"]["gym"] == ["HOT0011", "HOT0015"]
    assert "Price gap between cheapest and priciest: \u20b9580 per night." in result["summary"]


def test_compare_without_ids_picks_the_best_value_hotels():
    result = compare_hotels("Goa", max_hotels=3)
    assert [h["hotel_id"] for h in result["hotels"]] == ["HOT0014", "HOT0015", "HOT0011"]  # by value score
    assert result["highlights"]["best_value"] == "HOT0014"


def test_compare_hotel_errors():
    assert "not in Goa" in compare_hotels("Goa", ["HOT0011", "HOT0001"])["error"]  # HOT0001 is a Delhi hotel
    assert "At least two" in compare_hotels("Goa", ["HOT0011"])["error"]
    assert "error" in compare_hotels("Atlantis")


# ----------------------------------------------------------- flight compare ----
def test_compare_flights_states_the_tradeoff():
    result = compare_flights("Bangalore", "Goa")
    assert result["highlights"] == {"cheapest": "FL0021", "fastest": "FL0020"}
    assert "The fastest flight costs \u20b91,268 more and saves 1h 00m." in result["summary"]
    assert [f["price_per_hour_inr"] for f in result["flights"]] == [2678, 6624]
    assert result["travel_date"] is None


def test_compare_flights_edge_cases():
    single = compare_flights("Hyderabad", "Goa")
    assert "Only one flight operates" in single["summary"]
    dated = compare_flights("Bangalore", "Goa", "2026-12-05")
    assert dated["flights"][0]["departure_time"].startswith("2026-12-05")
    no_route = compare_flights("Delhi", "Goa")
    assert "error" in no_route and no_route["destinations_from_origin"] == ["Kolkata"]
    gone = compare_flights("Bangalore", "Goa", "2026-12-05", now=datetime(2026, 12, 5, 12, 0))
    assert "left today" in gone["error"]


# ------------------------------------------------------------ tool wrappers ----
def test_wrappers_return_compact_json():
    out = json.loads(hotel_amenities_tool.invoke({"city": "Goa", "amenity": "pool"}))
    assert out["count"] == 5
    out = json.loads(compare_hotels_tool.invoke({"city": "Goa", "hotel_ids": ["HOT0011", "HOT0014"]}))
    assert out["highlights"]["cheapest"] == "HOT0014"
    out = json.loads(compare_flights_tool.invoke({"origin": "Bangalore", "destination": "Goa"}))
    assert out["highlights"]["fastest"] == "FL0020"


@pytest.mark.parametrize("bad_max", [1, 7])
def test_compare_hotels_tool_validates_the_hotel_count(bad_max):
    with pytest.raises(Exception, match="max_hotels"):
        compare_hotels_tool.invoke({"city": "Goa", "max_hotels": bad_max})


# ------------------------------------------------------------------- tables ----
def test_comparison_and_amenity_tables():
    def table(tool, data):
        return results_to_tables([{"tool": tool, "args": {}, "data": data}])[0]

    title, hotels, note = table("compare_hotels", compare_hotels("Goa", ["HOT0011", "HOT0015"]))
    assert title == "Hotel comparison in Goa" and "Value score" in hotels.columns
    assert note.startswith("Cheapest: Budget Stay Inn") and "Value score = stars per Rs 1000" in note

    title, flights, note = table("compare_flights", compare_flights("Bangalore", "Goa"))
    assert title == "Flight comparison Bangalore \u2192 Goa (daily schedule)" and "Per hour" in flights.columns
    assert "saves 1h 00m" in note

    title, overview, _ = table("hotel_amenities", hotel_amenities("Goa"))
    assert title == "Amenities in Goa" and list(overview.columns) == ["Amenity", "Hotels", "Cheapest"]
    assert overview["Cheapest"][3] == "Royal Heritage (\u20b91,232/night)"  # pool

    title, filtered, _ = table("hotel_amenities", hotel_amenities("Goa", "gym"))
    assert title == "Hotels with gym in Goa" and len(filtered) == 3 and "Value score" not in filtered.columns


def test_failed_comparison_shows_the_reason():
    tables = results_to_tables([{"tool": "compare_hotels", "args": {}, "data": compare_hotels("Goa", ["HOT0011"])}])
    assert tables[0][1] is None and "At least two" in tables[0][2]
