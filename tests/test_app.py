"""Headless UI tests for the Streamlit app (the agent is replaced by a canned result)."""

from datetime import date
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import agent as agent_pkg
import config
from agent.agent import PlanResult
from agent.formatter import format_itinerary
from agent.planner import build_plan
from tests.test_agent import GOOD_DRAFT, REQUEST, fake_weather
from ui_helpers import build_preferences, results_to_tables, trip_end_date, weather_emoji

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def canned_ok(**kwargs):
    plan = build_plan(REQUEST, GOOD_DRAFT, weather_fn=fake_weather)
    return PlanResult(
        status="ok",
        request=REQUEST,
        plan=plan,
        text=format_itinerary(plan),
        json_text=plan.model_dump_json(indent=2),
        steps=[{"tool": "search_flights", "args": {"origin": "Bangalore"}}],
    )


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-key")
    calls = []

    def fake_plan_trip(**kwargs):
        calls.append(kwargs)
        return canned_ok()

    monkeypatch.setattr(agent_pkg, "plan_trip", fake_plan_trip)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.calls = calls
    return at


def test_initial_page_has_no_errors(app):
    assert not app.exception
    assert app.title[0].value.endswith("Agentic AI Travel Planner")
    assert app.button(key="plan_form").disabled is False


def test_form_submits_a_proper_request(app):
    app.selectbox(key="destination").select("Mumbai")
    app.slider(key="length").set_value(4)
    app.number_input(key="travelers").set_value(2)
    app.text_input(key="interests").set_value("beaches")
    app.radio(key="flight_pref").set_value("Fastest")
    app.button(key="plan_form").click().run()
    request = app.calls[0]["request"]
    assert (request.origin, request.destination) == ("Bangalore", "Mumbai")
    assert request.travelers == 2 and request.budget_limit_inr == 25000
    assert "beaches" in request.preferences and "fastest" in request.preferences
    start = date.fromisoformat(request.start_date)
    assert (date.fromisoformat(request.end_date) - start).days == 3


def test_results_are_rendered(app):
    app.button(key="plan_form").click().run()
    assert not app.exception
    assert any("Trip to Goa" in h.value for h in app.header)
    assert any(m.value.startswith("A 3-day trip from Bangalore to Goa") for m in app.markdown)  # trip summary
    labels = [m.label for m in app.metric]
    assert labels == ["Total cost", "Per person", "Nights", "Budget"]
    assert app.metric[0].value == "\u20b915,957"
    assert any("beach" in w.value for w in app.warning)  # unavailable preference is surfaced


def test_text_tab_sends_the_query(app):
    app.text_area(key="query").set_value("3 days in Goa from Bangalore")
    app.button(key="plan_text").click().run()
    call = dict(app.calls[0])
    assert callable(call.pop("on_progress"))  # live progress hook is always passed
    assert call == {"query": "3 days in Goa from Bangalore"}


def test_same_city_blocks_planning(app):
    app.selectbox(key="destination").select("Bangalore").run()
    assert app.button(key="plan_form").disabled is True
    assert any("different" in e.value for e in app.error)


def test_needs_info_and_error_messages(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-key")
    for status, bucket in (("needs_info", "warning"), ("error", "error")):
        monkeypatch.setattr(
            agent_pkg, "plan_trip", lambda status=status, **kw: PlanResult(status=status, message="Please add dates.")
        )
        at = AppTest.from_file(APP, default_timeout=30).run()
        at.button(key="plan_form").click().run()
        assert any("Please add dates." in e.value for e in getattr(at, bucket))


def test_missing_api_key_disables_buttons(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert at.button(key="plan_form").disabled and at.button(key="plan_text").disabled
    assert any("GROQ_API_KEY" in e.value for e in at.error)


# ----------------------------------------------------------------- helpers ----
def test_build_preferences():
    assert build_preferences() is None
    text = build_preferences("beaches", "Fastest", 4, ["pool", "wifi"])
    assert (
        text
        == "interests: beaches; prefer the fastest flights; hotel must be at least 4 stars; hotel must have: pool, wifi"
    )


def test_trip_end_date_and_weather_emoji():
    assert trip_end_date(date(2026, 12, 5), 3) == date(2026, 12, 7)
    assert weather_emoji("Light drizzle") == weather_emoji("Rain showers")
    assert weather_emoji("Clear sky") != weather_emoji("Overcast")
    assert weather_emoji(None)


def test_one_way_and_no_hotel_checkboxes_reach_the_request(app):
    app.checkbox(key="one_way").check()
    app.checkbox(key="no_hotel").check()
    app.button(key="plan_form").click().run()
    request = app.calls[0]["request"]
    assert request.one_way is True and request.needs_hotel is False


def test_defaults_are_round_trip_with_hotel(app):
    app.button(key="plan_form").click().run()
    request = app.calls[0]["request"]
    assert request.one_way is False and request.needs_hotel is True


# ------------------------------------------------------- quick-question mode ----
from tools.flight_tool import list_flight_routes, search_flights  # noqa: E402
from tools.hotel_tool import search_hotels  # noqa: E402
from tools.places_tool import search_places  # noqa: E402


def canned_answer(**kwargs):
    return PlanResult(
        status="ok",
        kind="answer",
        answer="Royal Heritage is the cheapest Goa hotel at \u20b91,232 a night.",
        tool_results=[
            {"tool": "search_hotels", "args": {"city": "Goa"}, "data": search_hotels("Goa", max_results=3)},
            {"tool": "search_flights", "args": {}, "data": search_flights("Delhi", "Goa")},
        ],
        steps=[{"tool": "search_hotels", "args": {"city": "Goa"}}],
        timings={"answer": 3.2},
    )


def test_answer_mode_renders_text_and_tables(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(agent_pkg, "plan_trip", canned_answer)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_area(key="query").set_value("cheapest hotels in Goa")
    at.button(key="plan_text").click().run()
    assert not at.exception
    assert any("Royal Heritage is the cheapest" in m.value for m in at.markdown)
    assert len(at.dataframe) == 1  # hotels table; the failed Delhi->Goa lookup shows a caption instead
    assert any("No flights from Delhi to Goa" in c.value for c in at.caption)
    assert not at.metric  # no trip-plan widgets in answer mode


def test_text_tab_has_no_example_buttons(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-key")
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert [b.key for b in at.button] == ["plan_form", "plan_text"]  # nothing but the two action buttons


def test_out_of_scope_reply_is_informational_not_a_warning(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(
        agent_pkg, "plan_trip", lambda **kw: PlanResult(status="needs_info", kind="other", message="I can plan trips.")
    )
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_area(key="query").set_value("capital of France")
    at.button(key="plan_text").click().run()
    assert any("I can plan trips." in i.value for i in at.info)
    assert not any("I can plan trips." in w.value for w in at.warning)


def test_results_to_tables_for_every_tool():
    flights = results_to_tables(
        [{"tool": "search_flights", "args": {}, "data": search_flights("Kolkata", "Jaipur", "2026-12-10", "fastest")}]
    )
    title, table, _ = flights[0]
    assert title == "Flights Kolkata \u2192 Jaipur on 2026-12-10"
    assert list(table.columns) == ["Airline", "Route", "Departs", "Arrives", "Duration", "Price"]
    assert any("(+1 day)" in arrives for arrives in table["Arrives"])  # overnight flight dated by ISO strings

    undated = results_to_tables([{"tool": "search_flights", "args": {}, "data": search_flights("Kolkata", "Jaipur")}])
    assert undated[0][0].endswith("(daily schedule)")
    assert any("(+1 day)" in arrives for arrives in undated[0][1]["Arrives"])  # and by the arrives_next_day flag

    hotels = results_to_tables([{"tool": "search_hotels", "args": {}, "data": search_hotels("Goa", nights=2)}])[0][1]
    assert "2 nights" in hotels.columns and hotels["Per night"][0] == "\u20b91,232"
    places = results_to_tables([{"tool": "search_places", "args": {}, "data": search_places("Goa")}])[0][1]
    assert list(places.columns) == ["Place", "Type", "Rating"]
    routes = results_to_tables([{"tool": "list_flight_routes", "args": {}, "data": list_flight_routes("Goa", "to")}])
    assert routes[0][0] == "Flight routes to Goa" and list(routes[0][1].columns)[0] == "From"
    weather = {
        "city": "Goa",
        "daily": [
            {
                "date": "2026-12-05",
                "available": True,
                "weather": "Clear sky",
                "temp_min_c": 22,
                "temp_max_c": 33,
                "rain_probability_percent": None,
                "source": "historical_estimate",
            },
            {"date": "2026-12-06", "available": False, "source": "forecast"},
        ],
    }
    wt = results_to_tables([{"tool": "get_weather_forecast", "args": {}, "data": weather}])[0][1]
    assert (
        wt["Source"][0] == "Typical (last year)" and wt["Rain chance"][0] == "-" and wt["Weather"][1] == "not available"
    )


def test_failed_lookups_and_untabled_tools_are_handled():
    tables = results_to_tables(
        [
            {"tool": "search_hotels", "args": {"city": "Atlantis"}, "data": {"error": "Unknown city 'Atlantis'."}},
            {"tool": "estimate_budget", "args": {}, "data": {"total": 1}},
            {"tool": "search_places", "args": {}, "data": {"city": "Goa", "places": []}},
        ]
    )
    assert len(tables) == 1 and tables[0][1] is None and "Unknown city" in tables[0][2]
