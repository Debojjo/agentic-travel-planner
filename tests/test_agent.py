"""Tests for the agent pipeline using a scripted fake LLM (no network, no Groq)."""

import json
from datetime import date
from typing import Any

import groq
import httpx
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

import agent.agent as agent_module
from agent import plan_trip
from agent.formatter import format_inr, format_itinerary
from agent.planner import build_plan, normalize_request
from agent.schemas import DraftDay, PlanDraft, TripRequest

TODAY = date(2026, 11, 10)


class ScriptedChatModel(BaseChatModel):
    """Returns pre-written AI messages and pre-built structured outputs."""

    responses: list[Any] = []
    structured: dict[Any, Any] = {}
    cursor: int = 0
    fail_first: int = 0  # raise `error` on the first N model calls
    error: Any = None
    attempts: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.attempts += 1
        if self.attempts <= self.fail_first:
            raise self.error
        message = self.responses[self.cursor]
        self.cursor += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(lambda _messages: self.structured[schema])


def fake_weather(city, start, end):
    days = [
        {
            "day_number": i + 1,
            "date": f"2026-12-0{5 + i}",
            "available": True,
            "weather": "Partly cloudy",
            "temp_max_c": 31.0,
            "temp_min_c": 23.0,
            "rain_probability_percent": 10,
            "precipitation_mm": 0.0,
            "rain_expected": False,
            "source": "forecast",
        }
        for i in range(3)
    ]
    return {"city": city, "daily": days}


REQUEST = TripRequest(
    origin="Bangalore",
    destination="Goa",
    start_date="2026-12-05",
    end_date="2026-12-07",
    budget_limit_inr=25000,
    preferences="beaches",
)

GOOD_DRAFT = PlanDraft(
    outbound_flight_id="FL0021",
    return_flight_id="FL0023",
    hotel_id="HOT0014",
    days=[
        DraftDay(day_number=1, place_ids=["PLC0012"]),
        DraftDay(day_number=2, place_ids=["PLC0014", "PLC0011"]),
        DraftDay(day_number=3, place_ids=[]),
    ],
    flight_reason="Cheapest options on both legs.",
    hotel_reason="Best value: 5-star at Rs 1232.",
    itinerary_reason="Highest rated places first.",
)


# ------------------------------------------------------------ validation ----
def test_normalize_accepts_valid_request():
    clean, problems = normalize_request(
        REQUEST.model_copy(update={"origin": "bengaluru", "destination": "goa"}), today=TODAY
    )
    assert problems == [] and clean.origin == "Bangalore" and clean.destination == "Goa"


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({"destination": None}, "I still need"),
        ({"destination": "Paris"}, "Unknown city"),
        ({"destination": "Bangalore"}, "different"),
        ({"start_date": "2026-11-01", "end_date": "2026-11-03"}, "past"),
        ({"end_date": "2026-12-05"}, "after"),
        ({"end_date": "2026-12-20"}, "limited"),
        ({"start_date": "05/12/2026"}, "Invalid date"),
    ],
)
def test_normalize_rejects_bad_requests(changes, expected):
    clean, problems = normalize_request(REQUEST.model_copy(update=changes), today=TODAY)
    assert clean is None and any(expected in p for p in problems)


# -------------------------------------------------------------- grounding ----
def test_build_plan_is_grounded_and_budget_recomputed():
    plan = build_plan(REQUEST, GOOD_DRAFT, weather_fn=fake_weather)
    assert plan.title == "Your 3-Day Trip to Goa (Dec 5\u20137)"
    assert plan.outbound_flight.price_inr == 5356
    assert plan.outbound_flight.departure_time == "2026-12-05T07:27"
    assert plan.return_flight.departure_time.startswith("2026-12-07")
    assert plan.hotel.total_price_inr == 2464
    assert [p.name for p in plan.days[1].places] == ["Beautiful Park", "Popular Lake"]
    assert plan.days[0].weather == "Partly cloudy"
    # 5356 + 4537 flights, 2 nights x 1232, 3 days x 1200 local
    assert plan.budget.total == 15957 and plan.budget.within_budget is True and plan.budget.remaining == 9043
    assert len(plan.warnings) == 1 and "beach" in plan.warnings[0]  # Goa has no beach places


def test_build_plan_removes_invented_or_wrong_items():
    bad = PlanDraft(
        outbound_flight_id="FL9999",
        return_flight_id="FL0016",  # Hyderabad -> Goa, wrong direction for the return leg
        hotel_id="HOT0001",  # a Delhi hotel
        days=[DraftDay(day_number=1, place_ids=["PLC0001", "PLC0012"])],
    )
    plan = build_plan(REQUEST, bad, weather_fn=fake_weather)
    assert plan.outbound_flight is None and plan.return_flight is None and plan.hotel is None
    assert [p.place_id for p in plan.days[0].places] == ["PLC0012"]
    joined = " ".join(plan.warnings)
    assert "were removed: outbound flight, return flight, hotel, 1 place." in joined
    assert not any(raw in joined for raw in ("FL9999", "FL0016", "HOT0001", "PLC0001"))  # no raw IDs shown to users
    assert plan.budget.flights == 0 and plan.budget.hotel == 0


def test_weather_failure_becomes_warning():
    plan = build_plan(REQUEST, GOOD_DRAFT, weather_fn=lambda *a: {"error": "network down"})
    assert all(d.weather is None for d in plan.days)
    assert any("Weather unavailable" in w for w in plan.warnings)


# -------------------------------------------------------------- formatter ----
def test_format_inr_indian_grouping():
    assert format_inr(13700) == "\u20b913,700"
    assert format_inr(1370000) == "\u20b913,70,000"
    assert format_inr(999) == "\u20b9999"


def test_formatted_itinerary_has_brief_sections():
    text = format_itinerary(build_plan(REQUEST, GOOD_DRAFT, weather_fn=fake_weather))
    for section in ("Flights", "Hotel", "Weather", "Itinerary", "Estimated Budget", "Why we chose this"):
        assert section in text
    assert "Total Cost: \u20b915,957" in text and "Royal Heritage" in text


# --------------------------------------------------------- full pipeline ----
def tool_call(name, args, call_id):
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def make_llm(request=REQUEST, draft=GOOD_DRAFT, responses=None):
    if responses is None:
        responses = [
            AIMessage(
                content="",
                tool_calls=[
                    tool_call(
                        "search_flights",
                        {"origin": "Bangalore", "destination": "Goa", "travel_date": "2026-12-05"},
                        "c1",
                    ),
                    tool_call(
                        "search_flights",
                        {"origin": "Goa", "destination": "Bangalore", "travel_date": "2026-12-07"},
                        "c2",
                    ),
                    tool_call("search_hotels", {"city": "Goa", "nights": 2}, "c3"),
                ],
            ),
            AIMessage(content="Chose FL0021, FL0023 and HOT0014."),
        ]
    return ScriptedChatModel(responses=responses, structured={TripRequest: request, PlanDraft: draft})


def test_full_pipeline_with_form_request():
    llm = make_llm()
    result = plan_trip(request=REQUEST, llm=llm, today=TODAY, weather_fn=fake_weather)
    assert result.status == "ok", result.message
    assert [s["tool"] for s in result.steps] == ["search_flights", "search_flights", "search_hotels"]
    assert result.plan.budget.total == 15957
    assert '"title"' in result.json_text and "Total Cost" in result.text


def test_full_pipeline_from_free_text():
    result = plan_trip(query="3 days Goa from Bangalore", llm=make_llm(), today=TODAY, weather_fn=fake_weather)
    assert result.status == "ok" and result.plan.destination == "Goa"


def test_missing_details_asks_instead_of_planning():
    incomplete = TripRequest(origin="Bangalore")
    llm = make_llm(request=incomplete, responses=[])  # any agent call would raise IndexError
    result = plan_trip(query="I want to travel", llm=llm, today=TODAY)
    assert result.status == "needs_info" and "where you want to go" in result.message


def test_llm_failure_returns_clean_error():
    broken = make_llm(responses=[])  # first agent call fails
    result = plan_trip(request=REQUEST, llm=broken, today=TODAY, weather_fn=fake_weather)
    assert result.status == "error" and result.message


# ------------------------------------------------------------- retries ----
def groq_error(cls, message, status):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return cls(message, response=httpx.Response(status, request=request), body=None)


MALFORMED = (
    "Error code: 400 - {'error': {'message': 'Failed to parse tool call arguments as JSON', 'code': 'tool_use_failed'}}"
)


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    monkeypatch.setattr(agent_module, "RETRY_INITIAL_DELAY", 0.0)


def test_should_retry_rules():
    assert agent_module.should_retry(groq_error(groq.BadRequestError, MALFORMED, 400))
    assert not agent_module.should_retry(groq_error(groq.BadRequestError, "model_not_found", 400))
    assert agent_module.should_retry(groq_error(groq.RateLimitError, "slow down", 429))
    assert agent_module.should_retry(groq_error(groq.InternalServerError, "oops", 500))
    assert not agent_module.should_retry(groq_error(groq.AuthenticationError, "bad key", 401))
    assert not agent_module.should_retry(ValueError("nope"))


def test_malformed_tool_call_is_retried_and_recovers():
    llm = make_llm()
    llm.fail_first, llm.error = 2, groq_error(groq.BadRequestError, MALFORMED, 400)
    result = plan_trip(request=REQUEST, llm=llm, today=TODAY, weather_fn=fake_weather)
    assert result.status == "ok", result.message
    assert llm.attempts >= 4


def test_persistent_malformed_tool_calls_give_clean_error():
    llm = make_llm()
    llm.fail_first, llm.error = 99, groq_error(groq.BadRequestError, MALFORMED, 400)
    result = plan_trip(request=REQUEST, llm=llm, today=TODAY, weather_fn=fake_weather)
    assert result.status == "error" and "malformed tool calls" in result.message
    assert llm.attempts == agent_module.MODEL_RETRIES + 1


def test_non_retryable_error_fails_fast():
    llm = make_llm()
    llm.fail_first, llm.error = 99, groq_error(groq.AuthenticationError, "Invalid API Key", 401)
    result = plan_trip(request=REQUEST, llm=llm, today=TODAY, weather_fn=fake_weather)
    assert result.status == "error" and llm.attempts == 1


def test_unavailable_place_type_detection():
    from agent.planner import unavailable_place_types
    from data_loader import load_all

    store = load_all()
    goa = REQUEST.model_copy(update={"destination": "Goa", "preferences": "Beaches and parking please"})
    assert unavailable_place_types(goa, store) == ["beach"]  # 'parking' must not match 'park'
    mumbai = goa.model_copy(update={"destination": "Mumbai"})
    assert unavailable_place_types(mumbai, store) == []
    assert unavailable_place_types(goa.model_copy(update={"preferences": None}), store) == []


# ------------------------------------------- warnings, weather reuse, progress ----
from langchain_core.messages import ToolMessage  # noqa: E402


def test_missing_flights_and_hotel_give_one_clear_warning_each():
    nothing = PlanDraft(days=[DraftDay(day_number=1, place_ids=["PLC0012"])])
    plan = build_plan(REQUEST, nothing, weather_fn=fake_weather)
    flight_notes = [w for w in plan.warnings if "flight" in w.lower()]
    hotel_notes = [w for w in plan.warnings if "hotel" in w.lower()]
    assert len(flight_notes) == 1 and "not included in the budget" in flight_notes[0]
    assert len(hotel_notes) == 1 and "not included in the budget" in hotel_notes[0]
    assert plan.budget.flights == 0 and plan.budget.hotel == 0


def test_one_missing_leg_is_named():
    one_way = GOOD_DRAFT.model_copy(update={"return_flight_id": None})
    plan = build_plan(REQUEST, one_way, weather_fn=fake_weather)
    notes = [w for w in plan.warnings if "return flight" in w]
    assert len(notes) == 1 and "Goa to Bangalore" in notes[0]


def test_over_budget_warning():
    tight = REQUEST.model_copy(update={"budget_limit_inr": 5000})
    plan = build_plan(tight, GOOD_DRAFT, weather_fn=fake_weather)
    assert plan.budget.within_budget is False
    assert any("over your \u20b95,000 budget" in w for w in plan.warnings)


def test_prefetched_weather_skips_the_second_api_call():
    def must_not_run(*args):
        raise AssertionError("weather_fn should not be called")

    prefetched = fake_weather("Goa", "2026-12-05", "2026-12-07")
    plan = build_plan(REQUEST, GOOD_DRAFT, weather_fn=must_not_run, weather=prefetched)
    assert plan.days[0].weather == "Partly cloudy"


def test_extract_weather_reuses_only_matching_results():
    good = {"city": "Goa", "start_date": "2026-12-05", "end_date": "2026-12-07", "daily": []}
    wrong_dates = {**good, "end_date": "2026-12-09"}
    failed = {"error": "network down"}

    def msg(payload):
        return ToolMessage(content=json.dumps(payload), name="get_weather_forecast", tool_call_id="x")

    assert agent_module.extract_weather([msg(good)], REQUEST) == good
    assert agent_module.extract_weather([msg(wrong_dates)], REQUEST) is None
    assert agent_module.extract_weather([msg(failed)], REQUEST) is None
    assert agent_module.extract_weather([msg(failed), msg(good)], REQUEST) == good
    assert agent_module.extract_weather([], REQUEST) is None


def test_progress_messages_and_timings():
    seen = []
    result = plan_trip(
        query="3 days Goa from Bangalore",
        llm=make_llm(),
        today=TODAY,
        weather_fn=fake_weather,
        on_progress=seen.append,
    )
    assert result.status == "ok"
    assert seen[0] == "Understanding your request"
    assert "Searching flights Bangalore \u2192 Goa" in seen and "Finding hotels in Goa" in seen
    assert seen[-2:] == ["Choosing the best options", "Building your itinerary"]
    assert set(result.timings) == {"understand_request", "research", "choose", "build"}


def test_failing_progress_callback_does_not_break_planning():
    def explode(message):
        raise RuntimeError("UI went away")

    result = plan_trip(request=REQUEST, llm=make_llm(), today=TODAY, weather_fn=fake_weather, on_progress=explode)
    assert result.status == "ok"


def test_describe_call_labels():
    d = agent_module.describe_call
    assert d({"name": "search_hotels", "args": {"city": "Goa"}}) == "Finding hotels in Goa"
    assert d({"name": "get_weather_forecast", "args": {"city": "Goa"}}) == "Checking the weather in Goa"
    assert d({"name": "estimate_budget", "args": {}}) == "Calculating the budget"
    assert d({"name": "mystery", "args": None}) == "Running mystery"


def test_reasoning_effort_only_for_gpt_oss(monkeypatch):
    monkeypatch.setattr(agent_module, "GROQ_API_KEY", "key")
    monkeypatch.setattr(agent_module, "GROQ_REASONING_EFFORT", "low")
    monkeypatch.setattr(agent_module, "GROQ_MODEL", "openai/gpt-oss-120b")
    assert agent_module.build_llm().reasoning_effort == "low"
    monkeypatch.setattr(agent_module, "GROQ_MODEL", "some/other-model")
    assert agent_module.build_llm().reasoning_effort is None


def test_tool_output_is_compact_json():
    from tools.hotel_tool import hotel_search_tool

    out = hotel_search_tool.invoke({"city": "Goa"})
    assert "\n" not in out and json.loads(out)["hotels"]


# ------------------------------------- fixes from live runs (rooms, dupes, places) ----
def test_hotel_total_and_budget_agree_for_a_group():
    group = REQUEST.model_copy(update={"travelers": 5, "budget_limit_inr": 30000})
    plan = build_plan(group, GOOD_DRAFT, weather_fn=fake_weather)
    assert plan.hotel.rooms == 3
    assert plan.hotel.total_price_inr == 1232 * 2 * 3
    assert plan.budget.hotel == plan.hotel.total_price_inr  # card and budget must match
    assert any("3 hotel rooms for 5 travelers" in a for a in plan.budget.assumptions)
    assert "3 rooms" in format_itinerary(plan)


def test_single_traveler_has_one_room_and_no_room_note():
    plan = build_plan(REQUEST, GOOD_DRAFT, weather_fn=fake_weather)
    assert plan.hotel.rooms == 1 and not any("two per room" in a for a in plan.budget.assumptions)


def test_llm_warnings_repeating_system_notes_are_dropped():
    noisy = PlanDraft(
        days=[DraftDay(day_number=1, place_ids=["PLC0012"])],
        warnings=[
            "Both outbound and return flights are missing; you will need to use train or road transport.",
            "Estimated total cost exceeds budget by 5088.",
            "Hotel criteria too restrictive.",
            "Day 2 is a rest day because the museum is closed on Mondays.",
        ],
    )
    tight = REQUEST.model_copy(update={"budget_limit_inr": 3000})
    plan = build_plan(tight, noisy, weather_fn=fake_weather)
    joined = " | ".join(plan.warnings)
    assert "train or road transport" not in joined and "exceeds budget by 5088" not in joined
    assert "too restrictive" not in joined
    assert "museum is closed on Mondays" in joined  # a genuinely new note survives
    assert sum("flight" in w.lower() for w in plan.warnings) == 1
    assert sum("over your" in w for w in plan.warnings) == 1


def test_llm_budget_note_is_kept_when_system_reports_no_overrun():
    draft = GOOD_DRAFT.model_copy(update={"warnings": ["Budget leaves room for an extra activity."]})
    plan = build_plan(REQUEST, draft, weather_fn=fake_weather)
    assert "Budget leaves room for an extra activity." in plan.warnings


def test_strict_hotel_filters_get_a_hint():
    strict = REQUEST.model_copy(
        update={"preferences": "hotel must be at least 3 stars; hotel must have: wifi, parking, gym"}
    )
    plan = build_plan(strict, PlanDraft(days=[DraftDay(day_number=1, place_ids=["PLC0012"])]), weather_fn=fake_weather)
    note = next(w for w in plan.warnings if "No hotel was selected" in w)
    assert "filters may be too strict" in note


def test_places_are_filled_in_when_the_agent_chose_none():
    plan = build_plan(REQUEST, PlanDraft(outbound_flight_id="FL0021", hotel_id="HOT0014"), weather_fn=fake_weather)
    assert [p.place_id for p in plan.days[0].places] == ["PLC0012", "PLC0014"]  # highest rated first
    assert [p.place_id for p in plan.days[1].places] == ["PLC0011", "PLC0013"]
    assert plan.days[2].places == []  # departure day stays free on a 3-day trip
    assert any("added automatically" in w for w in plan.warnings)


def test_places_are_not_filled_in_when_the_agent_chose_some():
    plan = build_plan(REQUEST, GOOD_DRAFT, weather_fn=fake_weather)
    assert not any("added automatically" in w for w in plan.warnings)
    assert plan.days[2].places == []  # intentionally free day is respected


def test_structured_output_attempts_are_logged_with_timings():
    result = plan_trip(query="3 days Goa from Bangalore", llm=make_llm(), today=TODAY, weather_fn=fake_weather)
    assert any(line.startswith("TripRequest via function_calling, attempt 1: ok (") for line in result.structured_log)
    assert any(line.startswith("PlanDraft via function_calling, attempt 1: ok (") for line in result.structured_log)


def test_failed_structured_attempts_are_logged():
    class FlakyModel(ScriptedChatModel):
        def with_structured_output(self, schema, **kwargs):
            def run(_messages):
                self.attempts += 1
                if self.attempts == 1:
                    raise ValueError("bad json from model")
                return self.structured[schema]

            return RunnableLambda(run)

    log = []
    llm = FlakyModel(structured={PlanDraft: GOOD_DRAFT})
    assert agent_module.draft_plan(llm, [], log) == GOOD_DRAFT
    assert "FAILED" in log[0] and "bad json from model" in log[0]
    assert log[1].startswith("PlanDraft via function_calling, attempt 2: ok")


# ----------------------------------------------- fixes from the 8 live scenario runs ----
from datetime import datetime  # noqa: E402

from agent.planner import describe_request, unsupported_interests  # noqa: E402
from tools.flight_tool import search_flights  # noqa: E402


def test_hallucinated_places_are_removed_quietly_and_notes_stay_factual():
    invented = PlanDraft(
        outbound_flight_id="FL0021",
        return_flight_id="FL0023",
        hotel_id="HOT0014",
        days=[
            DraftDay(day_number=1, place_ids=["PLACE_MANGUESH_TEMPLE", "PLACE_MAHDEV_TEMPLE"]),
            DraftDay(day_number=2, place_ids=["PLACE_BOM_JESUS"]),
        ],
        itinerary_reason="Selected temples (Manguesh, Mahadev) plus top attractions.",
    )
    plan = build_plan(REQUEST, invented, weather_fn=fake_weather)
    joined = " | ".join(plan.warnings)
    assert "PLACE_" not in joined and "removed: 3 places." in joined
    assert sum("removed" in w for w in plan.warnings) == 1  # one summary, not one warning per place
    assert "Manguesh" not in plan.reasoning["itinerary"]  # stale LLM prose is replaced
    assert all("Manguesh" not in d.note for d in plan.days)
    assert plan.days[0].places  # and the autofill still gave the trip real places


def test_day_notes_are_built_from_flight_times():
    plan = build_plan(REQUEST, GOOD_DRAFT, weather_fn=fake_weather)
    assert plan.days[0].note == "Arrive 09:27"
    assert plan.days[1].note == ""
    assert plan.days[2].note == "Flight home departs 17:10"
    overnight = GOOD_DRAFT.model_copy(update={"return_flight_id": "FL0006"})  # 22:52 -> 02:52 next day
    assert build_plan(REQUEST, overnight, weather_fn=fake_weather).days[2].note == "Flight home departs 22:52"


def test_llm_text_is_tidied():
    messy = GOOD_DRAFT.model_copy(
        update={
            "hotel_reason": "3\u2011star, price \u20b92\u202f554/night (total \u20b95\u202f108), best value_score 1.17",
            "warnings": ["Prices shown include \u20b91\u00a0200 per day extras."],
        }
    )
    plan = build_plan(REQUEST, messy, weather_fn=fake_weather)
    assert plan.reasoning["hotel"] == "3-star, price \u20b92,554/night (total \u20b95,108), best value score 1.17"
    assert "\u20b91,200 per day" in " ".join(plan.warnings)


def test_one_way_trip_has_no_return_leg_and_no_flight_warning():
    one_way = REQUEST.model_copy(update={"one_way": True})
    plan = build_plan(one_way, GOOD_DRAFT, weather_fn=fake_weather)  # draft wrongly includes a return flight
    assert plan.one_way and plan.return_flight is None
    assert plan.budget.flights == 5356
    assert not any("flight" in w.lower() for w in plan.warnings)
    assert plan.days[-1].note == ""
    assert "Return: not needed (one-way trip)" in format_itinerary(plan)


def test_staying_with_friends_means_no_hotel_and_no_hotel_warning():
    friends = REQUEST.model_copy(update={"needs_hotel": False, "travelers": 4})
    plan = build_plan(friends, GOOD_DRAFT, weather_fn=fake_weather)  # draft wrongly includes a hotel
    assert plan.hotel is None and plan.needs_hotel is False
    assert plan.budget.hotel == 0
    assert not any("hotel" in w.lower() for w in plan.warnings)
    assert not any("two per room" in a for a in plan.budget.assumptions)
    assert "Not needed (staying with friends or family)" in format_itinerary(plan)


def test_missing_flight_wording_is_truthful():
    # Route exists in the dataset but the agent chose nothing: must NOT claim it is unavailable.
    chose_nothing = PlanDraft(days=[DraftDay(day_number=1, place_ids=["PLC0012"])])
    note = next(w for w in build_plan(REQUEST, chose_nothing, weather_fn=fake_weather).warnings if "Flights" in w)
    assert "not selected" in note and "no such route" not in note and "Arrange" not in note
    # Route genuinely absent: say so and suggest arranging transport.
    delhi = REQUEST.model_copy(update={"origin": "Delhi"})
    note = next(w for w in build_plan(delhi, chose_nothing, weather_fn=fake_weather).warnings if "Flights" in w)
    assert "no such route in the dataset" in note and "Arrange that transport separately" in note


def test_request_limits_have_friendly_messages():
    base = REQUEST.model_copy(update={"origin": "Bangalore"})
    _, problems = normalize_request(base.model_copy(update={"travelers": 12}), today=TODAY)
    assert problems == ["Trips are limited to 10 travelers."]
    _, problems = normalize_request(base.model_copy(update={"travelers": 0}), today=TODAY)
    assert problems == ["At least 1 traveler is needed."]
    _, problems = normalize_request(base.model_copy(update={"budget_limit_inr": -5}), today=TODAY)
    assert problems == ["The budget cannot be negative."]
    clean, _ = normalize_request(base.model_copy(update={"budget_limit_inr": 0}), today=TODAY)
    assert clean.budget_limit_inr is None  # 0 means no limit, not "already over budget"


def test_missing_details_message_is_plain_language():
    _, problems = normalize_request(TripRequest(destination="Goa"), today=TODAY)
    assert "start_date" not in problems[0] and "where you are starting from" in problems[0]
    assert "travel dates" in problems[0]


def test_typo_gets_a_did_you_mean():
    _, problems = normalize_request(REQUEST.model_copy(update={"origin": "Banglore"}), today=TODAY)
    assert "Did you mean 'Bangalore'?" in problems[0]
    _, problems = normalize_request(REQUEST.model_copy(update={"destination": "Pune"}), today=TODAY)
    assert "Did you mean" not in problems[0] and "Supported cities" in problems[0]


def test_unsupported_interests_are_flagged():
    assert unsupported_interests("interests: trekking, nightlife, seafood") == [
        "trekking or hiking",
        "nightlife",
        "food and dining",
    ]
    assert unsupported_interests("hotel must have: parking, gym") == []  # 'parking' is not a 'park'/'bar'
    assert unsupported_interests("beaches and forts") == []
    plan = build_plan(REQUEST.model_copy(update={"preferences": "trekking"}), GOOD_DRAFT, weather_fn=fake_weather)
    assert any("no trekking or hiking options" in w for w in plan.warnings)


def test_task_message_describes_one_way_and_no_hotel():
    text = describe_request(REQUEST.model_copy(update={"one_way": True, "needs_hotel": False}))
    assert "ONE-WAY" in text and "NOT NEEDED" in text and "rooms:" not in text
    assert "round trip" in describe_request(REQUEST) and "search hotels" in describe_request(REQUEST)


def test_flights_that_already_left_today_are_excluded():
    # FL0020 departs 07:24 and FL0021 07:27 on Bangalore -> Goa
    before = search_flights("Bangalore", "Goa", "2026-12-05", now=datetime(2026, 12, 5, 7, 0))
    assert before["total_found"] == 2
    partly = search_flights("Bangalore", "Goa", "2026-12-05", now=datetime(2026, 12, 5, 7, 25))
    assert [f["flight_id"] for f in partly["flights"]] == ["FL0021"]
    gone = search_flights("Bangalore", "Goa", "2026-12-05", now=datetime(2026, 12, 5, 12, 23))
    assert "error" in gone and "left today" in gone["error"]
    future = search_flights("Bangalore", "Goa", "2026-12-06", now=datetime(2026, 12, 5, 23, 59))
    assert future["total_found"] == 2  # a later date is never filtered


# ------------------------------------------------ quick questions (lookup intent) ----
from langchain_core.messages import AIMessage as AiMessage  # noqa: E402


def lookup_llm(intent="lookup"):
    """Scripted model: classifies the message, calls a tool, then answers."""
    responses = [
        AiMessage(
            content="",
            tool_calls=[
                tool_call("search_hotels", {"city": "Goa", "max_price_per_night": 2500, "sort_by": "cheapest"}, "h1")
            ],
        ),
        AiMessage(content="The cheapest Goa hotel is Royal Heritage at \u20b91,232 per night."),
    ]
    return ScriptedChatModel(responses=responses, structured={TripRequest: TripRequest(intent=intent)})


def test_a_question_is_answered_without_planning_a_trip():
    seen = []
    result = plan_trip(
        query="cheapest hotels in Goa under 2500", llm=lookup_llm(), today=TODAY, on_progress=seen.append
    )
    assert result.status == "ok" and result.kind == "answer" and result.plan is None
    assert "Royal Heritage" in result.answer and result.text == result.answer
    assert [s["tool"] for s in result.steps] == ["search_hotels"]
    assert result.tool_results[0]["tool"] == "search_hotels"
    assert result.tool_results[0]["args"]["city"] == "Goa"
    assert result.tool_results[0]["data"]["hotels"][0]["hotel_id"] == "HOT0014"
    assert seen == ["Understanding your request", "Finding hotels in Goa"]
    assert set(result.timings) == {"understand_request", "answer"}


def test_unrelated_messages_get_a_capabilities_reply():
    result = plan_trip(query="what is the capital of France", llm=lookup_llm(intent="other"), today=TODAY)
    assert result.status == "needs_info" and result.kind == "other"
    assert "Goa" in result.message and "Plan a 3-day trip" in result.message


def test_a_form_request_always_plans_even_if_intent_says_lookup():
    asks = REQUEST.model_copy(update={"intent": "lookup"})
    result = plan_trip(request=asks, llm=make_llm(), today=TODAY, weather_fn=fake_weather)
    assert result.status == "ok" and result.kind == "plan"


def test_extract_tool_results_pairs_args_with_outputs():
    messages = [
        AiMessage(content="", tool_calls=[tool_call("search_places", {"city": "Jaipur"}, "p1")]),
        ToolMessage(content=json.dumps({"city": "Jaipur", "places": []}), name="search_places", tool_call_id="p1"),
        ToolMessage(content="not json", name="search_places", tool_call_id="p2"),
    ]
    results = agent_module.extract_tool_results(messages)
    assert results == [{"tool": "search_places", "args": {"city": "Jaipur"}, "data": {"city": "Jaipur", "places": []}}]


def test_final_answer_handles_plain_and_block_content():
    plain = [
        AiMessage(content="", tool_calls=[tool_call("search_places", {}, "x")]),
        AiMessage(content="Done \u20b91\u202f200."),
    ]
    assert agent_module.final_answer(plain) == "Done \u20b91,200."  # tidied like other LLM text
    blocks = [AiMessage(content=[{"type": "text", "text": "Hello"}, {"type": "text", "text": "there"}])]
    assert agent_module.final_answer(blocks) == "Hello there"
    assert agent_module.final_answer([]) == ""


def test_empty_answer_is_a_clean_error():
    llm = ScriptedChatModel(responses=[AiMessage(content="")], structured={TripRequest: TripRequest(intent="lookup")})
    result = plan_trip(query="hotels", llm=llm, today=TODAY)
    assert result.status == "error" and "rephrasing" in result.message


# ------------------------------------------------ quick answers are checked against the dataset ----
def places_lookup_llm(final_text):
    responses = [
        AiMessage(content="", tool_calls=[tool_call("search_places", {"city": "Kolkata", "max_results": 10}, "p1")]),
        AiMessage(content=final_text),
    ]
    return ScriptedChatModel(responses=responses, structured={TripRequest: TripRequest(intent="lookup")})


def test_a_four_of_five_place_list_from_the_model_is_corrected_end_to_end():
    four = (
        "Top places in Kolkata:\n- **Famous Fort** (beach) 4.9\n"
        "- **Historic Temple** (temple) 4.9\n- **Famous Temple** (market) 4.1"
    )
    result = plan_trip(query="Places to visit in Kolkata", llm=places_lookup_llm(four), today=TODAY)
    assert result.status == "ok" and result.answer.count("- **") == 5
    assert "incomplete place list" in result.grounding_note
    assert len(result.tool_results[0]["data"]["places"]) == 5  # the table data is unchanged


def test_a_faithful_reply_is_not_touched():
    faithful = "Kolkata has 5 places; the best rated is Famous Fort (beach) at 4.9."
    result = plan_trip(query="Places to visit in Kolkata", llm=places_lookup_llm(faithful), today=TODAY)
    assert result.answer == faithful and result.grounding_note == ""


# ------------------------------------------------------------ Trip Summary (brief, Step 4) ----
def test_trip_summary_states_the_essentials():
    plan = build_plan(REQUEST, GOOD_DRAFT, weather_fn=fake_weather)
    assert plan.summary == (
        "A 3-day trip from Bangalore to Goa (Dec 5\u20137) for 1 traveler. "
        "Estimated total cost: \u20b915,957, within your \u20b925,000 budget (\u20b99,043 to spare)."
    )
    assert plan.summary in format_itinerary(plan)  # shown in the readable text
    assert '"summary": "A 3-day trip' in plan.model_dump_json(indent=2)  # and in the JSON


def test_trip_summary_variants():
    over = build_plan(REQUEST.model_copy(update={"budget_limit_inr": 10000}), GOOD_DRAFT, weather_fn=fake_weather)
    assert over.summary.endswith("over your \u20b910,000 budget by \u20b95,957.")
    no_limit = build_plan(REQUEST.model_copy(update={"budget_limit_inr": None}), GOOD_DRAFT, weather_fn=fake_weather)
    assert no_limit.summary.endswith("Estimated total cost: \u20b915,957.")
    group = REQUEST.model_copy(update={"travelers": 4, "one_way": True})
    summary = build_plan(group, GOOD_DRAFT, weather_fn=fake_weather).summary
    assert "A 3-day one-way trip" in summary and "for 4 travelers" in summary
