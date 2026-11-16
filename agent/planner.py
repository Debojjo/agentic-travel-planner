"""Request validation and plan grounding.

The LLM only chooses dataset IDs. Everything shown to the user (names, prices,
times, notes, weather, budget) is rebuilt here from the datasets and tools, so
the final plan cannot contain invented flights, hotels, places or arithmetic.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Callable

from agent.formatter import format_inr
from agent.schemas import (
    BudgetSummary,
    DayItinerary,
    FlightLeg,
    HotelStay,
    PlaceVisit,
    PlanDraft,
    TripPlan,
    TripRequest,
)
from data_loader import DataStore, load_all
from tools.budget_tool import estimate_budget
from tools.common import ToolInputError, parse_date, resolve_city
from tools.flight_tool import rebase_flight
from tools.weather_tool import get_weather

logger = logging.getLogger(__name__)

MAX_PLAN_DAYS = 7  # the brief asks for 3-7 day itineraries
MAX_TRAVELERS = 10

# Activities the dataset has no concept of (it only has places with a type like beach, fort, museum).
_UNSUPPORTED_INTERESTS = {
    "trekking or hiking": r"\b(?:trek\w*|hik(?:e|es|ing))\b",
    "nightlife": r"\b(?:night ?life|clubs?|pubs?|bars?)\b",
    "food and dining": r"\b(?:seafood|food|restaurants?|cafes?|dining|cuisine)\b",
    "adventure sports": r"\b(?:adventure|water ?sports?|scuba|diving|snorkel\w*|rafting|paragliding)\b",
    "wildlife": r"\b(?:safari|wildlife)\b",
    "casinos": r"\bcasinos?\b",
    "yoga": r"\byoga\b",
}


def rooms_for(travelers: int) -> int:
    """Assume two travelers share a room."""
    return math.ceil(travelers / 2)


def _friendly_missing(missing: list[str]) -> str:
    """Describe missing request fields in plain language."""
    wording = {
        "origin": "where you are starting from",
        "destination": "where you want to go",
        "dates": "your travel dates (for example 'starting 5 December for 3 days')",
    }
    needed = [wording["origin"]] if "origin" in missing else []
    if "destination" in missing:
        needed.append(wording["destination"])
    if "start_date" in missing or "end_date" in missing:
        needed.append(wording["dates"])
    return "I still need: " + "; ".join(needed) + "."


def _validate_cities(request: TripRequest, store: DataStore) -> tuple[str | None, str | None, list[str]]:
    """Resolve origin and destination to dataset cities. Returns (origin, destination, problems)."""
    problems: list[str] = []
    cities: list[str | None] = []
    for label, name in (("Origin", request.origin), ("Destination", request.destination)):
        try:
            cities.append(resolve_city(store, name))
        except ToolInputError as exc:
            problems.append(f"{label}: {exc}")
            cities.append(None)
    origin, destination = cities
    if origin and destination and origin == destination:
        problems.append("Origin and destination must be different cities.")
    return origin, destination, problems


def _validate_dates(request: TripRequest, today: date) -> tuple[date | None, date | None, list[str]]:
    """Parse the trip dates and check them against today and the supported trip length."""
    try:
        start, end = parse_date(request.start_date), parse_date(request.end_date)
    except ToolInputError as exc:
        return None, None, [str(exc)]
    if start < today:
        return start, end, [f"Start date {start} is in the past (today is {today})."]
    if end <= start:
        return start, end, ["End date must be after the start date (at least a 2-day trip)."]
    if (end - start).days + 1 > MAX_PLAN_DAYS:
        return start, end, [f"Trips are limited to {MAX_PLAN_DAYS} days."]
    return start, end, []


def _validate_limits(request: TripRequest) -> list[str]:
    """Check the traveler count and budget."""
    problems = []
    if request.travelers < 1:
        problems.append("At least 1 traveler is needed.")
    elif request.travelers > MAX_TRAVELERS:
        problems.append(f"Trips are limited to {MAX_TRAVELERS} travelers.")
    if request.budget_limit_inr is not None and request.budget_limit_inr < 0:
        problems.append("The budget cannot be negative.")
    return problems


def normalize_request(
    request: TripRequest, store: DataStore | None = None, today: date | None = None
) -> tuple[TripRequest | None, list[str]]:
    """Validate a request. Returns (clean request, []) or (None, problems)."""
    store = store or load_all()
    today = today or date.today()

    missing = request.missing_fields()
    if missing:
        return None, [_friendly_missing(missing)]

    origin, destination, problems = _validate_cities(request, store)
    start, end, date_problems = _validate_dates(request, today)
    problems += date_problems + _validate_limits(request)
    if problems:
        return None, problems

    clean = request.model_copy(
        update={
            "origin": origin,
            "destination": destination,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "budget_limit_inr": request.budget_limit_inr or None,  # 0 means "no limit"
        }
    )
    return clean, []


def unavailable_place_types(request: TripRequest, store: DataStore) -> list[str]:
    """Place types named in the user's preferences that the destination has none of."""
    text = (request.preferences or "").lower()
    if not text:
        return []
    all_types = {p.type for p in store.places}
    in_city = {p.type for p in store.places if p.city == request.destination}
    return sorted(t for t in all_types - in_city if re.search(rf"\b{re.escape(t)}(?:s|es)?\b", text))


def unsupported_interests(preferences: str | None) -> list[str]:
    """Interests the dataset cannot serve at all (e.g. trekking, nightlife)."""
    text = (preferences or "").lower()
    return [label for label, pattern in _UNSUPPORTED_INTERESTS.items() if re.search(pattern, text)]


def describe_request(request: TripRequest) -> str:
    """Render a validated request as the agent's task message."""
    start, end = parse_date(request.start_date), parse_date(request.end_date)
    nights = (end - start).days
    budget_text = f"INR {round(request.budget_limit_inr)}" if request.budget_limit_inr else "not stated"
    lines = [
        "Plan this trip:",
        f"- From: {request.origin}",
        f"- To: {request.destination}",
        f"- Dates: {start} to {end} ({nights + 1} days, {nights} nights)",
        f"- Travelers: {request.travelers}"
        + (f" (rooms: {rooms_for(request.travelers)})" if request.needs_hotel else ""),
        f"- Budget limit: {budget_text}",
        f"- Preferences: {request.preferences or 'none stated'}",
        "- Trip type: "
        + (
            "ONE-WAY. Do not search for a return flight."
            if request.one_way
            else "round trip (outbound and return flights)"
        ),
        "- Accommodation: "
        + (
            "NOT NEEDED, the traveler is staying with friends or family. Do not search hotels."
            if not request.needs_hotel
            else "needed (search hotels)"
        ),
    ]
    return "\n".join(lines)


_THIN_SPACES = "\u202f\u00a0\u2009\u2007"


def _tidy(text: str) -> str:
    """Clean LLM prose: thin-space thousand separators, non-breaking hyphens, internal field names."""
    text = re.sub(rf"(?<=\d)[{_THIN_SPACES}](?=\d)", ",", text)
    text = re.sub(rf"[{_THIN_SPACES}]", " ", text)
    text = text.replace("\u2011", "-")
    text = re.sub(r"value[_ -]score", "value score", text, flags=re.IGNORECASE)
    return text.strip()


tidy_text = _tidy


def _autofill_places(days: list[DayItinerary], places_in_city: list) -> None:
    """Fallback: give the trip top-rated places (two a day, last day free on trips of 3+ days)."""
    ranked = sorted(places_in_city, key=lambda p: (-p.rating, p.name))
    sightseeing_days = days[:-1] if len(days) >= 3 else days
    for index, day in enumerate(sightseeing_days):
        for p in ranked[index * 2 : index * 2 + 2]:
            day.places.append(PlaceVisit(place_id=p.place_id, name=p.name, type=p.type, rating=p.rating))


def _day_note(number: int, total: int, outbound: FlightLeg | None, inbound: FlightLeg | None) -> str:
    """Factual note built from the chosen flights (never from LLM prose)."""
    parts = []
    if number == 1 and outbound:
        departs = datetime.fromisoformat(outbound.departure_time)
        arrives = datetime.fromisoformat(outbound.arrival_time)
        parts.append(f"Arrive {arrives:%H:%M}" + (" (next day)" if arrives.date() > departs.date() else ""))
    if number == total and inbound:
        parts.append(f"Flight home departs {datetime.fromisoformat(inbound.departure_time):%H:%M}")
    return "; ".join(parts)


# Topics the system reports itself, with words that signal the LLM is repeating them.
# Note: 'budget' only counts when the system actually reported an overrun.
_COVERED_TOPICS = {
    "flight": ("flight", "transport", "train", "road"),
    "hotel": ("hotel", "accommodation"),
    "budget": ("budget", "exceed"),
    "weather": ("forecast", "weather"),
}


def _novel_notes(llm_notes: list[str], covered: set[str]) -> list[str]:
    """Keep the LLM's notes only when they are not about a topic the system already reported."""
    return [
        _tidy(note)
        for note in llm_notes
        if not any(word in note.lower() for topic in covered for word in _COVERED_TOPICS[topic])
    ]


def _date_range_label(start: date, end: date) -> str:
    if start.month == end.month:
        return f"{start.strftime('%b')} {start.day}\u2013{end.day}"
    return f"{start.strftime('%b')} {start.day} \u2013 {end.strftime('%b')} {end.day}"


@dataclass
class _PlanNotes:
    """Messages collected while a plan is assembled."""

    warnings: list[str] = field(default_factory=list)  # reported by the system
    removed: list[str] = field(default_factory=list)  # draft items that did not match the dataset
    covered: set[str] = field(default_factory=set)  # topics already reported, so LLM repeats are dropped


def _validated_leg(
    flights: dict, flight_id: str | None, route: tuple[str, str], day: date, label: str, notes: _PlanNotes
) -> FlightLeg | None:
    """Look a drafted flight up in the dataset; reject it if it is missing or flies a different route."""
    if not flight_id:
        return None
    flight = flights.get(flight_id)
    if flight is None or (flight.origin, flight.destination) != route:
        logger.warning("Removed %s flight %r: not a %s to %s flight in the dataset", label, flight_id, *route)
        notes.removed.append(f"{label} flight")
        return None
    return FlightLeg(**rebase_flight(flight, day))


def _note_missing_flights(
    request: TripRequest, store: DataStore, outbound: FlightLeg | None, inbound: FlightLeg | None, notes: _PlanNotes
) -> None:
    """Say which flights are missing and whether the route exists at all (and so cannot be booked)."""
    routes = {(f.origin, f.destination) for f in store.flights}
    needed = [("outbound", (request.origin, request.destination), outbound)]
    if not request.one_way:
        needed.append(("return", (request.destination, request.origin), inbound))
    missing = [(label, route) for label, route, leg in needed if leg is None]
    if not missing:
        return
    notes.covered.add("flight")
    phrases = [
        f"{label} flight {route[0]} to {route[1]} "
        f"({'not selected' if route in routes else 'no such route in the dataset'})"
        for label, route in missing
    ]
    note = f"Flights are not included in the budget: {'; '.join(phrases)}."
    if any(route not in routes for _, route in missing):
        note += " Arrange that transport separately."
    notes.warnings.append(note)


def _choose_flights(
    request: TripRequest, draft: PlanDraft, store: DataStore, notes: _PlanNotes
) -> tuple[FlightLeg | None, FlightLeg | None]:
    """Ground the outbound and (unless one-way) return flights in the dataset."""
    flights = {f.flight_id: f for f in store.flights}
    start, end = parse_date(request.start_date), parse_date(request.end_date)
    forward, back = (request.origin, request.destination), (request.destination, request.origin)
    outbound = _validated_leg(flights, draft.outbound_flight_id, forward, start, "outbound", notes)
    inbound = None
    if not request.one_way:
        inbound = _validated_leg(flights, draft.return_flight_id, back, end, "return", notes)
    _note_missing_flights(request, store, outbound, inbound, notes)
    return outbound, inbound


def _choose_hotel(
    request: TripRequest, draft: PlanDraft, store: DataStore, nights: int, notes: _PlanNotes
) -> HotelStay | None:
    """Ground the drafted hotel in the dataset, or explain why there is none."""
    notes.covered.add("hotel")  # the system reports every hotel outcome itself (including 'not needed')
    if not request.needs_hotel:
        return None
    hotel = next((h for h in store.hotels if h.hotel_id == draft.hotel_id), None) if draft.hotel_id else None
    if draft.hotel_id and (hotel is None or hotel.city != request.destination):
        logger.warning("Removed hotel %r: not a hotel in %s", draft.hotel_id, request.destination)
        notes.removed.append("hotel")
        hotel = None
    if hotel is None:
        hint = ""
        if "hotel must" in (request.preferences or "").lower():
            hint = " Your hotel filters may be too strict; try fewer amenities or a lower star rating."
        notes.warnings.append(
            f"No hotel was selected in {request.destination}, so hotel cost is not included in the budget.{hint}"
        )
        return None
    rooms = rooms_for(request.travelers)
    return HotelStay(
        hotel_id=hotel.hotel_id,
        name=hotel.name,
        city=hotel.city,
        stars=hotel.stars,
        price_per_night_inr=hotel.price_per_night,
        nights=nights,
        rooms=rooms,
        total_price_inr=hotel.price_per_night * nights * rooms,
        amenities=hotel.amenities,
    )


def _weather_by_date(
    request: TripRequest, weather: dict | None, weather_fn: Callable[[str, str, str], dict], notes: _PlanNotes
) -> dict[str, dict]:
    """Index the trip's weather by date, fetching it only if the agent did not already."""
    if weather is None:
        weather = weather_fn(request.destination, request.start_date, request.end_date)
    if "error" in weather:
        notes.covered.add("weather")
        notes.warnings.append(f"Weather unavailable: {str(weather['error'])[:120]}")
        return {}
    if weather.get("note"):
        notes.covered.add("weather")
        notes.warnings.append(weather["note"])
    return {d["date"]: d for d in weather["daily"] if d.get("available")}


def _note_preference_gaps(request: TripRequest, store: DataStore, notes: _PlanNotes) -> None:
    """Tell the user when a wish (a place type or activity) cannot be met from the dataset."""
    for missing_type in unavailable_place_types(request, store):
        notes.warnings.append(
            f"You asked for {missing_type} places, but {request.destination} has none in the dataset; "
            "the best-rated alternatives were used."
        )
    unsupported = unsupported_interests(request.preferences)
    if unsupported:
        notes.warnings.append(
            f"The dataset has no {', '.join(unsupported)} options; it only lists places such as "
            "beaches, forts, museums, temples, parks, lakes, markets and monuments."
        )


def _visits_for_day(place_ids: list[str], places: dict, destination: str, number: int) -> tuple[list[PlaceVisit], int]:
    """Look up one day's drafted places. Returns (visits, how many were rejected)."""
    visits, rejected = [], 0
    for place_id in dict.fromkeys(place_ids):
        place = places.get(place_id)
        if place is None or place.city != destination:
            logger.warning("Removed place %r on day %d: not a place in %s", place_id, number, destination)
            rejected += 1
            continue
        visits.append(PlaceVisit(place_id=place.place_id, name=place.name, type=place.type, rating=place.rating))
    return visits, rejected


def _build_days(
    request: TripRequest,
    draft: PlanDraft,
    store: DataStore,
    weather_by_date: dict[str, dict],
    flights: tuple[FlightLeg | None, FlightLeg | None],
) -> tuple[list[DayItinerary], int]:
    """Build every day of the trip. Returns (days, number of drafted places that were rejected)."""
    start, end = parse_date(request.start_date), parse_date(request.end_date)
    dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    places = {p.place_id: p for p in store.places}
    draft_days = {d.day_number: d for d in draft.days}
    days: list[DayItinerary] = []
    rejected_total = 0
    for number, day in enumerate(dates, start=1):
        draft_day = draft_days.get(number)
        visits, rejected = _visits_for_day(
            draft_day.place_ids if draft_day else [], places, request.destination, number
        )
        rejected_total += rejected
        w = weather_by_date.get(day.isoformat(), {})
        days.append(
            DayItinerary(
                day_number=number,
                date=day.isoformat(),
                weather=w.get("weather"),
                temp_max_c=w.get("temp_max_c"),
                temp_min_c=w.get("temp_min_c"),
                rain_expected=w.get("rain_expected"),
                weather_source=w.get("source"),
                places=visits,
                note=_day_note(number, len(dates), *flights),
            )
        )
    return days, rejected_total


def _autofill_if_empty(request: TripRequest, store: DataStore, days: list[DayItinerary], notes: _PlanNotes) -> bool:
    """If the agent chose no places at all, add the destination's top-rated ones. Returns True if it did."""
    places_in_city = [p for p in store.places if p.city == request.destination]
    if not places_in_city or any(day.places for day in days):
        return False
    _autofill_places(days, places_in_city)
    notes.warnings.append(
        f"The agent did not choose any places, so the top-rated places in {request.destination} "
        "were added automatically."
    )
    return True


def _note_removed_items(rejected_places: int, notes: _PlanNotes) -> None:
    """One summary warning for everything the agent proposed that is not in the dataset."""
    if rejected_places:
        notes.removed.append(f"{rejected_places} place{'s' if rejected_places > 1 else ''}")
    if notes.removed:
        notes.warnings.append(
            "The agent proposed items that are not in the dataset, so they were removed: "
            + ", ".join(notes.removed)
            + "."
        )


def _summarise_budget(
    request: TripRequest,
    flights: tuple[FlightLeg | None, FlightLeg | None],
    hotel: HotelStay | None,
    nights: int,
    notes: _PlanNotes,
) -> BudgetSummary:
    """Recompute the budget in code (never copied from the LLM) and flag an overrun."""
    rooms = rooms_for(request.travelers)
    raw = estimate_budget(
        flight_prices=[leg.price_inr for leg in flights if leg],
        hotel_price_per_night=hotel.price_per_night_inr if hotel else 0,
        nights=nights,
        travelers=request.travelers,
        rooms=rooms,
        budget_limit=request.budget_limit_inr,
    )
    budget = BudgetSummary(**{k: v for k, v in raw.items() if k in BudgetSummary.model_fields}, **raw["breakdown"])
    if request.travelers > 1 and hotel:
        budget.assumptions.append(
            f"{rooms} hotel room{'s' if rooms > 1 else ''} for {request.travelers} travelers (two per room)."
        )
    if budget.within_budget is False:
        notes.covered.add("budget")
        notes.warnings.append(
            f"Estimated cost is {format_inr(budget.over_by)} over your {format_inr(budget.budget_limit)} budget."
        )
    return budget


def _reasoning(draft: PlanDraft, request: TripRequest, places_replaced: bool) -> dict[str, str]:
    """The 'why we chose this' texts, tidied; the itinerary reason is replaced if its places were not usable."""
    itinerary = _tidy(draft.itinerary_reason)
    if places_replaced:
        itinerary = f"The itinerary uses the best-rated places in {request.destination} found in the dataset."
    texts = {"flights": _tidy(draft.flight_reason), "hotel": _tidy(draft.hotel_reason), "itinerary": itinerary}
    return {key: text for key, text in texts.items() if text}


def _trip_summary(request: TripRequest, trip_days: int, date_label: str, budget: BudgetSummary) -> str:
    """One-sentence trip summary (the first item of the brief's required output)."""
    people = "1 traveler" if request.travelers == 1 else f"{request.travelers} travelers"
    kind = "one-way " if request.one_way else ""
    text = (
        f"A {trip_days}-day {kind}trip from {request.origin} to {request.destination} ({date_label}) "
        f"for {people}. Estimated total cost: {format_inr(budget.total)}"
    )
    if budget.within_budget is True:
        return (
            text + f", within your {format_inr(budget.budget_limit)} budget ({format_inr(budget.remaining)} to spare)."
        )
    if budget.within_budget is False:
        return text + f", over your {format_inr(budget.budget_limit)} budget by {format_inr(budget.over_by)}."
    return text + "."


def build_plan(
    request: TripRequest,
    draft: PlanDraft,
    store: DataStore | None = None,
    weather_fn: Callable[[str, str, str], dict] | None = None,
    weather: dict | None = None,
) -> TripPlan:
    """Turn the LLM's draft (IDs + reasons) into a grounded TripPlan.

    `weather` is the result the agent already fetched (saves a second API call);
    when omitted, `weather_fn` is called.
    """
    store = store or load_all()
    start, end = parse_date(request.start_date), parse_date(request.end_date)
    nights = (end - start).days
    notes = _PlanNotes()

    flights = _choose_flights(request, draft, store, notes)
    hotel = _choose_hotel(request, draft, store, nights, notes)
    weather_by_date = _weather_by_date(request, weather, weather_fn or get_weather, notes)
    _note_preference_gaps(request, store, notes)

    days, rejected_places = _build_days(request, draft, store, weather_by_date, flights)
    autofilled = _autofill_if_empty(request, store, days, notes)
    _note_removed_items(rejected_places, notes)
    budget = _summarise_budget(request, flights, hotel, nights, notes)

    date_label = _date_range_label(start, end)
    return TripPlan(
        title=f"Your {len(days)}-Day Trip to {request.destination} ({date_label})",
        summary=_trip_summary(request, len(days), date_label, budget),
        origin=request.origin,
        destination=request.destination,
        start_date=request.start_date,
        end_date=request.end_date,
        travelers=request.travelers,
        one_way=request.one_way,
        needs_hotel=request.needs_hotel,
        outbound_flight=flights[0],
        return_flight=flights[1],
        hotel=hotel,
        days=days,
        budget=budget,
        reasoning=_reasoning(draft, request, autofilled or rejected_places > 0),
        warnings=list(dict.fromkeys(notes.warnings + _novel_notes(draft.warnings, notes.covered))),
    )
