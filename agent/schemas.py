"""Pydantic schemas used by the agent.

TripRequest  - what the user wants (from the form or extracted from free text)
PlanDraft    - what the LLM decides (dataset IDs + reasons only)
TripPlan     - the final plan, built in Python from dataset records so every
               price, name and time is grounded in data rather than LLM text
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ------------------------------------------------------------------ request ----
class TripRequest(BaseModel):
    """A user's trip request. Fields are None when the user did not provide them."""

    intent: Literal["plan_trip", "lookup", "other"] = Field(
        default="plan_trip",
        description=(
            "plan_trip: wants a trip or itinerary planned. lookup: asks a specific question about flights, "
            "hotel prices, places to visit or weather. other: unrelated to travel in the supported cities."
        ),
    )
    origin: str | None = Field(default=None, description="Departure city")
    destination: str | None = Field(default=None, description="Destination city")
    start_date: str | None = Field(default=None, description="First day of the trip, YYYY-MM-DD")
    end_date: str | None = Field(default=None, description="Last day of the trip, YYYY-MM-DD")
    travelers: int = Field(default=1, description="Number of travelers ('my wife and I' = 2, 'family of 4' = 4)")
    budget_limit_inr: float | None = Field(default=None, description="TOTAL budget in INR for all travelers, if stated")
    one_way: bool = Field(default=False, description="True only if the user wants a one-way trip (no return flight)")
    needs_hotel: bool = Field(
        default=True, description="False if the user will stay with friends or family or already has accommodation"
    )
    preferences: str | None = Field(
        default=None, description="Interests or constraints, e.g. 'beaches, 4-star hotel, fastest flight'"
    )

    def missing_fields(self) -> list[str]:
        """Essential fields the user has not provided."""
        required = {
            "origin": self.origin,
            "destination": self.destination,
            "start_date": self.start_date,
            "end_date": self.end_date,
        }
        return [name for name, value in required.items() if not value]


# -------------------------------------------------------------------- draft ----
class DraftDay(BaseModel):
    """The LLM's choice of places for one day."""

    day_number: int = Field(description="1-based day number")
    place_ids: list[str] = Field(default_factory=list, description="place_id values from search_places results")


class PlanDraft(BaseModel):
    """The LLM's decisions, expressed as dataset IDs plus reasons."""

    outbound_flight_id: str | None = Field(default=None, description="flight_id of the chosen outbound flight")
    return_flight_id: str | None = Field(default=None, description="flight_id of the chosen return flight")
    hotel_id: str | None = Field(default=None, description="hotel_id of the chosen hotel")
    days: list[DraftDay] = Field(default_factory=list)
    flight_reason: str = Field(default="", description="Why these flights were chosen")
    hotel_reason: str = Field(default="", description="Why this hotel was chosen")
    itinerary_reason: str = Field(default="", description="Why these places and this day-by-day order")
    warnings: list[str] = Field(default_factory=list, description="Anything missing or compromised")


# --------------------------------------------------------------- final plan ----
class FlightLeg(BaseModel):
    """One chosen flight, dated and priced from the dataset."""

    flight_id: str
    airline: str
    origin: str
    destination: str
    departure_time: str
    arrival_time: str
    duration: str
    price_inr: int


class HotelStay(BaseModel):
    """The chosen hotel with nights, rooms and total price."""

    hotel_id: str
    name: str
    city: str
    stars: int
    price_per_night_inr: int
    nights: int
    rooms: int = 1
    total_price_inr: int  # price per night x nights x rooms
    amenities: list[str]


class PlaceVisit(BaseModel):
    """A place to visit, copied from the places dataset."""

    place_id: str
    name: str
    type: str
    rating: float


class DayItinerary(BaseModel):
    """One day of the trip: weather, places and a factual note."""

    day_number: int
    date: str
    weather: str | None = None
    temp_max_c: float | None = None
    temp_min_c: float | None = None
    rain_expected: bool | None = None
    weather_source: str | None = None
    places: list[PlaceVisit] = Field(default_factory=list)
    note: str = ""


class BudgetSummary(BaseModel):
    """Cost breakdown in rupees, with the comparison to the user's limit if one was given."""

    flights: int
    hotel: int
    food_and_local_travel: int
    total: int
    per_person: int
    budget_limit: int | None = None
    within_budget: bool | None = None
    remaining: int | None = None
    over_by: int | None = None
    assumptions: list[str] = Field(default_factory=list)


class TripPlan(BaseModel):
    """The complete, grounded trip plan."""

    title: str
    summary: str = ""
    origin: str
    destination: str
    start_date: str
    end_date: str
    travelers: int
    one_way: bool = False
    needs_hotel: bool = True
    outbound_flight: FlightLeg | None = None
    return_flight: FlightLeg | None = None
    hotel: HotelStay | None = None
    days: list[DayItinerary]
    budget: BudgetSummary
    reasoning: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
