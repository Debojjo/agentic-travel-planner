"""Budget estimation tool: flights + hotel + local expenses."""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from tools.common import ToolInputError, error_payload, to_json

DEFAULT_DAILY_LOCAL_EXPENSE_INR = 1200  # food + local transport per person per day (assumption)


def estimate_budget(
    flight_prices: list[float],
    hotel_price_per_night: float,
    nights: int,
    travelers: int = 1,
    rooms: int = 1,
    daily_local_expense: float = DEFAULT_DAILY_LOCAL_EXPENSE_INR,
    budget_limit: float | None = None,
) -> dict:
    """Total trip cost in INR.

    flight_prices are per traveler (one entry per leg). A trip of N nights spans N+1 days.
    """
    try:
        if nights < 1:
            raise ToolInputError("nights must be at least 1.")
        if travelers < 1 or rooms < 1:
            raise ToolInputError("travelers and rooms must be at least 1.")
        if hotel_price_per_night < 0 or daily_local_expense < 0 or any(p < 0 for p in flight_prices):
            raise ToolInputError("Prices cannot be negative.")
    except ToolInputError as exc:
        return error_payload(str(exc))

    days = nights + 1
    flights_total = sum(flight_prices) * travelers
    hotel_total = hotel_price_per_night * nights * rooms
    local_total = daily_local_expense * days * travelers
    total = flights_total + hotel_total + local_total

    result = {
        "currency": "INR",
        "trip_days": days,
        "nights": nights,
        "travelers": travelers,
        "rooms": rooms,
        "breakdown": {
            "flights": round(flights_total),
            "hotel": round(hotel_total),
            "food_and_local_travel": round(local_total),
        },
        "total": round(total),
        "per_person": round(total / travelers),
        "assumptions": [
            f"Food and local travel estimated at INR {round(daily_local_expense)} per person per day.",
            "Flight prices are per traveler; hotel price is per room per night.",
        ],
    }
    if budget_limit is not None:
        difference = round(budget_limit - total)
        result["budget_limit"] = round(budget_limit)
        result["within_budget"] = difference >= 0
        result["remaining" if difference >= 0 else "over_by"] = abs(difference)
    return result


class BudgetInput(BaseModel):
    """Arguments for the estimate_budget tool."""

    flight_prices: list[float] = Field(
        description="Price of each flight leg for ONE traveler in INR, e.g. [5356, 6200] for outbound and return"
    )
    hotel_price_per_night: float = Field(description="Hotel price per room per night in INR")
    nights: int = Field(ge=1, description="Number of nights staying (trip days = nights + 1)")
    travelers: int = Field(default=1, ge=1, description="Number of travelers")
    rooms: int = Field(default=1, ge=1, description="Number of hotel rooms")
    daily_local_expense: float = Field(
        default=DEFAULT_DAILY_LOCAL_EXPENSE_INR, ge=0, description="Food and local transport per person per day in INR"
    )
    budget_limit: float | None = Field(default=None, ge=0, description="User's total budget in INR, if they gave one")


@tool("estimate_budget", args_schema=BudgetInput)
def budget_estimate_tool(
    flight_prices: list[float],
    hotel_price_per_night: float,
    nights: int,
    travelers: int = 1,
    rooms: int = 1,
    daily_local_expense: float = DEFAULT_DAILY_LOCAL_EXPENSE_INR,
    budget_limit: float | None = None,
) -> str:
    """Estimate total trip cost (flights + hotel + food/local travel) and compare it with the user's budget."""
    return to_json(
        estimate_budget(
            flight_prices, hotel_price_per_night, nights, travelers, rooms, daily_local_expense, budget_limit
        )
    )
