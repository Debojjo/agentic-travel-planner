"""Weather lookup via Open-Meteo (free, no API key).

- Dates inside the forecast window use the live forecast API.
- Dates further ahead use the same calendar dates from a past year (archive API)
  as a rough guide, and are labelled 'historical_estimate'.

Smoke test (needs internet):
    python -m tools.weather_tool Goa 2026-11-12 2026-11-14
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from config import OPEN_METEO_URL, REQUEST_TIMEOUT_SECONDS
from data_loader import DataStore, load_all
from tools.common import ToolInputError, error_payload, parse_date, resolve_city, to_json

logger = logging.getLogger(__name__)

FORECAST_URL = OPEN_METEO_URL
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Conservative: Open-Meteo documents up to 16 forecast days; we use 14 to stay safely inside.
FORECAST_WINDOW_DAYS = 14
ARCHIVE_LAG_DAYS = 7  # archive data lags real time by a few days
MAX_TRIP_DAYS = 14
RAIN_PROBABILITY_THRESHOLD = 50  # percent
RAIN_AMOUNT_THRESHOLD_MM = 5.0

FORECAST_VARIABLES = [
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_probability_max",
    "precipitation_sum",
]
ARCHIVE_VARIABLES = ["weather_code", "temperature_2m_max", "temperature_2m_min", "precipitation_sum"]

# WMO weather interpretation codes used by Open-Meteo
WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Fog",
    51: "Light drizzle",
    53: "Drizzle",
    55: "Heavy drizzle",
    56: "Freezing drizzle",
    57: "Freezing drizzle",
    61: "Light rain",
    63: "Rain",
    65: "Heavy rain",
    66: "Freezing rain",
    67: "Freezing rain",
    71: "Light snow",
    73: "Snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Light rain showers",
    81: "Rain showers",
    82: "Violent rain showers",
    85: "Snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with hail",
    99: "Thunderstorm with heavy hail",
}


class WeatherServiceError(Exception):
    """Raised when the weather API cannot be reached or returns an error."""


def describe_weather(code: int | None) -> str:
    """Convert a WMO weather code to text."""
    return WEATHER_CODES.get(code, "Unknown") if code is not None else "Unknown"


def _get_json(url: str, params: dict[str, Any]) -> dict:
    """GET a JSON document, converting every failure into WeatherServiceError."""
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        payload = response.json()
    except requests.RequestException as exc:
        raise WeatherServiceError(f"Could not reach the weather service: {exc}") from exc
    except ValueError as exc:
        raise WeatherServiceError("Weather service returned invalid data.") from exc

    if isinstance(payload, dict) and payload.get("error"):
        raise WeatherServiceError(f"Weather service error: {payload.get('reason', 'unknown reason')}")
    if not response.ok:
        raise WeatherServiceError(f"Weather service returned HTTP {response.status_code}.")
    return payload


def _fetch_daily(url: str, latitude: float, longitude: float, start: date, end: date, variables: list[str]) -> dict:
    """Return {'YYYY-MM-DD': {variable: value}} for the requested range."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ",".join(variables),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "timezone": "auto",
    }
    daily = _get_json(url, params).get("daily") or {}
    times = daily.get("time") or []
    return {
        day: {var: (daily.get(var) or [None] * len(times))[index] for var in variables}
        for index, day in enumerate(times)
    }


def _shift_back(day: date, years: int) -> date:
    """Same calendar day `years` earlier (29 Feb becomes 28 Feb)."""
    try:
        return day.replace(year=day.year - years)
    except ValueError:
        return day.replace(year=day.year - years, day=28)


def _years_back(latest_day: date, today: date) -> int:
    """Smallest number of years that lands safely inside the archive."""
    years = 1
    while _shift_back(latest_day, years) > today - timedelta(days=ARCHIVE_LAG_DAYS):
        years += 1
    return years


def _build_day(day_number: int, day: date, values: dict | None, source: str) -> dict:
    """Format one day's weather for the agent."""
    if not values or values.get("weather_code") is None:
        return {"day_number": day_number, "date": day.isoformat(), "available": False, "source": source}

    probability = values.get("precipitation_probability_max")
    rain_mm = values.get("precipitation_sum")
    rain_expected = (probability is not None and probability >= RAIN_PROBABILITY_THRESHOLD) or (
        rain_mm is not None and rain_mm >= RAIN_AMOUNT_THRESHOLD_MM
    )
    return {
        "day_number": day_number,
        "date": day.isoformat(),
        "available": True,
        "weather": describe_weather(values["weather_code"]),
        "temp_max_c": values.get("temperature_2m_max"),
        "temp_min_c": values.get("temperature_2m_min"),
        "rain_probability_percent": probability,
        "precipitation_mm": rain_mm,
        "rain_expected": rain_expected,
        "source": source,
    }


def _validate_weather_request(
    store: DataStore, city: str, start_date: str, end_date: str, today: date
) -> tuple[str, date, date]:
    """Resolve the city and check the dates. Returns (city, start, end) or raises ToolInputError."""
    city_name = resolve_city(store, city)
    start, end = parse_date(start_date), parse_date(end_date)
    if end < start:
        raise ToolInputError("end_date must not be before start_date.")
    if start < today:
        raise ToolInputError(f"start_date {start} is in the past. Weather is provided from today ({today}) onwards.")
    if (end - start).days + 1 > MAX_TRIP_DAYS:
        raise ToolInputError(f"Trip too long: at most {MAX_TRIP_DAYS} days are supported.")
    return city_name, start, end


def _forecast_results(coords, days: list[date], numbering: dict[date, int]) -> dict[date, dict]:
    """Live forecast for days inside the forecast window."""
    raw = _fetch_daily(FORECAST_URL, coords.latitude, coords.longitude, days[0], days[-1], FORECAST_VARIABLES)
    return {d: _build_day(numbering[d], d, raw.get(d.isoformat()), "forecast") for d in days}


def _typical_results(coords, days: list[date], numbering: dict[date, int], today: date) -> dict[date, dict]:
    """Last year's weather for the same dates, for days beyond the forecast window."""
    shifted = {d: _shift_back(d, _years_back(days[-1], today)) for d in days}
    raw = _fetch_daily(
        ARCHIVE_URL,
        coords.latitude,
        coords.longitude,
        min(shifted.values()),
        max(shifted.values()),
        ARCHIVE_VARIABLES,
    )
    return {
        d: _build_day(numbering[d], d, raw.get(past_day.isoformat()), "historical_estimate")
        for d, past_day in shifted.items()
    }


def get_weather(
    city: str,
    start_date: str,
    end_date: str,
    today: date | None = None,
    store: DataStore | None = None,
) -> dict:
    """Day-by-day weather for a city between two dates (inclusive)."""
    store = store or load_all()
    today = today or date.today()
    try:
        city_name, start, end = _validate_weather_request(store, city, start_date, end_date, today)
    except ToolInputError as exc:
        return error_payload(str(exc))

    days = [start + timedelta(days=offset) for offset in range((end - start).days + 1)]
    numbering = {d: index + 1 for index, d in enumerate(days)}
    forecast_limit = today + timedelta(days=FORECAST_WINDOW_DAYS - 1)
    forecast_days = [d for d in days if d <= forecast_limit]
    typical_days = [d for d in days if d > forecast_limit]
    coords = store.cities[city_name]

    results: dict[date, dict] = {}
    try:
        if forecast_days:
            results.update(_forecast_results(coords, forecast_days, numbering))
        if typical_days:
            results.update(_typical_results(coords, typical_days, numbering, today))
    except WeatherServiceError as exc:
        logger.warning("Weather lookup failed: %s", exc)
        return error_payload(str(exc), hint="Continue planning without weather and tell the user it was unavailable.")

    payload = {
        "city": city_name,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": [results[d] for d in days],
    }
    if typical_days:
        payload["note"] = (
            f"Dates beyond the {FORECAST_WINDOW_DAYS}-day forecast window show typical weather: "
            "last year's actual weather for the same dates, not a forecast."
        )
    return payload


class WeatherInput(BaseModel):
    """Arguments for the get_weather_forecast tool."""

    city: str = Field(description="Destination city, e.g. 'Goa'")
    start_date: str = Field(description="First day of the trip, YYYY-MM-DD")
    end_date: str = Field(description="Last day of the trip, YYYY-MM-DD (inclusive)")


@tool("get_weather_forecast", args_schema=WeatherInput)
def weather_forecast_tool(city: str, start_date: str, end_date: str) -> str:
    """Get day-by-day weather (conditions, temperatures, rain chance) for a destination across the trip dates."""
    return to_json(get_weather(city, start_date, end_date))


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python -m tools.weather_tool <city> <start YYYY-MM-DD> <end YYYY-MM-DD>")
    else:
        print(to_json(get_weather(*sys.argv[1:4])))
