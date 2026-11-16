"""Tests for the weather and budget tools. Weather HTTP calls are mocked."""

import json
from datetime import date, timedelta

import pytest

import tools.weather_tool as wt
from tools.budget_tool import budget_estimate_tool, estimate_budget
from tools.weather_tool import get_weather

TODAY = date(2026, 11, 10)


@pytest.fixture
def fake_api(monkeypatch):
    """Replace the HTTP layer with a fake that records calls."""
    calls = []

    def fake_get(url, params):
        calls.append((url, params))
        start = date.fromisoformat(params["start_date"])
        end = date.fromisoformat(params["end_date"])
        days = [(start + timedelta(days=i)).isoformat() for i in range((end - start).days + 1)]
        n = len(days)
        daily = {
            "time": days,
            "weather_code": [2] * n,
            "temperature_2m_max": [31.0] * n,
            "temperature_2m_min": [24.0] * n,
            "precipitation_sum": [0.0] * n,
        }
        if "archive" not in url:
            daily["precipitation_probability_max"] = [10] * n
        return {"daily": daily}

    monkeypatch.setattr(wt, "_get_json", fake_get)
    return calls


# ---------------------------------------------------------------- weather ----
def test_forecast_inside_window(fake_api):
    result = get_weather("Goa", "2026-11-12", "2026-11-14", today=TODAY)
    assert [d["day_number"] for d in result["daily"]] == [1, 2, 3]
    day = result["daily"][0]
    assert day["weather"] == "Partly cloudy" and day["temp_max_c"] == 31.0
    assert day["source"] == "forecast" and day["rain_expected"] is False
    assert "note" not in result
    assert len(fake_api) == 1 and "archive" not in fake_api[0][0]
    assert fake_api[0][1]["latitude"] == 15.4909


def test_far_future_uses_historical_estimate(fake_api):
    result = get_weather("Goa", "2027-02-12", "2027-02-14", today=TODAY)
    assert all(d["source"] == "historical_estimate" for d in result["daily"])
    assert "note" in result
    url, params = fake_api[0]
    assert "archive" in url and params["start_date"] == "2026-02-12"


def test_trip_spanning_forecast_and_estimate(fake_api):
    result = get_weather("Goa", "2026-11-22", "2026-11-26", today=TODAY)
    sources = [d["source"] for d in result["daily"]]
    assert sources == ["forecast", "forecast", "historical_estimate", "historical_estimate", "historical_estimate"]
    assert len(fake_api) == 2


def test_rain_expected_flag(monkeypatch):
    def rainy(url, params):
        day = params["start_date"]
        return {
            "daily": {
                "time": [day],
                "weather_code": [63],
                "temperature_2m_max": [27],
                "temperature_2m_min": [22],
                "precipitation_probability_max": [80],
                "precipitation_sum": [12.0],
            }
        }

    monkeypatch.setattr(wt, "_get_json", rainy)
    day = get_weather("Goa", "2026-11-12", "2026-11-12", today=TODAY)["daily"][0]
    assert day["weather"] == "Rain" and day["rain_expected"] is True


def test_service_failure_returns_error(monkeypatch):
    def boom(url, params):
        raise wt.WeatherServiceError("Could not reach the weather service: timeout")

    monkeypatch.setattr(wt, "_get_json", boom)
    result = get_weather("Goa", "2026-11-12", "2026-11-14", today=TODAY)
    assert "error" in result and "hint" in result


def test_missing_data_marks_day_unavailable(monkeypatch):
    monkeypatch.setattr(wt, "_get_json", lambda url, params: {"daily": {"time": []}})
    day = get_weather("Goa", "2026-11-12", "2026-11-12", today=TODAY)["daily"][0]
    assert day["available"] is False


@pytest.mark.parametrize(
    "args",
    [
        ("Atlantis", "2026-11-12", "2026-11-14"),
        ("Goa", "2026-11-14", "2026-11-12"),
        ("Goa", "2026-11-01", "2026-11-03"),
        ("Goa", "12-11-2026", "14-11-2026"),
        ("Goa", "2026-11-12", "2026-12-20"),
    ],
)
def test_weather_bad_inputs(args, fake_api):
    assert "error" in get_weather(*args, today=TODAY)
    assert fake_api == []


def test_leap_day_shift():
    assert wt._shift_back(date(2028, 2, 29), 1) == date(2027, 2, 28)
    assert wt._years_back(date(2027, 2, 14), TODAY) == 1


def test_weather_code_description():
    assert wt.describe_weather(0) == "Clear sky"
    assert wt.describe_weather(12345) == "Unknown"
    assert wt.describe_weather(None) == "Unknown"


# ----------------------------------------------------------------- budget ----
def test_budget_totals():
    result = estimate_budget([4800], 3200, nights=2, daily_local_expense=1000)
    assert result["trip_days"] == 3
    assert result["breakdown"] == {"flights": 4800, "hotel": 6400, "food_and_local_travel": 3000}
    assert result["total"] == 14200


def test_budget_multiple_travelers_and_legs():
    result = estimate_budget([5000, 6000], 2000, nights=3, travelers=2, rooms=1, daily_local_expense=1000)
    assert result["breakdown"]["flights"] == 22000
    assert result["breakdown"]["hotel"] == 6000
    assert result["breakdown"]["food_and_local_travel"] == 8000
    assert result["total"] == 36000 and result["per_person"] == 18000


def test_budget_limit_comparison():
    under = estimate_budget([4800], 3200, nights=2, daily_local_expense=1000, budget_limit=20000)
    assert under["within_budget"] is True and under["remaining"] == 5800
    over = estimate_budget([4800], 3200, nights=2, daily_local_expense=1000, budget_limit=10000)
    assert over["within_budget"] is False and over["over_by"] == 4200


def test_budget_invalid_inputs():
    assert "error" in estimate_budget([4800], 3200, nights=0)
    assert "error" in estimate_budget([-1], 3200, nights=2)
    assert "error" in estimate_budget([4800], 3200, nights=2, travelers=0)


def test_budget_tool_invoke():
    out = budget_estimate_tool.invoke({"flight_prices": [4800], "hotel_price_per_night": 3200, "nights": 2})
    data = json.loads(out)
    assert data["total"] == 4800 + 6400 + 3600
