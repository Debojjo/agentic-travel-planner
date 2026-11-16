"""Shared helpers for the LangChain tools."""

from __future__ import annotations

import difflib
import json
from datetime import date, datetime
from typing import Any

from data_loader import DataStore, normalize_city


class ToolInputError(ValueError):
    """Raised when a tool receives input it cannot act on."""


def resolve_city(store: DataStore, name: str) -> str:
    """Return the canonical city name, or raise ToolInputError if unsupported."""
    city = normalize_city(name)
    if city not in store.cities:
        supported = ", ".join(sorted(store.cities))
        close = difflib.get_close_matches(city, list(store.cities), n=1, cutoff=0.7)
        hint = f" Did you mean '{close[0]}'?" if close else ""
        raise ToolInputError(f"Unknown city '{name}'.{hint} Supported cities: {supported}.")
    return city


def parse_date(value: str) -> date:
    """Parse a YYYY-MM-DD string into a date."""
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d").date()
    except (ValueError, AttributeError) as exc:
        raise ToolInputError(f"Invalid date '{value}'. Use YYYY-MM-DD.") from exc


def format_duration(minutes: int) -> str:
    """Format minutes as e.g. '2h 05m'."""
    hours, mins = divmod(minutes, 60)
    return f"{hours}h {mins:02d}m" if hours else f"{mins}m"


def format_inr(amount: float) -> str:
    """Format rupees with Indian digit grouping, e.g. 1370000 -> Rs 13,70,000."""
    digits = str(int(round(amount)))
    if len(digits) <= 3:
        return "\u20b9" + digits
    head, tail = digits[:-3], digits[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return "\u20b9" + ",".join(groups + [tail])


def clamp(value: int, low: int, high: int) -> int:
    """Limit value to the range [low, high]."""
    return max(low, min(high, value))


def error_payload(message: str, **extra: Any) -> dict:
    """Standard error result. Tools return this instead of raising, so the agent can recover."""
    return {"error": message, **extra}


def to_json(payload: Any) -> str:
    """Serialise a tool result for the LLM (compact JSON uses far fewer tokens than pretty-printed)."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
