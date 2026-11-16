"""Load, validate and normalise the flights, hotels and places datasets.

Usage:
    from data_loader import load_all
    store = load_all()
    store.flights, store.hotels, store.places, store.cities

Run `python data_loader.py` for a data-quality report.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from config import CITY_ALIASES, DATA_DIR

logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Raised when a dataset file is missing or is not valid JSON."""


def normalize_city(name: str) -> str:
    """Return the canonical city name (handles case, whitespace and aliases)."""
    cleaned = name.strip()
    alias = CITY_ALIASES.get(cleaned.lower())
    return alias if alias else cleaned.title()


# ---------------------------------------------------------------- models ----
class Flight(BaseModel):
    """A one-way flight. Dataset keys 'from'/'to' map to origin/destination."""

    model_config = ConfigDict(populate_by_name=True)

    flight_id: str
    airline: str
    origin: str = Field(alias="from")
    destination: str = Field(alias="to")
    departure_time: datetime
    arrival_time: datetime
    price: int = Field(gt=0)

    @field_validator("origin", "destination")
    @classmethod
    def _normalize_cities(cls, value: str) -> str:
        return normalize_city(value)

    @model_validator(mode="after")
    def _check_times(self) -> "Flight":
        if self.arrival_time <= self.departure_time:
            raise ValueError("arrival_time must be after departure_time")
        if self.origin == self.destination:
            raise ValueError("origin and destination are the same")
        return self

    @property
    def duration_minutes(self) -> int:
        """Flight duration in minutes."""
        return int((self.arrival_time - self.departure_time).total_seconds() // 60)


class Hotel(BaseModel):
    """A hotel. The dataset has star rating only (no guest rating)."""

    hotel_id: str
    name: str
    city: str
    stars: int = Field(ge=1, le=5)
    price_per_night: int = Field(gt=0)
    amenities: list[str] = Field(default_factory=list)

    @field_validator("city")
    @classmethod
    def _normalize_city(cls, value: str) -> str:
        return normalize_city(value)

    @field_validator("amenities")
    @classmethod
    def _normalize_amenities(cls, value: list[str]) -> list[str]:
        return [item.strip().lower() for item in value]


class Place(BaseModel):
    """A point of interest."""

    place_id: str
    name: str
    city: str
    type: str
    rating: float = Field(ge=0, le=5)

    @field_validator("city")
    @classmethod
    def _normalize_city(cls, value: str) -> str:
        return normalize_city(value)

    @field_validator("type")
    @classmethod
    def _normalize_type(cls, value: str) -> str:
        return value.strip().lower()


class City(BaseModel):
    """City coordinates, needed for the Open-Meteo weather call."""

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


# ----------------------------------------------------------------- store ----
@dataclass
class DataStore:
    """All validated data plus any issues found while loading."""

    flights: list[Flight] = field(default_factory=list)
    hotels: list[Hotel] = field(default_factory=list)
    places: list[Place] = field(default_factory=list)
    cities: dict[str, City] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)


T = TypeVar("T", bound=BaseModel)


def _read_json(path: Path) -> Any:
    """Read a JSON file, raising DataLoadError with a clear message on failure."""
    try:
        with path.open(encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as exc:
        raise DataLoadError(f"Data file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DataLoadError(f"Invalid JSON in {path}: {exc}") from exc


def _parse_records(records: list[dict], model: type[T], label: str, issues: list[str]) -> list[T]:
    """Validate each record; skip bad ones and log them in `issues`."""
    valid: list[T] = []
    for index, record in enumerate(records):
        try:
            valid.append(model.model_validate(record))
        except ValidationError as exc:
            message = f"{label}[{index}] skipped: {exc.errors()[0]['msg']}"
            logger.warning(message)
            issues.append(message)
    return valid


def _check_duplicate_ids(items: list, id_attr: str, label: str, issues: list[str]) -> None:
    seen: set[str] = set()
    for item in items:
        value = getattr(item, id_attr)
        if value in seen:
            issues.append(f"Duplicate {label} id: {value}")
        seen.add(value)


def _check_city_coverage(store: DataStore) -> None:
    """Flag cities used in data that have no coordinates."""
    used = {f.origin for f in store.flights} | {f.destination for f in store.flights}
    used |= {h.city for h in store.hotels} | {p.city for p in store.places}
    for city in sorted(used - set(store.cities)):
        store.issues.append(f"City '{city}' has no coordinates in cities.json")


@lru_cache(maxsize=1)
def load_all(data_dir: Path = DATA_DIR) -> DataStore:
    """Load and validate every dataset. Cached after the first call."""
    store = DataStore()
    store.flights = _parse_records(_read_json(data_dir / "flights.json"), Flight, "flight", store.issues)
    store.hotels = _parse_records(_read_json(data_dir / "hotels.json"), Hotel, "hotel", store.issues)
    store.places = _parse_records(_read_json(data_dir / "places.json"), Place, "place", store.issues)

    raw_cities = _read_json(data_dir / "cities.json")
    store.cities = {normalize_city(name): City.model_validate(value) for name, value in raw_cities.items()}

    _check_duplicate_ids(store.flights, "flight_id", "flight", store.issues)
    _check_duplicate_ids(store.hotels, "hotel_id", "hotel", store.issues)
    _check_duplicate_ids(store.places, "place_id", "place", store.issues)
    _check_city_coverage(store)
    return store


def available_routes(store: DataStore) -> set[tuple[str, str]]:
    """Distinct (origin, destination) pairs that have at least one flight."""
    return {(f.origin, f.destination) for f in store.flights}


def print_report() -> None:
    """Print a data-quality summary."""
    store = load_all()
    cities = sorted(store.cities)
    routes = available_routes(store)
    possible = len(cities) * (len(cities) - 1)
    years = sorted({f.departure_time.year for f in store.flights})

    print(f"Flights: {len(store.flights)} | Hotels: {len(store.hotels)} | Places: {len(store.places)}")
    print(f"Cities ({len(cities)}): {', '.join(cities)}")
    print(f"Flight routes covered: {len(routes)} of {possible} possible directed pairs")
    print(f"Flight years in data: {years}")
    print(f"Issues: {len(store.issues)}")
    for issue in store.issues:
        print(f"  - {issue}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print_report()
