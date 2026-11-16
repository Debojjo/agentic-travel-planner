"""Tests for data_loader."""

import pytest

from data_loader import available_routes, load_all, normalize_city


@pytest.fixture(scope="module")
def store():
    return load_all()


def test_counts(store):
    assert len(store.flights) == 30
    assert len(store.hotels) == 40
    assert len(store.places) == 40


def test_no_load_issues(store):
    assert store.issues == []


def test_every_city_has_coordinates(store):
    used = {h.city for h in store.hotels} | {p.city for p in store.places}
    assert used <= set(store.cities)


def test_flight_durations_positive(store):
    assert all(f.duration_minutes > 0 for f in store.flights)


def test_ids_unique(store):
    assert len({f.flight_id for f in store.flights}) == len(store.flights)
    assert len({h.hotel_id for h in store.hotels}) == len(store.hotels)
    assert len({p.place_id for p in store.places}) == len(store.places)


def test_known_route_exists(store):
    assert ("Hyderabad", "Goa") in available_routes(store)


@pytest.mark.parametrize("raw,expected", [("bengaluru", "Bangalore"), (" goa ", "Goa"), ("Calcutta", "Kolkata")])
def test_normalize_city(raw, expected):
    assert normalize_city(raw) == expected
