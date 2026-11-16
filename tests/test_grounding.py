"""Tests that quick-answer text is checked against the dataset."""

import pytest

from agent.grounding import ground_answer, places_listing, unknown_place_names
from data_loader import load_all
from tools.flight_tool import search_flights
from tools.hotel_tool import search_hotels
from tools.places_tool import search_places
from ui_helpers import results_to_tables

STORE = load_all()


def places_result(city="Kolkata", **kwargs):
    return [{"tool": "search_places", "args": {"city": city, **kwargs}, "data": search_places(city, **kwargs)}]


KOLKATA_FOUR_OF_FIVE = (
    "Here are a few top-rated attractions in Kolkata:\n\n"
    "- **Famous Fort** \u2013 beach | rating 4.9\n"
    "- **Historic Temple** \u2013 temple | rating 4.9\n"
    "- **Scenic Museum** \u2013 museum | rating 4.5\n"
    "- **Famous Temple** \u2013 market | rating 4.1\n\n"
    "Let me know if you'd like more."
)


# ------------------------------------------------------------- invented names ----
@pytest.mark.parametrize(
    "text,expected",
    [
        ("Don't miss the Red Fort and Calangute Beach.", ["Red Fort", "Calangute Beach"]),
        ("Visit Victoria Memorial in the evening.", ["Victoria Memorial"]),
        ("The Basilica of Bom Jesus is lovely", []),  # no capitalised-word + noun pair, so not detected
        ("Try the **Famous Fort** (beach) and the Beautiful Park.", []),  # dataset places
        ("The Famous Fort is rated 4.9.", []),  # leading 'The' is ignored
        ("Stay at Grand Palace Hotel or Sea View Resort.", []),  # hotel names containing place words
        ("Fly IndiGo from Goa to Bangalore.", []),
        ("Best Market for souvenirs", []),  # 'Best' is praise, not part of a name
        ("The Market opens early. Top Beach views.", []),  # bare nouns after filler words are not names
        ("Try Jaipur City Palace tonight.", ["City Palace"]),
    ],
)
def test_unknown_place_names(text, expected):
    assert unknown_place_names(text, STORE) == expected


# -------------------------------------------------------------- completeness ----
def test_a_complete_listing_is_left_alone():
    results = places_result()
    complete = places_listing([r["data"] for r in results])
    answer, note = ground_answer(complete, results, STORE)
    assert answer == complete and note == ""


def test_your_kolkata_screenshot_the_incomplete_list_is_replaced_with_all_five():
    answer, note = ground_answer(KOLKATA_FOUR_OF_FIVE, places_result(), STORE)
    assert "Here are all 5 places in Kolkata from the dataset" in answer
    assert answer.count("- **") == 5
    assert "Scenic Museum** (temple) \u2013 rating 4.9" in answer  # the one the model had left out
    assert "Scenic Museum** (museum) \u2013 rating 4.5" in answer
    assert "incomplete place list" in note


def test_a_targeted_answer_naming_one_place_is_left_alone():
    text = "The highest-rated place in Kolkata is Famous Fort (beach) at 4.9."
    answer, note = ground_answer(text, places_result(), STORE)
    assert answer == text and note == ""


def test_a_filtered_result_is_compared_with_what_was_returned():
    results = places_result(place_type="temple")  # returns the two temples
    both = "Temples in Kolkata: Historic Temple (4.9) and Scenic Museum (4.9)."
    assert ground_answer(both, results, STORE)[1] == ""
    assert ground_answer("Temples: Historic Temple only.", results, STORE)[1] == ""  # one named = targeted


# --------------------------------------------------------------- invented text ----
def test_an_invented_landmark_in_a_sentence_is_removed_and_the_rest_kept():
    text = "Famous Fort is the top-rated place at 4.9. You could also see the Red Fort nearby. Enjoy!"
    answer, note = ground_answer(text, places_result(), STORE)
    assert "Red Fort" not in answer and "Famous Fort is the top-rated place at 4.9." in answer
    assert "Red Fort" in note


def test_an_invented_bullet_breaks_the_list_so_the_full_dataset_list_is_used():
    text = (
        "Top places in Kolkata:\n- **Famous Fort** (beach) 4.9\n- **Historic Temple** (temple) 4.9\n"
        "- **Victoria Memorial** (monument) 4.8"
    )
    answer, note = ground_answer(text, places_result(), STORE)
    assert "Victoria Memorial" not in answer and answer.count("- **") == 5
    assert "not in the dataset" in note


def test_invented_places_in_a_non_place_answer_are_removed_without_a_replacement_list():
    results = [{"tool": "search_hotels", "args": {}, "data": search_hotels("Goa", max_results=1)}]
    text = "The cheapest Goa hotel is Royal Heritage at \u20b91,232 a night. It is close to Calangute Beach."
    answer, note = ground_answer(text, results, STORE)
    assert answer == "The cheapest Goa hotel is Royal Heritage at \u20b91,232 a night."
    assert "Calangute Beach" in note


def test_empty_after_removal_falls_back_gracefully():
    answer, _ = ground_answer("Visit the Red Fort.", [], STORE)
    assert answer == "Here is what I found in the dataset."


def test_clean_answers_pass_through_untouched():
    flights = [{"tool": "search_flights", "args": {}, "data": search_flights("Bangalore", "Goa")}]
    text = "SpiceJet is cheapest at \u20b95,356."
    assert ground_answer(text, flights, STORE) == (text, "")
    assert ground_answer(text, [], STORE) == (text, "")


# ------------------------------------------------------------------ the listing ----
def test_listing_states_when_only_some_places_are_shown():
    results = places_result("Kolkata", max_results=3)
    text = places_listing([r["data"] for r in results])
    assert "Here are 3 of 5 places in Kolkata" in text and text.count("- **") == 3


def test_listing_explains_an_unavailable_type():
    text = places_listing([r["data"] for r in places_result("Goa", place_type="temple")])
    assert "No 'temple' places in Goa" in text


def test_tables_say_when_they_are_truncated():
    title, table, note = results_to_tables(places_result("Kolkata", max_results=3))[0]
    assert len(table) == 3 and note.startswith("Showing 3 of 5.")
    assert not results_to_tables(places_result("Kolkata"))[0][2].startswith("Showing")  # complete table: no note


def test_two_places_sharing_a_name_must_both_be_listed():
    both = (
        "- Famous Fort (beach) 4.9\n- Historic Temple (temple) 4.9\n- Scenic Museum (temple) 4.9\n"
        "- Scenic Museum (museum) 4.5\n- Famous Temple (market) 4.1"
    )
    assert ground_answer(both, places_result(), STORE) == (both, "")  # all five present: untouched
    dropped_one_duplicate = both.replace("- Scenic Museum (temple) 4.9\n", "")
    answer, note = ground_answer(dropped_one_duplicate, places_result(), STORE)
    assert answer.count("- **") == 5 and "incomplete place list" in note
