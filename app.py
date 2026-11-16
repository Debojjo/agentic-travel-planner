"""Streamlit UI for the Agentic AI Travel Planner.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pandas as pd
import streamlit as st

import agent as travel_agent
from agent import PlanResult, TripRequest
from agent.formatter import format_inr
from config import GROQ_API_KEY, GROQ_MODEL
from data_loader import load_all
from ui_helpers import AMENITY_OPTIONS, build_preferences, results_to_tables, trip_end_date, weather_emoji

st.set_page_config(page_title="AI Travel Planner", page_icon="\u2708\ufe0f", layout="wide")

store = load_all()
CITIES = sorted(store.cities)
TODAY = date.today()


# ------------------------------------------------------------------ sidebar ----
def render_sidebar() -> None:
    """Show the About panel, dataset coverage and limits, and a warning if the API key is missing."""
    with st.sidebar:
        st.header("About")
        st.write(
            "An agentic AI travel assistant built with LangChain. It plans whole trips or answers quick "
            "questions, deciding for itself which tools to call (flights, hotels, places, weather, budget)."
        )
        st.caption(f"Model: `{GROQ_MODEL}`")
        st.subheader("Coverage")
        st.caption("Cities: " + ", ".join(CITIES))
        st.caption(
            "Flights, hotels and places come from a fixed sample dataset, so not every route exists "
            "(e.g. there is no Delhi to Goa flight). Trips are 2\u20137 days for up to 10 travelers."
        )
        st.caption("Child fares and discounts are not modeled. Two travelers share a hotel room.")
        if not GROQ_API_KEY:
            st.error("GROQ_API_KEY is missing. Add it to your .env file and restart the app.")


# ------------------------------------------------------------------- inputs ----
def run_planning(**kwargs) -> None:
    """Call the agent, showing each step as it happens, and keep the result across reruns."""
    with st.status("Processing your request...", expanded=True) as status:
        result = travel_agent.plan_trip(on_progress=st.write, **kwargs)
        if result.status == "ok":
            ready = "Here is what I found" if result.kind == "answer" else "Your trip is ready"
            status.update(label=ready, state="complete", expanded=False)
        elif result.status == "needs_info" and result.kind == "other":
            status.update(label="I can only help with travel requests", state="complete", expanded=False)
        elif result.status == "needs_info":
            status.update(label="A few more details are needed", state="complete", expanded=False)
        else:
            status.update(label="Could not complete your request", state="error", expanded=False)
    st.session_state["result"] = result


def render_form_tab() -> None:
    """Draw the structured trip form and start planning when the button is pressed."""
    c1, c2 = st.columns(2)
    origin = c1.selectbox("From", CITIES, index=CITIES.index("Bangalore"), key="origin")
    destination = c2.selectbox("To", CITIES, index=CITIES.index("Goa"), key="destination")

    c3, c4, c5 = st.columns(3)
    start = c3.date_input("Start date", value=TODAY + timedelta(days=30), min_value=TODAY, key="start_date")
    length = c4.slider("Trip length (days)", 2, 7, 3, key="length")
    travelers = c5.number_input("Travelers", min_value=1, max_value=10, value=1, key="travelers")
    c3.caption(
        f"Trip: {start:%b} {start.day} \u2013 {trip_end_date(start, length):%b} {trip_end_date(start, length).day}"
    )

    c6, c7 = st.columns(2)
    budget = c6.number_input("Total budget (\u20b9, 0 = no limit)", min_value=0, value=25000, step=1000, key="budget")
    interests = c7.text_input("Interests (optional)", placeholder="beaches, temples, museums", key="interests")

    with st.expander("Filters"):
        f1, f2 = st.columns(2)
        flight_pref = f1.radio("Flights", ["Cheapest", "Fastest"], horizontal=True, key="flight_pref")
        min_stars = f2.select_slider("Minimum hotel stars", options=[1, 2, 3, 4, 5], value=1, key="min_stars")
        amenities = st.multiselect("Hotel must have", AMENITY_OPTIONS, key="amenities")
        t1, t2 = st.columns(2)
        one_way = t1.checkbox("One-way trip (no return flight)", key="one_way")
        no_hotel = t2.checkbox("I don't need a hotel (staying with friends or family)", key="no_hotel")

    same_city = origin == destination
    if same_city:
        st.error("Origin and destination must be different.")
    if st.button("Plan my trip", type="primary", key="plan_form", disabled=same_city or not GROQ_API_KEY):
        request = TripRequest(
            origin=origin,
            destination=destination,
            start_date=start.isoformat(),
            end_date=trip_end_date(start, length).isoformat(),
            travelers=int(travelers),
            budget_limit_inr=float(budget) if budget else None,
            preferences=build_preferences(interests, flight_pref, min_stars, amenities),
            one_way=one_way,
            needs_hotel=not no_hotel,
        )
        run_planning(request=request)


def render_text_tab() -> None:
    """Draw the free-text box for describing a trip or asking a quick question."""
    st.caption("Plan a whole trip, or just ask a question about flights, hotel prices, places or weather.")
    query = st.text_area(
        "Ask or describe your trip",
        placeholder=(
            "e.g. Cheapest hotels in Goa under 3000, or: Plan a 4 day trip from Hyderabad to Goa starting 12 December"
        ),
        height=110,
        key="query",
    )
    if st.button("Go", type="primary", key="plan_text", disabled=not GROQ_API_KEY):
        run_planning(query=query)


# ------------------------------------------------------------------ results ----
def render_flight(label: str, leg, not_needed: bool = False) -> None:
    """Draw one flight card (or a note when the leg is not needed or unavailable)."""
    with st.container(border=True):
        st.markdown(f"**{label}**")
        if not_needed:
            st.caption("Not needed (one-way trip).")
            return
        if leg is None:
            st.caption("No flight available in the dataset for this leg.")
            return
        st.markdown(f"{leg.airline} \u00b7 **{format_inr(leg.price_inr)}**")
        st.caption(f"{leg.origin} \u2192 {leg.destination}")
        st.caption(f"Departs {leg.departure_time[11:]} \u00b7 arrives {leg.arrival_time[11:]} \u00b7 {leg.duration}")


def render_day(day) -> None:
    """Draw one day of the itinerary: date, weather, places and the factual flight note."""
    pretty = date.fromisoformat(day.date)
    with st.container(border=True):
        st.markdown(f"**Day {day.day_number}** \u00b7 {pretty:%a}, {pretty:%b} {pretty.day}")
        if day.weather:
            text = f"{weather_emoji(day.weather)} {day.weather}, {day.temp_min_c:.0f}\u2013{day.temp_max_c:.0f}\u00b0C"
            if day.rain_expected:
                text += " \u00b7 rain likely"
            st.write(text)
            if day.weather_source == "historical_estimate":
                st.caption("Typical weather from last year (beyond forecast range)")
        else:
            st.caption("Weather not available")
        if day.places:
            for place in day.places:
                st.markdown(f"- {place.name} _({place.type})_ \u2b50 {place.rating}")
        elif not day.note:
            st.caption("Leisure day")
        if day.note:
            st.caption(day.note)


def render_agent_trace(result: PlanResult) -> None:
    """Expander showing what the agent did: tool calls, timings and any reply adjustments."""
    with st.expander(f"How the agent got here ({len(result.steps)} tool calls)"):
        if result.grounding_note:
            st.caption(result.grounding_note)
        if result.timings:
            st.caption(
                " \u00b7 ".join(f"{name.replace('_', ' ')}: {seconds}s" for name, seconds in result.timings.items())
            )
        for line in result.structured_log:
            st.caption(line)
        for number, step in enumerate(result.steps, start=1):
            st.code(f"{number}. {step['tool']}({json.dumps(step['args'])})", language="text")


def render_answer(result: PlanResult) -> None:
    """Show a quick-question answer: the written reply, then the data it is based on."""
    with st.container(border=True):
        st.markdown(result.answer)
    for title, table, note in results_to_tables(result.tool_results):
        st.subheader(title)
        if table is None:
            st.caption(note)
            continue
        st.dataframe(table, hide_index=True)
        if note:
            st.caption(note)
    render_agent_trace(result)


def render_metrics(plan) -> None:
    """The four headline numbers: total, per person, nights and budget status."""
    budget = plan.budget
    total, per_person, nights, status = st.columns(4)
    total.metric("Total cost", format_inr(budget.total))
    per_person.metric("Per person", format_inr(budget.per_person))
    nights.metric("Nights", len(plan.days) - 1)
    if budget.within_budget is True:
        status.metric("Budget", "Within budget", f"{format_inr(budget.remaining)} left")
    elif budget.within_budget is False:
        status.metric("Budget", "Over budget", f"-{format_inr(budget.over_by)}")
    else:
        status.metric("Budget", "No limit set")


def render_hotel(plan) -> None:
    """The hotel card (or a note when no hotel is needed or available)."""
    with st.container(border=True):
        if not plan.needs_hotel:
            st.caption("Not needed (staying with friends or family).")
        elif plan.hotel:
            hotel = plan.hotel
            star_icons = "\u2b50" * hotel.stars
            st.markdown(f"**{hotel.name}** \u00b7 {star_icons}")
            rooms = f" \u00d7 {hotel.rooms} rooms" if hotel.rooms > 1 else ""
            st.markdown(
                f"{format_inr(hotel.price_per_night_inr)}/night \u00d7 {hotel.nights} nights{rooms} "
                f"= **{format_inr(hotel.total_price_inr)}**"
            )
            st.caption("Amenities: " + ", ".join(hotel.amenities))
        else:
            st.caption("No hotel available.")


def render_budget(budget) -> None:
    """Budget chart and line items, with the assumptions behind them."""
    chart_col, table_col = st.columns([3, 2])
    chart_col.bar_chart(
        pd.Series(
            {"Flights": budget.flights, "Hotel": budget.hotel, "Food & local travel": budget.food_and_local_travel}
        )
    )
    with table_col:
        st.markdown(
            f"- Flights: **{format_inr(budget.flights)}**\n"
            f"- Hotel: **{format_inr(budget.hotel)}**\n"
            f"- Food & local travel: **{format_inr(budget.food_and_local_travel)}**\n"
            f"- **Total: {format_inr(budget.total)}**"
        )
        for assumption in budget.assumptions:
            st.caption(assumption)


def render_reasoning_and_notes(plan) -> None:
    """'Why we chose this' cards and any warnings."""
    if plan.reasoning:
        st.subheader("\U0001f9e0 Why we chose this")
        for key, text in plan.reasoning.items():
            st.info(f"**{key.capitalize()}:** {text}")
    for warning in plan.warnings:
        st.warning(warning)


def render_plan(result: PlanResult) -> None:
    """Show a full trip plan: summary, flights, hotel, days, budget, reasoning and downloads."""
    plan = result.plan
    st.header(plan.title)
    if plan.summary:
        st.write(plan.summary)
    render_metrics(plan)

    left, right = st.columns(2)
    with left:
        st.subheader("\u2708\ufe0f Flights")
        render_flight("Outbound", plan.outbound_flight)
        render_flight("Return", plan.return_flight, not_needed=plan.one_way)
    with right:
        st.subheader("\U0001f3e8 Hotel")
        render_hotel(plan)

    st.subheader("\U0001f5d3\ufe0f Day by day")
    columns = st.columns(min(len(plan.days), 4))
    for index, day in enumerate(plan.days):
        with columns[index % len(columns)]:
            render_day(day)

    st.subheader("\U0001f4b0 Budget")
    render_budget(plan.budget)
    render_reasoning_and_notes(plan)
    render_agent_trace(result)

    json_col, text_col, _ = st.columns([1, 1, 3])
    json_col.download_button("Download JSON", result.json_text, file_name="trip_plan.json", mime="application/json")
    text_col.download_button("Download text", result.text, file_name="trip_plan.md", mime="text/markdown")


def render_result(result: PlanResult) -> None:
    """Show whatever the planner returned: a message, a quick answer or a full trip plan."""
    if result.status == "needs_info":
        (st.info if result.kind == "other" else st.warning)(result.message)
    elif result.status == "error":
        st.error(result.message)
    elif result.kind == "answer":
        render_answer(result)
    else:
        render_plan(result)


# --------------------------------------------------------------------- page ----
def main() -> None:
    """Lay out the page: sidebar, the two input tabs, and the latest result."""
    render_sidebar()
    st.title("\u2708\ufe0f Agentic AI Travel Planner")
    st.caption("Plan a trip with an AI agent that searches flights, hotels, places and weather for you.")

    form_tab, text_tab = st.tabs(["Plan with a form", "Ask or describe your trip"])
    with form_tab:
        render_form_tab()
    with text_tab:
        render_text_tab()

    result = st.session_state.get("result")
    if result is not None:
        st.divider()
        render_result(result)


main()
