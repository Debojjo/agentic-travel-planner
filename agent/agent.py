"""The travel planning agent.

Pipeline:
  1. extract_request - free text -> TripRequest (skipped when a form supplies it)
  2. normalize_request - validate; ask the user for missing or invalid details
  3. research agent   - a LangChain tool-calling agent decides which tools to call
  4. draft_plan       - the LLM names the chosen flight/hotel/place IDs and reasons
  5. build_plan       - Python rebuilds everything from the datasets and recomputes the budget
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

import groq
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel

from agent.formatter import format_itinerary
from agent.grounding import ground_answer
from agent.planner import build_plan, describe_request, normalize_request, tidy_text
from agent.prompts import DRAFT_INSTRUCTION, DRAFT_SYSTEM, extraction_prompt, lookup_prompt, research_prompt
from agent.schemas import PlanDraft, TripPlan, TripRequest
from config import GROQ_API_KEY, GROQ_MODEL, GROQ_REASONING_EFFORT, LLM_TEMPERATURE
from data_loader import DataStore, load_all
from tools import get_lookup_tools, get_tools

logger = logging.getLogger(__name__)

RECURSION_LIMIT = 30  # max agent steps (each tool round trip uses 2)
STRUCTURED_METHODS = ("function_calling", "json_schema")
STRUCTURED_ATTEMPTS = 2  # attempts per structured-output method
MODEL_RETRIES = 4  # retries per model call inside the agent loop
RETRY_INITIAL_DELAY = 1.0  # seconds; doubles after each retry

ProgressCallback = Callable[[str], None]


class AgentError(Exception):
    """Raised when the LLM or agent fails in a way the user should be told about."""


@dataclass
class PlanResult:
    """Outcome of plan_trip."""

    status: str  # "ok" | "needs_info" | "error"
    kind: str = "plan"  # "plan" (a full itinerary) | "answer" (a quick question) | "other" (not about travel)
    message: str = ""
    request: TripRequest | None = None
    plan: TripPlan | None = None
    text: str = ""
    json_text: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)  # seconds per phase
    structured_log: list[str] = field(default_factory=list)  # one line per structured-output attempt
    answer: str = ""  # the written answer for kind == "answer"
    tool_results: list[dict[str, Any]] = field(default_factory=list)  # parsed tool outputs, for tables
    grounding_note: str = ""  # what, if anything, was changed in the model's reply to match the dataset


def should_retry(exc: Exception) -> bool:
    """Which Groq failures are worth retrying.

    gpt-oss occasionally emits a malformed tool call, which Groq rejects with a 400
    'tool_use_failed'. A fresh attempt almost always succeeds.
    """
    if isinstance(exc, groq.BadRequestError):
        text = str(exc)
        return "tool_use_failed" in text or "Failed to parse tool call" in text
    if isinstance(exc, (groq.RateLimitError, groq.APIConnectionError, groq.APITimeoutError)):
        return True
    if isinstance(exc, groq.APIStatusError):
        return exc.status_code >= 500
    return False


def build_llm() -> ChatGroq:
    """Create the Groq chat model."""
    if not GROQ_API_KEY:
        raise AgentError("GROQ_API_KEY is missing. Add it to your .env file.")
    options: dict[str, Any] = {}
    if GROQ_REASONING_EFFORT and GROQ_MODEL.startswith("openai/gpt-oss"):
        options["reasoning_effort"] = GROQ_REASONING_EFFORT  # only gpt-oss models accept this
    return ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=LLM_TEMPERATURE, max_retries=2, **options)


def _emit(callback: ProgressCallback | None, message: str) -> None:
    """Send a progress message; a failing UI callback must never break planning."""
    if callback is None:
        return
    try:
        callback(message)
    except Exception:  # noqa: BLE001
        logger.debug("Progress callback failed", exc_info=True)


def describe_call(call: dict) -> str:
    """Plain-language label for a tool call, shown to the user while planning."""
    name, args = call.get("name", ""), call.get("args") or {}
    if name == "search_flights":
        return f"Searching flights {args.get('origin', '?')} \u2192 {args.get('destination', '?')}"
    if name == "search_hotels":
        return f"Finding hotels in {args.get('city', '?')}"
    if name == "search_places":
        return f"Discovering places in {args.get('city', '?')}"
    if name == "get_weather_forecast":
        return f"Checking the weather in {args.get('city', '?')}"
    if name == "estimate_budget":
        return "Calculating the budget"
    if name == "list_flight_routes":
        return f"Looking up flight routes for {args.get('city', '?')}"
    if name == "hotel_amenities":
        return f"Checking hotel amenities in {args.get('city', '?')}"
    if name == "compare_hotels":
        return f"Comparing hotels in {args.get('city', '?')}"
    if name == "compare_flights":
        return f"Comparing flights {args.get('origin', '?')} \u2192 {args.get('destination', '?')}"
    return f"Running {name}"


def _structured_call(llm: Any, schema: type[BaseModel], messages: list, log: list[str] | None = None) -> BaseModel:
    """Get a validated `schema` instance from the LLM, trying several methods and attempts.

    Every attempt is recorded in `log` (method, outcome, seconds) so slow or flaky calls can be diagnosed.
    """
    errors: list[str] = []
    for method in STRUCTURED_METHODS:
        for attempt in range(1, STRUCTURED_ATTEMPTS + 1):
            began = time.perf_counter()
            label = f"{schema.__name__} via {method}, attempt {attempt}"
            try:
                result = llm.with_structured_output(schema, method=method).invoke(messages)
                if isinstance(result, dict):
                    result = schema.model_validate(result)
                if isinstance(result, schema):
                    if log is not None:
                        log.append(f"{label}: ok ({time.perf_counter() - began:.1f}s)")
                    return result
                errors.append(f"{method}: unexpected result type {type(result).__name__}")
                outcome = "unexpected result type"
            except Exception as exc:  # noqa: BLE001 - any provider/validation failure triggers a retry
                logger.warning("Structured output via %s (attempt %d) failed: %s", method, attempt, exc)
                errors.append(f"{method}: {str(exc)[:120]}")
                outcome = str(exc).replace("\n", " ")[:90]
            if log is not None:
                log.append(f"{label}: FAILED ({time.perf_counter() - began:.1f}s) {outcome}")
    raise AgentError("The model could not produce structured output. " + " | ".join(errors)[:400])


def extract_request(llm: Any, query: str, today: date, log: list[str] | None = None) -> TripRequest:
    """Turn a free-text query into a TripRequest."""
    messages = [SystemMessage(content=extraction_prompt(today)), HumanMessage(content=query)]
    return _structured_call(llm, TripRequest, messages, log)


def _run_agent(
    llm: Any,
    tools: list,
    system_prompt: str,
    user_text: str,
    on_progress: ProgressCallback | None = None,
) -> tuple[list, list[dict]]:
    """Run a tool-calling agent. Returns (messages, tool-call trace)."""
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        middleware=[
            ModelRetryMiddleware(
                max_retries=MODEL_RETRIES,
                retry_on=should_retry,
                on_failure="error",
                initial_delay=RETRY_INITIAL_DELAY,
            )
        ],
    )
    final_state: dict | None = None
    announced: set[int] = set()
    for state in agent.stream(
        {"messages": [HumanMessage(content=user_text)]},
        config={"recursion_limit": RECURSION_LIMIT},
        stream_mode="values",
    ):
        final_state = state
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls and id(last) not in announced:
            announced.add(id(last))
            for call in last.tool_calls:
                _emit(on_progress, describe_call(call))

    messages = (final_state or {}).get("messages", [])
    steps = [
        {"tool": call["name"], "args": call["args"]}
        for message in messages
        if isinstance(message, AIMessage)
        for call in (message.tool_calls or [])
    ]
    return messages, steps


def run_research(
    llm: Any,
    request: TripRequest,
    store: DataStore,
    today: date,
    on_progress: ProgressCallback | None = None,
) -> tuple[list, list[dict]]:
    """Run the trip-planning agent for a validated request."""
    return _run_agent(
        llm, get_tools(), research_prompt(today, sorted(store.cities)), describe_request(request), on_progress
    )


def extract_tool_results(messages: list) -> list[dict[str, Any]]:
    """Parsed tool outputs ({tool, args, data}) in call order, used to render result tables."""
    args_by_call = {
        call.get("id"): call.get("args") or {}
        for message in messages
        if isinstance(message, AIMessage)
        for call in (message.tool_calls or [])
    }
    results = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        try:
            data = json.loads(message.content)
        except (TypeError, ValueError):
            continue
        if isinstance(data, dict):
            results.append({"tool": message.name, "args": args_by_call.get(message.tool_call_id, {}), "data": data})
    return results


def final_answer(messages: list) -> str:
    """The agent's closing message as plain text."""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            content = message.content
            if isinstance(content, list):
                content = " ".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
            return tidy_text(str(content))
    return ""


def answer_question(
    llm: Any,
    query: str,
    store: DataStore,
    today: date,
    on_progress: ProgressCallback | None = None,
) -> PlanResult:
    """Answer a quick question (flights, hotel prices, places, weather) with the tools."""
    began = time.perf_counter()
    messages, steps = _run_agent(
        llm, get_lookup_tools(), lookup_prompt(today, sorted(store.cities)), query, on_progress
    )
    answer = final_answer(messages)
    if not answer:
        return PlanResult(status="error", message="I could not put together an answer. Please try rephrasing.")
    tool_results = extract_tool_results(messages)
    answer, grounding_note = ground_answer(answer, tool_results, store)
    return PlanResult(
        status="ok",
        kind="answer",
        answer=answer,
        text=answer,
        tool_results=tool_results,
        grounding_note=grounding_note,
        steps=steps,
        timings={"answer": round(time.perf_counter() - began, 1)},
    )


def capabilities_message(store: DataStore) -> str:
    """What the assistant can do, shown when a message is not about travel."""
    return (
        "I can plan trips and answer questions about flights, hotels, places and weather in "
        + ", ".join(sorted(store.cities))
        + ". Try: 'Cheapest hotels in Goa', 'Flights from Bangalore to Goa', 'Things to do in Jaipur', "
        "or 'Plan a 3-day trip from Bangalore to Goa starting 5 December, budget 25000'."
    )


def extract_weather(messages: list, request: TripRequest) -> dict | None:
    """Reuse the weather result the agent already fetched, if it matches this trip."""
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and message.name == "get_weather_forecast":
            try:
                data = json.loads(message.content)
            except (TypeError, ValueError):
                continue
            if (
                isinstance(data, dict)
                and "error" not in data
                and data.get("city") == request.destination
                and data.get("start_date") == request.start_date
                and data.get("end_date") == request.end_date
            ):
                return data
    return None


def draft_plan(llm: Any, messages: list, log: list[str] | None = None) -> PlanDraft:
    """Ask the LLM to state its final choices as dataset IDs."""
    prompt = [SystemMessage(content=DRAFT_SYSTEM), *messages, HumanMessage(content=DRAFT_INSTRUCTION)]
    return _structured_call(llm, PlanDraft, prompt, log)


def _friendly_error(exc: Exception) -> str:
    text = str(exc)
    lowered = text.lower()
    if "tool_use_failed" in lowered or "failed to parse tool call" in lowered:
        return "The model kept producing malformed tool calls. Please try again."
    if "rate" in lowered and "limit" in lowered:
        return "The Groq rate limit was reached. Wait a minute and try again."
    if "model" in lowered and ("not exist" in lowered or "not found" in lowered):
        return "The configured Groq model is not available to your account. Check GROQ_MODEL in .env."
    if "api key" in lowered or "401" in lowered or "authentication" in lowered:
        return "Groq rejected the API key. Check GROQ_API_KEY in .env."
    return f"Planning failed: {text[:300]}"


@contextmanager
def _timed(timings: dict[str, float], name: str):
    """Record how long the enclosed block took, in seconds, under `name`."""
    began = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = time.perf_counter() - began


def _understand(
    llm: Any,
    query: str | None,
    store: DataStore,
    today: date,
    on_progress: ProgressCallback | None,
    timings: dict[str, float],
    structured_log: list[str],
) -> tuple[TripRequest | None, PlanResult | None]:
    """Turn free text into a TripRequest, or finish early. Returns (request, None) or (None, early result)."""
    if not query or not query.strip():
        return None, PlanResult(status="needs_info", message="Tell me where you want to go, from where, and when.")
    _emit(on_progress, "Understanding your request")
    with _timed(timings, "understand_request"):
        request = extract_request(llm, query, today, structured_log)
    if request.intent == "lookup":
        answer = answer_question(llm, query, store, today, on_progress)
        answer.timings = {**{name: round(seconds, 1) for name, seconds in timings.items()}, **answer.timings}
        answer.structured_log = structured_log
        return None, answer
    if request.intent == "other":
        return None, PlanResult(status="needs_info", kind="other", message=capabilities_message(store))
    return request, None


def _plan(
    llm: Any,
    request: TripRequest,
    store: DataStore,
    today: date,
    weather_fn: Callable[[str, str, str], dict] | None,
    on_progress: ProgressCallback | None,
    timings: dict[str, float],
    structured_log: list[str],
) -> PlanResult:
    """Validate a request, then run research, drafting and grounding to produce a full trip plan."""
    clean, problems = normalize_request(request, store, today)
    if problems:
        return PlanResult(
            status="needs_info",
            message=" ".join(problems)
            + " Example: 'Plan a 3-day trip from Bangalore to Goa starting 5 December, budget 25000'.",
            request=request,
        )

    with _timed(timings, "research"):
        messages, steps = run_research(llm, clean, store, today, on_progress)
    _emit(on_progress, "Choosing the best options")
    with _timed(timings, "choose"):
        draft = draft_plan(llm, messages, structured_log)
    _emit(on_progress, "Building your itinerary")
    with _timed(timings, "build"):
        plan = build_plan(clean, draft, store, weather_fn, weather=extract_weather(messages, clean))

    return PlanResult(
        status="ok",
        request=clean,
        plan=plan,
        text=format_itinerary(plan),
        json_text=plan.model_dump_json(indent=2),
        steps=steps,
        timings={name: round(seconds, 1) for name, seconds in timings.items()},
        structured_log=structured_log,
    )


def plan_trip(
    query: str | None = None,
    request: TripRequest | None = None,
    llm: Any = None,
    today: date | None = None,
    store: DataStore | None = None,
    weather_fn: Callable[[str, str, str], dict] | None = None,
    on_progress: ProgressCallback | None = None,
) -> PlanResult:
    """Plan a trip, or answer a quick question, from free text (`query`) or a ready-made `request`.

    Free text is classified first: a trip to plan runs the full planning pipeline, a question about
    flights, hotels, places or weather is answered directly with the tools, and anything else gets a
    short description of what the assistant can do. A `request` (e.g. from the form) always plans.

    `on_progress` receives short plain-language status messages while the agent works.
    """
    store = store or load_all()
    today = today or date.today()
    timings: dict[str, float] = {}
    structured_log: list[str] = []
    try:
        llm = llm or build_llm()
        if request is None:
            request, early_result = _understand(llm, query, store, today, on_progress, timings, structured_log)
            if early_result is not None:
                return early_result
        return _plan(llm, request, store, today, weather_fn, on_progress, timings, structured_log)
    except AgentError as exc:
        return PlanResult(status="error", message=str(exc))
    except Exception as exc:  # noqa: BLE001 - surface any provider failure as a clean message
        logger.exception("Trip planning failed")
        return PlanResult(status="error", message=_friendly_error(exc))
