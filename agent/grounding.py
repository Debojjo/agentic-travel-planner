"""Checks that the written reply in quick answers agrees with the dataset.

The tables under an answer are built from tool results, so they are always correct. The model's
sentence above them is not, so it is checked here:
  * place names that are not in the dataset (e.g. a real landmark the model remembered) are removed;
  * a list of places that leaves some out is replaced by the complete list from the tool result.
"""

from __future__ import annotations

import re
from collections import Counter

from data_loader import DataStore

_PLACE_NOUNS = (
    "Fort|Museum|Temple|Park|Lake|Beach|Market|Monument|Palace|Garden|Gardens|Bazaar|Basilica|Church|"
    "Cathedral|Mosque|Tomb|Gate|Memorial|Gallery|Zoo|Square|Tower|Bridge|Island|Falls|Cave|Caves"
)
# 1-3 capitalised words followed by a place noun, e.g. "Famous Fort", "The Red Fort", "Calangute Beach"
_NAME_PATTERN = re.compile(rf"\b((?:[A-Z][\w'\u2019.-]*\s+){{1,3}}(?:{_PLACE_NOUNS}))\b")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
# Capitalised words that start sentences or praise a place but are not part of its name.
_FILLER = {
    "the",
    "a",
    "an",
    "this",
    "that",
    "these",
    "those",
    "our",
    "your",
    "its",
    "visit",
    "see",
    "try",
    "explore",
    "top",
    "best",
    "here",
    "there",
    "in",
    "at",
    "on",
    "near",
    "enjoy",
    "also",
    "and",
    "for",
    "of",
    "to",
    "with",
    "from",
    "is",
    "are",
    "one",
    "some",
    "most",
    "other",
    "more",
}


def _known_names(store: DataStore) -> set[str]:
    """Every proper name the dataset contains (places, hotels, airlines, cities)."""
    names = {p.name for p in store.places} | {h.name for h in store.hotels}
    names |= {f.airline for f in store.flights} | set(store.cities)
    return names


def unknown_place_names(text: str, store: DataStore) -> list[str]:
    """Place-like names in `text` (e.g. 'Red Fort', 'Calangute Beach') that are not in the dataset."""
    known = _known_names(store)
    unknown: list[str] = []
    for match in _NAME_PATTERN.finditer(text):
        words = match.group(1).split()
        while len(words) > 1 and words[0].lower() in _FILLER:
            words.pop(0)  # 'The Red Fort' -> 'Red Fort'
        if len(words) < 2:
            continue  # only a bare noun such as 'The Market': not a name
        suffixes = [" ".join(words[i:]) for i in range(len(words) - 1)]
        if any(any(suffix in name for name in known) for suffix in suffixes):
            continue
        name = " ".join(words[-2:])
        if name not in unknown:
            unknown.append(name)
    return unknown


def _drop_segments_with(text: str, bad_names: list[str]) -> str:
    """Remove every sentence or bullet line that mentions one of `bad_names`."""
    kept_lines = []
    for line in text.split("\n"):
        parts = _SENTENCE_SPLIT.split(line)
        kept = [part for part in parts if not any(name in part for name in bad_names)]
        if kept:
            kept_lines.append(" ".join(kept))
        elif not line.strip():
            kept_lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept_lines)).strip()


def _place_results(tool_results: list[dict]) -> list[dict]:
    """Successful search_places outputs that actually returned places."""
    return [
        result["data"]
        for result in tool_results
        if result.get("tool") == "search_places" and result["data"].get("places")
    ]


def _listing_is_incomplete(text: str, data: dict) -> bool:
    """True if the text lists two or more of the returned places but leaves some out.

    Counts items, not just names: two different places can share a name (Kolkata has two 'Scenic Museum'
    entries), and a list that mentions that name once has dropped one of them.
    """
    wanted = Counter(p["name"] for p in data["places"])
    if sum(1 for name in wanted if name in text) < 2:
        return False  # a targeted answer about one place is not a list
    return any(text.count(name) < count for name, count in wanted.items())


def places_listing(place_results: list[dict]) -> str:
    """The complete list of returned places, written from the data (never by the model)."""
    blocks = []
    for data in place_results:
        shown, total = len(data["places"]), data.get("total_found", len(data["places"]))
        count = f"all {shown}" if shown == total else f"{shown} of {total}"
        lines = [f"Here are {count} places in {data['city']} from the dataset, best rated first:"]
        lines += [f"- **{p['name']}** ({p['type']}) \u2013 rating {p['rating']}" for p in data["places"]]
        if data.get("requested_type_unavailable"):
            lines.append(f"_{data['note']}_")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def ground_answer(answer: str, tool_results: list[dict], store: DataStore) -> tuple[str, str]:
    """Return (checked answer, note). The note is empty when the model's reply needed no change."""
    notes: list[str] = []
    unknown = unknown_place_names(answer, store)
    if unknown:
        answer = _drop_segments_with(answer, unknown)
        notes.append("removed mentions of places that are not in the dataset (" + ", ".join(unknown) + ")")

    place_results = _place_results(tool_results)
    if place_results and (not answer.strip() or any(_listing_is_incomplete(answer, d) for d in place_results)):
        answer = places_listing(place_results)
        notes.append("replaced an incomplete place list with the full list from the dataset")
    if not answer.strip():
        answer = "Here is what I found in the dataset."
    return answer, "Reply checked against the dataset: " + "; ".join(notes) + "." if notes else ""
