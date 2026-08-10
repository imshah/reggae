"""Routing: decide local vs. remote for an `ask`, and expose task types.

Routine questions stay on the local model; heavyweight tasks (gaps, critique,
diagram) always go remote. For `ask`, a lightweight heuristic flags
high-complexity questions for escalation; the caller may also force a route.
"""
from __future__ import annotations

from enum import Enum

# words that suggest an analytical / cross-cutting question worth escalating
_HEAVY_HINTS = (
    "gap",
    "risk",
    "improve",
    "recommend",
    "critique",
    "compare",
    "trade-off",
    "tradeoff",
    "architecture",
    "scalab",
    "bottleneck",
    "single point of failure",
    "why",
    "assess",
    "evaluate",
    "strategy",
    "roadmap",
)


class Route(str, Enum):
    LOCAL = "local"
    REMOTE = "remote"


def route_for_ask(question: str, force: Route | None = None) -> Route:
    if force is not None:
        return force
    q = question.lower()
    if len(question.split()) > 40:
        return Route.REMOTE
    if any(h in q for h in _HEAVY_HINTS):
        return Route.REMOTE
    return Route.LOCAL
