"""
LangGraph agent state — langgraph 1.1.3 style.

TypedDict is the canonical state container for LangGraph StateGraph.
Annotated[list[X], operator.add] declares a merge reducer: if two nodes
both write to the same list field, LangGraph concatenates the values
rather than overwriting, making the schema forward-compatible with
parallel branches.

All fields are optional (total=False) so nodes only need to return the
keys they actually modify — partial updates, not full state replacement.
"""

from __future__ import annotations

import operator
from typing import Any, Optional
from typing_extensions import Annotated, TypedDict


class CountryAgentState(TypedDict, total=False):
    # ── Input ──────────────────────────────────────────────────────────────
    user_query: str

    # ── Intent node output ─────────────────────────────────────────────────
    country_name: str
    requested_fields: Annotated[list[str], operator.add]  # merge-safe
    intent_error: Optional[str]

    # ── Fetch node output ──────────────────────────────────────────────────
    raw_country_data: Optional[dict[str, Any]]
    fetch_error: Optional[str]

    # ── Synthesis node output ──────────────────────────────────────────────
    answer: str
    sources: Annotated[list[str], operator.add]           # merge-safe
