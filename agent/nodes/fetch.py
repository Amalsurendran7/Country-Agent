"""
Node 2 — Tool invocation (REST Countries API fetch).

Thin delegation layer: calls fetch_country() from agent.tools and stores
the result in state. Keeping API logic in tools.py means this node stays
easy to swap out (mock, cache, retry-decorator) without touching the graph.
"""

from __future__ import annotations

import logging

from agent.state import CountryAgentState
from agent.tools import fetch_country

logger = logging.getLogger(__name__)


async def fetch_node(state: CountryAgentState) -> CountryAgentState:
    """
    Call REST Countries API → raw_country_data.
    Writes fetch_error on any failure.
    """
    country_name = state.get("country_name", "").strip()

    if not country_name:
        return {**state, "fetch_error": "No country name available to look up."}

    data, error = await fetch_country(country_name)

    if error:
        logger.warning("Fetch failed for '%s': %s", country_name, error)
        return {**state, "raw_country_data": None, "fetch_error": error}

    logger.info("Fetched data for '%s'", data.get("common_name", country_name))
    return {**state, "raw_country_data": data, "fetch_error": None}
