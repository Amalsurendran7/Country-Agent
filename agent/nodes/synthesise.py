"""
Node 3 — Answer synthesis.

Uses the native Together.ai SDK (AsyncTogether) directly — no langchain wrapper.

The LLM is given only the data fetched from REST Countries via a system prompt
that explicitly forbids using training knowledge. Plain text response — no
structured output needed here since the answer is free-form prose.

Pattern mirrors the SDK example:
    response = await client.chat.completions.create(model=..., messages=[...])
    answer = response.choices[0].message.content
"""

from __future__ import annotations

import json
import logging
from typing import Any

from together import AsyncTogether

from config.settings import settings
from agent.state import CountryAgentState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are the answer-generation component of a country information service.\n"
    "Write a concise, factual answer using ONLY the JSON data supplied.\n"
    "Do NOT use any country knowledge from your training weights.\n"
    "If a requested field is absent from the JSON, say so plainly.\n"
    "Use plain prose. Avoid bullet points unless listing multiple items.\n"
    "Stay under 120 words unless many fields were requested.\n"
    "Do not open with 'Based on the data' or similar meta-phrases."
)

# ── Client ─────────────────────────────────────────────────────────────────

_client = AsyncTogether(api_key=settings.together_api_key)

# ── Field → data key mapping ───────────────────────────────────────────────

_FIELD_KEYS: dict[str, list[str]] = {
    "capital":      ["capital"],
    "population":   ["population"],
    "currency":     ["currency_list"],
    "currencies":   ["currency_list"],
    "language":     ["language_list"],
    "languages":    ["language_list"],
    "region":       ["region", "subregion"],
    "subregion":    ["subregion"],
    "area":         ["area"],
    "flag":         ["flag_emoji", "flag_svg"],
    "timezone":     ["timezones"],
    "timezones":    ["timezones"],
    "continent":    ["continents"],
    "demonym":      ["demonyms"],
    "independence": ["independent"],
    "borders":      ["borders"],
    "calling_code": ["calling_codes"],
    "tld":          ["tld"],
}


def _build_context(data: dict[str, Any], fields: list[str]) -> str:
    """Minimal JSON containing only the fields the user asked about."""
    ctx: dict[str, Any] = {
        "common_name":   data.get("common_name"),
        "official_name": data.get("official_name"),
    }
    for field in fields:
        for key in _FIELD_KEYS.get(field, []):
            if key in data:
                ctx[key] = data[key]
    return json.dumps(ctx, ensure_ascii=False)


# ── Node ───────────────────────────────────────────────────────────────────

async def synthesise_node(state: CountryAgentState) -> CountryAgentState:
    """Produce a grounded natural-language answer from raw_country_data."""
    data   = state.get("raw_country_data") or {}
    fields = state.get("requested_fields") or []
    query  = state.get("user_query", "")

    user_message = (
        f"User question: {query}\n\n"
        f"Fields requested: {', '.join(fields)}\n\n"
        f"Country data (JSON):\n{_build_context(data, fields)}"
    )

    try:
        response = await _client.chat.completions.create(
            model=settings.synthesis_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=0,
            max_tokens=512,
        )
        answer = response.choices[0].message.content.strip()

    except Exception as exc:
        logger.error("Synthesis node error: %s", exc)
        answer = "Sorry, I was unable to compose an answer at this time."

    common_name = data.get("common_name") or state.get("country_name", "")
    source_url  = f"{settings.countries_base_url}/name/{common_name}"

    logger.info("Synthesis complete for '%s'", common_name)
    return {
        **state,
        "answer":  answer,
        "sources": [f"REST Countries API — {source_url}"],
    }
