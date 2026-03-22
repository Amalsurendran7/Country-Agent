"""
Node 1 — Intent extraction.

Uses the native Together.ai SDK (AsyncTogether) directly — no langchain wrapper.

Structured output strategy:
  Together.ai supports response_format={"type": "json_schema", "json_schema": {...}}
  We pass the Pydantic model's JSON schema so the model is constrained to return
  exactly the shape we need. The response is then parsed by Pydantic — no manual
  JSON wrangling required.

Pattern mirrors the SDK example:
    client = AsyncTogether()
    response = await client.chat.completions.create(
        model=..., messages=[...], response_format={...}
    )
    content = response.choices[0].message.content
"""

from __future__ import annotations

import json
import logging
from typing import Literal

from together import AsyncTogether
from pydantic import BaseModel, Field

from config.settings import settings
from agent.state import CountryAgentState

logger = logging.getLogger(__name__)

# ── Output schema ──────────────────────────────────────────────────────────

FieldName = Literal[
    "capital", "population", "currency", "language",
    "region", "subregion", "area", "flag", "timezone",
    "continent", "demonym", "independence", "borders",
    "calling_code", "tld",
]

_DEFAULT_FIELDS: list[str] = ["capital", "population", "region"]


class IntentOutput(BaseModel):
    """Schema the LLM must return for every intent extraction call."""
    country: str = Field(
        description=(
            "The country name exactly as stated by the user. "
            "Empty string if no country is identifiable."
        )
    )
    fields: list[FieldName] = Field(
        default_factory=list,
        description="Information fields the user is asking about.",
    )


# Build once at import time — schema never changes at runtime
_INTENT_JSON_SCHEMA = IntentOutput.model_json_schema()

_SYSTEM_PROMPT = (
    "You are an intent-parsing component for a country information service. "
    "Extract the country name and the specific data fields the user wants.\n"
    "Rules:\n"
    "- If no country is mentioned, set country to an empty string.\n"
    "- If no specific fields are asked but a country is given, "
    "default fields are: capital, population, region.\n"
    "- Only use the allowed field names defined in the JSON schema."
)

# ── Client ─────────────────────────────────────────────────────────────────

# AsyncTogether is instantiated once — it manages its own connection pool
_client = AsyncTogether(api_key=settings.together_api_key)


# ── Node ───────────────────────────────────────────────────────────────────

async def intent_node(state: CountryAgentState) -> CountryAgentState:
    """
    Parse user_query → country_name + requested_fields.

    Calls Together.ai with json_schema response_format so the model is
    constrained to return valid IntentOutput JSON. Pydantic validates the
    parsed result — no manual field checking needed.
    """
    query = state.get("user_query", "").strip()

    if not query:
        return {**state, "intent_error": "Empty query received."}

    try:
        response = await _client.chat.completions.create(
            model=settings.intent_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": query},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name":   "IntentOutput",
                    "schema": _INTENT_JSON_SCHEMA,
                    "strict": True,
                },
            },
            temperature=0,
            max_tokens=256,
        )

        raw_content = response.choices[0].message.content
        result = IntentOutput.model_validate(json.loads(raw_content))

    except Exception as exc:
        logger.error("Intent node error: %s", exc)
        return {**state, "intent_error": f"Intent extraction failed: {exc}"}

    if not result.country:
        return {
            **state,
            "intent_error": (
                "I could not identify a country in your question. "
                "Please include a country name — for example: "
                "'What is the capital of France?'"
            ),
        }

    resolved_fields: list[str] = list(result.fields) if result.fields else _DEFAULT_FIELDS
    logger.info("Intent — country=%r  fields=%s", result.country, resolved_fields)

    return {
        **state,
        "country_name":     result.country,
        "requested_fields": resolved_fields,
        "intent_error":     None,
    }
