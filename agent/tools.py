"""
REST Countries v3.1 API wrapper — httpx 0.28.1

Uses httpx.Timeout with separate connect / read / write / pool values
instead of a single scalar — the recommended approach in httpx 0.20+.

Contract: always returns (data | None, error_str | None).
Never raises — all exceptions are caught and converted to error strings
so the fetch_node can handle every failure mode uniformly.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)

# Granular timeout — connect can be tight; read needs headroom for slow upstreams
_TIMEOUT = httpx.Timeout(
    connect=settings.countries_connect_timeout,
    read=settings.countries_read_timeout,
    write=5.0,
    pool=2.0,
)

# Explicit allowlist — only keys we understand are forwarded downstream
_RAW_FIELD_MAP: dict[str, str] = {
    "capital":    "capital",
    "population": "population",
    "currencies": "currencies",
    "languages":  "languages",
    "region":     "region",
    "subregion":  "subregion",
    "area":       "area",
    "flags":      "flags",
    "timezones":  "timezones",
    "continents": "continents",
    "demonyms":   "demonyms",
    "independent":"independent",
    "borders":    "borders",
    "idd":        "idd",
    "tld":        "tld",
    "cca2":       "cca2",
    "cca3":       "cca3",
    "unMember":   "un_member",
    "latlng":     "latlng",
}


def _normalise(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten one REST Countries v3.1 entry into a flat, typed dict.
    All downstream nodes depend on this schema — changes here are the
    single source of truth for what country data looks like in the graph.
    """
    out: dict[str, Any] = {}

    for api_key, our_key in _RAW_FIELD_MAP.items():
        if api_key in raw:
            out[our_key] = raw[api_key]

    # ── Name ───────────────────────────────────────────────────────────────
    name_obj             = raw.get("name", {})
    out["common_name"]   = name_obj.get("common", "")
    out["official_name"] = name_obj.get("official", "")
    out["native_names"]  = name_obj.get("nativeName", {})

    # ── Currencies → [{"code": "USD", "name": "US dollar", "symbol": "$"}]
    out["currency_list"] = [
        {"code": code, **details}
        for code, details in raw.get("currencies", {}).items()
    ]

    # ── Languages → ["English", "French", …]
    out["language_list"] = list(raw.get("languages", {}).values())

    # ── Calling codes → ["+1", "+44", …]
    idd      = raw.get("idd", {})
    root     = idd.get("root", "")
    suffixes = idd.get("suffixes", [])
    out["calling_codes"] = (
        [f"{root}{s}" for s in suffixes] if suffixes else ([root] if root else [])
    )

    # ── Flags
    flags             = raw.get("flags", {})
    out["flag_emoji"] = raw.get("flag", "")
    out["flag_svg"]   = flags.get("svg", "")
    out["flag_png"]   = flags.get("png", "")

    return out


async def fetch_country(
    country_name: str,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """
    Fetch and normalise country data by name from REST Countries API.

    Returns
    -------
    (data, None)   — success
    (None, message)— any failure; caller decides how to surface it
    """
    url = f"{settings.countries_base_url}/name/{country_name.strip()}"

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            response = await client.get(url, params={"fullText": "false"})

        if response.status_code == 404:
            return None, f"No country found matching '{country_name}'."

        if response.status_code != 200:
            return None, f"REST Countries API returned HTTP {response.status_code}."

        payload = response.json()
        if not isinstance(payload, list) or not payload:
            return None, f"Empty response for '{country_name}'."

        # Prefer exact common-name match; fall back to first result
        query_lower = country_name.lower()
        best = next(
            (
                entry for entry in payload
                if entry.get("name", {}).get("common", "").lower() == query_lower
            ),
            payload[0],
        )
        return _normalise(best), None

    except httpx.TimeoutException:
        logger.warning("Timeout fetching country '%s'", country_name)
        return None, "The REST Countries API did not respond in time. Please try again."

    except httpx.RequestError as exc:
        logger.error("Network error fetching '%s': %s", country_name, exc)
        return None, "A network error occurred while contacting the REST Countries API."

    except Exception as exc:
        logger.exception("Unexpected error fetching '%s'", country_name)
        return None, f"Unexpected error: {exc}"
