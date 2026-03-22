"""
Test suite — together SDK 2.5.0 native client edition.

All Together.ai API calls and HTTP calls are mocked — no API key required.

Run:  pytest tests/ -v -W error::DeprecationWarning
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── Helpers ────────────────────────────────────────────────────────────────

MOCK_COUNTRY_DATA = {
    "common_name":   "Germany",
    "official_name": "Federal Republic of Germany",
    "capital":       ["Berlin"],
    "population":    84607016,
    "currency_list": [{"code": "EUR", "name": "Euro", "symbol": "€"}],
    "language_list": ["German"],
    "region":        "Europe",
    "subregion":     "Western Europe",
    "area":          357114.0,
    "flag_emoji":    "🇩🇪",
    "flag_svg":      "https://flagcdn.com/de.svg",
    "timezones":     ["UTC+01:00"],
    "continents":    ["Europe"],
    "borders":       ["AUT", "BEL", "CZE", "DNK", "FRA", "LUX", "NLD", "POL", "CHE"],
    "calling_codes": ["+49"],
    "tld":           [".de"],
    "independent":   True,
}


def _together_response(content: str) -> MagicMock:
    """
    Build a mock that matches the Together SDK response shape:
        response.choices[0].message.content
    """
    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    return response


def _http_mock(status: int, payload=None):
    """Mock httpx AsyncClient returning the given HTTP status and JSON payload."""
    resp = MagicMock()
    resp.status_code = status
    if payload is not None:
        resp.json.return_value = payload

    client = AsyncMock()
    client.get = AsyncMock(return_value=resp)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__  = AsyncMock(return_value=False)
    return client


# ── Settings tests ─────────────────────────────────────────────────────────

class TestSettings:
    def test_instantiates_without_api_key(self):
        from config.settings import Settings
        s = Settings(TOGETHER_API_KEY="")
        assert s.together_api_key == ""

    def test_together_api_key_stored(self):
        from config.settings import Settings
        s = Settings(TOGETHER_API_KEY="test-key-123")
        assert s.together_api_key == "test-key-123"

    def test_port_lower_bound(self):
        from config.settings import Settings
        with pytest.raises(Exception):
            Settings(PORT=0)

    def test_port_upper_bound(self):
        from config.settings import Settings
        with pytest.raises(Exception):
            Settings(PORT=99999)

    def test_connect_timeout_must_be_positive(self):
        from config.settings import Settings
        with pytest.raises(Exception):
            Settings(COUNTRIES_CONNECT_TIMEOUT=0)

    def test_read_timeout_must_be_positive(self):
        from config.settings import Settings
        with pytest.raises(Exception):
            Settings(COUNTRIES_READ_TIMEOUT=-1.0)

    def test_llm_max_retries_bounds(self):
        from config.settings import Settings
        with pytest.raises(Exception):
            Settings(LLM_MAX_RETRIES=0)
        with pytest.raises(Exception):
            Settings(LLM_MAX_RETRIES=11)

    def test_default_models_are_llama(self):
        from config.settings import settings
        assert "Llama" in settings.intent_model or "Meta-Llama" in settings.intent_model
        assert "Llama" in settings.synthesis_model or "Meta-Llama" in settings.synthesis_model


# ── Tools tests ────────────────────────────────────────────────────────────

class TestFetchCountry:
    _RAW_GERMANY = [{
        "name": {
            "common": "Germany",
            "official": "Federal Republic of Germany",
            "nativeName": {"deu": {"official": "Bundesrepublik Deutschland"}},
        },
        "capital":    ["Berlin"],
        "population": 84607016,
        "currencies": {"EUR": {"name": "Euro", "symbol": "€"}},
        "languages":  {"deu": "German"},
        "region":     "Europe",
        "subregion":  "Western Europe",
        "area":       357114.0,
        "flags":      {"svg": "https://flagcdn.com/de.svg", "png": ""},
        "flag":       "🇩🇪",
        "timezones":  ["UTC+01:00"],
        "continents": ["Europe"],
        "borders":    ["AUT", "BEL"],
        "idd":        {"root": "+4", "suffixes": ["9"]},
        "tld":        [".de"],
        "independent": True,
    }]

    @pytest.mark.asyncio
    async def test_success_normalises_all_key_fields(self):
        from agent.tools import fetch_country
        with patch("httpx.AsyncClient") as cls:
            cls.return_value = _http_mock(200, self._RAW_GERMANY)
            data, err = await fetch_country("Germany")

        assert err is None
        assert data["common_name"]               == "Germany"
        assert data["official_name"]             == "Federal Republic of Germany"
        assert data["population"]                == 84607016
        assert data["capital"]                   == ["Berlin"]
        assert data["currency_list"][0]["code"]  == "EUR"
        assert data["language_list"]             == ["German"]
        assert data["calling_codes"]             == ["+49"]
        assert data["flag_emoji"]                == "🇩🇪"

    @pytest.mark.asyncio
    async def test_exact_name_match_preferred_over_first_result(self):
        from agent.tools import fetch_country
        raw = [
            {"name": {"common": "New Guinea", "official": "New Guinea", "nativeName": {}},
             "currencies": {}, "languages": {}, "idd": {}, "flags": {}},
            {"name": {"common": "Papua New Guinea",
                      "official": "Independent State of Papua New Guinea",
                      "nativeName": {}},
             "currencies": {}, "languages": {}, "idd": {}, "flags": {}},
        ]
        with patch("httpx.AsyncClient") as cls:
            cls.return_value = _http_mock(200, raw)
            data, err = await fetch_country("Papua New Guinea")

        assert err is None
        assert data["common_name"] == "Papua New Guinea"

    @pytest.mark.asyncio
    async def test_404_returns_not_found_message(self):
        from agent.tools import fetch_country
        with patch("httpx.AsyncClient") as cls:
            cls.return_value = _http_mock(404)
            data, err = await fetch_country("Narnia")

        assert data is None
        assert "No country found" in err

    @pytest.mark.asyncio
    async def test_non_200_includes_status_code(self):
        from agent.tools import fetch_country
        with patch("httpx.AsyncClient") as cls:
            cls.return_value = _http_mock(503)
            data, err = await fetch_country("Germany")

        assert data is None
        assert "503" in err

    @pytest.mark.asyncio
    async def test_timeout_returns_friendly_message(self):
        import httpx
        from agent.tools import fetch_country

        client = AsyncMock()
        client.get = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__  = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient") as cls:
            cls.return_value = client
            data, err = await fetch_country("Germany")

        assert data is None
        assert "did not respond in time" in err

    @pytest.mark.asyncio
    async def test_network_error_returns_message(self):
        import httpx
        from agent.tools import fetch_country

        client = AsyncMock()
        client.get = AsyncMock(side_effect=httpx.RequestError("dns fail"))
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__  = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient") as cls:
            cls.return_value = client
            data, err = await fetch_country("Germany")

        assert data is None
        assert "network error" in err.lower()

    @pytest.mark.asyncio
    async def test_empty_list_response_handled(self):
        from agent.tools import fetch_country
        with patch("httpx.AsyncClient") as cls:
            cls.return_value = _http_mock(200, [])
            data, err = await fetch_country("Germany")

        assert data is None
        assert "Empty response" in err


# ── Intent node tests ──────────────────────────────────────────────────────

class TestIntentNode:
    """
    Mocks _client.chat.completions.create — the AsyncTogether SDK call.
    The mock returns a Together-shaped response object with
    response.choices[0].message.content = JSON string.
    """

    def _mock_intent_response(self, country: str, fields: list) -> MagicMock:
        payload = json.dumps({"country": country, "fields": fields})
        return _together_response(payload)

    @pytest.mark.asyncio
    async def test_extracts_country_and_fields(self):
        from agent.nodes.intent import intent_node

        with patch("agent.nodes.intent._client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                return_value=self._mock_intent_response("Japan", ["currency", "capital"])
            )
            result = await intent_node({"user_query": "What currency does Japan use?"})

        assert result["country_name"]              == "Japan"
        assert "currency" in result["requested_fields"]
        assert "capital"  in result["requested_fields"]
        assert result.get("intent_error") is None

    @pytest.mark.asyncio
    async def test_empty_query_does_not_call_sdk(self):
        from agent.nodes.intent import intent_node

        with patch("agent.nodes.intent._client") as mock_client:
            result = await intent_node({"user_query": "   "})
            mock_client.chat.completions.create.assert_not_called()

        assert result["intent_error"] == "Empty query received."

    @pytest.mark.asyncio
    async def test_no_country_sets_intent_error(self):
        from agent.nodes.intent import intent_node

        with patch("agent.nodes.intent._client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                return_value=self._mock_intent_response("", ["population"])
            )
            result = await intent_node({"user_query": "What is the biggest country?"})

        assert result.get("intent_error") is not None
        assert "country" in result["intent_error"].lower()

    @pytest.mark.asyncio
    async def test_empty_fields_defaults_to_basics(self):
        from agent.nodes.intent import intent_node

        with patch("agent.nodes.intent._client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                return_value=self._mock_intent_response("Brazil", [])
            )
            result = await intent_node({"user_query": "Tell me about Brazil"})

        assert result["country_name"]           == "Brazil"
        assert "capital"    in result["requested_fields"]
        assert "population" in result["requested_fields"]
        assert "region"     in result["requested_fields"]

    @pytest.mark.asyncio
    async def test_sdk_exception_sets_intent_error(self):
        from agent.nodes.intent import intent_node

        with patch("agent.nodes.intent._client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("Together.ai 429 rate limit")
            )
            result = await intent_node({"user_query": "Capital of Spain?"})

        assert result.get("intent_error") is not None
        assert "Intent extraction failed" in result["intent_error"]

    @pytest.mark.asyncio
    async def test_whitespace_only_query_does_not_call_sdk(self):
        from agent.nodes.intent import intent_node

        with patch("agent.nodes.intent._client") as mock_client:
            result = await intent_node({"user_query": "\t\n  "})
            mock_client.chat.completions.create.assert_not_called()

        assert result.get("intent_error") is not None

    @pytest.mark.asyncio
    async def test_response_format_includes_json_schema(self):
        """Verify the SDK call passes json_schema response_format."""
        from agent.nodes.intent import intent_node

        with patch("agent.nodes.intent._client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                return_value=self._mock_intent_response("France", ["capital"])
            )
            await intent_node({"user_query": "Capital of France?"})

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["response_format"]["type"] == "json_schema"
            assert "json_schema" in call_kwargs["response_format"]
            assert call_kwargs["temperature"] == 0


# ── Fetch node tests ───────────────────────────────────────────────────────

class TestFetchNode:
    @pytest.mark.asyncio
    async def test_successful_fetch_stored_in_state(self):
        from agent.nodes.fetch import fetch_node
        with patch("agent.nodes.fetch.fetch_country", return_value=(MOCK_COUNTRY_DATA, None)):
            result = await fetch_node({
                "user_query":       "Capital of Germany?",
                "country_name":     "Germany",
                "requested_fields": ["capital"],
            })

        assert result["raw_country_data"]["common_name"] == "Germany"
        assert result.get("fetch_error") is None

    @pytest.mark.asyncio
    async def test_api_error_stored_in_state(self):
        from agent.nodes.fetch import fetch_node
        with patch("agent.nodes.fetch.fetch_country",
                   return_value=(None, "No country found matching 'Narnia'.")):
            result = await fetch_node({
                "user_query":       "Capital of Narnia?",
                "country_name":     "Narnia",
                "requested_fields": ["capital"],
            })

        assert result.get("raw_country_data") is None
        assert "Narnia" in result["fetch_error"]

    @pytest.mark.asyncio
    async def test_missing_country_name_sets_error(self):
        from agent.nodes.fetch import fetch_node
        result = await fetch_node({"user_query": "something"})
        assert result.get("fetch_error") is not None

    @pytest.mark.asyncio
    async def test_blank_country_name_sets_error(self):
        from agent.nodes.fetch import fetch_node
        result = await fetch_node({"user_query": "?", "country_name": "   "})
        assert result.get("fetch_error") is not None


# ── Synthesis node tests ───────────────────────────────────────────────────

class TestSynthesiseNode:
    @pytest.mark.asyncio
    async def test_answer_returned_and_source_populated(self):
        from agent.nodes.synthesise import synthesise_node

        with patch("agent.nodes.synthesise._client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                return_value=_together_response("The capital of Germany is Berlin.")
            )
            result = await synthesise_node({
                "user_query":       "What is the capital of Germany?",
                "country_name":     "Germany",
                "requested_fields": ["capital"],
                "raw_country_data": MOCK_COUNTRY_DATA,
            })

        assert result["answer"] == "The capital of Germany is Berlin."
        assert len(result["sources"]) == 1
        assert "restcountries.com" in result["sources"][0]
        assert "Germany" in result["sources"][0]

    @pytest.mark.asyncio
    async def test_sdk_failure_returns_graceful_fallback(self):
        from agent.nodes.synthesise import synthesise_node

        with patch("agent.nodes.synthesise._client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("model overloaded")
            )
            result = await synthesise_node({
                "user_query":       "Capital?",
                "country_name":     "Germany",
                "requested_fields": ["capital"],
                "raw_country_data": MOCK_COUNTRY_DATA,
            })

        assert "unable to compose" in result["answer"].lower()

    @pytest.mark.asyncio
    async def test_source_url_uses_common_name(self):
        from agent.nodes.synthesise import synthesise_node

        with patch("agent.nodes.synthesise._client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                return_value=_together_response("France uses Euro.")
            )
            result = await synthesise_node({
                "user_query":       "Currency of France?",
                "country_name":     "France",
                "requested_fields": ["currency"],
                "raw_country_data": {**MOCK_COUNTRY_DATA, "common_name": "France"},
            })

        assert "France" in result["sources"][0]

    @pytest.mark.asyncio
    async def test_empty_data_does_not_crash(self):
        from agent.nodes.synthesise import synthesise_node

        with patch("agent.nodes.synthesise._client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                return_value=_together_response("No data available.")
            )
            result = await synthesise_node({
                "user_query":       "Capital?",
                "country_name":     "Unknown",
                "requested_fields": ["capital"],
                "raw_country_data": {},
            })

        assert isinstance(result["answer"], str)

    @pytest.mark.asyncio
    async def test_sdk_called_with_correct_model(self):
        """Verify the synthesis node uses synthesis_model, not intent_model."""
        from agent.nodes.synthesise import synthesise_node
        from config.settings import settings

        with patch("agent.nodes.synthesise._client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                return_value=_together_response("Answer.")
            )
            await synthesise_node({
                "user_query":       "Capital?",
                "country_name":     "Germany",
                "requested_fields": ["capital"],
                "raw_country_data": MOCK_COUNTRY_DATA,
            })

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["model"] == settings.synthesis_model
            assert call_kwargs["temperature"] == 0


# ── Graph (end-to-end) tests ───────────────────────────────────────────────

class TestGraph:
    def _cfg(self, tid: str = "test") -> dict:
        return {"configurable": {"thread_id": tid}}

    def _intent_resp(self, country: str, fields: list) -> MagicMock:
        return _together_response(json.dumps({"country": country, "fields": fields}))

    @pytest.mark.asyncio
    async def test_happy_path_all_nodes_run(self):
        from agent.graph import country_agent

        with (
            patch("agent.nodes.intent._client") as mi,
            patch("agent.nodes.fetch.fetch_country",
                  return_value=(MOCK_COUNTRY_DATA, None)),
            patch("agent.nodes.synthesise._client") as ms,
        ):
            mi.chat.completions.create = AsyncMock(
                return_value=self._intent_resp("Germany", ["capital", "population"])
            )
            ms.chat.completions.create = AsyncMock(
                return_value=_together_response(
                    "The capital of Germany is Berlin. Population: ~84M."
                )
            )
            state = await country_agent.ainvoke(
                {"user_query": "What is the capital and population of Germany?"},
                config=self._cfg("t-happy"),
            )

        assert state["country_name"]              == "Germany"
        assert "capital" in state["requested_fields"]
        assert state.get("intent_error") is None
        assert state.get("fetch_error")  is None
        assert "Berlin" in state["answer"]

    @pytest.mark.asyncio
    async def test_unknown_country_routes_to_error_node(self):
        from agent.graph import country_agent

        with (
            patch("agent.nodes.intent._client") as mi,
            patch("agent.nodes.fetch.fetch_country",
                  return_value=(None, "No country found matching 'Narnia'.")),
        ):
            mi.chat.completions.create = AsyncMock(
                return_value=self._intent_resp("Narnia", ["capital"])
            )
            state = await country_agent.ainvoke(
                {"user_query": "Capital of Narnia?"},
                config=self._cfg("t-narnia"),
            )

        assert "No country found" in state["answer"]

    @pytest.mark.asyncio
    async def test_vague_query_routes_to_error_node(self):
        from agent.graph import country_agent

        with patch("agent.nodes.intent._client") as mi:
            mi.chat.completions.create = AsyncMock(
                return_value=self._intent_resp("", [])
            )
            state = await country_agent.ainvoke(
                {"user_query": "What is the biggest country?"},
                config=self._cfg("t-vague"),
            )

        assert state.get("intent_error") or "country" in state["answer"].lower()

    @pytest.mark.asyncio
    async def test_answer_always_present_on_sdk_crash(self):
        from agent.graph import country_agent

        with patch("agent.nodes.intent._client") as mi:
            mi.chat.completions.create = AsyncMock(
                side_effect=Exception("network down")
            )
            state = await country_agent.ainvoke(
                {"user_query": "Capital of France?"},
                config=self._cfg("t-crash"),
            )

        assert "answer" in state
        assert isinstance(state["answer"], str)
        assert len(state["answer"]) > 0

    @pytest.mark.asyncio
    async def test_separate_thread_ids_are_isolated(self):
        from agent.graph import country_agent

        with (
            patch("agent.nodes.intent._client") as mi,
            patch("agent.nodes.fetch.fetch_country",
                  return_value=(MOCK_COUNTRY_DATA, None)),
            patch("agent.nodes.synthesise._client") as ms,
        ):
            mi.chat.completions.create = AsyncMock(
                return_value=self._intent_resp("Germany", ["population"])
            )
            ms.chat.completions.create = AsyncMock(
                return_value=_together_response("Population: 84M.")
            )

            state_a = await country_agent.ainvoke(
                {"user_query": "Population of Germany?"},
                config=self._cfg("thread-A"),
            )
            state_b = await country_agent.ainvoke(
                {"user_query": "Population of Germany?"},
                config=self._cfg("thread-B"),
            )

        assert state_a["answer"] == state_b["answer"]

    @pytest.mark.asyncio
    async def test_sources_populated_on_success(self):
        from agent.graph import country_agent

        with (
            patch("agent.nodes.intent._client") as mi,
            patch("agent.nodes.fetch.fetch_country",
                  return_value=(MOCK_COUNTRY_DATA, None)),
            patch("agent.nodes.synthesise._client") as ms,
        ):
            mi.chat.completions.create = AsyncMock(
                return_value=self._intent_resp("Germany", ["population"])
            )
            ms.chat.completions.create = AsyncMock(
                return_value=_together_response("Population: 84M.")
            )

            state = await country_agent.ainvoke(
                {"user_query": "Population?"},
                config=self._cfg("t-sources"),
            )

        assert len(state.get("sources", [])) > 0
        assert "restcountries.com" in state["sources"][0]


# ── FastAPI endpoint tests ─────────────────────────────────────────────────

class TestAPIEndpoints:
    @pytest.mark.asyncio
    async def test_health_returns_correct_shape(self):
        from httpx import AsyncClient, ASGITransport
        from api.app import app

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.get("/health")

        assert r.status_code == 200
        body = r.json()
        assert body["status"]  == "ok"
        assert body["version"] == "3.0.0"
        assert "intent_model"    in body
        assert "synthesis_model" in body

    @pytest.mark.asyncio
    async def test_root_advertises_together_ai(self):
        from httpx import AsyncClient, ASGITransport
        from api.app import app

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.get("/")

        assert r.status_code == 200
        assert r.json()["llm_provider"] == "Together.ai"

    @pytest.mark.asyncio
    async def test_query_success_returns_full_response_shape(self):
        from httpx import AsyncClient, ASGITransport
        from api.app import app

        with (
            patch("agent.nodes.intent._client") as mi,
            patch("agent.nodes.fetch.fetch_country",
                  return_value=(MOCK_COUNTRY_DATA, None)),
            patch("agent.nodes.synthesise._client") as ms,
        ):
            mi.chat.completions.create = AsyncMock(
                return_value=_together_response(
                    json.dumps({"country": "Germany", "fields": ["population"]})
                )
            )
            ms.chat.completions.create = AsyncMock(
                return_value=_together_response("Germany has ~84 million people.")
            )

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                r = await client.post(
                    "/query",
                    json={"question": "What is the population of Germany?"},
                )

        assert r.status_code == 200
        body = r.json()
        assert "answer"          in body
        assert "request_id"      in body
        assert "latency_ms"      in body
        assert "sources"         in body
        assert "fields_resolved" in body
        assert body["country"]   == "Germany"
        assert "X-Request-ID"    in r.headers

    @pytest.mark.asyncio
    async def test_query_rejects_blank_after_strip(self):
        from httpx import AsyncClient, ASGITransport
        from api.app import app

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post("/query", json={"question": "   "})

        assert r.status_code == 422

    @pytest.mark.asyncio
    async def test_query_rejects_too_short(self):
        from httpx import AsyncClient, ASGITransport
        from api.app import app

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post("/query", json={"question": "hi"})

        assert r.status_code == 422

    @pytest.mark.asyncio
    async def test_query_rejects_too_long(self):
        from httpx import AsyncClient, ASGITransport
        from api.app import app

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post("/query", json={"question": "x" * 501})

        assert r.status_code == 422

    @pytest.mark.asyncio
    async def test_request_id_header_is_valid_uuid(self):
        from httpx import AsyncClient, ASGITransport
        from api.app import app
        import uuid as _uuid

        with (
            patch("agent.nodes.intent._client") as mi,
            patch("agent.nodes.fetch.fetch_country",
                  return_value=(MOCK_COUNTRY_DATA, None)),
            patch("agent.nodes.synthesise._client") as ms,
        ):
            mi.chat.completions.create = AsyncMock(
                return_value=_together_response(
                    json.dumps({"country": "Germany", "fields": ["capital"]})
                )
            )
            ms.chat.completions.create = AsyncMock(
                return_value=_together_response("Berlin.")
            )

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                r = await client.post(
                    "/query", json={"question": "Capital of Germany?"}
                )

        rid = r.headers.get("X-Request-ID", "")
        parsed = _uuid.UUID(rid)   # raises ValueError if not a valid UUID
        assert str(parsed) == rid
