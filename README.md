# Country Information Agent — v3

AI agent answering natural-language questions about countries.  
Exact library versions: **langgraph 1.1.3 · langchain 1.2.13 · pydantic 2.12.5**

---

## Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI 0.135.1                         │
│  POST /query  ──►  LangGraph 1.1.3 pipeline                 │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────┐   ┌────────────────────┐   ┌──────────────────────┐
│  1. intent_node    │──►│  2. fetch_node     │──►│  3. synthesise_node  │
│  ChatOpenAI→       │   │  REST Countries    │   │  ChatOpenAI→         │
│  Together.ai 8B    │   │  API (httpx)       │   │  Together.ai 70B     │
│                    │   │                    │   │                      │
│  with_structured_  │   │  Normalised dict   │   │  LCEL chain:         │
│  output(Intent     │   │  in state          │   │  prompt | llm |      │
│  Output)           │   │                    │   │  StrOutputParser()   │
└────────────────────┘   └────────────────────┘   └──────────────────────┘
        │                         │
      error                     error
        └──────────┬──────────────┘
                   ▼
             error_node ──► END
```

Each LLM node has a `RetryPolicy` (exponential back-off + jitter).  
`MemorySaver` checkpointer gives every request isolated, inspectable state.

---

## Library versions & compatibility notes

| Package | Version | Notes |
|---------|---------|-------|
| `langgraph` | 1.1.3 | `START` constant; `retry_policy=` on `add_node`; `MemorySaver` |
| `langchain` | 1.2.13 | LCEL `|` composition; `ChatPromptTemplate`; `StrOutputParser` |
| `langchain-core` | 1.2.20 | Pulled by langchain 1.2.13 |
| `langchain-openai` | 0.3.35 | `ChatOpenAI` pointed at Together.ai base URL |
| `pydantic` | 2.12.5 | `ConfigDict`; `field_validator(mode="after")`; `Annotated` |
| `pydantic-settings` | 2.13.1 | `BaseSettings` + `SettingsConfigDict` |
| `fastapi` | 0.135.1 | `asynccontextmanager` lifespan; `Annotated` + `Depends` |
| `httpx` | 0.28.1 | `Timeout(connect=, read=, write=, pool=)` granular |
| `uvicorn` | 0.42.0 | ASGI server |
| `together` | 2.5.0 | SDK (installed; Together.ai also reachable via OpenAI-compat URL) |

> **Why `ChatOpenAI` instead of `ChatTogether`?**  
> `langchain-together 0.3.1` requires `langchain-core < 0.4`, which conflicts  
> with `langchain 1.2.13` (needs `langchain-core ≥ 1.2`).  
> Together.ai exposes an OpenAI-compatible REST API at `https://api.together.xyz/v1`,  
> so `ChatOpenAI(openai_api_base=..., openai_api_key=<together_key>)` is the  
> correct approach for this version combination.

---

## Quickstart

```bash
git clone <repo-url> && cd country-agent-v3
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your TOGETHER_API_KEY
python main.py                # http://localhost:8000
```

### Docker

```bash
cp .env.example .env          # add TOGETHER_API_KEY
docker-compose up --build
```

---

## API

### `POST /query`

```jsonc
// Request
{ "question": "What currency does Japan use?" }

// Response
{
  "request_id": "3f2a1b…",
  "question": "What currency does Japan use?",
  "answer": "Japan uses the Japanese Yen (JPY, symbol ¥).",
  "country": "Japan",
  "fields_resolved": ["currency"],
  "sources": ["REST Countries API — https://restcountries.com/v3.1/name/Japan"],
  "latency_ms": 940
}
```

### `GET /health`

```json
{ "status": "ok", "version": "3.0.0",
  "intent_model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
  "synthesis_model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo" }
```

Interactive docs: `http://localhost:8000/docs`

---

## Running tests

```bash
# All 42 tests — no API key needed (LLM + HTTP fully mocked)
pytest tests/ -v -W error::DeprecationWarning
```

---

## Project structure

```
country-agent-v3/
├── agent/
│   ├── graph.py          # LangGraph 1.1.3 pipeline
│   ├── state.py          # TypedDict + Annotated reducers
│   ├── tools.py          # REST Countries wrapper (httpx 0.28.1)
│   └── nodes/
│       ├── intent.py     # with_structured_output → IntentOutput
│       ├── fetch.py      # calls tools.fetch_country
│       └── synthesise.py # LCEL: prompt | llm | StrOutputParser
├── api/app.py            # FastAPI 0.135.1
├── config/settings.py    # pydantic-settings BaseSettings
├── tests/test_agent.py   # 42 tests (fully mocked)
├── main.py
├── requirements.txt      # exact pinned versions
├── Dockerfile
└── docker-compose.yml
```

---

## Known limitations

- **One country per query** — multi-country comparisons need a loop/fan-out extension.
- **No response caching** — every request hits REST Countries; a short TTL cache would reduce p99.
- **No auth** — add OAuth2/API-key middleware for production exposure.
- **`langchain-together` incompatible** — see table above for why `ChatOpenAI` is used instead.
