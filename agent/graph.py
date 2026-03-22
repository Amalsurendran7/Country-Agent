"""
LangGraph 1.1.3 graph definition.

Key API points for this version:
  - START constant from langgraph.constants (not set_entry_point — deprecated)
  - retry_policy= kwarg on add_node (not retry= — renamed in 0.5)
  - MemorySaver checkpointer passed to compile(); every ainvoke() call must
    supply {"configurable": {"thread_id": "<unique>"}} in its config
  - RetryPolicy from langgraph.types with max_attempts, backoff_factor, jitter

Graph topology
──────────────
        START
          │
   ┌──────▼───────┐
   │ intent_node  │   extract country + requested fields
   └──────┬───────┘
          │
   ┌──────▼──────────────┐
   │ route_after_intent  │   guard: bail if intent_error is set
   └───┬──────────┬───────┘
    (ok)        (err)
       │           │
┌──────▼──┐   ┌───▼──────┐
│  fetch  │   │  error   │──► END
└──────┬──┘   └──────────┘
       │
┌──────▼─────────────┐
│ route_after_fetch  │   guard: bail if fetch_error is set
└───┬──────────┬──────┘
 (ok)        (err)
    │           │
┌───▼──────┐ ┌─▼──────┐
│synthesise│ │ error  │──► END
└───┬──────┘ └────────┘
    │
   END
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END
from langgraph.constants import START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RetryPolicy

from agent.state import CountryAgentState
from agent.nodes.intent     import intent_node
from agent.nodes.fetch      import fetch_node
from agent.nodes.synthesise import synthesise_node

# Retry policies — applied per node, not globally
# LLM nodes get more attempts + exponential back-off for Together.ai rate limits
_LLM_RETRY = RetryPolicy(
    max_attempts=3,
    initial_interval=0.5,
    backoff_factor=2.0,
    max_interval=30.0,
    jitter=True,
)
# API node gets fewer attempts — REST Countries failures tend to be hard (404)
_API_RETRY = RetryPolicy(
    max_attempts=2,
    initial_interval=1.0,
    backoff_factor=2.0,
    max_interval=10.0,
    jitter=False,
)


# ── Error terminal ─────────────────────────────────────────────────────────

async def error_node(state: CountryAgentState) -> CountryAgentState:
    """
    Terminal node that surfaces the first error as the final answer.
    Guarantees the `answer` key is always populated — callers never
    need to check for missing keys.
    """
    message = (
        state.get("intent_error")
        or state.get("fetch_error")
        or "An unexpected error occurred."
    )
    return {**state, "answer": message, "sources": []}


# ── Routing functions ──────────────────────────────────────────────────────

def _route_after_intent(state: CountryAgentState) -> str:
    return "error" if state.get("intent_error") else "fetch"


def _route_after_fetch(state: CountryAgentState) -> str:
    return "error" if state.get("fetch_error") else "synthesise"


# ── Graph builder ──────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(CountryAgentState)

    # ── Nodes ──────────────────────────────────────────────────────────────
    # retry_policy= is the correct kwarg in langgraph ≥ 0.5 / 1.x
    graph.add_node("intent",     intent_node,     retry_policy=_LLM_RETRY)
    graph.add_node("fetch",      fetch_node,      retry_policy=_API_RETRY)
    graph.add_node("synthesise", synthesise_node, retry_policy=_LLM_RETRY)
    graph.add_node("error",      error_node)

    # ── Entry point — use START constant, not deprecated set_entry_point()
    graph.add_edge(START, "intent")

    # ── Conditional routing ────────────────────────────────────────────────
    graph.add_conditional_edges(
        "intent",
        _route_after_intent,
        {"fetch": "fetch", "error": "error"},
    )
    graph.add_conditional_edges(
        "fetch",
        _route_after_fetch,
        {"synthesise": "synthesise", "error": "error"},
    )

    # ── Terminal edges ─────────────────────────────────────────────────────
    graph.add_edge("synthesise", END)
    graph.add_edge("error",      END)

    return graph.compile(
        checkpointer=MemorySaver(),   # enables per-request state inspection
    )


# Compiled once at import time — reused across all requests
country_agent = build_graph()
