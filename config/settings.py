"""
Application configuration — pydantic-settings 2.13.1

LLM calls use the native Together.ai SDK (together==2.5.0) directly.
No langchain LLM wrapper is needed — the SDK's AsyncTogether client is used
in the intent and synthesis nodes.
"""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # ── Together.ai ────────────────────────────────────────────────────────
    together_api_key: str = Field(
        default="",
        alias="TOGETHER_API_KEY",
    )

    # Intent node — small/fast model, produces structured JSON only
    intent_model: str = Field(
        default="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        alias="INTENT_MODEL",
    )
    # Synthesis node — larger model for user-facing prose quality
    synthesis_model: str = Field(
        default="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        alias="SYNTHESIS_MODEL",
    )

    # ── REST Countries API ─────────────────────────────────────────────────
    countries_base_url: str = Field(
        default="https://restcountries.com/v3.1",
        alias="COUNTRIES_BASE_URL",
    )
    countries_connect_timeout: float = Field(
        default=3.0, alias="COUNTRIES_CONNECT_TIMEOUT", gt=0,
    )
    countries_read_timeout: float = Field(
        default=8.0, alias="COUNTRIES_READ_TIMEOUT", gt=0,
    )

    # ── Server ─────────────────────────────────────────────────────────────
    host: str  = Field(default="0.0.0.0", alias="HOST")
    port: int  = Field(default=8000, alias="PORT", ge=1, le=65535)
    debug: bool = Field(default=False, alias="DEBUG")

    # ── Agent retry behaviour ──────────────────────────────────────────────
    llm_max_retries: int = Field(default=3, alias="LLM_MAX_RETRIES", ge=1, le=10)
    api_max_retries: int = Field(default=2, alias="API_MAX_RETRIES", ge=1, le=5)

    @field_validator("together_api_key")
    @classmethod
    def key_present_at_runtime(cls, v: str) -> str:
        return v


settings = Settings()
