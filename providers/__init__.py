"""
providers/__init__.py — Public surface of the providers package.

Phase 1: Groq support.
Phase 2 (revised): OpenAI replaces Gemini as the second provider.
               Groq remains the default.
Phase 5: PROVIDER_REGISTRY added as single source of truth for all-model execution.
"""

from __future__ import annotations

from providers.base_provider import LLMProvider
from providers.groq_provider import GroqProvider
from providers.openai_provider import OpenAIProvider

# ── Single source of truth for all registered providers ───────────────────────
# Add new provider names here when integrating additional backends.
# Used by run_all_providers() in backend.py to drive multi-provider execution.
PROVIDER_REGISTRY: list[str] = ["groq", "openai"]

__all__ = ["LLMProvider", "GroqProvider", "OpenAIProvider", "PROVIDER_REGISTRY", "get_provider"]


def get_provider(name: str = "groq") -> LLMProvider:
    """
    Return an initialised LLM provider by name.

    Groq is the default and is returned for all existing call-sites
    that use get_provider() with no arguments — behaviour is unchanged.
    OpenAI is only activated when explicitly requested.

    Args:
        name: Provider identifier string.
              "groq"   → GroqProvider   (default, existing behaviour)
              "openai" → OpenAIProvider (Phase 2)

    Returns:
        An initialised LLMProvider subclass instance.

    Raises:
        ValueError: If the requested provider name is unknown.
    """
    if name == "groq":
        return GroqProvider()

    if name == "openai":
        return OpenAIProvider()

    raise ValueError(
        f"Unknown provider '{name}'. "
        f"Currently supported providers: {PROVIDER_REGISTRY}"
    )
