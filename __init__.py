"""
providers/__init__.py — Public API for the PSAF provider package.

Phase 1: exports the abstract LLMProvider interface only.
Concrete implementations (GroqProvider, GeminiProvider, …) will be
added in subsequent phases without requiring changes to this file or
to any existing PSAF module.
"""

from providers.base_provider import LLMProvider

__all__ = ["LLMProvider"]
