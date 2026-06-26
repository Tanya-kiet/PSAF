"""
providers/base_provider.py — Abstract base class defining the LLM provider
interface for the Prompt Stability Analysis Framework (PSAF).

All concrete provider implementations (Groq, Gemini, OpenAI, Claude, …)
must subclass LLMProvider and implement every abstract method declared here.
No logic lives in this file — it is a pure interface contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract interface for a pluggable LLM backend.

    PSAF components depend only on this interface, not on any
    SDK-specific class.  Swapping providers therefore requires no
    downstream changes.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send *prompt* to the model and return the response as a plain string.

        Args:
            prompt: The user message to send to the model.

        Returns:
            The model's response text, or an empty string on failure.
        """

    @abstractmethod
    def generate_variations(self, prompt: str) -> list[str]:
        """Generate paraphrased rewrites of *prompt*.

        Args:
            prompt: The seed prompt to rewrite.

        Returns:
            A list of rewritten prompt strings.
        """
