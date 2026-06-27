"""
providers/base_provider.py — Abstract base class for all LLM providers.

Phase 1: Architecture preparation only.
No implementation logic lives here.
"""

from __future__ import annotations


class LLMProvider:
    """
    Abstract base for all LLM provider implementations.

    Subclasses must override both methods.
    No default behaviour is provided intentionally —
    callers will get a clear NotImplementedError if they
    accidentally use the base class directly.
    """

    def generate_response(self, prompt: str) -> str:
        """
        Send a single prompt to the LLM and return the response text.

        Args:
            prompt: The user-facing prompt string to send.

        Returns:
            The raw response string from the LLM.

        Raises:
            NotImplementedError: Always — subclasses must implement this.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement generate_response()"
        )

    def generate_variations(self, prompt: str) -> list[str]:
        """
        Generate paraphrased variations of the given prompt.

        Args:
            prompt: The seed prompt to paraphrase.

        Returns:
            A list of paraphrased prompt strings.

        Raises:
            NotImplementedError: Always — subclasses must implement this.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement generate_variations()"
        )
