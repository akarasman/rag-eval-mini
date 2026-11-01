"""Common LLM interface.

An LLM backend turns a prompt into text. It knows nothing about retrieval, passages or citations:
generate.py builds the prompt and parses the reply, so the same logic works with any backend.

Contract shared by every backend (ollama, transformers, openai_compatible, ...):
- Settings such as temperature, token limit and timeout are constructor parameters, so they come from
  the `params` of a config and never change between calls.
- generate() returns the model's RAW text. JSON parsing and retries are not a backend's job.
- Bad parameters raise ValueError when the backend is constructed, before any request or model load.
  Transport or API failures at call time raise LLMError.
"""
from typing import Optional, Protocol


class LLMError(RuntimeError):
    """The LLM could not be reached or answered badly: connection refused, a non-2xx status, or a
    response body that is not shaped as expected. Not used for bad parameters (those are ValueError)."""


class LLM(Protocol):
    name: str

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Return the model's raw text for this prompt.

        Args:
            prompt: the user message.
            system: an optional system message that sets the model's behaviour.

        Raises:
            LLMError: if the model could not be reached or its reply is malformed.
        """
        ...
