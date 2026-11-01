"""Ollama backend: a local model served by an Ollama server (https://ollama.com).

Talks to the server's /api/chat endpoint over HTTP, so nothing heavy is loaded in this process and the
model runs wherever the server runs. Start the server and pull a model first, e.g. `ollama pull llama3.2`.
"""
from typing import Any, Dict, List, Optional

from ._http import post_json, validate_server_settings
from .base import LLMError
from .registry import register_llm


@register_llm("ollama")
class OllamaLLM:
    """Args:
        model: the model tag as the server knows it, e.g. "llama3.2".
        host: address of the Ollama server.
        temperature: sampling temperature; 0 makes the output as repeatable as the server allows.
        max_tokens: the most tokens the model may generate (Ollama's `num_predict`).
        timeout: seconds to wait for a reply. A local CPU model can be slow, hence the high default.
        json_mode: ask the server to constrain the output to valid JSON (Ollama's `format: "json"`).

    Raises:
        ValueError: if model is empty, host is not an http(s) URL, max_tokens or timeout is not positive,
            or temperature is negative.
    """
    name = "ollama"

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: float = 120.0,
        json_mode: bool = True,
    ):
        validate_server_settings(model, "host", host, max_tokens, timeout, temperature)
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.json_mode = json_mode

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Return the model's raw reply text. See LLM.generate.

        Args:
            prompt: the user message.
            system: an optional system message that sets the model's behaviour.

        Raises:
            LLMError: if the server cannot be reached, answers with an error, or replies in an
                unexpected shape.
        """
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        if self.json_mode:
            payload["format"] = "json"

        data = post_json(
            f"{self.host}/api/chat", payload, self.timeout,
            hint=f"is the Ollama server running at {self.host}?",
        )
        try:
            content = data["message"]["content"]
        except (KeyError, TypeError) as error:
            raise LLMError("unexpected reply from Ollama: no message.content") from error
        if not isinstance(content, str):
            raise LLMError("unexpected reply from Ollama: message.content is not text")
        return content
