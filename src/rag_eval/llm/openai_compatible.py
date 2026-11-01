"""Backend for hosted APIs that speak the OpenAI chat-completions protocol.

Many providers (including several with free tiers) offer this protocol, so the provider is chosen by
`base_url` in the config and nothing here is specific to one of them. The API key is read from an
environment variable that the config NAMES; the key itself never appears in a config, a log line, an
error message or repr().
"""
import os
from typing import Any, Dict, List, Optional

from ._http import post_json, validate_server_settings
from .base import LLMError
from .registry import register_llm


@register_llm("openai_compatible")
class OpenAICompatibleLLM:
    """Args:
        model: the provider's model name.
        base_url: the API root; "/chat/completions" is appended, e.g. "https://api.example.com/v1".
        api_key_env: NAME of the environment variable that holds the API key (not the key itself).
        temperature: sampling temperature; 0 makes the output as repeatable as the provider allows.
        max_tokens: the most tokens the model may generate.
        timeout: seconds to wait for a reply.

    Raises:
        ValueError: if model is empty, base_url is not an http(s) URL, max_tokens or timeout is not
            positive, temperature is negative, or the environment variable is unset or empty.
    """
    name = "openai_compatible"

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key_env: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: float = 60.0,
    ):
        validate_server_settings(model, "base_url", base_url, max_tokens, timeout, temperature)
        if not api_key_env.strip():
            raise ValueError("api_key_env must name an environment variable")
        # Read now, not at the first request: a missing key should stop the run before any work is done.
        api_key = os.environ.get(api_key_env, "").strip()
        if not api_key:
            raise ValueError(f"environment variable {api_key_env} is not set (or is empty); it must hold the API key")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._api_key = api_key

    def __repr__(self) -> str:
        # Written out so the key can never end up here, however attributes are added later.
        return f"OpenAICompatibleLLM(model={self.model!r}, base_url={self.base_url!r}, api_key_env={self.api_key_env!r})"

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Return the model's raw reply text. See LLM.generate.

        Args:
            prompt: the user message.
            system: an optional system message that sets the model's behaviour.

        Raises:
            LLMError: if the API cannot be reached, answers with an error, or replies in an
                unexpected shape.
        """
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        data = post_json(
            f"{self.base_url}/chat/completions", payload, self.timeout,
            headers={"Authorization": f"Bearer {self._api_key}"},
            hint="check base_url and your network",
        )
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise LLMError("unexpected reply from the API: no choices[0].message.content") from error
        if not isinstance(content, str):
            raise LLMError("unexpected reply from the API: choices[0].message.content is not text")
        return content
