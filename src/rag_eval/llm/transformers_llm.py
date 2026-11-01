"""Hugging Face transformers backend: a chat model that runs inside this process, on the CPU by default.

The model is downloaded and loaded on the first generate(), not when the backend is built, so building
one from a config is cheap and tests need no model. Small instruct models (about 0.1 to 1.5 billion
parameters) are practical on a laptop CPU; larger ones are slow.
"""
from typing import Any, Dict, List, Optional

from .base import LLMError
from .registry import register_llm


@register_llm("transformers")
class TransformersLLM:
    """Args:
        model_name: Hugging Face model id of a chat model, e.g. "HuggingFaceTB/SmolLM2-360M-Instruct".
            There is no default: the choice decides speed and answer quality.
        device: "cpu", "cuda", ...
        temperature: 0 means greedy decoding (repeatable); above 0 the output is sampled.
        max_new_tokens: the most tokens the model may generate.

    Raises:
        ValueError: if model_name is empty, max_new_tokens is not positive, or temperature is negative.
    """
    name = "transformers"

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        temperature: float = 0.0,
        max_new_tokens: int = 512,
    ):
        if not model_name.strip():
            raise ValueError("model_name must not be empty")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if temperature < 0:
            raise ValueError("temperature must not be negative")
        self.model_name = model_name
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self._tokenizer: Any = None
        self._model: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer  # heavy import, kept out of module load

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
        except (OSError, ValueError) as error:  # unknown model id, no network, unsupported architecture
            raise LLMError(f"could not load the model {self.model_name!r}: {error}") from error
        model.to(self.device)
        model.eval()
        self._tokenizer, self._model = tokenizer, model

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Return only the newly generated text, not the prompt. See LLM.generate.

        Args:
            prompt: the user message.
            system: an optional system message that sets the model's behaviour.

        Raises:
            LLMError: if the model cannot be loaded, or its tokenizer has no chat template.
        """
        import torch

        self._ensure_loaded()
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            inputs = self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
            )
        except ValueError as error:  # e.g. the tokenizer has no chat template
            raise LLMError(f"cannot format the prompt for {self.model_name!r}: {error}") from error
        inputs = inputs.to(self.device)

        options: Dict[str, Any] = {"max_new_tokens": self.max_new_tokens, "do_sample": self.temperature > 0}
        if self.temperature > 0:
            options["temperature"] = self.temperature  # passing it with greedy decoding only makes transformers warn
        pad_token_id = self._tokenizer.pad_token_id
        options["pad_token_id"] = pad_token_id if pad_token_id is not None else self._tokenizer.eos_token_id

        with torch.inference_mode():
            output = self._model.generate(**inputs, **options)
        new_tokens = output[0][inputs["input_ids"].shape[1]:] # slice off the prompt, leaving only the newly generated tokens
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)
