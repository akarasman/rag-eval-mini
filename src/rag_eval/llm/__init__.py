"""LLM backends behind one interface: a local server, an in-process model, or a hosted API."""
# Importing a backend module registers it by name (see registry.py).
from . import ollama  # noqa: F401
from . import openai_compatible  # noqa: F401
from . import transformers_llm  # noqa: F401
