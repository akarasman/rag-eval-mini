"""Registry of LLM backends a config can ask for by name.

To add a backend: decorate its class with @register_llm("name") and import its module in
llm/__init__.py.
"""
from typing import Any, Callable, Dict, List, TypeVar

from .base import LLM

_LLMT = TypeVar("_LLMT", bound=Callable[..., LLM])

LLM_REGISTRY: Dict[str, Callable[..., LLM]] = {}


def register_llm(name: str) -> Callable[[_LLMT], _LLMT]:
    """Decorator that registers an LLM class (or factory) under `name`, returned unchanged.

    Raises:
        ValueError: if the name is already registered.
    """
    def decorator(factory: _LLMT) -> _LLMT:
        if name in LLM_REGISTRY:
            raise ValueError(f"LLM {name!r} is already registered")
        LLM_REGISTRY[name] = factory
        return factory
    return decorator


def available_llms() -> List[str]:
    return sorted(LLM_REGISTRY)


def build_llm(name: str, params: Dict[str, Any]) -> LLM:
    """Construct a registered LLM backend from keyword parameters.

    Args:
        name: the registered name, e.g. "ollama".
        params: keyword arguments for the backend's constructor.

    Raises:
        ValueError: if the name is not registered, or the parameters do not fit the backend.
    """
    if name not in LLM_REGISTRY:
        raise ValueError(f"Unknown LLM {name!r}; available: {available_llms()}")
    try:
        return LLM_REGISTRY[name](**params)
    except TypeError as error:  # unknown or missing keyword argument
        raise ValueError(f"Invalid parameters for LLM {name!r}: {error}") from error
