"""Registry of retrievers an experiment config can ask for by name.

To add a retriever: decorate its class with @register_retriever("name") and make sure its module is
imported in retrievers/__init__.py. The oracle is deliberately not registered: it is a test double
that needs the queries and qrels, so it cannot be built from a config.
"""
from typing import Any, Callable, Dict, List, TypeVar

from .base import Retriever

_RetrieverT = TypeVar("_RetrieverT", bound=Callable[..., Retriever])

RETRIEVER_REGISTRY: Dict[str, Callable[..., Retriever]] = {}


def register_retriever(name: str) -> Callable[[_RetrieverT], _RetrieverT]:
    """Decorator that registers a retriever class (or factory) under `name`, returned unchanged.

    Raises:
        ValueError: if the name is already registered.
    """
    def decorator(factory: _RetrieverT) -> _RetrieverT:
        if name in RETRIEVER_REGISTRY:
            raise ValueError(f"Retriever {name!r} is already registered")
        RETRIEVER_REGISTRY[name] = factory
        return factory
    return decorator


def available_retrievers() -> List[str]:
    return sorted(RETRIEVER_REGISTRY)


def build_retriever(name: str, params: Dict[str, Any]) -> Retriever:
    """Construct a registered retriever from keyword parameters.

    Args:
        name: the registered name, e.g. "bm25".
        params: keyword arguments for the retriever's constructor (the config's `params`).

    Raises:
        ValueError: if the name is not registered, or the parameters do not fit the retriever
            (the message names the retriever and the offending parameters).
    """
    if name not in RETRIEVER_REGISTRY:
        raise ValueError(f"Unknown retriever {name!r}; available: {available_retrievers()}")
    try:
        return RETRIEVER_REGISTRY[name](**params)
    except TypeError as error:  # unknown or missing keyword argument
        raise ValueError(f"Invalid parameters for retriever {name!r}: {error}") from error
