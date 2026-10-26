"""Registry of vector stores a retriever can ask for by name.

To add a store: decorate its class with @register_vector_store("name") and import its module in
vectorstores/__init__.py.
"""
from typing import Any, Callable, Dict, List, TypeVar

from .base import VectorStore

_StoreT = TypeVar("_StoreT", bound=Callable[..., VectorStore])

VECTOR_STORE_REGISTRY: Dict[str, Callable[..., VectorStore]] = {}


def register_vector_store(name: str) -> Callable[[_StoreT], _StoreT]:
    """Decorator that registers a vector store class (or factory) under `name`, returned unchanged.

    Raises:
        ValueError: if the name is already registered.
    """
    def decorator(factory: _StoreT) -> _StoreT:
        if name in VECTOR_STORE_REGISTRY:
            raise ValueError(f"Vector store {name!r} is already registered")
        VECTOR_STORE_REGISTRY[name] = factory
        return factory
    return decorator


def available_vector_stores() -> List[str]:
    return sorted(VECTOR_STORE_REGISTRY)


def build_vector_store(name: str, params: Dict[str, Any]) -> VectorStore:
    """Construct a registered vector store from keyword parameters.

    Raises:
        ValueError: if the name is not registered, or the parameters do not fit the store.
    """
    if name not in VECTOR_STORE_REGISTRY:
        raise ValueError(f"Unknown vector store {name!r}; available: {available_vector_stores()}")
    try:
        return VECTOR_STORE_REGISTRY[name](**params)
    except TypeError as error:  # unknown or missing keyword argument
        raise ValueError(f"Invalid parameters for vector store {name!r}: {error}") from error
