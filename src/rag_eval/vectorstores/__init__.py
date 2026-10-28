"""Vector stores: exact and approximate nearest-neighbour search behind one interface."""
# Importing a store module registers it by name (see registry.py).
from . import exact  # noqa: F401
from . import faiss_store  # noqa: F401
from . import qdrant_store  # noqa: F401
