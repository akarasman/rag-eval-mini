"""Retrievers: lexical, dense and hybrid, all behind one interface."""
# Importing a retriever module registers it by name (see registry.py).
from . import bm25, dense  # noqa: F401
