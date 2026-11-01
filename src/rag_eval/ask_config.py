"""Configuration for asking a question: which corpus, which retriever, which LLM, and generation settings.

Kept apart from the experiment config (config.py): an experiment evaluates a retriever against qrels, while
asking only needs a corpus to retrieve from and an LLM to answer. The two share the building blocks
(dataset, retriever) and the strictness: configs are validated up front, so a typo or an unknown name fails
before any data is loaded, any model is built or any request is made. Unknown keys are rejected.

The LLM itself is NOT built during validation (that happens in ask.py), so loading a config never reads an
API key or contacts a server.
"""
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml
from pydantic import Field, field_validator, model_validator

from . import llm  # noqa: F401  (importing the package registers the backends)
from .config import DatasetConfig, RetrieverConfig, _StrictModel, load_config
from .llm.registry import LLM_REGISTRY, available_llms


class LLMConfig(_StrictModel):
    """An LLM backend picked by its registered name, with keyword parameters for its constructor."""
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _name_is_registered(cls, name: str) -> str:
        if name not in LLM_REGISTRY:
            raise ValueError(f"unknown LLM {name!r}; available: {available_llms()}")
        return name


class GenerationSettings(_StrictModel):
    """How passages are chosen and how the answer is checked.

    Attributes:
        k: passages retrieved and sent to the model.
        max_attempts: LLM calls allowed when a reply cannot be parsed (1 means no retry).
        on_invalid_citation: "report" lists cited labels that match no passage in the result and logs a
            warning; "ignore" drops them silently.
        max_passage_chars: cut each passage to this many characters in the prompt; None keeps it whole.
        min_score: drop retrieved passages scoring below this before generation; None keeps all. Only
            meaningful for a plain BM25 or dense retriever: hybrid RRF scores depend only on ranks, and
            scores are not comparable across retrievers.
    """
    k: int = Field(default=5, gt=0)
    max_attempts: int = Field(default=2, ge=1)
    on_invalid_citation: Literal["report", "ignore"] = "report"
    max_passage_chars: Optional[int] = Field(default=None, gt=0)
    min_score: Optional[float] = None


class AskConfig(_StrictModel):
    """Everything needed to answer questions from a corpus.

    Give the retriever either inline (`retriever`) or by pointing at an experiment config
    (`retriever_from`, whose `retriever` section is reused), never both. A relative `retriever_from` is
    resolved against the working directory.

    Attributes:
        dataset: only the corpus is used (name and data_dir); queries and qrels are not needed.
    """
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    retriever: Optional[RetrieverConfig] = None
    retriever_from: Optional[Path] = None
    llm: LLMConfig
    generation: GenerationSettings = Field(default_factory=GenerationSettings)

    @model_validator(mode="after")
    def _exactly_one_retriever_source(self) -> "AskConfig":
        if (self.retriever is None) == (self.retriever_from is None):
            raise ValueError("give exactly one of `retriever` and `retriever_from`")
        if self.retriever_from is not None:
            _retriever_from_file(self.retriever_from)  # fail now if the file is missing or invalid
        return self

    @property
    def resolved_retriever(self) -> RetrieverConfig:
        """The retriever to use: the inline one, or the one in the `retriever_from` experiment config."""
        if self.retriever is not None:
            return self.retriever
        return _retriever_from_file(self.retriever_from)


def _retriever_from_file(path: Path) -> RetrieverConfig:
    try:
        return load_config(path).retriever
    except OSError as error:
        raise ValueError(f"retriever_from: cannot read {path}: {error}") from error
    except ValueError as error:  # includes pydantic's ValidationError
        raise ValueError(f"retriever_from: {path} is not a valid experiment config: {error}") from error


def load_ask_config(path: Path) -> AskConfig:
    """Read and validate a YAML ask config.

    Args:
        path: the YAML file, e.g. configs/ask/ollama.yaml.

    Raises:
        ValueError: if the file is not a YAML mapping, or does not pass validation
            (pydantic's ValidationError is a ValueError and names every offending field).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level, got {type(data).__name__}")
    return AskConfig.model_validate(data)
