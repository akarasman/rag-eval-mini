"""Experiment configuration.

One config describes ONE run: a single retriever evaluated on a single dataset split. Configs live in
YAML files, are loaded with load_config() and validated up front, so a typo or an unknown retriever or
metric fails before any data is loaded or any model is built. Unknown keys are rejected.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .metrics import get_metric
from .retrievers.registry import RETRIEVER_REGISTRY, available_retrievers


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetConfig(_StrictModel):
    """Which dataset split to evaluate on. The dataset lives in <data_dir>/<name>."""
    name: str = "scifact"
    data_dir: Path = Path("data")
    split: str = "test"

    @property
    def path(self) -> Path:
        return self.data_dir / self.name


class RetrieverConfig(_StrictModel):
    """A retriever picked by its registered name, with keyword parameters for its constructor."""
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _name_is_registered(cls, name: str) -> str:
        if name not in RETRIEVER_REGISTRY:
            raise ValueError(f"unknown retriever {name!r}; available: {available_retrievers()}")
        return name


class MetricsConfig(_StrictModel):
    """Metrics (by registered name) evaluated at each k. Results are keyed "<name>@<k>"."""
    names: Tuple[str, ...] = ("recall", "mrr", "ndcg")
    ks: Tuple[int, ...] = (1, 5, 10, 100)

    @field_validator("names")
    @classmethod
    def _names_are_registered(cls, names: Tuple[str, ...]) -> Tuple[str, ...]:
        if not names:
            raise ValueError("at least one metric is required")
        for name in names:
            get_metric(name)  # raises ValueError listing the available metrics
        return names

    @field_validator("ks")
    @classmethod
    def _ks_are_positive(cls, ks: Tuple[int, ...]) -> Tuple[int, ...]:
        if not ks or any(k <= 0 for k in ks):
            raise ValueError(f"ks must be a non-empty list of positive integers, got {list(ks)}")
        return ks

    @property
    def result_names(self) -> Tuple[str, ...]:
        """Every "<name>@<k>" key that evaluation will produce."""
        return tuple(f"{name}@{k}" for name in self.names for k in self.ks)


class ExamplesConfig(_StrictModel):
    """Query examples to log: the n best and n worst queries by one result metric. n = 0 disables."""
    n: int = Field(default=10, ge=0)
    metric: str = "ndcg@10"


class MlflowConfig(_StrictModel):
    """MLflow options. The experiment name is the top-level `experiment` of the config."""
    tracking_uri: Optional[str] = None  # None: MLflow's default (a SQLite file, mlflow.db, in the working directory)
    run_name: Optional[str] = None  # None: derived from the retriever and its parameters
    tags: Dict[str, str] = Field(default_factory=dict)
    log_per_query: bool = True  # store the per-query scores as an artifact
    examples: ExamplesConfig = Field(default_factory=ExamplesConfig)


class ExperimentConfig(_StrictModel):
    """Everything needed to reproduce one run.

    Attributes:
        experiment: MLflow experiment name. Group comparable runs under one name, e.g. "scifact-baselines".
    """
    experiment: str = Field(min_length=1)
    retriever: RetrieverConfig
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    mlflow: MlflowConfig = Field(default_factory=MlflowConfig)

    @model_validator(mode="after")
    def _example_metric_is_computed(self) -> "ExperimentConfig":
        if self.mlflow.examples.n > 0 and self.mlflow.examples.metric not in self.metrics.result_names:
            raise ValueError(
                f"mlflow.examples.metric {self.mlflow.examples.metric!r} is not computed; "
                f"choose one of {list(self.metrics.result_names)} or set mlflow.examples.n to 0"
            )
        return self

    @property
    def run_name(self) -> str:
        """The configured run name, or e.g. "bm25 k1=1.2 stem=True" derived from the retriever."""
        if self.mlflow.run_name:
            return self.mlflow.run_name
        params = " ".join(f"{key}={value}" for key, value in self.retriever.params.items())
        return f"{self.retriever.name} {params}".strip()

    def flat_params(self) -> Dict[str, Any]:
        """Flat key -> value view of the config, for logging as MLflow params."""
        params: Dict[str, Any] = {
            "dataset.name": self.dataset.name,
            "dataset.split": self.dataset.split,
            "retriever.name": self.retriever.name,
            "metrics.names": ",".join(self.metrics.names),
            "metrics.ks": ",".join(str(k) for k in self.metrics.ks),
        }
        params.update({f"retriever.{key}": value for key, value in self.retriever.params.items()})
        return params


def load_config(path: Path) -> ExperimentConfig:
    """Read and validate a YAML config file.

    Raises:
        ValueError: if the file is not a YAML mapping, or does not pass validation
            (pydantic's ValidationError is a ValueError and names every offending field).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level, got {type(data).__name__}")
    return ExperimentConfig.model_validate(data)
