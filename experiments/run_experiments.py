"""Experiment runner.

Defines a set of retriever configurations, evaluates each on the same queries
and qrels, and logs params + metrics to MLflow (one run per config).
"""


def main() -> None:
    # TODO: load dataset once
    # TODO: define configs (e.g. bm25 k1/b values, dense model, hybrid weights/method)
    # TODO: for each config: build retriever, evaluate, mlflow.log_params / log_metrics
    # TODO: decide what else to log (timings, index size, config as artifact)
    raise NotImplementedError


if __name__ == "__main__":
    main()
