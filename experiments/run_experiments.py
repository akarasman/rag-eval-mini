"""Experiment runner.

Runs one or more experiment configs (YAML) and logs each to MLflow (one run per config).
All configs are loaded and validated before the first one runs, so a typo in the last file
does not waste the earlier runs.

    python experiments/run_experiments.py experiments/configs/bm25_baseline.yaml
"""
import argparse
import logging
from pathlib import Path
from typing import List, Optional

from rag_eval.config import load_config
from rag_eval.experiment import run_experiment


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run retrieval experiments from YAML configs.")
    parser.add_argument("configs", nargs="+", type=Path, help="one or more experiment config files")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    configs = [load_config(path) for path in args.configs]

    for config in configs:
        result = run_experiment(config)
        print(f"\n{config.experiment} / {config.run_name}")
        for name, value in result.mean.items():
            print(f"  {name:<12} {value:.4f}")


if __name__ == "__main__":
    main()
