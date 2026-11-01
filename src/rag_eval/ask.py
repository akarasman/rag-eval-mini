"""'Ask a question' entry point.

Glue: load the corpus, retrieve the top-k passages for a question, let an LLM answer from them (with
citations, or an honest "the passages do not answer this"), and print the answer with its sources.

    python -m rag_eval.ask --config configs/ask/ollama.yaml "your question"

Everything that can be wrong with the setup fails before the slow work: the config is validated, the
question is checked, and the LLM backend is built (which checks its parameters and API key) before the
corpus is loaded and indexed.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .ask_config import AskConfig, load_ask_config
from .data import download_dataset, load_corpus
from .generate import AnswerStatus, GenerationResult, answer_question
from .llm.base import LLMError
from .llm.registry import build_llm
from .retrievers.base import Retriever
from .retrievers.registry import build_retriever
from .text import doc_to_text

Corpus = Dict[str, Dict[str, str]]


def retrieve_passages(
    retriever: Retriever, corpus: Corpus, question: str, k: int, min_score: Optional[float] = None
) -> List[Tuple[str, str]]:
    """Search the corpus and return the passages to show the model, best first.

    Args:
        retriever: an indexed retriever.
        corpus: doc_id -> {"title", "text"}, the corpus the retriever was indexed on.
        question: the question to search for.
        k: how many passages to retrieve.
        min_score: drop passages scoring below this; None keeps them all. Only meaningful for a plain
            BM25 or dense retriever (hybrid RRF scores depend only on ranks).

    Returns:
        [(doc_id, doc_to_text(doc)), ...], possibly empty.
    """
    ranking = retriever.search(question, k)
    if min_score is not None:
        ranking = [(doc_id, score) for doc_id, score in ranking if score >= min_score]
    return [(doc_id, doc_to_text(corpus[doc_id])) for doc_id, _ in ranking]


def format_result(result: GenerationResult, corpus: Corpus) -> str:
    """The text shown to the user: the status, the answer or why there is none, and the cited sources.

    Args:
        result: what answer_question returned.
        corpus: doc_id -> {"title", "text"}, used to show the title of each cited source.
    """
    lines = [f"Status: {result.status.value}", ""]

    if result.status is AnswerStatus.ANSWERED:
        lines.append(result.answer)
        lines += ["", "Sources:"]
        for number, doc_id in enumerate(result.citations, start=1):
            title = corpus.get(doc_id, {}).get("title", "")
            lines.append(f"  {number}. {doc_id}" + (f" - {title}" if title else ""))
    elif result.status is AnswerStatus.ABSTAINED:
        lines.append("The retrieved passages do not answer this question.")
    elif result.status is AnswerStatus.NO_PASSAGES:
        lines.append("No passages were retrieved for this question, so no answer was generated.")
    elif result.status is AnswerStatus.UNGROUNDED:
        lines.append("The model gave an answer without a valid citation; treat it as unverified.")
        if result.answer:
            lines += ["", f"Unverified answer: {result.answer}"]
    else:  # INVALID_OUTPUT
        lines.append(
            f"The model did not return a valid JSON reply in {result.attempts} attempt(s). "
            "Run with --json to see its raw output."
        )

    if result.invalid_citations:
        lines += ["", f"Note: the model cited passage numbers that do not exist: {result.invalid_citations}."]
    return "\n".join(lines)


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _answer(config: AskConfig, question: str, k: int) -> Tuple[GenerationResult, Corpus]:
    # Build the LLM first: a bad backend setting or a missing API key should not cost an indexing run.
    llm = build_llm(config.llm.name, config.llm.params)

    corpus = load_corpus(download_dataset(config.dataset.name, config.dataset.data_dir))
    retriever_config = config.resolved_retriever
    retriever = build_retriever(retriever_config.name, retriever_config.params)
    retriever.index(corpus)

    settings = config.generation
    passages = retrieve_passages(retriever, corpus, question, k, settings.min_score)
    result = answer_question(
        question,
        passages,
        llm,
        max_attempts=settings.max_attempts,
        on_invalid_citation=settings.on_invalid_citation,
        max_passage_chars=settings.max_passage_chars,
    )
    return result, corpus


def main(argv: Optional[List[str]] = None) -> None:
    """Answer one question from the command line.

    Args:
        argv: the arguments after the program name; None reads sys.argv.

    Exits with status 1 and a one-line message on a bad config or an LLM failure, unless -v is given, in
    which case the full traceback is shown.
    """
    parser = argparse.ArgumentParser(description="Answer a question from the corpus, with citations.")
    parser.add_argument("question", help="the question to answer")
    parser.add_argument("--config", required=True, type=Path, help="an ask config file, e.g. configs/ask/ollama.yaml")
    parser.add_argument("--k", type=_positive_int, help="passages to send to the model (overrides generation.k)")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON instead of text")
    parser.add_argument("-v", "--verbose", action="store_true", help="log progress, and show tracebacks on errors")
    args = parser.parse_args(argv)

    if not args.question.strip():
        parser.error("the question must not be empty")
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")

    try:
        config = load_ask_config(args.config)
        result, corpus = _answer(config, args.question, args.k or config.generation.k)
    except (ValueError, LLMError) as error:
        if args.verbose:
            raise
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    print(result.model_dump_json(indent=2) if args.json else format_result(result, corpus))


if __name__ == "__main__":
    main()
