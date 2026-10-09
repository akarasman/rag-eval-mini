"""'Ask a question' entry point.

Glue: load/index the corpus, retrieve top-k passages for a question, pass them
to generate.answer_question, print the answer with its sources.
Run as: python -m rag_eval.ask "your question"
"""


def main() -> None:
    # TODO: parse args (question, retriever choice, k)
    # TODO: build the chosen retriever, retrieve, generate, print answer + cited sources
    raise NotImplementedError


if __name__ == "__main__":
    main()
