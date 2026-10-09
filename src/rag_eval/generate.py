"""Answer generation.

Builds a prompt from the question and retrieved passages, calls an LLM, and
returns an answer that cites the passages it used.
"""


def build_prompt(question: str, passages: list[tuple[str, str]]) -> str:
    """passages: [(doc_id, text), ...]. Return the full prompt string."""
    # TODO: number/label passages so the model can cite them
    # TODO: instruct the model to answer only from the passages and say so if it can't
    raise NotImplementedError


def call_llm(prompt: str) -> str:
    """Send the prompt to the LLM and return its raw text."""
    # TODO: choose backend (local model vs. free-tier API)
    raise NotImplementedError


def extract_citations(answer: str, passages: list[tuple[str, str]]) -> list[str]:
    """Return the doc_ids the answer cites."""
    # TODO: parse citation markers; decide how to handle markers that match no passage
    raise NotImplementedError


def answer_question(question: str, passages: list[tuple[str, str]]) -> dict:
    """Return {"answer": str, "citations": [doc_id, ...]}."""
    raise NotImplementedError
