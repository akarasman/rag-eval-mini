"""Tests for the LLM registry, using throwaway backends so the real registry is never polluted."""
import pytest

from rag_eval.llm import registry
from rag_eval.llm.registry import available_llms, build_llm, register_llm


class _FakeLLM:
    name = "fake"

    def __init__(self, model: str, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

    def generate(self, prompt, system=None):
        return "reply"


@pytest.fixture
def empty_registry(monkeypatch):
    """Swap in an empty registry for the test; the real one is restored afterwards."""
    monkeypatch.setattr(registry, "LLM_REGISTRY", {})
    return registry.LLM_REGISTRY


def test_register_llm_returns_the_class_unchanged_and_makes_it_buildable(empty_registry):
    decorated = register_llm("fake")(_FakeLLM)

    assert decorated is _FakeLLM
    llm = build_llm("fake", {"model": "m", "temperature": 0.5})
    assert isinstance(llm, _FakeLLM)
    assert (llm.model, llm.temperature) == ("m", 0.5)


def test_register_llm_rejects_a_duplicate_name(empty_registry):
    register_llm("fake")(_FakeLLM)

    with pytest.raises(ValueError, match="already registered"):
        register_llm("fake")(_FakeLLM)


def test_build_llm_rejects_an_unknown_name_and_lists_the_available_ones(empty_registry):
    register_llm("fake")(_FakeLLM)

    with pytest.raises(ValueError, match="nope.*fake"):
        build_llm("nope", {})


def test_build_llm_rejects_parameters_the_backend_does_not_take(empty_registry):
    register_llm("fake")(_FakeLLM)

    with pytest.raises(ValueError, match="Invalid parameters for LLM 'fake'.*typo"):
        build_llm("fake", {"model": "m", "typo": 1})


def test_build_llm_rejects_a_missing_required_parameter(empty_registry):
    register_llm("fake")(_FakeLLM)

    with pytest.raises(ValueError, match="Invalid parameters for LLM 'fake'.*model"):
        build_llm("fake", {})


def test_a_value_error_from_the_backend_constructor_keeps_its_own_message(empty_registry):
    class _Picky:
        name = "picky"

        def __init__(self, model: str):
            raise ValueError("model must not be empty")

    register_llm("picky")(_Picky)

    with pytest.raises(ValueError) as error:
        build_llm("picky", {"model": ""})

    assert str(error.value) == "model must not be empty"  # not wrapped in "Invalid parameters"


def test_available_llms_is_sorted(empty_registry):
    register_llm("b")(_FakeLLM)
    register_llm("a")(_FakeLLM)

    assert available_llms() == ["a", "b"]
