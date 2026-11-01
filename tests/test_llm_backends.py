"""Tests for the LLM backends. HTTP is faked by replacing requests.post, and the transformers classes
by fakes, so nothing here needs a network, a server or a model (one real-model test is marked slow)."""
import numpy as np
import pytest
import requests

from rag_eval.llm.base import LLMError
from rag_eval.llm.ollama import OllamaLLM
from rag_eval.llm.openai_compatible import OpenAICompatibleLLM
from rag_eval.llm.registry import available_llms, build_llm
from rag_eval.llm.transformers_llm import TransformersLLM

_NOT_JSON = object()


class _FakeResponse:
    def __init__(self, json_data=None, status_code=200, text=""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self):
        if self._json_data is _NOT_JSON:
            raise ValueError("not json")
        return self._json_data


class _PostRecorder:
    """Stands in for requests.post: records every call, then returns a canned response or raises."""

    def __init__(self):
        self.calls = []
        self.response = _FakeResponse()
        self.error = None

    def __call__(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        if self.error is not None:
            raise self.error
        return self.response


@pytest.fixture
def post(monkeypatch) -> _PostRecorder:
    recorder = _PostRecorder()
    monkeypatch.setattr(requests, "post", recorder)
    return recorder


def test_every_backend_is_registered():
    assert {"ollama", "openai_compatible", "transformers"} <= set(available_llms())


# --- Ollama --------------------------------------------------------------------------

OLLAMA_REPLY = {"message": {"role": "assistant", "content": '{"answerable": false}'}}


def test_ollama_sends_the_chat_request_and_returns_the_reply_text(post):
    post.response = _FakeResponse(OLLAMA_REPLY)
    llm = OllamaLLM(model="llama3.2", temperature=0.2, max_tokens=64, timeout=9.0)

    text = llm.generate("the prompt", system="the system")

    assert text == '{"answerable": false}'
    (call,) = post.calls
    assert call["url"] == "http://localhost:11434/api/chat"
    assert call["timeout"] == 9.0
    assert call["json"] == {
        "model": "llama3.2",
        "messages": [{"role": "system", "content": "the system"}, {"role": "user", "content": "the prompt"}],
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 64},
        "format": "json",
    }


def test_ollama_without_a_system_message_sends_only_the_user_message(post):
    post.response = _FakeResponse(OLLAMA_REPLY)

    OllamaLLM(model="m").generate("hi")

    assert post.calls[0]["json"]["messages"] == [{"role": "user", "content": "hi"}]


def test_ollama_json_mode_off_sends_no_format(post):
    post.response = _FakeResponse(OLLAMA_REPLY)

    OllamaLLM(model="m", json_mode=False).generate("hi")

    assert "format" not in post.calls[0]["json"]


def test_ollama_uses_the_configured_host_and_ignores_a_trailing_slash(post):
    post.response = _FakeResponse(OLLAMA_REPLY)

    OllamaLLM(model="m", host="http://gpu-box:8000/").generate("hi")

    assert post.calls[0]["url"] == "http://gpu-box:8000/api/chat"


def test_ollama_connection_error_says_what_to_check(post):
    post.error = requests.ConnectionError("refused")

    with pytest.raises(LLMError, match="Ollama server running at http://localhost:11434"):
        OllamaLLM(model="m").generate("hi")


def test_a_timeout_is_an_llm_error(post):
    post.error = requests.Timeout("slow")

    with pytest.raises(LLMError, match="timed out after 120.0s"):
        OllamaLLM(model="m").generate("hi")


def test_any_other_request_failure_is_an_llm_error(post):
    post.error = requests.exceptions.InvalidURL("bad")

    with pytest.raises(LLMError, match="failed"):
        OllamaLLM(model="m").generate("hi")


def test_a_non_2xx_status_is_an_llm_error_with_the_status_and_the_start_of_the_body(post):
    post.response = _FakeResponse(status_code=404, text='{"error": "model \'x\' not found"}' + " padding" * 100)

    with pytest.raises(LLMError) as error:
        OllamaLLM(model="x").generate("hi")

    message = str(error.value)
    assert "404" in message and "model 'x' not found" in message
    assert len(message) < 400  # only the start of the body


@pytest.mark.parametrize("reply", [
    _NOT_JSON,
    ["not", "an", "object"],
    {},
    {"message": {}},
    {"message": None},
    {"message": {"content": None}},
    {"message": {"content": 5}},
])
def test_ollama_malformed_replies_are_llm_errors(post, reply):
    post.response = _FakeResponse(reply)

    with pytest.raises(LLMError):
        OllamaLLM(model="m").generate("hi")


@pytest.mark.parametrize("params, match", [
    ({"model": ""}, "model"),
    ({"model": "  "}, "model"),
    ({"model": "m", "max_tokens": 0}, "max_tokens"),
    ({"model": "m", "timeout": 0}, "timeout"),
    ({"model": "m", "temperature": -0.1}, "temperature"),
    ({"model": "m", "host": "localhost:11434"}, "host"),
])
def test_ollama_rejects_bad_parameters_when_built(params, match):
    with pytest.raises(ValueError, match=match):
        OllamaLLM(**params)


def test_ollama_builds_from_a_config_and_requires_a_model():
    assert isinstance(build_llm("ollama", {"model": "m", "temperature": 0.1}), OllamaLLM)
    with pytest.raises(ValueError, match="Invalid parameters for LLM 'ollama'"):
        build_llm("ollama", {})


# --- OpenAI-compatible ---------------------------------------------------------------

KEY = "sk-secret-key-123"
API_REPLY = {"choices": [{"message": {"role": "assistant", "content": "the reply"}}]}


@pytest.fixture
def api_key(monkeypatch) -> str:
    monkeypatch.setenv("TEST_LLM_API_KEY", KEY)
    return KEY


def _api(**overrides) -> OpenAICompatibleLLM:
    params = {"model": "m", "base_url": "https://api.example.com/v1", "api_key_env": "TEST_LLM_API_KEY"}
    params.update(overrides)
    return OpenAICompatibleLLM(**params)


def test_openai_compatible_sends_the_request_with_the_key_and_returns_the_text(post, api_key):
    post.response = _FakeResponse(API_REPLY)

    text = _api(model="big-model", temperature=0.3, max_tokens=99, timeout=7.0).generate("the prompt", system="sys")

    assert text == "the reply"
    (call,) = post.calls
    assert call["url"] == "https://api.example.com/v1/chat/completions"
    assert call["headers"]["Authorization"] == f"Bearer {KEY}"
    assert call["timeout"] == 7.0
    assert call["json"] == {
        "model": "big-model",
        "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "the prompt"}],
        "temperature": 0.3,
        "max_tokens": 99,
    }


def test_openai_compatible_ignores_a_trailing_slash_in_the_base_url(post, api_key):
    post.response = _FakeResponse(API_REPLY)

    _api(base_url="https://api.example.com/v1/").generate("hi")

    assert post.calls[0]["url"] == "https://api.example.com/v1/chat/completions"


def test_a_missing_key_fails_when_built_and_names_the_variable(monkeypatch):
    monkeypatch.delenv("TEST_LLM_API_KEY", raising=False)

    with pytest.raises(ValueError, match="TEST_LLM_API_KEY"):
        _api()


def test_an_empty_key_is_the_same_as_a_missing_one(monkeypatch):
    monkeypatch.setenv("TEST_LLM_API_KEY", "   ")

    with pytest.raises(ValueError, match="TEST_LLM_API_KEY"):
        _api()


def test_the_key_never_appears_in_repr_str_or_the_construction_error(api_key, monkeypatch):
    llm = _api()

    assert KEY not in repr(llm) and KEY not in str(llm)
    assert "TEST_LLM_API_KEY" in repr(llm)  # the variable NAME is fine

    monkeypatch.setenv("OTHER_KEY", KEY)
    with pytest.raises(ValueError) as error:
        _api(api_key_env="OTHER_KEY", base_url="not-a-url")
    assert KEY not in str(error.value)


def test_the_key_never_appears_in_an_error_message(post, api_key):
    post.response = _FakeResponse(status_code=401, text="invalid credentials")

    with pytest.raises(LLMError) as error:
        _api().generate("hi")

    assert "401" in str(error.value)
    assert KEY not in str(error.value)


def test_openai_compatible_connection_error_is_an_llm_error(post, api_key):
    post.error = requests.ConnectionError("refused")

    with pytest.raises(LLMError, match="https://api.example.com/v1/chat/completions"):
        _api().generate("hi")


@pytest.mark.parametrize("reply", [
    _NOT_JSON,
    {},
    {"choices": []},
    {"choices": [{}]},
    {"choices": [{"message": {}}]},
    {"choices": [{"message": {"content": None}}]},
    {"choices": "nope"},
    [],
])
def test_openai_compatible_malformed_replies_are_llm_errors(post, api_key, reply):
    post.response = _FakeResponse(reply)

    with pytest.raises(LLMError):
        _api().generate("hi")


@pytest.mark.parametrize("overrides, match", [
    ({"model": ""}, "model"),
    ({"base_url": "api.example.com"}, "base_url"),
    ({"max_tokens": 0}, "max_tokens"),
    ({"timeout": -1}, "timeout"),
    ({"temperature": -1}, "temperature"),
    ({"api_key_env": ""}, "api_key_env"),
])
def test_openai_compatible_rejects_bad_parameters_when_built(api_key, overrides, match):
    with pytest.raises(ValueError, match=match):
        _api(**overrides)


def test_openai_compatible_requires_base_url_and_key_variable(api_key):
    with pytest.raises(ValueError, match="Invalid parameters for LLM 'openai_compatible'"):
        build_llm("openai_compatible", {"model": "m"})


# --- transformers: no model is loaded by these tests -----------------------------------

@pytest.mark.parametrize("params, match", [
    ({"model_name": ""}, "model_name"),
    ({"model_name": "m", "max_new_tokens": 0}, "max_new_tokens"),
    ({"model_name": "m", "temperature": -1}, "temperature"),
])
def test_transformers_rejects_bad_parameters_when_built(params, match):
    with pytest.raises(ValueError, match=match):
        TransformersLLM(**params)


def test_transformers_has_no_default_model():
    with pytest.raises(ValueError, match="Invalid parameters for LLM 'transformers'"):
        build_llm("transformers", {})


class _FakeInputs(dict):
    def to(self, device):
        self.device = device
        return self


class _FakeTokenizer:
    pad_token_id = None
    eos_token_id = 7

    def __init__(self):
        self.template_calls = []

    def apply_chat_template(self, messages, add_generation_prompt, return_tensors, return_dict):
        self.template_calls.append(messages)
        return _FakeInputs(input_ids=np.array([[1, 2, 3]]), attention_mask=np.array([[1, 1, 1]]))

    def decode(self, tokens, skip_special_tokens):
        assert skip_special_tokens
        return " ".join(str(int(token)) for token in tokens)


class _FakeModel:
    def __init__(self):
        self.generate_calls = []
        self.device = None

    def to(self, device):
        self.device = device

    def eval(self):
        pass

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return np.array([[1, 2, 3, 10, 11]])  # the 3 prompt tokens, then 2 new ones


@pytest.fixture
def fake_transformers(monkeypatch):
    transformers = pytest.importorskip("transformers")
    tokenizer, model = _FakeTokenizer(), _FakeModel()
    loads = []
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda name: loads.append(name) or tokenizer)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda name: model)
    return tokenizer, model, loads


def test_building_a_transformers_backend_loads_nothing(monkeypatch):
    transformers = pytest.importorskip("transformers")

    def fail(name):
        raise AssertionError("the model must not be loaded at construction")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fail)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", fail)

    llm = build_llm("transformers", {"model_name": "no/such-model"})

    assert llm._model is None and llm._tokenizer is None


def test_transformers_returns_only_the_new_text_and_loads_once(fake_transformers):
    tokenizer, model, loads = fake_transformers
    llm = TransformersLLM(model_name="some/model", device="cpu")

    assert llm.generate("question", system="rules") == "10 11"  # not "1 2 3 10 11"
    llm.generate("question again")

    assert loads == ["some/model"]
    assert tokenizer.template_calls[0] == [
        {"role": "system", "content": "rules"}, {"role": "user", "content": "question"},
    ]
    assert tokenizer.template_calls[1] == [{"role": "user", "content": "question again"}]
    assert model.device == "cpu"


def test_zero_temperature_means_greedy_decoding_without_a_temperature_argument(fake_transformers):
    _, model, _ = fake_transformers

    TransformersLLM(model_name="m", temperature=0.0, max_new_tokens=33).generate("hi")

    options = model.generate_calls[0]
    assert options["do_sample"] is False
    assert "temperature" not in options
    assert options["max_new_tokens"] == 33
    assert options["pad_token_id"] == 7  # falls back to the end-of-text token when there is no pad token


def test_a_positive_temperature_samples(fake_transformers):
    _, model, _ = fake_transformers

    TransformersLLM(model_name="m", temperature=0.7).generate("hi")

    options = model.generate_calls[0]
    assert options["do_sample"] is True and options["temperature"] == 0.7


def test_the_prompt_tensors_are_passed_to_the_model(fake_transformers):
    _, model, _ = fake_transformers

    TransformersLLM(model_name="m").generate("hi")

    assert set(model.generate_calls[0]) >= {"input_ids", "attention_mask"}


def test_a_model_that_cannot_be_loaded_is_an_llm_error_and_loading_is_retried(monkeypatch):
    transformers = pytest.importorskip("transformers")
    attempts = []

    def fail(name):
        attempts.append(name)
        raise OSError("not found on the hub")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fail)
    llm = TransformersLLM(model_name="no/such-model")

    for _ in range(2):
        with pytest.raises(LLMError, match="could not load the model 'no/such-model'.*not found on the hub"):
            llm.generate("hi")

    assert len(attempts) == 2


def test_a_tokenizer_without_a_chat_template_is_an_llm_error(fake_transformers):
    tokenizer, _, _ = fake_transformers

    def no_template(*args, **kwargs):
        raise ValueError("Cannot use chat template functions")

    tokenizer.apply_chat_template = no_template

    with pytest.raises(LLMError, match="cannot format the prompt"):
        TransformersLLM(model_name="m").generate("hi")


@pytest.mark.slow
def test_a_real_small_chat_model_generates_text():
    llm = TransformersLLM(model_name="HuggingFaceTB/SmolLM2-135M-Instruct", max_new_tokens=8)

    reply = llm.generate("Say hello.", system="You are brief.")

    assert isinstance(reply, str) and reply.strip()
    assert "Say hello." not in reply  # only the new text, not the prompt
