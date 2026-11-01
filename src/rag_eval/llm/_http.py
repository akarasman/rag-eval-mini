"""HTTP plumbing shared by the backends that talk to a server (private to the llm package).

One place turns every way a request can go wrong into an LLMError with a message that says what to check,
so the backends only describe their request and where the text sits in the reply.
"""
from typing import Any, Dict, Optional

import requests

from .base import LLMError

_BODY_PREVIEW_CHARS = 200


def require_http_url(parameter: str, value: str) -> None:
    """Check a server address, so a typo fails when the backend is built and not at the first request.

    Args:
        parameter: the parameter's name, used in the message.
        value: the address to check.

    Raises:
        ValueError: if it does not start with http:// or https://.
    """
    if not value.strip().lower().startswith(("http://", "https://")):
        raise ValueError(f"{parameter} must start with http:// or https://, got {value!r}")


def validate_server_settings(
    model: str, url_parameter: str, url: str, max_tokens: int, timeout: float, temperature: float
) -> None:
    """Check the settings every server backend shares, so a mistake fails when the backend is built.

    Args:
        model: the model name; must not be blank.
        url_parameter: the name of the address parameter, used in the message ("host", "base_url").
        url: the server address.
        max_tokens: the most tokens the model may generate.
        timeout: seconds to wait for a reply.
        temperature: the sampling temperature.

    Raises:
        ValueError: if model is blank, url is not an http(s) address, max_tokens or timeout is not
            positive, or temperature is negative.
    """
    if not model.strip():
        raise ValueError("model must not be empty")
    require_http_url(url_parameter, url)
    if max_tokens <= 0 or timeout <= 0:
        raise ValueError("max_tokens and timeout must be positive")
    if temperature < 0:
        raise ValueError("temperature must not be negative")


def post_json(
    url: str, payload: Dict[str, Any], timeout: float, headers: Optional[Dict[str, str]] = None, hint: str = ""
) -> Any:
    """POST a JSON body and return the decoded JSON reply.

    Headers are never put into an error message, so an Authorization header cannot leak.

    Args:
        url: the full endpoint URL.
        payload: the JSON request body.
        timeout: seconds to wait for the server.
        headers: extra request headers, e.g. Authorization.
        hint: appended to a connection error to say what to check, e.g. "is the server running?".

    Raises:
        LLMError: if the server cannot be reached or times out, answers with a non-2xx status, or the
            body is not JSON.
    """
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.Timeout as error:
        raise LLMError(f"request to {url} timed out after {timeout}s") from error
    except requests.ConnectionError as error:
        raise LLMError(f"cannot connect to {url}{f' ({hint})' if hint else ''}") from error
    except requests.RequestException as error:
        raise LLMError(f"request to {url} failed: {type(error).__name__}") from error

    if not 200 <= response.status_code < 300:
        preview = response.text[:_BODY_PREVIEW_CHARS]
        raise LLMError(f"{url} answered with status {response.status_code}: {preview}")
    try:
        return response.json()
    except ValueError as error:
        raise LLMError(f"{url} did not return JSON: {response.text[:_BODY_PREVIEW_CHARS]}") from error
