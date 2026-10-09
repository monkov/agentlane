"""Validate MCP endpoint URLs before they reach an HTTP client."""

from typing import Any
from urllib.parse import parse_qsl, urlsplit

_CREDENTIAL_QUERY_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "authorization",
        "authorization_code",
        "client_secret",
        "code",
        "id_token",
        "key",
        "password",
        "refresh_token",
        "secret",
        "session_state",
        "state",
        "token",
    }
)


def is_credential_query_key(key: str) -> bool:
    """Return whether a URL field can hold credentials or OAuth state."""
    normalized = key.lower().replace("-", "_")
    return normalized in _CREDENTIAL_QUERY_KEYS or normalized.endswith(
        ("_token", "_secret", "_api_key", "_credentials")
    )


def validate_http_url(url: str, allow_insecure_http: bool = False) -> None:
    """Reject insecure endpoints and credentials without echoing the URL."""
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        _ = parsed.port
        query = parse_qsl(parsed.query, keep_blank_values=True, max_num_fields=1000)
    except ValueError:
        raise ValueError("MCP Streamable HTTP URL is invalid.") from None
    if parsed.scheme not in {"http", "https"} or not hostname:
        raise ValueError("MCP Streamable HTTP URL must be an absolute HTTP URL.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("MCP Streamable HTTP URL must not contain credentials.")
    if parsed.fragment or any(is_credential_query_key(key) for key, _ in query):
        raise ValueError(
            "MCP Streamable HTTP URL must not contain credentials, OAuth "
            "callback parameters, or a fragment."
        )
    if parsed.scheme != "https" and not allow_insecure_http:
        raise ValueError(
            "MCP Streamable HTTP requires HTTPS. Set allow_insecure_http=True "
            "only for development."
        )


async def validate_redirect(response: Any) -> None:
    """Reject unsafe redirects before the HTTP client follows them."""
    if response.status_code < 300 or response.status_code >= 400:
        return
    location = response.headers.get("location")
    if location is None:
        return
    source = response.request.url
    target = source.join(location)
    validate_http_url(str(target), allow_insecure_http=source.scheme == "http")
    same_origin = (
        source.scheme == target.scheme
        and source.host == target.host
        and source.port == target.port
    )
    secure_upgrade = (
        source.scheme == "http"
        and target.scheme == "https"
        and source.host == target.host
        and source.port in {None, 80}
        and target.port in {None, 443}
    )
    if not same_origin and not secure_upgrade:
        raise ValueError("MCP redirect to a different origin is not allowed.")
