"""Fixtures shared by the MCP integration tests."""

import pytest

from .helpers import ConnectionInstaller


@pytest.fixture(name="install_connection")
def fixture_install_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> ConnectionInstaller:
    return ConnectionInstaller(monkeypatch)
