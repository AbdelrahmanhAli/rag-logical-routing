"""Tests for the routing decision schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from route import Route


def test_document_search_route_is_valid():
    route = Route(destination="document_search", reason="asks about the corpus")
    assert route.destination == "document_search"


def test_general_route_is_valid():
    assert Route(destination="general", reason="small talk").destination == "general"


def test_unknown_destination_is_rejected():
    with pytest.raises(ValidationError):
        Route(destination="web_search", reason="not a supported route")


def test_reason_is_required():
    with pytest.raises(ValidationError):
        Route(destination="general")
