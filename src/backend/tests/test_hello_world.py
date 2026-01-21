"""Tests for hello_world module."""

import sys
import os

# Add parent directory to path to import hello_world
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hello_world import hello_world


def test_hello_world_returns_string():
    """Test that hello_world returns a string."""
    result = hello_world()
    assert isinstance(result, str)


def test_hello_world_contains_expected_text():
    """Test that hello_world contains expected text."""
    result = hello_world()
    assert "Hello World" in result
    assert "4dt907" in result


def test_hello_world_exact_message():
    """Test the exact message returned by hello_world."""
    result = hello_world()
    assert result == "Hello World from 4dt907!"
