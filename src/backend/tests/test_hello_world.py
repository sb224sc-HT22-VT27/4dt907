"""Tests for hello_world module."""

import sys
import os

# Add parent directory to path to import hello_world
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hello_world import hello_world


def test_hello_world_returns_expected_string():
    assert hello_world() == "Hello World from 4dt907!"
