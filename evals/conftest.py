"""
conftest.py — Shared pytest fixtures for the evals suite.

Adds the project root to sys.path so all eval tests can import from
agents/, core/, and data/ without installing the package.
"""

import pathlib
import sys

# Ensure project root is on the path regardless of how pytest is invoked.
PROJECT_ROOT = pathlib.Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
