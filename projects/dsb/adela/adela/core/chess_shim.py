"""Compatibility shim to use bulletchess as the core chess engine.

This module exposes a `chess` namespace backed by `bulletchess` for core
board/move operations. It also optionally exposes a `pgn` namespace backed by
`python-chess` for PGN parsing utilities if available.

Usage:
  from adela.core.chess_shim import chess
  # Optional PGN parsing
  from adela.core.chess_shim import pgn as chess_pgn

If `python-chess` is not installed, `pgn` will be None. Code paths that rely on
PGN parsing should guard against this and provide a helpful error.
"""

from __future__ import annotations

from typing import Any


# Prefer bulletchess for core engine functionality
try:
  import bulletchess as chess  # type: ignore
except Exception as err:  # pragma: no cover - fail fast with clear message
  raise ImportError(
    "bulletchess is required as the chess engine. Please install it with 'pip install bulletchess'."
  ) from err


# Optional PGN parsing support via python-chess
try:
  import chess.pgn as pgn  # type: ignore
except Exception:
  pgn = None  # type: ignore[assignment]


def require_pgn() -> Any:
  """Return the PGN module or raise a clear error if unavailable."""
  if pgn is None:
    raise ImportError(
      "PGN parsing requires the 'python-chess' package. Install it to enable PGN features."
    )
  return pgn


