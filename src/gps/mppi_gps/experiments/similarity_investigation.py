"""Compatibility CLI for the J1 library-owned lifecycle."""

from ..similarity_launch import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
