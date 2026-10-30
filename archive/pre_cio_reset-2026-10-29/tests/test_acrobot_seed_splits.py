"""Validation episodes must not reuse the search reset/noise streams."""

import sys

import pytest

from scripts.tuning.tune_acrobot import parse_args


def test_default_search_and_validation_seeds_are_disjoint(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["tune_acrobot"])
    args = parse_args()
    assert set(args.search_seeds).isdisjoint(args.validation_seeds)


def test_explicit_overlapping_seed_sets_are_rejected(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["tune_acrobot", "--search-seeds", "1,2",
                                      "--validation-seeds", "2,3"])
    with pytest.raises(SystemExit, match="2"):
        parse_args()
