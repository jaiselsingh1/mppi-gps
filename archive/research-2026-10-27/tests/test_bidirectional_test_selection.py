"""Pure checks of frozen accountable-test selection; no engine or actor calls."""
from pathlib import Path
import pytest
from scripts.diagnostics.run_bidirectional_tests import select_test_roster


def test_default_and_explicit_rosters_are_regular_sorted_local_files(tmp_path):
    for name in ("test_bidirectional_b.py","test_bidirectional_a.py","test_other.py"):
        (tmp_path/name).write_text("# fixture\n")
    assert [p.name for p in select_test_roster(tmp_path)]==["test_bidirectional_a.py","test_bidirectional_b.py"]
    assert [p.name for p in select_test_roster(tmp_path,["test_bidirectional_b.py"])]==["test_bidirectional_b.py"]


@pytest.mark.parametrize("names",[[],["../test_bidirectional_a.py"],["test_bidirectional_missing.py"],["test_other.py"],["test_bidirectional_a.py"]*2])
def test_invalid_subset_rejected_before_test_execution(tmp_path,names):
    (tmp_path/"test_bidirectional_a.py").write_text("# fixture\n")
    (tmp_path/"test_other.py").write_text("# fixture\n")
    with pytest.raises(ValueError):select_test_roster(tmp_path,names)


def test_symlink_cannot_escape_frozen_test_inventory(tmp_path):
    original=tmp_path/"outside.py";original.write_text("# fixture\n")
    (tmp_path/"test_bidirectional_link.py").symlink_to(original)
    assert select_test_roster(tmp_path)==[]
    with pytest.raises(ValueError):select_test_roster(tmp_path,["test_bidirectional_link.py"])
