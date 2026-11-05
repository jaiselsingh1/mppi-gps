"""Metadata recovery must retain failed execution and its original controller."""

import copy

import numpy as np
import pytest

from src.gps.mppi_gps.reporting.acrobot_attractor_recovery import derive_packet, validate_failure


def rows():
    values = [
        dict(kind="shell", factor=f, capture=f <= 4)
        for f in (0.25, 1.0, 4.0, 16.0)
        for _ in range(24)
    ]
    values += [dict(kind="confirm_boundary", capture=True) for _ in range(64)]
    values += [dict(kind="confirm_interior", capture=True) for _ in range(64)]
    return values


def test_recovered_packet_keeps_gains_and_cannot_bypass_confirmation_failure():
    initial = dict(
        P=np.eye(4).tolist(), K=[[1.0, 2.0, 3.0, 4.0]], unsaturated_rho=2.0, A=np.eye(4).tolist()
    )
    before = copy.deepcopy(initial)
    got = derive_packet(initial, rows(), dict(shell_factors=[0.25, 1.0, 4.0, 16.0]))
    assert got["rho"] == 8 and got["join_validation_passed"]
    assert initial == before and all(got[k] == v for k, v in initial.items())
    broken = rows()
    broken[-1]["capture"] = False
    assert not derive_packet(initial, broken, dict(shell_factors=[0.25, 1.0, 4.0, 16.0]))[
        "join_validation_passed"
    ]
    with pytest.raises(ValueError, match="128"):
        derive_packet(initial, rows()[:-1], dict(shell_factors=[0.25, 1.0, 4.0, 16.0]))


def test_only_exact_storage_finalization_error_can_be_recovered():
    message = "FileExistsError: frozen original path"
    good = dict(
        status="STOPPED_INVALID_EXECUTION",
        stage="lqr",
        error=message,
        unknown_partial_work={"native_controls": 0},
    )
    validate_failure(good, dict(original_error_exact=message))
    for field, value in [
        ("error", "PhysicsError: invalid"),
        ("status", "STOPPED_RESOURCES"),
        ("unknown_partial_work", {"native_controls": 1}),
    ]:
        altered = dict(good, **{field: value})
        with pytest.raises(ValueError):
            validate_failure(altered, dict(original_error_exact=message))
