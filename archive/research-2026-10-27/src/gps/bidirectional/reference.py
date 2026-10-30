"""A disclosed energy-shaping/upright-LQR Acrobot reference.

This is a fixed analytic benchmark, not a teacher for global-policy fitting.
Reading model parameters and computing actions performs no integration, model
rollout, ``mj_forward``, or policy inference. Every executed control goes through
the caller's QueryExecutor. Success is measured externally, never assumed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_continuous_are

from .evidence import EpisodeTrace
from .task import Snapshot


_XML_SHA256 = hashlib.sha256((Path(__file__).resolve().parents[3]/
                            "assets/acrobot.xml").read_bytes()).hexdigest()


@dataclass(frozen=True)
class ReferenceConfig:
    energy_gain: float = .1
    excitation_amplitude: float = .2
    excitation_frequency_hz: float = 1.
    switch_angle_norm: float = .3
    switch_velocity_norm: float = 2.
    Q: tuple[float, float, float, float] = (10., 10., 1., 1.)
    R: float = .1

    def __post_init__(self):
        for name in ("energy_gain", "excitation_frequency_hz", "switch_angle_norm",
                     "switch_velocity_norm", "R"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"Positive finite {name} required")
        if not np.isfinite(self.excitation_amplitude) or not 0 <= self.excitation_amplitude <= 1:
            raise ValueError("Excitation amplitude must lie in [0,1]")
        q = tuple(self.Q)
        if len(q) != 4 or not np.isfinite(q).all() or min(q) <= 0:
            raise ValueError("Q must contain four positive finite diagonal weights")
        object.__setattr__(self, "Q", q)


def _id(model, kind, name):
    return int(getattr(model, kind)(name).id)


def _inertia_y(principal, quat):
    """Body-local yy inertia from MuJoCo's principal-inertia quaternion."""
    w, x, y, z = np.asarray(quat, dtype=float)
    if not np.isclose(w*w+x*x+y*y+z*z, 1., rtol=0, atol=1e-10):
        raise ValueError("Model inertia quaternion is not normalized")
    row_y = np.array([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)])
    return float(np.dot(np.square(row_y), principal))


@dataclass(frozen=True)
class AcrobotParameters:
    m1: float
    m2: float
    c1: float
    c2: float
    l1: float
    I1: float
    I2: float
    gravity: float
    damping1: float
    damping2: float
    gear: float
    pivot_height: float
    armature1: float = 0.
    armature2: float = 0.

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Finite Acrobot parameters required")
        if min(self.m1, self.m2, self.c1, self.c2, self.l1, self.I1,
               self.I2, self.gravity, self.gear) <= 0:
            raise ValueError("Positive physical mass/geometry/inertia/gear required")
        if min(self.damping1, self.damping2, self.armature1, self.armature2) < 0:
            raise ValueError("Damping and armature cannot be negative")

    @classmethod
    def from_model(cls, model):
        if (model.nq, model.nv, model.nu) != (2, 2, 1):
            raise ValueError("Reference requires the unchanged two-link Acrobot")
        upper, lower = _id(model, "body", "upper_arm"), _id(model, "body", "lower_arm")
        shoulder, elbow = _id(model, "joint", "shoulder"), _id(model, "joint", "elbow")
        actuator = _id(model, "actuator", "elbow")
        if int(model.body_parentid[lower]) != upper:
            raise ValueError("Lower body must be attached to upper arm")
        for body in (upper, lower):
            if (not np.allclose(model.body_ipos[body, :2], 0., atol=1e-12, rtol=0)
                    or not np.allclose(model.body_quat[body], [1., 0., 0., 0.], atol=1e-12, rtol=0)):
                raise ValueError("Reference requires original planar link COMs/frames")
        if not np.allclose(model.body_pos[lower, :2], 0., atol=1e-12, rtol=0):
            raise ValueError("Reference requires original lower-link attachment")
        for joint in (shoulder, elbow):
            if not np.allclose(model.jnt_axis[joint], [0., 1., 0.], atol=1e-12, rtol=0):
                raise ValueError("Reference requires original positive-y hinge axes")
        gravity = np.asarray(model.opt.gravity)
        if not np.allclose(gravity[:2], 0., atol=1e-12, rtol=0) or gravity[2] >= 0:
            raise ValueError("Reference requires downward world-z gravity")
        d0, d1 = int(model.jnt_dofadr[shoulder]), int(model.jnt_dofadr[elbow])
        if (d0, d1) != (0, 1):
            raise ValueError("State coordinates must be shoulder then relative elbow")
        return cls(float(model.body_mass[upper]), float(model.body_mass[lower]),
            float(model.body_ipos[upper, 2]), float(model.body_ipos[lower, 2]),
            float(model.body_pos[lower, 2]),
            _inertia_y(model.body_inertia[upper], model.body_iquat[upper]),
            _inertia_y(model.body_inertia[lower], model.body_iquat[lower]),
            float(-gravity[2]), float(model.dof_damping[d0]), float(model.dof_damping[d1]),
            float(model.actuator_gear[actuator, 0]), float(model.body_pos[upper, 2]),
            float(model.dof_armature[d0]), float(model.dof_armature[d1]))

    def mass_matrix(self, q):
        elbow = float(np.asarray(q)[1])
        cross = self.m2*self.l1*self.c2*np.cos(elbow)
        lower = self.I2+self.m2*self.c2**2
        return np.array([[self.I1+self.m1*self.c1**2+lower+self.m2*self.l1**2+2*cross+self.armature1,
                          lower+cross], [lower+cross, lower+self.armature2]])

    def energy(self, q, velocity):
        q, velocity = np.asarray(q, dtype=float), np.asarray(velocity, dtype=float)
        if q.shape != (2,) or velocity.shape != (2,) or not np.isfinite([q, velocity]).all():
            raise ValueError("Energy requires finite two-joint positions and velocities")
        potential = self.gravity*((self.m1+self.m2)*self.pivot_height
            +(self.m1*self.c1+self.m2*self.l1)*np.cos(q[0])
            +self.m2*self.c2*np.cos(q[0]+q[1]))
        return float(.5*velocity@self.mass_matrix(q)@velocity+potential)

    def upright_linearization(self):
        mass = self.mass_matrix(np.zeros(2))
        b = self.m2*self.c2
        a = self.m1*self.c1+self.m2*self.l1
        destabilizing_gravity = self.gravity*np.array([[a+b, b], [b, b]])
        damping = np.diag([self.damping1, self.damping2])
        A = np.block([[np.zeros((2,2)), np.eye(2)],
                      [np.linalg.solve(mass, destabilizing_gravity), -np.linalg.solve(mass, damping)]])
        B = np.concatenate([np.zeros(2), np.linalg.solve(mass, [0., self.gear])])[:, None]
        return A, B


class EnergyLQRReference:
    def __init__(self, task_or_model, config: ReferenceConfig | dict | None = None):
        self.config = ReferenceConfig() if config is None else (
            ReferenceConfig(**config) if isinstance(config, dict) else config)
        if not isinstance(self.config, ReferenceConfig):
            raise TypeError("ReferenceConfig required")
        value = task_or_model
        if isinstance(value, AcrobotParameters):
            self.parameters = value
        else:
            model = value.env.model if hasattr(value, "env") else value
            self.parameters = AcrobotParameters.from_model(model)
        self.A, self.B = self.parameters.upright_linearization()
        P = solve_continuous_are(self.A, self.B, np.diag(self.config.Q), np.array([[self.config.R]]))
        self.K = (self.B.T@P)/self.config.R
        if np.max(np.linalg.eigvals(self.A-self.B@self.K).real) >= 0:
            raise ValueError("Analytic upright LQR failed its stability check")
        self.target_energy = self.parameters.energy(np.zeros(2), np.zeros(2))
        payload = dict(algorithm="bounded-energy-shaping-upright-lqr-v1",
            config=asdict(self.config), model_parameters=asdict(self.parameters),
            acrobot_xml_sha256=_XML_SHA256, lqr_gain=self.K.tolist())
        self.identity_payload = payload
        self.controller_identity = hashlib.sha256(json.dumps(payload, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        self.last_diagnostics = None

    @property
    def identity(self):
        return self.controller_identity

    def act(self, snapshot: Snapshot):
        if not isinstance(snapshot, Snapshot):
            raise TypeError("Reference actions require an explicit saved snapshot")
        q, velocity = snapshot.state[1:3], snapshot.state[3:5]
        wrapped = (q+np.pi)%(2*np.pi)-np.pi
        energy = self.parameters.energy(q, velocity)
        if (np.linalg.norm(wrapped) < self.config.switch_angle_norm
                and np.linalg.norm(velocity) < self.config.switch_velocity_norm):
            raw = -float((self.K@np.concatenate([wrapped, velocity])).item())
            mode, excitation = "upright_lqr", 0.
        else:
            # Deterministic bounded excitation breaks the hanging-rest symmetry.
            # It is disclosed task-independent arithmetic, not a hidden search.
            excitation = self.config.excitation_amplitude*np.cos(
                2*np.pi*self.config.excitation_frequency_hz*snapshot.state[0])
            raw = self.config.energy_gain*(self.target_energy-energy)*velocity[1]+excitation
            mode = "energy_shaping"
        action = np.array([np.clip(raw, -1., 1.)])
        self.last_diagnostics = dict(mode=mode, energy=energy, target_energy=self.target_energy,
            energy_error=self.target_energy-energy, excitation=float(excitation),
            unsaturated_action=float(raw), action=action.tolist(),
            controller_hash=self.controller_identity)
        return action


def execute_reference(executor, initial, episode_id, *, reference=None, source=None):
    """Run one complete remaining analytic continuation, without a rescue."""
    reference = EnergyLQRReference(executor.task) if reference is None else reference
    executor.task.restore(initial)
    owner = dict(actor_hash=None, controller_hash=reference.controller_identity,
        semantics="analytic_reference", excluded_from_teacher_aux=True)
    provenance = dict(source={} if source is None else source,
        controller="analytic_reference", excluded_from_teacher_aux=True,
        reference_identity=reference.identity_payload)
    executor.archive.append("reference_controller", controller_hash=reference.controller_identity,
        identity=reference.identity_payload, excluded_from_teacher_aux=True,
        new_simulator_controls=0, actor_inference_rows=0)
    steps, query_ids = [], []
    while executor.task.remaining:
        snapshot = executor.task.snapshot()
        action = reference.act(snapshot)
        executor.archive.append("reference_arithmetic", episode_id=episode_id,
            context_id=initial.context_id, elapsed=snapshot.elapsed, rows=1,
            new_simulator_controls=0, actor_inference_rows=0, **reference.last_diagnostics)
        result = executor.step(action, metadata={"owner": owner, "episode_id": episode_id,
            "ancestry": initial.to_dict(), "reference": reference.identity_payload,
            "excluded_from_teacher_aux": True})
        steps.append(result)
        query_ids.append(executor.latest_query_id)
    trace = EpisodeTrace.from_steps(initial, steps, config=executor.task.config,
        actor_owner=owner, execution_semantics="analytic_reference", source=provenance,
        query_ids=query_ids)
    saved = trace.archive(executor.archive, episode_id)
    summary = dict(episode_id=episode_id, context_id=initial.context_id, cost=trace.total_cost,
        capture=trace.captured, complete=trace.continuation_complete,
        ordinary_start=initial.elapsed == 0, remaining_steps=executor.task.config.T-initial.elapsed,
        local=bool(np.any((trace.elapsed <= executor.task.config.local_by)&
                          (trace.tip_distances <= executor.task.config.local_distance))),
        actor_hash=None, controller_hash=reference.controller_identity, family_id=None,
        task_hash=initial.task_hash, query_ids=query_ids, influenced=0, changed=0,
        raw=str(executor.archive.path/saved["file"]), deterministic=True,
        excluded_from_teacher_aux=True, semantics="analytic_reference")
    executor.archive.append("episode_outcome", **summary)
    return trace, None, summary
