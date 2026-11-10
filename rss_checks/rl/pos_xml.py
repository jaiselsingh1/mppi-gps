"""Position-controlled Walker (PREREG_pos.md; rss_checks/rl). Writes assets/walker2d_pos_kp<KP>.xml: the stock
assets/walker2d.xml with its 6 torque motors replaced by MuJoCo general actuators acting as PD position servos.
Action u in [-1, 1] is an offset from the standing pose (all joints at 0): target q* = u rad (amendment ~00:30, PREREG_pos.md:
the range mapping made u = 0 a deep crouch, and MPPI's plan starts at 0). One unit = 1 rad = about the motor's full
100 N m at kp 100, so MPPI's noise is the same push size as with torques; targets past a joint limit press on the limit.
Torque = kp (q* - q) - kv qdot, clipped to +-100 N m (the motors' gear-100 limit), kv = kp / 10 (0.1 s time constant).
  pos_xml.py KP [KP ...]       -> the XML files
  pos_xml.py hold KP [KP ...]  -> stand check: constant targets at the standing pose (joints at 0), 8 s, 16 starts"""
import re, sys
from pathlib import Path
import numpy as np

ROOT = Path.home() / "mppi-gps-claude-sac"; SRC = ROOT / "assets/walker2d.xml"
RANGES = dict(thigh_joint=(-150, 0), leg_joint=(-150, 0), foot_joint=(-45, 45),
              thigh_left_joint=(-150, 0), leg_left_joint=(-150, 0), foot_left_joint=(-45, 45))


def write(kp, scale=1.0):
    kv = kp / 10.0; lines = []
    for j, (lo, hi) in RANGES.items():
        a, b = scale, 0.0
        lines.append(f'    <general joint="{j}" ctrllimited="true" ctrlrange="-1 1" forcelimited="true" forcerange="-100 100" '
                     f'gaintype="fixed" gainprm="{kp * a:.6f}" biastype="affine" biasprm="{kp * b:.6f} {-kp:.6f} {-kv:.6f}"/>')
    xml = SRC.read_text()
    xml = re.sub(r"(<actuator>).*?(</actuator>)", "\\1\n" + "\n".join(lines) + "\n  \\2", xml, flags=re.S)
    name = f"walker2d_pos_off_kp{kp:g}.xml" if scale == 1.0 else f"walker2d_pos_off_s{scale:g}_kp{kp:g}.xml"
    out = ROOT / "assets" / name; out.write_text(xml); return out


def hold(kp):
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(write(kp))); X = np.load(ROOT / "rss_checks/rl/standing_starts.npz")["states"][:16]
    u = np.zeros(6)   # targets at joint angle 0 (the standing pose)
    up, sat = 0, []
    for x in X:
        d = mujoco.MjData(m); mujoco.mj_setState(m, d, x, mujoco.mjtState.mjSTATE_FULLPHYSICS); mujoco.mj_forward(m, d); ok = True
        for t in range(4000):
            d.ctrl[:] = u; mujoco.mj_step(m, d); sat.append(np.mean(np.abs(d.actuator_force) >= 99.9))
            if not (0.8 < d.qpos[1] < 2.0 and abs(d.qpos[2]) < 1.0):
                ok = False; break
        up += ok
    return dict(kp=kp, standing_targets=u.round(3).tolist(), stood_8s=up, of=len(X), saturated_share=float(np.mean(sat)))


if __name__ == "__main__":
    if sys.argv[1] == "hold":
        for kp in map(float, sys.argv[2:]):
            print(hold(kp), flush=True)
    elif sys.argv[1] == "grid":   # pos_xml.py grid SCALE:KP ...
        for sk in sys.argv[2:]:
            sc, kp = map(float, sk.split(":")); print(write(kp, sc))
    else:
        for kp in map(float, sys.argv[1:]):
            print(write(kp))
