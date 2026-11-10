"""Render an S2 policy alone (u = pi(x), deterministic) from held-out standing starts, side by side.
  MUJOCO_GL=egl s2_render.py ARM PT START... -> runs/root/render_<ARM>_<PT stem>.mp4 / .gif"""
import sys
from pathlib import Path

ROOT = Path.home() / "mppi-gps-claude-sac"; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rss_checks"))
sys.path.insert(0, str(ROOT / "rss_checks" / "rl"))
import numpy as np


def main():
    import imageio.v2 as imageio, mujoco, torch
    from PIL import Image, ImageDraw
    from pos_check import obs17
    from posloop import Student, standing
    from rootcause import FS, T, mlp, xml
    from s2_gps import d0
    arm, pt, starts = sys.argv[1], sys.argv[2], [int(a) for a in sys.argv[3:]]
    O0, _ = d0(); pol = Student(O0, seed=0); pol.net.load_state_dict(torch.load(ROOT / f"runs/root/s2_{arm}" / pt)); net = (pol.layers(), pol.m, pol.s)
    EV = standing(61_000_000 + np.arange(32)); mm = mujoco.MjModel.from_xml_path(xml("pos")); dt = mm.opt.timestep * FS
    W, H = 360, 270; ren = mujoco.Renderer(mm, height=H, width=W)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_BODY, "torso"); cam.distance, cam.elevation, cam.azimuth = 4.0, -15.0, 90.0
    panels = []
    for i in starts:
        d = mujoco.MjData(mm); mujoco.mj_setState(mm, d, EV[i], mujoco.mjtState.mjSTATE_FULLPHYSICS); mujoco.mj_forward(mm, d)
        frames, fell, vx = [], None, []
        for t in range(T):
            if t % 4 == 0:
                ren.update_scene(d, camera=cam); im = Image.fromarray(ren.render().copy()); dr = ImageDraw.Draw(im)
                v = np.mean(vx[-25:]) if vx else 0.0
                dr.rectangle((0, 0, W, 20), fill=(250, 250, 248))
                dr.text((6, 5), f"held-out start {i}   t {t * dt:4.1f} s   x {d.qpos[0]:5.1f} m   vx {v:4.2f}" + ("   FELL" if fell else ""), fill=(20, 20, 20))
                frames.append(np.asarray(im))
            o = obs17(np.r_[0.0, d.qpos, d.qvel][None])
            d.ctrl[:] = np.clip(mlp(net, o)[0], -1, 1)
            for _ in range(FS):
                mujoco.mj_step(mm, d)
            vx.append(d.qvel[0])
            if fell is None and not (0.8 < d.qpos[1] < 2.0 and abs(d.qpos[2]) < 1.0):
                fell = t + 1
        print(dict(start=i, fell=fell, x_end=round(float(d.qpos[0]), 2), mean_vx_after_100=round(float(np.mean(vx[100:])), 3)), flush=True)
        panels.append(frames)
    n = min(len(p) for p in panels); video = [np.concatenate([p[k] for p in panels], 1) for k in range(n)]
    stem = ROOT / f"runs/root/render_{arm}_{Path(pt).stem}"
    import subprocess
    hh, ww = video[0].shape[:2]
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{ww}x{hh}",
                           "-r", f"{1 / (4 * dt):.4f}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", f"{stem}.mp4"],
                          stdin=subprocess.PIPE)
    for f in video:
        ff.stdin.write(np.ascontiguousarray(f[:, :, :3]).tobytes())
    ff.stdin.close(); ff.wait()
    gif = [Image.fromarray(f).resize((f.shape[1] * 2 // 3, f.shape[0] * 2 // 3)) for f in video[::2]]
    gif[0].save(f"{stem}.gif", save_all=True, append_images=gif[1:], duration=round(2 * 4 * dt * 1000), loop=0)
    print(stem, "fps", round(1 / (4 * dt), 2), "frames", n, flush=True)


if __name__ == "__main__":
    main()
